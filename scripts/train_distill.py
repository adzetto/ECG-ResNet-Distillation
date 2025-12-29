import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecg_utils import compute_class_weights, split_by_subject_stratified


def augment_multilead(x):
    x = x.copy()
    if np.random.rand() < 0.5:
        x *= np.random.uniform(0.85, 1.15)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-x.shape[1] // 20, x.shape[1] // 20 + 1)
        x = np.roll(x, shift, axis=1)
    if np.random.rand() < 0.5:
        noise_std = np.random.uniform(0.01, 0.05)
        x += np.random.normal(0.0, noise_std, size=x.shape)
    if np.random.rand() < 0.4:
        amp = np.random.uniform(0.02, 0.1)
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, x.shape[1], endpoint=False)
        baseline = amp * np.sin(2 * np.pi * freq * t + phase)
        x += baseline[None, :]
    return x.astype(np.float32)


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam


class ECGWindowDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (channels, time), got {x.shape}")
        if self.augment:
            x = augment_multilead(x)
        y = int(self.y[idx])
        return torch.from_numpy(x), y


class ECGNet(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ECGResNet(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        if bottleneck > 0 and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            conv_in = bottleneck
        else:
            self.bottleneck = None
            conv_in = in_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(conv_in, out_channels, kernel_size=k, padding=k // 2, bias=False)
                for k in kernel_sizes
            ]
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(conv_in, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.pool_conv(self.pool(x)))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        return self.relu(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        self.module1 = InceptionModule(in_channels, out_channels, kernel_sizes, bottleneck)
        mid_channels = out_channels * (len(kernel_sizes) + 1)
        self.module2 = InceptionModule(mid_channels, out_channels, kernel_sizes, bottleneck)
        self.module3 = InceptionModule(mid_channels, out_channels, kernel_sizes, bottleneck)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.module1(x)
        out = self.module2(out)
        out = self.module3(out)
        res = self.residual(x)
        return self.relu(out + res)


class ECGInception(nn.Module):
    def __init__(self, num_classes, in_channels=1, blocks=2, out_channels=32):
        super().__init__()
        channels = in_channels
        stack = []
        for _ in range(blocks):
            block = InceptionBlock(channels, out_channels)
            stack.append(block)
            channels = out_channels * 4
        self.features = nn.Sequential(*stack)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(name, num_classes, in_channels=1):
    if name == "basic":
        return ECGNet(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet":
        return ECGResNet(num_classes=num_classes, in_channels=in_channels)
    if name == "inception":
        return ECGInception(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown model: {name}")


def resolve_label_names(npz_data, override):
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    if "label_names" in npz_data.files:
        names = npz_data["label_names"]
        if isinstance(names, np.ndarray):
            names = names.tolist()
        normalized = []
        for item in names:
            if isinstance(item, (bytes, np.bytes_)):
                normalized.append(item.decode("utf-8"))
            else:
                normalized.append(str(item))
        return normalized
    return []


def align_label_names(label_names, num_classes):
    names = list(label_names)
    if len(names) < num_classes:
        for idx in range(len(names), num_classes):
            names.append(f"class_{idx}")
    elif len(names) > num_classes:
        names = names[:num_classes]
    return names


def compute_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    for idx in range(num_classes):
        denom = cm[idx].sum()
        per_class_acc[idx] = (cm[idx, idx] / denom) if denom > 0 else 0.0
    overall = np.trace(cm) / np.maximum(cm.sum(), 1)
    return overall, per_class_acc, cm


def parse_args():
    parser = argparse.ArgumentParser(description="Train student model with distillation")
    parser.add_argument("--data", default="data/processed/ecg_windows_ptbxl_stage2_12lead.npz")
    parser.add_argument("--teacher", required=True, help="Teacher checkpoint path")
    parser.add_argument("--out-dir", default="models/student")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model", choices=["basic", "resnet", "inception"], default="resnet")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CE loss")
    parser.add_argument("--lead-name", default="II")
    parser.add_argument("--lead-index", type=int, default=-1)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Learning rate schedule",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--label-names",
        default="",
        help="Comma-separated label names to override dataset labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.data, mmap_mode="r", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n, channels, time), got {X.shape}")

    num_classes = int(np.max(y)) + 1 if y.size else 0
    label_names = resolve_label_names(data, args.label_names)
    label_names = align_label_names(label_names, num_classes)

    lead_names = []
    if "lead_names" in data.files:
        names = data["lead_names"]
        if isinstance(names, np.ndarray):
            names = names.tolist()
        lead_names = [n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n) for n in names]

    if args.lead_index >= 0:
        lead_index = args.lead_index
    elif lead_names:
        upper = [name.upper() for name in lead_names]
        target = args.lead_name.upper()
        if target not in upper:
            raise SystemExit(f"Lead {args.lead_name} not found in {lead_names}")
        lead_index = upper.index(target)
    else:
        lead_index = 1

    train_mask, val_mask, test_mask = split_by_subject_stratified(
        subjects,
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_path = out_dir / "split_subjects.json"
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_subjects": sorted(np.unique(subjects[train_mask]).tolist()),
                "val_subjects": sorted(np.unique(subjects[val_mask]).tolist()),
                "test_subjects": sorted(np.unique(subjects[test_mask]).tolist()),
                "label_names": label_names,
                "lead_index": lead_index,
                "lead_name": args.lead_name,
            },
            handle,
            indent=2,
        )

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    train_ds = ECGWindowDataset(X_train, y_train, augment=args.augment)
    val_ds = ECGWindowDataset(X_val, y_val, augment=False)

    class_weights = compute_class_weights(y_train, num_classes)
    if args.balanced_sampler:
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False
        )
        ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).float().to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
            label_smoothing=args.label_smoothing,
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_ckpt = torch.load(args.teacher, map_location=device)
    teacher_model_name = teacher_ckpt.get("model_name", "resnet")
    teacher_in_channels = teacher_ckpt.get("in_channels", X.shape[1])
    teacher = build_model(teacher_model_name, num_classes, in_channels=teacher_in_channels).to(device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()

    student = build_model(args.model, num_classes, in_channels=1).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    kldiv = nn.KLDivLoss(reduction="batchmean")
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )

    best_val = 0.0
    patience = 0
    best_path = out_dir / "ecg_student.pt"

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            if args.mixup > 0.0 and batch_x.size(0) > 1:
                mixed_x, y_a, y_b, lam = mixup_batch(batch_x, batch_y, args.mixup)
                with torch.no_grad():
                    teacher_logits = teacher(mixed_x)
                student_in = mixed_x[:, lead_index : lead_index + 1, :]
                student_logits = student(student_in)
                loss_ce = lam * ce_loss(student_logits, y_a) + (1.0 - lam) * ce_loss(
                    student_logits, y_b
                )
            else:
                with torch.no_grad():
                    teacher_logits = teacher(batch_x)
                student_in = batch_x[:, lead_index : lead_index + 1, :]
                student_logits = student(student_in)
                loss_ce = ce_loss(student_logits, batch_y)
            t = args.temperature
            loss_kd = kldiv(
                torch.log_softmax(student_logits / t, dim=1),
                torch.softmax(teacher_logits / t, dim=1),
            ) * (t * t)
            loss = args.alpha * loss_ce + (1.0 - args.alpha) * loss_kd

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        student.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                student_in = batch_x[:, lead_index : lead_index + 1, :]
                logits = student(student_in)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(batch_y.numpy().tolist())

        overall, per_class_acc, _ = compute_metrics(
            np.array(y_true), np.array(y_pred), num_classes
        )
        balanced = float(np.mean(per_class_acc))
        mean_loss = sum(train_losses) / max(len(train_losses), 1)
        print(
            f"Epoch {epoch:02d} | loss={mean_loss:.4f} | val_acc={overall:.4f} | "
            f"bal_acc={balanced:.4f} | per_class={per_class_acc.round(4).tolist()}"
        )

        if balanced > best_val:
            best_val = balanced
            patience = 0
            torch.save(
                {
                    "model": student.state_dict(),
                    "label_names": label_names,
                    "model_name": args.model,
                    "in_channels": 1,
                    "lead_index": lead_index,
                    "lead_name": args.lead_name,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping")
                break
        if scheduler is not None:
            scheduler.step()

    print(f"Best val balanced accuracy: {best_val:.4f}")
    print(f"Student model saved to {best_path}")


if __name__ == "__main__":
    main()
