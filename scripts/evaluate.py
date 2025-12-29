import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecg_utils import LABELS, split_by_subject_stratified


class ECGWindowDataset(Dataset):
    def __init__(self, X, y, subjects=None):
        self.X = X
        self.y = y
        self.subjects = subjects

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"Unexpected sample shape: {x.shape}")
        x = torch.from_numpy(x)
        y = int(self.y[idx])
        if self.subjects is None:
            return x, y
        return x, y, self.subjects[idx]


class ECGNet(torch.nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_ch, out_ch, kernel_size=7, stride=stride, padding=3
        )
        self.bn1 = torch.nn.BatchNorm1d(out_ch)
        self.conv2 = torch.nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3)
        self.bn2 = torch.nn.BatchNorm1d(out_ch)
        self.relu = torch.nn.ReLU()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                torch.nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ECGResNet(torch.nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        if bottleneck > 0 and in_channels > 1:
            self.bottleneck = torch.nn.Conv1d(
                in_channels, bottleneck, kernel_size=1, bias=False
            )
            conv_in = bottleneck
        else:
            self.bottleneck = None
            conv_in = in_channels
        self.branches = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    conv_in, out_channels, kernel_size=k, padding=k // 2, bias=False
                )
                for k in kernel_sizes
            ]
        )
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = torch.nn.Conv1d(conv_in, out_channels, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.pool_conv(self.pool(x)))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        return self.relu(x)


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        self.module1 = InceptionModule(in_channels, out_channels, kernel_sizes, bottleneck)
        mid_channels = out_channels * (len(kernel_sizes) + 1)
        self.module2 = InceptionModule(mid_channels, out_channels, kernel_sizes, bottleneck)
        self.module3 = InceptionModule(mid_channels, out_channels, kernel_sizes, bottleneck)
        self.residual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(mid_channels),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.module1(x)
        out = self.module2(out)
        out = self.module3(out)
        res = self.residual(x)
        return self.relu(out + res)


class ECGInception(torch.nn.Module):
    def __init__(self, num_classes, in_channels=1, blocks=2, out_channels=32):
        super().__init__()
        channels = in_channels
        stack = []
        for _ in range(blocks):
            block = InceptionBlock(channels, out_channels)
            stack.append(block)
            channels = out_channels * 4
        self.features = torch.nn.Sequential(*stack)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(channels, num_classes)

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


def load_split_subjects(split_path, subjects):
    if not split_path.exists():
        return None
    with split_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    test_subjects = set(data.get("test_subjects", []))
    if not test_subjects:
        return None
    return np.isin(subjects, list(test_subjects))


def normalize_label(item):
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8")
    return str(item)


def resolve_label_names(npz_data, checkpoint, override):
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    if checkpoint and "label_names" in checkpoint:
        names = checkpoint["label_names"]
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return [normalize_label(item) for item in names]
    if "label_names" in npz_data.files:
        names = npz_data["label_names"]
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return [normalize_label(item) for item in names]
    return list(LABELS)


def align_label_names(label_names, num_classes):
    names = list(label_names)
    if len(names) < num_classes:
        for idx in range(len(names), num_classes):
            names.append(f"class_{idx}")
    elif len(names) > num_classes:
        names = names[:num_classes]
    return names


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ECG classifier")
    parser.add_argument("--data", default="data/processed/ecg_windows.npz")
    parser.add_argument("--model", default="models/ecg_cnn.pt")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--by-subject",
        action="store_true",
        help="Report subject-level majority-vote accuracy in addition to window-level",
    )
    parser.add_argument(
        "--label-names",
        default="",
        help="Comma-separated label names to override dataset labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data, mmap_mode="r", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    split_path = Path(args.out_dir) / "split_subjects.json"
    test_mask = load_split_subjects(split_path, subjects)
    if test_mask is None:
        _, _, test_mask = split_by_subject_stratified(subjects, y, seed=args.seed)

    X_test = X[test_mask]
    y_test = y[test_mask]

    subject_ids = subjects[test_mask]
    dataset = ECGWindowDataset(X_test, y_test, subjects=subject_ids if args.by_subject else None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    model_name = checkpoint.get("model_name", "basic")

    num_classes = int(np.max(y)) + 1 if y.size else 0
    label_names = resolve_label_names(data, checkpoint, args.label_names)
    label_names = align_label_names(label_names, num_classes)
    if X.ndim == 2:
        in_channels = 1
    elif X.ndim == 3:
        in_channels = X.shape[1]
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    y_true = []
    y_pred = []
    probs = []
    subject_list = []
    with torch.no_grad():
        for batch in loader:
            if args.by_subject:
                batch_x, batch_y, batch_subjects = batch
                subject_list.extend(
                    batch_subjects.cpu().numpy().tolist()
                    if isinstance(batch_subjects, torch.Tensor)
                    else list(batch_subjects)
                )
            else:
                batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(batch_probs, axis=1)
            y_pred.extend(preds.tolist())
            y_true.extend(batch_y.numpy().tolist())
            if args.by_subject:
                probs.extend(batch_probs.tolist())

    overall, per_class_acc, cm = compute_metrics(
        np.array(y_true), np.array(y_pred), num_classes
    )

    print(f"Test accuracy: {overall:.4f}")
    for idx, acc in enumerate(per_class_acc):
        name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
        print(f"{name} accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)

    if args.by_subject:
        subj_pred = {}
        subj_prob = {}
        subj_true = {}
        for subj, true_label, prob in zip(subject_list, y_true, probs):
            if subj not in subj_prob:
                subj_prob[subj] = np.zeros(num_classes, dtype=np.float64)
                subj_true[subj] = np.zeros(num_classes, dtype=np.int64)
            subj_prob[subj] += np.asarray(prob, dtype=np.float64)
            subj_true[subj][true_label] += 1
        subj_true_labels = []
        subj_pred_labels = []
        for subj, summed in subj_prob.items():
            subj_pred_labels.append(int(np.argmax(summed)))
            subj_true_labels.append(int(np.argmax(subj_true[subj])))
        overall_s, per_class_s, cm_s = compute_metrics(
            np.array(subj_true_labels), np.array(subj_pred_labels), num_classes
        )
        print("Subject-level accuracy:")
        print(f"Test accuracy: {overall_s:.4f}")
        for idx, acc in enumerate(per_class_s):
            name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
            print(f"{name} accuracy: {acc:.4f}")
        print("Subject-level confusion matrix:")
        print(cm_s)


if __name__ == "__main__":
    main()
