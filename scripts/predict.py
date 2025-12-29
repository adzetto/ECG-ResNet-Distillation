import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecg_utils import bandpass_filter, iter_windows, resample_signal, standardize


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


def load_signal(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.loadtxt(path, dtype=np.float32, delimiter=",")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECG inference on a single-lead signal")
    parser.add_argument("--model", default="models/ecg_cnn.pt")
    parser.add_argument("--input", required=True, help="CSV or NPY file with signal samples")
    parser.add_argument("--input-rate", type=int, required=True, help="Input sampling rate (Hz)")
    parser.add_argument("--model-rate", type=int, default=250, help="Model sampling rate (Hz)")
    parser.add_argument("--window-sec", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    signal = load_signal(args.input).astype(np.float32)

    resampled = resample_signal(signal, float(args.input_rate), float(args.model_rate))
    resampled = bandpass_filter(resampled, float(args.model_rate))

    window_size = int(args.window_sec * args.model_rate)
    stride = window_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    label_names = checkpoint.get("label_names") or []
    if isinstance(label_names, np.ndarray):
        label_names = label_names.tolist()
    normalized = []
    for item in label_names:
        if isinstance(item, (bytes, np.bytes_)):
            normalized.append(item.decode("utf-8"))
        else:
            normalized.append(str(item))
    label_names = normalized
    num_classes = len(label_names) if label_names else 3
    model_name = checkpoint.get("model_name", "basic")
    in_channels = int(checkpoint.get("in_channels", 1))
    if in_channels != 1:
        raise SystemExit(
            f"Model expects {in_channels} channels; predict.py only supports single-lead input."
        )

    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for _, _, segment in iter_windows(resampled, window_size, stride):
            segment = standardize(segment)
            x = (
                torch.from_numpy(segment.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).cpu().item())
            preds.append(pred)

    if not preds:
        print("No windows available for prediction.")
        return

    votes = np.bincount(np.array(preds), minlength=num_classes)
    final_idx = int(np.argmax(votes))
    final_label = label_names[final_idx] if final_idx < len(label_names) else str(final_idx)
    print(f"Prediction: {final_label}")
    print(f"Vote counts: {votes.tolist()}")


if __name__ == "__main__":
    main()
