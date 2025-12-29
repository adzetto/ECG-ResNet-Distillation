import argparse
import sys
from pathlib import Path

import numpy as np
import wfdb
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecg_utils import (
    LABEL_TO_ID,
    bandpass_filter,
    is_arrhythmia_symbol,
    is_beat_symbol,
    pick_lead,
    resample_positions,
    resample_signal,
    standardize,
    iter_windows,
)

DEFAULT_LEADS = ["MLII", "II", "V1", "V2", "I"]


def infer_ptb_label(comments):
    text = " ".join(comments).lower()
    if "myocardial infarction" in text:
        return "MI"
    if "healthy control" in text or "healthy" in text and "control" in text:
        return "NORMAL"
    return None


def load_mitdb(
    raw_dir,
    fs_out,
    window_size,
    stride,
    max_windows,
    preferred_leads,
    min_arrhythmia_count,
    min_arrhythmia_ratio,
):
    base = Path(raw_dir) / "mitdb"
    records = wfdb.get_record_list("mitdb")
    windows = []
    labels = []
    subjects = []
    for record_name in tqdm(records, desc="mitdb", unit="record"):
        record_path = base / record_name
        record = wfdb.rdrecord(str(record_path))
        ann = wfdb.rdann(str(record_path), "atr")
        lead_idx = pick_lead(record.sig_name, preferred_leads)
        signal = record.p_signal[:, lead_idx].astype(np.float32)
        fs_in = float(record.fs)
        resampled = resample_signal(signal, fs_in, fs_out)
        resampled = bandpass_filter(resampled, fs_out)
        ann_samples = np.asarray(ann.sample)
        arr_mask = np.array([is_arrhythmia_symbol(sym) for sym in ann.symbol])
        beat_mask = np.array([is_beat_symbol(sym) for sym in ann.symbol])
        arr_positions = resample_positions(ann_samples[arr_mask], fs_in, fs_out)
        beat_positions = resample_positions(ann_samples[beat_mask], fs_in, fs_out)
        arr_positions.sort()
        beat_positions.sort()
        if beat_positions.size == 0:
            continue

        def count_in_window(pos_arr, start, end):
            left = np.searchsorted(pos_arr, start, side="left")
            right = np.searchsorted(pos_arr, end, side="left")
            return right - left

        candidate_windows = []
        for start, end, segment in iter_windows(resampled, window_size, stride):
            arr_count = count_in_window(arr_positions, start, end)
            if arr_count == 0:
                continue
            beat_count = count_in_window(beat_positions, start, end)
            if beat_count == 0:
                continue
            arr_ratio = arr_count / beat_count
            if arr_count >= min_arrhythmia_count and arr_ratio >= min_arrhythmia_ratio:
                candidate_windows.append((start, end, segment))
        if max_windows and len(candidate_windows) > max_windows:
            rng = np.random.default_rng(13)
            rng.shuffle(candidate_windows)
            candidate_windows = candidate_windows[:max_windows]
        for _, _, segment in candidate_windows:
            windows.append(standardize(segment))
            labels.append(LABEL_TO_ID["ARRHYTHMIA"])
            subjects.append(record_name)
    return windows, labels, subjects


def load_nsrdb(raw_dir, fs_out, window_size, stride, max_windows, preferred_leads):
    base = Path(raw_dir) / "nsrdb"
    records = wfdb.get_record_list("nsrdb")
    windows = []
    labels = []
    subjects = []
    for record_name in tqdm(records, desc="nsrdb", unit="record"):
        record_path = base / record_name
        record = wfdb.rdrecord(str(record_path))
        lead_idx = pick_lead(record.sig_name, preferred_leads)
        signal = record.p_signal[:, lead_idx].astype(np.float32)
        fs_in = float(record.fs)
        resampled = resample_signal(signal, fs_in, fs_out)
        resampled = bandpass_filter(resampled, fs_out)
        candidate_windows = list(iter_windows(resampled, window_size, stride))
        if max_windows and len(candidate_windows) > max_windows:
            rng = np.random.default_rng(13)
            rng.shuffle(candidate_windows)
            candidate_windows = candidate_windows[:max_windows]
        for _, _, segment in candidate_windows:
            windows.append(standardize(segment))
            labels.append(LABEL_TO_ID["NORMAL"])
            subjects.append(record_name)
    return windows, labels, subjects


def load_ptbdb(raw_dir, fs_out, window_size, stride, max_windows, preferred_leads):
    base = Path(raw_dir) / "ptbdb"
    records = wfdb.get_record_list("ptbdb")
    windows = []
    labels = []
    subjects = []
    for record_name in tqdm(records, desc="ptbdb", unit="record"):
        record_path = base / record_name
        record = wfdb.rdrecord(str(record_path))
        label = infer_ptb_label(record.comments)
        if label is None:
            continue
        lead_idx = pick_lead(record.sig_name, preferred_leads)
        signal = record.p_signal[:, lead_idx].astype(np.float32)
        fs_in = float(record.fs)
        resampled = resample_signal(signal, fs_in, fs_out)
        resampled = bandpass_filter(resampled, fs_out)
        candidate_windows = list(iter_windows(resampled, window_size, stride))
        if max_windows and len(candidate_windows) > max_windows:
            rng = np.random.default_rng(13)
            rng.shuffle(candidate_windows)
            candidate_windows = candidate_windows[:max_windows]
        for _, _, segment in candidate_windows:
            windows.append(standardize(segment))
            labels.append(LABEL_TO_ID[label])
            subjects.append(record_name.split("/")[0])
    return windows, labels, subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ECG dataset windows")
    parser.add_argument("--raw-dir", default="data/raw", help="Input data dir")
    parser.add_argument("--out-dir", default="data/processed", help="Output dir")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--max-windows-per-record", type=int, default=600)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--arrhythmia-min-count", type=int, default=1)
    parser.add_argument("--arrhythmia-min-ratio", type=float, default=0.0)
    parser.add_argument(
        "--leads",
        nargs="+",
        default=DEFAULT_LEADS,
        help="Preferred leads in order (e.g., MLII II V1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_out = float(args.sample_rate)
    window_size = int(args.window_sec * fs_out)
    stride = int(args.stride_sec * fs_out)

    windows = []
    labels = []
    subjects = []

    mit_w, mit_y, mit_s = load_mitdb(
        args.raw_dir,
        fs_out,
        window_size,
        stride,
        args.max_windows_per_record,
        args.leads,
        args.arrhythmia_min_count,
        args.arrhythmia_min_ratio,
    )
    windows.extend(mit_w)
    labels.extend(mit_y)
    subjects.extend(mit_s)

    nsr_w, nsr_y, nsr_s = load_nsrdb(
        args.raw_dir,
        fs_out,
        window_size,
        stride,
        args.max_windows_per_record,
        args.leads,
    )
    windows.extend(nsr_w)
    labels.extend(nsr_y)
    subjects.extend(nsr_s)

    ptb_w, ptb_y, ptb_s = load_ptbdb(
        args.raw_dir,
        fs_out,
        window_size,
        stride,
        args.max_windows_per_record,
        args.leads,
    )
    windows.extend(ptb_w)
    labels.extend(ptb_y)
    subjects.extend(ptb_s)

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject_arr = np.asarray(subjects)

    if args.balance:
        rng = np.random.default_rng(13)
        idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
        min_count = min(len(v) for v in idx_by_class.values())
        selected = []
        for cls, idxs in idx_by_class.items():
            rng.shuffle(idxs)
            selected.append(idxs[:min_count])
        selected = np.concatenate(selected)
        rng.shuffle(selected)
        X = X[selected]
        y = y[selected]
        subject_arr = subject_arr[selected]

    out_path = out_dir / "ecg_windows.npz"
    np.savez_compressed(out_path, X=X, y=y, subjects=subject_arr)
    class_counts = np.bincount(y, minlength=len(LABEL_TO_ID))
    count_map = {name: int(class_counts[idx]) for name, idx in LABEL_TO_ID.items()}
    print(f"Saved {len(y)} windows to {out_path}")
    print(f"Class counts: {count_map}")


if __name__ == "__main__":
    main()
