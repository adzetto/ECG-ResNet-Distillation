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
    bandpass_filter,
    is_arrhythmia_symbol,
    is_beat_symbol,
    iter_windows,
    pick_lead,
    resample_positions,
    resample_signal,
    standardize,
)


LABEL_NAMES = ["SINUS", "ARRHYTHMIA"]
DEFAULT_LEADS = ["MLII", "II", "V1", "V2", "I"]
DEFAULT_ARR_DB = ["mitdb", "afdb", "incartdb", "cudb", "sddb"]
DEFAULT_NORM_DB = ["nsrdb"]


def list_record_names(base_dir):
    records = []
    for hea in sorted(base_dir.rglob("*.hea")):
        rel = hea.relative_to(base_dir)
        records.append(str(rel.with_suffix("")))
    return records


def load_arrhythmia_db(
    db_name,
    base_dir,
    fs_out,
    window_size,
    stride,
    max_windows,
    preferred_leads,
    min_arrhythmia_count,
    min_arrhythmia_ratio,
    ann_ext="atr",
):
    windows = []
    labels = []
    subjects = []
    records = list_record_names(base_dir)
    if not records:
        return windows, labels, subjects

    for record_name in tqdm(records, desc=db_name, unit="record"):
        record_path = base_dir / record_name
        try:
            record = wfdb.rdrecord(str(record_path))
        except Exception:
            continue
        try:
            ann = wfdb.rdann(str(record_path), ann_ext)
        except Exception:
            continue

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
                candidate_windows.append(segment)

        if max_windows and len(candidate_windows) > max_windows:
            rng = np.random.default_rng(13)
            rng.shuffle(candidate_windows)
            candidate_windows = candidate_windows[:max_windows]

        for segment in candidate_windows:
            windows.append(standardize(segment))
            labels.append(1)
            subjects.append(f"{db_name}:{record_name}")

    return windows, labels, subjects


def load_normal_db(
    db_name,
    base_dir,
    fs_out,
    window_size,
    stride,
    max_windows,
    preferred_leads,
):
    windows = []
    labels = []
    subjects = []
    records = list_record_names(base_dir)
    if not records:
        return windows, labels, subjects

    for record_name in tqdm(records, desc=db_name, unit="record"):
        record_path = base_dir / record_name
        try:
            record = wfdb.rdrecord(str(record_path))
        except Exception:
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
            labels.append(0)
            subjects.append(f"{db_name}:{record_name}")

    return windows, labels, subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare multi-db stage1 dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--out-name", default="ecg_windows_stage1_multi.npz")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=5)
    parser.add_argument("--max-windows-per-record", type=int, default=600)
    parser.add_argument("--min-arrhythmia-count", type=int, default=2)
    parser.add_argument("--min-arrhythmia-ratio", type=float, default=0.1)
    parser.add_argument("--leads", nargs="+", default=DEFAULT_LEADS)
    parser.add_argument("--arrhythmia-dbs", nargs="+", default=DEFAULT_ARR_DB)
    parser.add_argument("--normal-dbs", nargs="+", default=DEFAULT_NORM_DB)
    parser.add_argument("--balance", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fs_out = float(args.sample_rate)
    window_size = int(args.window_sec * fs_out)
    stride = int(args.stride_sec * fs_out)

    windows = []
    labels = []
    subjects = []

    for db in args.arrhythmia_dbs:
        base = Path(args.raw_dir) / db
        if not base.exists():
            print(f"Skipping missing arrhythmia dataset: {db}")
            continue
        w, y, s = load_arrhythmia_db(
            db,
            base,
            fs_out,
            window_size,
            stride,
            args.max_windows_per_record,
            args.leads,
            args.min_arrhythmia_count,
            args.min_arrhythmia_ratio,
        )
        windows.extend(w)
        labels.extend(y)
        subjects.extend(s)

    for db in args.normal_dbs:
        base = Path(args.raw_dir) / db
        if not base.exists():
            print(f"Skipping missing normal dataset: {db}")
            continue
        w, y, s = load_normal_db(
            db,
            base,
            fs_out,
            window_size,
            stride,
            args.max_windows_per_record,
            args.leads,
        )
        windows.extend(w)
        labels.extend(y)
        subjects.extend(s)

    if not windows:
        raise SystemExit("No windows collected. Check dataset paths.")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    label_arr = np.asarray(LABEL_NAMES, dtype=str)
    np.savez_compressed(out_path, X=X, y=y, subjects=subject_arr, label_names=label_arr)
    counts = np.bincount(y, minlength=len(LABEL_NAMES))
    count_map = {LABEL_NAMES[i]: int(counts[i]) for i in range(len(LABEL_NAMES))}
    print(f"Saved {len(y)} windows to {out_path}")
    print(f"Class counts: {count_map}")


if __name__ == "__main__":
    main()
