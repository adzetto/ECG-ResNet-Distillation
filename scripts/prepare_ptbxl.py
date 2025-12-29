import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import wfdb
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecg_utils import LABEL_TO_ID, bandpass_filter, iter_windows, pick_lead, resample_signal, standardize

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required for PTB-XL prep. Install with: pip install pandas") from exc


def load_mappings(ptbxl_dir):
    scp_path = Path(ptbxl_dir) / "scp_statements.csv"
    scp = pd.read_csv(scp_path, index_col=0)
    diag_map = {}
    rhythm_codes = set()
    for code, row in scp.iterrows():
        diagnostic = row.get("diagnostic", 0)
        rhythm = row.get("rhythm", 0)
        diagnostic = 0 if pd.isna(diagnostic) else int(diagnostic)
        rhythm = 0 if pd.isna(rhythm) else int(rhythm)
        if diagnostic == 1:
            diag_map[code] = str(row.get("diagnostic_class", "")).strip()
        if rhythm == 1:
            rhythm_codes.add(code)
    return diag_map, rhythm_codes


def parse_scp_codes(raw):
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    return ast.literal_eval(raw)


def assign_label(diag_classes, rhythm_codes, mode):
    diag_classes = {cls for cls in diag_classes if cls}
    rhythm_codes = {code for code in rhythm_codes if code}

    has_mi = "MI" in diag_classes
    has_norm = "NORM" in diag_classes
    has_non_sr = any(code != "SR" for code in rhythm_codes)

    if mode == "loose":
        if has_mi:
            return "MI"
        if has_non_sr:
            return "ARRHYTHMIA"
        if has_norm and all(cls == "NORM" for cls in diag_classes):
            return "NORMAL"
        return None

    if mode == "strict":
        if diag_classes == {"MI"} and not has_non_sr:
            return "MI"
        if diag_classes == {"NORM"} and not has_non_sr:
            return "NORMAL"
        if has_non_sr and diag_classes == {"NORM"}:
            return "ARRHYTHMIA"
        return None

    if mode == "semi":
        if diag_classes.issubset({"MI"}) and has_mi and not has_non_sr:
            return "MI"
        if diag_classes == {"NORM"} and not has_non_sr:
            return "NORMAL"
        if has_non_sr and diag_classes.issubset({"NORM"}):
            return "ARRHYTHMIA"
        return None

    raise ValueError(f"Unknown label mode: {mode}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PTB-XL ECG windows")
    parser.add_argument("--ptbxl-dir", default="data/raw/ptbxl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--out-name", default="ecg_windows_ptbxl.npz")
    parser.add_argument("--use-hr", action="store_true", help="Use 500 Hz records")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--lead", default="II")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=["loose", "semi", "strict"],
        default="loose",
        help="Label purity: loose (default), semi, strict",
    )
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ptbxl_dir = Path(args.ptbxl_dir)
    db_path = ptbxl_dir / "ptbxl_database.csv"
    if not db_path.exists():
        raise SystemExit(f"Missing {db_path}. Place PTB-XL files under {ptbxl_dir}.")

    diag_map, rhythm_codes = load_mappings(ptbxl_dir)
    db = pd.read_csv(db_path)

    windows = []
    labels = []
    subjects = []

    window_size = int(args.window_sec * args.sample_rate)
    stride = int(args.stride_sec * args.sample_rate)

    rows = db.itertuples(index=False)
    if args.max_records and args.max_records > 0:
        rows = list(rows)[: args.max_records]

    for row in tqdm(rows, desc="ptbxl", unit="record"):
        scp_codes = parse_scp_codes(getattr(row, "scp_codes"))
        diag_classes = {diag_map[c] for c in scp_codes if c in diag_map}
        rhythm_in_record = {c for c in scp_codes if c in rhythm_codes}
        label = assign_label(diag_classes, rhythm_in_record, args.label_mode)
        if label is None:
            continue

        filename = getattr(row, "filename_hr" if args.use_hr else "filename_lr")
        record_path = ptbxl_dir / filename
        record = wfdb.rdrecord(str(record_path))
        lead_idx = pick_lead(record.sig_name, [args.lead])
        signal = record.p_signal[:, lead_idx].astype(np.float32)

        fs_in = float(record.fs)
        resampled = resample_signal(signal, fs_in, float(args.sample_rate))
        resampled = bandpass_filter(resampled, float(args.sample_rate))

        for _, _, segment in iter_windows(resampled, window_size, stride):
            if len(segment) != window_size:
                continue
            windows.append(standardize(segment))
            labels.append(LABEL_TO_ID[label])
            subjects.append(getattr(row, "patient_id"))

    if not windows:
        raise SystemExit("No windows collected. Check PTB-XL path and label rules.")

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject_arr = np.asarray(subjects)

    class_counts = np.bincount(y, minlength=len(LABEL_TO_ID))
    pre_balance_map = {
        name: int(class_counts[idx]) for name, idx in LABEL_TO_ID.items()
    }
    print(f"Pre-balance class counts: {pre_balance_map}")

    if args.balance:
        if any(class_counts[idx] == 0 for idx in range(len(LABEL_TO_ID))):
            raise SystemExit(
                f"Cannot balance; missing class in PTB-XL labels: {pre_balance_map}"
            )
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
    np.savez_compressed(out_path, X=X, y=y, subjects=subject_arr)

    class_counts = np.bincount(y, minlength=len(LABEL_TO_ID))
    final_map = {name: int(class_counts[idx]) for name, idx in LABEL_TO_ID.items()}
    print(f"Saved {len(y)} windows to {out_path}")
    print(f"Class counts: {final_map}")


if __name__ == "__main__":
    main()
