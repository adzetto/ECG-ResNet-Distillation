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

from ecg_utils import bandpass_filter, iter_windows, resample_signal, standardize

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required. Install with: pip install pandas") from exc

LABEL_NAMES = ["NORMAL", "MI"]


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


def classify_stage2(diag_classes, rhythm_in_record):
    if any(code != "SR" for code in rhythm_in_record):
        return None
    if diag_classes == {"MI"}:
        return 1
    if diag_classes == {"NORM"}:
        return 0
    return None


def balance_dataset(X, y, subjects, seed=13):
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=len(LABEL_NAMES))
    count_map = {LABEL_NAMES[i]: int(counts[i]) for i in range(len(LABEL_NAMES))}
    print(f"Pre-balance class counts: {count_map}")
    if any(counts[i] == 0 for i in range(len(LABEL_NAMES))):
        raise SystemExit(f"Cannot balance; missing class: {count_map}")
    rng = np.random.default_rng(seed)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    min_count = min(len(v) for v in idx_by_class.values())
    selected = []
    for cls, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        selected.append(idxs[:min_count])
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return X[selected], y[selected], subjects[selected]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PTB-XL stage2 multi-lead dataset")
    parser.add_argument("--ptbxl-dir", default="data/raw/ptbxl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--out-name", default="ecg_windows_ptbxl_stage2_12lead.npz")
    parser.add_argument("--use-hr", action="store_true", help="Use 500 Hz records")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--balance", action="store_true")
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
    lead_names = None

    window_size = int(args.window_sec * args.sample_rate)
    stride = int(args.stride_sec * args.sample_rate)

    rows = db.itertuples(index=False)
    if args.max_records and args.max_records > 0:
        rows = list(rows)[: args.max_records]

    for row in tqdm(rows, desc="ptbxl", unit="record"):
        scp_codes = parse_scp_codes(getattr(row, "scp_codes"))
        diag_classes = {diag_map[c] for c in scp_codes if c in diag_map and diag_map[c]}
        rhythm_in = {c for c in scp_codes if c in rhythm_codes}
        label = classify_stage2(diag_classes, rhythm_in)
        if label is None:
            continue

        filename = getattr(row, "filename_hr" if args.use_hr else "filename_lr")
        record_path = ptbxl_dir / filename
        record = wfdb.rdrecord(str(record_path))
        if lead_names is None:
            lead_names = list(record.sig_name)
            lead_index_map = None
        else:
            if list(record.sig_name) != lead_names:
                lead_index_map = [record.sig_name.index(name) for name in lead_names]
            else:
                lead_index_map = None

        signal = record.p_signal
        if lead_index_map is not None:
            signal = signal[:, lead_index_map]

        fs_in = float(record.fs)
        filtered = []
        for lead_idx in range(signal.shape[1]):
            lead_sig = signal[:, lead_idx].astype(np.float32)
            resampled = resample_signal(lead_sig, fs_in, float(args.sample_rate))
            resampled = bandpass_filter(resampled, float(args.sample_rate))
            filtered.append(standardize(resampled))
        resampled = np.stack(filtered, axis=0)

        for start, end, _ in iter_windows(resampled[0], window_size, stride):
            segment = resampled[:, start:end]
            if segment.shape[1] != window_size:
                continue
            windows.append(segment)
            labels.append(label)
            subjects.append(getattr(row, "patient_id"))

    if not windows:
        raise SystemExit("No windows collected. Check PTB-XL path and label rules.")

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject_arr = np.asarray(subjects)

    if args.balance:
        X, y, subject_arr = balance_dataset(X, y, subject_arr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    label_arr = np.asarray(LABEL_NAMES, dtype=str)
    lead_arr = np.asarray(lead_names or [], dtype=str)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        subjects=subject_arr,
        label_names=label_arr,
        lead_names=lead_arr,
    )

    counts = np.bincount(y, minlength=len(LABEL_NAMES))
    count_map = {LABEL_NAMES[i]: int(counts[i]) for i in range(len(LABEL_NAMES))}
    print(f"Saved {len(y)} windows to {out_path}")
    print(f"Class counts: {count_map}")


if __name__ == "__main__":
    main()
