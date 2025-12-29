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

from ecg_utils import bandpass_filter, iter_windows, pick_lead, resample_signal, standardize

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required. Install with: pip install pandas") from exc


STAGE1_LABELS = ["SINUS", "ARRHYTHMIA"]
STAGE2_LABELS = ["NORMAL", "MI"]


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


def classify_stage1(rhythm_in_record, assume_sr_if_missing):
    if any(code != "SR" for code in rhythm_in_record):
        return 1
    if "SR" in rhythm_in_record:
        return 0
    return 0 if assume_sr_if_missing else None


def classify_stage2(diag_classes, rhythm_in_record):
    if any(code != "SR" for code in rhythm_in_record):
        return None
    if diag_classes == {"MI"}:
        return 1
    if diag_classes == {"NORM"}:
        return 0
    return None


def balance_dataset(X, y, subjects, label_names, seed=13):
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=len(label_names))
    count_map = {label_names[i]: int(counts[i]) for i in range(len(label_names))}
    print(f"Pre-balance class counts: {count_map}")
    if any(counts[i] == 0 for i in range(len(label_names))):
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


def save_npz(out_path, X, y, subjects, label_names):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_arr = np.asarray(label_names, dtype=str)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        subjects=subjects,
        label_names=label_arr,
    )
    counts = np.bincount(y, minlength=len(label_names))
    count_map = {label_names[i]: int(counts[i]) for i in range(len(label_names))}
    print(f"Saved {len(y)} windows to {out_path}")
    print(f"Class counts: {count_map}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PTB-XL two-stage datasets")
    parser.add_argument("--ptbxl-dir", default="data/raw/ptbxl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--out-stage1", default="ecg_windows_ptbxl_stage1.npz")
    parser.add_argument("--out-stage2", default="ecg_windows_ptbxl_stage2.npz")
    parser.add_argument("--use-hr", action="store_true", help="Use 500 Hz records")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--lead", default="II")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument(
        "--strict-rhythm",
        action="store_true",
        help="Require explicit SR code for sinus rhythm class",
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

    stage1_windows = []
    stage1_labels = []
    stage1_subjects = []

    stage2_windows = []
    stage2_labels = []
    stage2_subjects = []

    window_size = int(args.window_sec * args.sample_rate)
    stride = int(args.stride_sec * args.sample_rate)

    rows = db.itertuples(index=False)
    if args.max_records and args.max_records > 0:
        rows = list(rows)[: args.max_records]

    for row in tqdm(rows, desc="ptbxl", unit="record"):
        scp_codes = parse_scp_codes(getattr(row, "scp_codes"))
        diag_classes = {diag_map[c] for c in scp_codes if c in diag_map and diag_map[c]}
        rhythm_in = {c for c in scp_codes if c in rhythm_codes}

        label_stage1 = classify_stage1(rhythm_in, not args.strict_rhythm)
        label_stage2 = classify_stage2(diag_classes, rhythm_in)

        if label_stage1 is None and label_stage2 is None:
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
            segment = standardize(segment)
            if label_stage1 is not None:
                stage1_windows.append(segment)
                stage1_labels.append(label_stage1)
                stage1_subjects.append(getattr(row, "patient_id"))
            if label_stage2 is not None:
                stage2_windows.append(segment)
                stage2_labels.append(label_stage2)
                stage2_subjects.append(getattr(row, "patient_id"))

    if not stage1_windows or not stage2_windows:
        raise SystemExit("No windows collected for one of the stages.")

    stage1_X = np.stack(stage1_windows).astype(np.float32)
    stage1_y = np.asarray(stage1_labels, dtype=np.int64)
    stage1_subjects = np.asarray(stage1_subjects)

    stage2_X = np.stack(stage2_windows).astype(np.float32)
    stage2_y = np.asarray(stage2_labels, dtype=np.int64)
    stage2_subjects = np.asarray(stage2_subjects)

    if args.balance:
        stage1_X, stage1_y, stage1_subjects = balance_dataset(
            stage1_X, stage1_y, stage1_subjects, STAGE1_LABELS
        )
        stage2_X, stage2_y, stage2_subjects = balance_dataset(
            stage2_X, stage2_y, stage2_subjects, STAGE2_LABELS
        )

    out_dir = Path(args.out_dir)
    save_npz(
        out_dir / args.out_stage1,
        stage1_X,
        stage1_y,
        stage1_subjects,
        STAGE1_LABELS,
    )
    save_npz(
        out_dir / args.out_stage2,
        stage2_X,
        stage2_y,
        stage2_subjects,
        STAGE2_LABELS,
    )


if __name__ == "__main__":
    main()
