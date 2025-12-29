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


LABEL_NAMES = ["NORMAL", "MI"]
DEFAULT_LEADS = ["MLII", "II", "V1", "V2", "I"]


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


def classify_ptbxl(diag_classes, rhythm_in_record, strict_rhythm):
    if strict_rhythm and any(code != "SR" for code in rhythm_in_record):
        return None
    if diag_classes == {"MI"}:
        return 1
    if diag_classes == {"NORM"}:
        return 0
    return None


def infer_ptb_label(comments):
    text = " ".join(comments).lower()
    if "myocardial infarction" in text:
        return "MI"
    if "healthy control" in text or ("healthy" in text and "control" in text):
        return "NORMAL"
    return None


def list_ptbdb_records(ptbdb_dir):
    records = []
    for hea in sorted(Path(ptbdb_dir).rglob("*.hea")):
        rel = hea.relative_to(ptbdb_dir)
        records.append(str(rel.with_suffix("")))
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MI vs Normal dataset from PTB-XL + PTBDB")
    parser.add_argument("--ptbxl-dir", default="data/raw/ptbxl")
    parser.add_argument("--ptbdb-dir", default="data/raw/ptbdb")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--out-name", default="ecg_windows_stage2_combo.npz")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=5)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--strict-rhythm", action="store_true")
    parser.add_argument("--leads", nargs="+", default=DEFAULT_LEADS)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    fs_out = float(args.sample_rate)
    window_size = int(args.window_sec * fs_out)
    stride = int(args.stride_sec * fs_out)

    windows = []
    labels = []
    subjects = []

    ptbxl_dir = Path(args.ptbxl_dir)
    db_path = ptbxl_dir / "ptbxl_database.csv"
    if db_path.exists():
        diag_map, rhythm_codes = load_mappings(ptbxl_dir)
        db = pd.read_csv(db_path)
        rows = db.itertuples(index=False)
        if args.max_records and args.max_records > 0:
            rows = list(rows)[: args.max_records]
        for row in tqdm(rows, desc="ptbxl", unit="record"):
            scp_codes = parse_scp_codes(getattr(row, "scp_codes"))
            diag_classes = {diag_map[c] for c in scp_codes if c in diag_map and diag_map[c]}
            rhythm_in = {c for c in scp_codes if c in rhythm_codes}
            label = classify_ptbxl(diag_classes, rhythm_in, args.strict_rhythm)
            if label is None:
                continue
            filename = getattr(row, "filename_lr")
            record_path = ptbxl_dir / filename
            record = wfdb.rdrecord(str(record_path))
            lead_idx = pick_lead(record.sig_name, args.leads)
            signal = record.p_signal[:, lead_idx].astype(np.float32)
            fs_in = float(record.fs)
            resampled = resample_signal(signal, fs_in, fs_out)
            resampled = bandpass_filter(resampled, fs_out)
            for _, _, segment in iter_windows(resampled, window_size, stride):
                if len(segment) != window_size:
                    continue
                windows.append(standardize(segment))
                labels.append(label)
                subjects.append(f"ptbxl:{getattr(row, 'patient_id')}")
    else:
        print(f"Skipping PTB-XL; missing {db_path}")

    ptbdb_dir = Path(args.ptbdb_dir)
    if ptbdb_dir.exists():
        records = list_ptbdb_records(ptbdb_dir)
        for record_name in tqdm(records, desc="ptbdb", unit="record"):
            record_path = ptbdb_dir / record_name
            try:
                record = wfdb.rdrecord(str(record_path))
            except Exception:
                continue
            label_name = infer_ptb_label(record.comments)
            if label_name is None:
                continue
            label = 1 if label_name == "MI" else 0
            lead_idx = pick_lead(record.sig_name, args.leads)
            signal = record.p_signal[:, lead_idx].astype(np.float32)
            fs_in = float(record.fs)
            resampled = resample_signal(signal, fs_in, fs_out)
            resampled = bandpass_filter(resampled, fs_out)
            for _, _, segment in iter_windows(resampled, window_size, stride):
                if len(segment) != window_size:
                    continue
                windows.append(standardize(segment))
                labels.append(label)
                subjects.append(f"ptbdb:{record_name.split('/')[0]}")
    else:
        print(f"Skipping PTBDB; missing {ptbdb_dir}")

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
