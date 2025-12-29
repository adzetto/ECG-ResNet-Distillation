import math
from fractions import Fraction

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

LABELS = ["NORMAL", "MI", "ARRHYTHMIA"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}

NORMAL_BEATS = {"N", "L", "R", "e", "j"}
NON_BEAT_SYMBOLS = {
    "|",
    "~",
    "+",
    "[",
    "]",
    "!",
    "x",
    "(",
    ")",
    "p",
    "t",
    "u",
    "`",
    "^",
    "=",
    '"',
    "?",
}


def is_arrhythmia_symbol(symbol):
    if symbol in NORMAL_BEATS:
        return False
    if symbol in NON_BEAT_SYMBOLS:
        return False
    return True


def is_beat_symbol(symbol):
    return symbol not in NON_BEAT_SYMBOLS


def pick_lead(sig_names, preferred):
    name_map = {name.upper(): idx for idx, name in enumerate(sig_names)}
    for name in preferred:
        idx = name_map.get(name.upper())
        if idx is not None:
            return idx
    return 0


def bandpass_filter(x, fs_hz, low_hz=0.5, high_hz=40.0, order=4):
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    nyq = 0.5 * fs_hz
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.99999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def resample_signal(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    ratio = Fraction(int(fs_out), int(fs_in)).limit_denominator()
    return resample_poly(x, ratio.numerator, ratio.denominator)


def resample_positions(samples, fs_in, fs_out):
    if fs_in == fs_out:
        return np.asarray(samples, dtype=int)
    scale = fs_out / fs_in
    return np.round(np.asarray(samples) * scale).astype(int)


def standardize(x):
    mean = float(np.mean(x))
    std = float(np.std(x))
    return (x - mean) / (std + 1e-6)


def iter_windows(x, window_size, stride):
    total = x.shape[0]
    for start in range(0, total - window_size + 1, stride):
        end = start + window_size
        yield start, end, x[start:end]


def split_by_subject(subjects, train_ratio=0.7, val_ratio=0.15, seed=13):
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_ratio)
    n_val = int(len(unique) * val_ratio)
    train_subjects = set(unique[:n_train])
    val_subjects = set(unique[n_train : n_train + n_val])
    test_subjects = set(unique[n_train + n_val :])
    train_mask = np.isin(subjects, list(train_subjects))
    val_mask = np.isin(subjects, list(val_subjects))
    test_mask = np.isin(subjects, list(test_subjects))
    return train_mask, val_mask, test_mask


def split_by_subject_stratified(subjects, labels, train_ratio=0.7, val_ratio=0.15, seed=13):
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    unique_subjects = np.unique(subjects)
    num_classes = int(labels.max()) + 1 if labels.size else 0

    subject_labels = []
    for subj in unique_subjects:
        subj_mask = subjects == subj
        counts = np.bincount(labels[subj_mask], minlength=num_classes)
        subject_labels.append(int(np.argmax(counts)))

    rng = np.random.default_rng(seed)
    train_subjects = set()
    val_subjects = set()
    test_subjects = set()

    for cls in range(num_classes):
        cls_subjects = [s for s, c in zip(unique_subjects, subject_labels) if c == cls]
        rng.shuffle(cls_subjects)
        n_total = len(cls_subjects)
        if n_total == 0:
            continue
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1
        n_test = max(0, n_total - n_train - n_val)
        train_subjects.update(cls_subjects[:n_train])
        val_subjects.update(cls_subjects[n_train : n_train + n_val])
        if n_test > 0:
            test_subjects.update(cls_subjects[n_train + n_val :])

    train_mask = np.isin(subjects, list(train_subjects))
    val_mask = np.isin(subjects, list(val_subjects))
    test_mask = np.isin(subjects, list(test_subjects))
    return train_mask, val_mask, test_mask


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = float(np.sum(counts))
    weights = total / (counts + 1e-6)
    weights = weights / np.mean(weights)
    return weights
