# note.txt Analysis (Experiment History)

This document summarizes the experimentation log in `note.txt`. The log captures the end-to-end path from initial dataset assembly to multi-stage modeling.
It includes the early failures, the rationale for each methodological change, and the academic framing of the decisions.

## Scope And Context

The log begins with hardware context for a single-lead AD8232 pipeline, then moves into dataset selection and model training attempts.
The core objective is a 3-class classifier with >= 95% per-label accuracy: NORMAL, MI, ARRHYTHMIA.

## Timeline Of Attempts

### Attempt 0: Scaffold And Import Failure

First struggle: `ModuleNotFoundError: No module named 'ecg_utils'` when running scripts directly.
Resolution: prepend repo root to `sys.path` in entrypoints so the module is importable in non-packaged execution.

Why this was chosen:
- It preserves the simple "python scripts/...py" workflow while keeping code in a flat repo layout.

### Attempt 1: Baseline CNN On Mixed Datasets

Data: MIT-BIH (mitdb) + NSRDB + PTBDB, 10s windows, single lead.

Observed issue:
- Per-class accuracies oscillated heavily across epochs and were not stable enough for 95% per-label.
- Early stopping on overall accuracy masked the worst-class behavior.

Reasoning:
- Group split was not stratified, so class representation across subject splits was unstable.
- Mixed datasets introduce domain shift: NORMAL windows from NSRDB and MI windows from PTBDB are from different acquisition pipelines.

### Attempt 2: Stratified Subject Split + Balanced Accuracy Target

Change:
- Stratified split by subject, and early stopping driven by balanced accuracy.

Why:
- The objective is per-class reliability, not just mean accuracy.
- Balanced accuracy is aligned with this target and reduces the effect of class imbalance.

$$
\mathrm{BAcc} = \frac{1}{K}\sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}
$$

Outcome:
- Stability improved, but 95% per-label was still not reached.

### Attempt 3: Arrhythmia Filtering + Balanced Dataset + ResNet

Change:
- Stricter arrhythmia window definition (minimum arrhythmia count and ratio).
- Balanced class sampling and stronger model (ResNet) with augmentation.

Why:
- The arrhythmia label was noisy in short windows; filtering reduces false positives.
- Stronger capacity and augmentation reduce underfitting and improve robustness.

Key statistic from the log:
- Balanced dataset created: 2,719 windows per class.

Outcome:
- ARRHYTHMIA improved, but NORMAL vs MI confusion remained.
- This indicates domain shift rather than purely model capacity.

### Attempt 4: PTB-XL Only (Single Domain)

Change:
- Build a PTB-XL-only dataset with explicit diagnostic and rhythm label rules.

Why:
- When $P_{train}(x,y)$ mixes domains, the classifier can overfit acquisition cues rather than pathology.
- Using a single domain reduces the mismatch:

$$
P_{train}(x,y) \neq P_{test}(x,y)
$$

Initial struggle:
- Missing PTB-XL files and NaN flags in metadata required handling and file verification.

Tradeoff observed:
- Stricter label modes increase purity but reduce dataset size.

### Attempt 5: Two-Stage Pipeline (Arrhythmia Then MI)

Change:
- Stage 1: SINUS vs ARRHYTHMIA from rhythm codes.
- Stage 2: MI vs NORMAL on sinus-only signals.

Why:
- Decomposes a 3-class task into two easier binary decisions.
- This reduces label overlap between arrhythmia and MI cases and focuses MI on sinus rhythm.

$$$$
p(y_{3} \mid x) \approx p(y_{2} \mid x, y_{1}=\text{SINUS}) \cdot p(y_{1}=\text{SINUS} \mid x)
$$$$

Observed issue:
- Still below 95% per-label, indicating limits of single-lead MI detection with current data.

### Attempt 6: Multi-Lead Teacher And Distillation (Proposed)

Motivation:
- Multi-lead signals provide stronger MI cues than a single lead.
- Distillation gives a practical upper bound and transfers knowledge to the single-lead student.

This was proposed as a ceiling-finding step rather than a final AD8232-equivalent model.

## Mathematical Framing Of Preprocessing

Standardization (per window):

$$
\\tilde{x}_t = \frac{x_t - \mu}{\sigma + \epsilon}
$$

Empirical risk objective (multi-class cross-entropy):

$$
\hat{R}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i), y_i\big)
$$

## Rationale Summary

- Stratified subject splits were chosen to reduce per-class variance across folds.
- Balanced accuracy was chosen to align optimization with the 95% per-label requirement.
- Arrhythmia filtering was chosen to reduce label noise in MIT-BIH windows.
- PTB-XL-only training was chosen to reduce domain shift from mixed sources.
- Two-stage modeling was chosen to decouple arrhythmia detection from MI discrimination.

If you want me to add numeric tables of the logged accuracy values, I can extract and format them from `note.txt`.
