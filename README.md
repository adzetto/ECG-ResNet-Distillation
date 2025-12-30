ECG 3-Class Classifier (Single-Lead)
===================================

Goal
----
Train a single-lead ECG classifier with three labels:
- NORMAL
- HEART ATTACK (MI)
- ARRHYTHMIA

This pipeline targets AD8232-like signals by standardizing to one lead and a
fixed sampling rate (default 250 Hz).

Datasets (PhysioNet)
--------------------
These are used by the scripts in this repo:
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- MIT-BIH Normal Sinus Rhythm Database: https://physionet.org/content/nsrdb/1.0.0/
- PTB Diagnostic ECG Database: https://physionet.org/content/ptbdb/1.0.0/
- Optional (large, multi-label): PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

Kaggle Mirrors / Alternatives
-----------------------------
Kaggle dataset pages are accessible from CLI via a text proxy, but Kaggle Code
search results are JS-heavy and not easily scraped. If you want Kaggle notebooks,
open kaggle.com/code in a browser and search for the dataset names below.

Dataset mirrors:
- MIT-BIH Database: https://www.kaggle.com/datasets/mondejar/mitbih-database
- MIT-BIH Arrhythmia (CSV): https://www.kaggle.com/datasets/protobioengineering/mit-bih-arrhythmia-database-modern-2023
- MIT-BIH Normal Sinus Rhythm: https://www.kaggle.com/datasets/shymammoth/mitbih-normal-sinus-rhythm-database
- PTB Diagnostic ECG: https://www.kaggle.com/datasets/physionet/ptb-diagnostic-ecg-database
- PTB-XL (PhysioNet): https://www.kaggle.com/datasets/physionet/ptbxl-electrocardiography-database
- PTB-XL (mirror): https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

Pipeline Summary
----------------
1) Download datasets from PhysioNet (or manually place Kaggle versions in data/raw)
2) Build 10-second windows with bandpass filtering and resampling to 250 Hz
3) Train a 1D CNN with patient-level splits
4) Evaluate per-class accuracy on a held-out subject set

Architecture And Figures
------------------------
Detailed model specs, input formats, parameter counts, and diagrams:
- docs/architecture.md

Labeling Strategy
-----------------
- ARRHYTHMIA: windows from MIT-BIH Arrhythmia that contain any non-normal beat
- NORMAL: windows from MIT-BIH Normal Sinus Rhythm (and PTB healthy controls)
- MI: windows from PTB Diagnostic ECG records labeled "myocardial infarction"

This keeps labels clean and avoids mixing normal windows from arrhythmia patients.

Alternative: PTB-XL Only (single dataset)
-----------------------------------------
If you need >95% per-class accuracy, reduce cross-dataset shift by using PTB-XL only.
PTB-XL provides diagnosis and rhythm labels per 10-second record.

Rules used by `scripts/prepare_ptbxl.py` (loose mode):
- MI: diagnostic_class contains MI (takes precedence)
- ARRHYTHMIA: any rhythm code not equal to SR
- NORMAL: diagnostic_class is NORM only and rhythm is SR or absent

Setup
-----
Dependencies (conda or pip):
- numpy, scipy, wfdb, torch, tqdm, pandas

Example (pip):
    pip install numpy scipy wfdb torch tqdm pandas

Download
--------
    python scripts/download_physionet.py --datasets mitdb nsrdb ptbdb

Prepackaged data (raw + processed) is available here:
    https://drive.google.com/drive/folders/1vfI3gWR8zDvml_UJBe-B6ZRh_TwtOt_t?usp=sharing
After downloading, place the `data/` folder at the repo root.

Prepare data
------------
    python scripts/prepare_dataset.py --balance

To reduce arrhythmia label noise (recommended for higher accuracy):
    python scripts/prepare_dataset.py --balance --arrhythmia-min-count 2 --arrhythmia-min-ratio 0.1

Prepare PTB-XL only dataset (place PTB-XL files in data/raw/ptbxl):
    python scripts/prepare_ptbxl.py --balance

Higher label purity options:
    python scripts/prepare_ptbxl.py --balance --label-mode semi
    python scripts/prepare_ptbxl.py --balance --label-mode strict

Two-stage PTB-XL datasets (arrhythmia vs sinus, then MI vs normal):
    python scripts/prepare_ptbxl_twostage.py --ptbxl-dir data/raw/ptbxl --balance

Stage2 multi-lead dataset for teacher training:
    python scripts/prepare_ptbxl_stage2_multilead.py --ptbxl-dir data/raw/ptbxl --balance

Defaults:
- 250 Hz output rate
- 10s windows, 10s stride (non-overlapping)
- max 600 windows per record to limit NSRDB size

Train
-----
    python scripts/train.py --epochs 40 --batch-size 64

Stronger model + augmentation:
    python scripts/train.py --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler

Train on PTB-XL dataset:
    python scripts/train.py --data data/processed/ecg_windows_ptbxl.npz --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler

Train two-stage models (use separate output dirs to avoid overwriting):
    python scripts/train.py --data data/processed/ecg_windows_ptbxl_stage1.npz --out-dir models/stage1 --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler
    python scripts/train.py --data data/processed/ecg_windows_ptbxl_stage2.npz --out-dir models/stage2 --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler

Teacher-student distillation (best shot for MI on single-lead II):
    python scripts/train.py --data data/processed/ecg_windows_ptbxl_stage2_12lead.npz --out-dir models/teacher --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler
    python scripts/train_distill.py --data data/processed/ecg_windows_ptbxl_stage2_12lead.npz --teacher models/teacher/ecg_resnet.pt --out-dir models/student --epochs 50 --batch-size 64 --model resnet --augment --balanced-sampler --lead-name II

Evaluate
--------
    python scripts/evaluate.py

If you trained ResNet:
    python scripts/evaluate.py --model models/ecg_resnet.pt

Evaluate PTB-XL model:
    python scripts/evaluate.py --data data/processed/ecg_windows_ptbxl.npz --model models/ecg_resnet.pt

Evaluate two-stage models:
    python scripts/evaluate.py --data data/processed/ecg_windows_ptbxl_stage1.npz --model models/stage1/ecg_resnet.pt --out-dir models/stage1
    python scripts/evaluate.py --data data/processed/ecg_windows_ptbxl_stage2.npz --model models/stage2/ecg_resnet.pt --out-dir models/stage2

Evaluate distilled student on single-lead stage2 data:
    python scripts/evaluate.py --data data/processed/ecg_windows_ptbxl_stage2.npz --model models/student/ecg_student.pt --out-dir models/student

Inference (single signal)
-------------------------
Save a single-lead signal as CSV or NPY, then:
    python scripts/predict.py --input path/to/signal.csv --input-rate 250

Two-stage inference (final 3-class output):
    python scripts/predict_twostage.py --stage1 models/stage1/ecg_resnet.pt --stage2 models/student/ecg_student.pt --input path/to/signal.csv --input-rate 250

Notes on 95% per-class accuracy
-------------------------------
Reaching >=95% per label is aggressive for cross-dataset ECG. You will likely
need to:
- Increase model capacity (ResNet1D, InceptionTime)
- Use more MI and arrhythmia cases (PTB-XL MI + AFIB subsets)
- Tune window size (5s/10s/20s) and stride
- Apply class-balanced sampling and calibration
- Validate with strict patient-level splits to avoid leakage

Safety
------
This is a research pipeline, not a medical device. Do not use outputs for
clinical decisions.
