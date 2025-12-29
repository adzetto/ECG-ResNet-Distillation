import argparse
from pathlib import Path

import wfdb

DEFAULT_DATASETS = ["mitdb", "nsrdb", "ptbdb"]


def download_dataset(name, out_dir):
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} -> {target}")
    wfdb.dl_database(name, str(target))


def parse_args():
    parser = argparse.ArgumentParser(description="Download PhysioNet ECG datasets")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory to store PhysioNet datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to download (e.g., mitdb nsrdb ptbdb)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for name in args.datasets:
        download_dataset(name, args.data_dir)


if __name__ == "__main__":
    main()
