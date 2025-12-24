import logging
from pathlib import Path
import pandas as pd


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_splits(splits_dir: Path) -> dict:
    splits = {}
    for split in ["train", "dev", "test"]:
        path = splits_dir / f"{split}.txt"
        if path.exists():
            with open(path, "r") as f:
                splits[split] = [line.strip() for line in f if line.strip()]
    return splits


def save_splits(splits: dict, splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, user_ids in splits.items():
        with open(splits_dir / f"{split_name}.txt", "w") as f:
            f.write("\n".join(user_ids))

