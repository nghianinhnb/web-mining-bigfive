import logging
import pickle
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
    """Load DataFrame from parquet or pickle format."""
    # Try pickle first (extension .pkl or .parquet with pickle fallback)
    pkl_path = path.with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    # Fallback to parquet if available
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            # Try pickle version
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    return pickle.load(f)
            raise
    raise FileNotFoundError(f"Neither {path} nor {pkl_path} found")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to pickle format (parquet fallback if pyarrow available)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use pickle for reliability
    pkl_path = path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved data to {pkl_path}")


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

