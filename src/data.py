"""Chronological loaders with train-only z-score (ETT 12/4/4 or 70/10/20)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════
# Standard ETT chronological splits (12 / 4 / 4  months)
# ═══════════════════════════════════════════════════════════════════════════
#   Dataset     Freq    Rows    Train       Val         Test
#   ETTh1/h2    1 h     17420   0–12*730    12*730–16*730  16*730–end
#   ETTm1/m2    15 min  69680   0–12*2920   12*2920–16*2920  16*2920–end
# ═══════════════════════════════════════════════════════════════════════════

ETT_SPLITS: Dict[str, Tuple[int, int]] = {
    "ETTh1": (12 * 30 * 24, 16 * 30 * 24),        # 8640 / 11520
    "ETTh2": (12 * 30 * 24, 16 * 30 * 24),
    "ETTm1": (12 * 30 * 24 * 4, 16 * 30 * 24 * 4),  # 34560 / 46080
    "ETTm2": (12 * 30 * 24 * 4, 16 * 30 * 24 * 4),
}


def _detect_ett_name(path: str | Path) -> Optional[str]:
    """Try to identify which ETT dataset a path belongs to."""
    stem = Path(path).stem.lower()
    for name in ETT_SPLITS:
        if name.lower() in stem:
            return name
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════
class TimeSeriesDataset(Dataset):
    """Sliding-window dataset over a pre-normalised contiguous array.

    Parameters
    ----------
    values      : (N, D) float32 array — already normalised
    context_len : look-back window
    pred_len    : forecast horizon
    stride      : step between consecutive windows (default 1)
    """

    def __init__(
        self,
        values: np.ndarray,
        context_len: int = 336,
        pred_len: int = 96,
        stride: int = 1,
    ):
        assert values.ndim == 2
        self.values = values
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_samples = max(
            0, (len(values) - context_len - pred_len) // stride + 1
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.values[start : start + self.context_len]
        y = self.values[
            start + self.context_len : start + self.context_len + self.pred_len
        ]
        return torch.from_numpy(x), torch.from_numpy(y)


# ═══════════════════════════════════════════════════════════════════════════
# CSV loader
# ═══════════════════════════════════════════════════════════════════════════
def _load_csv(path: str | Path) -> np.ndarray:
    """Load CSV, drop date/string columns, return float32 array (N, D)."""
    df = pd.read_csv(path)
    # drop common date-like columns
    for col in ("date", "Date", "datetime", "Datetime", "timestamp"):
        if col in df.columns:
            df = df.drop(columns=[col])
    values = df.select_dtypes(include=["number"]).to_numpy(dtype="float32")
    if values.size == 0:
        raise ValueError(f"No numeric columns found in {path}")
    return values


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def create_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    context_len: int = 336,
    pred_len: int = 96,
    stride: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train / val / test DataLoaders with chronological splits.

    For ETT datasets the standard 12/4/4 month split is used automatically.
    For other CSVs the data is split as  train_ratio / val_ratio / rest.

    Normalisation uses **train-only** mean and std.

    Returns
    -------
    train_loader, val_loader, test_loader, n_features
    """
    csv_path = Path(csv_path)
    values = _load_csv(csv_path)
    N, D = values.shape

    # ---- determine split boundaries ------------------------------------
    ett_name = _detect_ett_name(csv_path)
    if ett_name is not None:
        train_end, val_end = ETT_SPLITS[ett_name]
        # clamp to actual data length
        train_end = min(train_end, N)
        val_end = min(val_end, N)
    else:
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))

    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]

    # ---- normalise with training statistics only ------------------------
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-6

    train_norm = ((train_raw - mean) / std).astype("float32")
    val_norm = ((val_raw - mean) / std).astype("float32")
    test_norm = ((test_raw - mean) / std).astype("float32")

    # ---- create datasets ------------------------------------------------
    train_ds = TimeSeriesDataset(train_norm, context_len, pred_len, stride)
    val_ds = TimeSeriesDataset(val_norm, context_len, pred_len, stride)
    test_ds = TimeSeriesDataset(test_norm, context_len, pred_len, stride)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    print(
        f"[data] {csv_path.name}  D={D}  "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
        f"(split @ {train_end}/{val_end}/{N})"
    )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False), D
