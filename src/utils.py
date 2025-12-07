# src/utils.py
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset


def filter_and_encode_labels(y: np.ndarray, min_count: int = 5) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Filters out labels that appear fewer than min_count times and encodes remaining labels to ints.
    Returns (y_filtered, label_to_int).
    """
    counter = Counter(y)
    keep = {lbl for lbl, c in counter.items() if c >= min_count}
    mask = np.array([lbl in keep for lbl in y])
    y_filtered = y[mask]

    unique = sorted(list({lbl for lbl in y_filtered}))
    label_to_int = {lbl: i for i, lbl in enumerate(unique)}
    y_encoded = np.array([label_to_int[lbl] for lbl in y_filtered])

    return y_encoded, label_to_int, mask


class ECGDataset(Dataset):
    """
    PyTorch dataset for ECG beats.
    Expect X shaped (N, L, 1) and y shaped (N,)
    For CNN we will transpose in the collate/loader loop to (N, 1, L).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]           # shape (L, 1)
        y = self.y[idx]
        x = torch.from_numpy(x)   # float32 tensor
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def collate_cnn(batch):
    """
    Convert list of (x, y) to (batch, 1, L) and (batch,)
    """
    xs, ys = zip(*batch)
    xs = torch.stack(xs)             # (batch, L, 1)
    xs = xs.permute(0, 2, 1)         # (batch, 1, L)
    ys = torch.stack([y for y in ys])
    return xs, ys


def collate_lstm(batch):
    """
    Return (batch, L, 1), (batch,)
    """
    xs, ys = zip(*batch)
    xs = torch.stack(xs)             # (batch, L, 1)
    ys = torch.stack([y for y in ys])
    return xs, ys


def save_checkpoint(state: Dict[str, Any], path: str):
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device)
