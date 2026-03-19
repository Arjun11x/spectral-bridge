import os
import platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src import config


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for the Spectral Graffiti audio in-painting task.

    Each sample is a 100ms audio clip where some time steps are missing
    (Is_Context=0). The model sees surviving context points and must
    reconstruct the gaps.

    Args:
        csv_path: path to spectral_graffiti.csv. Defaults to config.TRAIN_FILE.

    Returns per index:
        x      : [SEQ_LEN, 2] — (masked_value, is_context) per time step
        y_true : [SEQ_LEN]    — ground truth voltage for all time steps
        mask   : [SEQ_LEN]    — 1 where signal is known, 0 where it is missing
    """

    REQUIRED_COLUMNS = {"Sample_ID", "Time_ms", "Is_Context", "Value"}

    def __init__(self, csv_path: str = None):
        csv_path = csv_path or config.TRAIN_FILE
        df = self._load_and_validate(csv_path)
        self.samples = self._build_samples(df)

    # ── public helpers ─────────────────────────────────────────────────────

    def get_by_sample_id(self, sample_id: int) -> dict:
        """Return the processed tensor dict for a specific Sample_ID.
        Useful for EDA and visualizations."""
        idx = self._id_to_idx.get(sample_id)
        if idx is None:
            raise KeyError(f"Sample_ID {sample_id} not found in dataset.")
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def __repr__(self) -> str:
        return (
            f"SpectralDataset("
            f"n_samples={len(self)}, "
            f"seq_len={config.SEQ_LEN}, "
            f"feature_dim={config.FEATURE_DIM})"
        )

    # ── internal ───────────────────────────────────────────────────────────

    def _load_and_validate(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}.\n"
                f"Download from Kaggle and place in {config.DATA_DIR}."
            )
        df = pd.read_csv(csv_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")
        return df

    def _build_samples(self, df: pd.DataFrame) -> list:
        samples = []
        self._id_to_idx = {}  # maps Sample_ID → list index for get_by_sample_id

        for sample_id, group in df.groupby("Sample_ID"):
            group = group.sort_values("Time_ms").reset_index(drop=True)

            values     = self._pad(group["Value"].values.astype(np.float32))
            is_context = self._pad(group["Is_Context"].values.astype(np.float32))

            values     = torch.from_numpy(values)
            is_context = torch.from_numpy(is_context)

            # zero out positions where signal is missing so the model
            # cannot see the ground truth through the input
            masked_value = values * is_context

            # stack into [SEQ_LEN, FEATURE_DIM]
            x = torch.stack([masked_value, is_context], dim=-1)

            self._id_to_idx[sample_id] = len(samples)
            samples.append({
                "x":      x,
                "y_true": values,
                "mask":   is_context,
            })

        return samples

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad a 1D array to config.SEQ_LEN."""
        arr = arr[: config.SEQ_LEN]
        if len(arr) < config.SEQ_LEN:
            arr = np.pad(arr, (0, config.SEQ_LEN - len(arr)))
        return arr


def get_dataloaders(
    csv_path:   str   = None,
    batch_size: int   = None,
    val_split:  float = None,
    seed:       int   = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from the dataset.

    The split is seeded so the same samples always land in train vs val,
    making experiments reproducible and comparable across runs.

    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    val_split  = val_split  or config.VAL_SPLIT
    seed       = seed       or config.SEED

    dataset    = SpectralDataset(csv_path)
    total      = len(dataset)
    val_size   = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # num_workers > 0 causes issues on Windows outside of a
    # __main__ guard, so we detect the platform and adjust
    workers = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader