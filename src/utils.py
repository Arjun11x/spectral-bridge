import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from src import config


def set_seed(seed: int = None) -> None:
    """
    Fix all random seeds for fully reproducible training runs.
    Covers Python, NumPy, PyTorch CPU and GPU.
    """
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """
    Returns the best available device and prints it once.
    Colab T4 GPU will show 'cuda — Tesla T4'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: cuda — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: cpu — no GPU found")
    return device


def ensure_dirs() -> None:
    """Create results/plots and results/checkpoints if they don't exist."""
    os.makedirs(config.PLOTS_DIR,       exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)


# ── checkpointing ──────────────────────────────────────────────────────────


def save_checkpoint(
    model:         torch.nn.Module,
    optimizer:     torch.optim.Optimizer,
    scheduler:     object,
    epoch:         int,
    best_val_loss: float,
    train_losses:  list,
    val_losses:    list,
    lr_history:    list = None,
    path:          str  = None,
) -> None:
    """
    Save full training state so training can resume exactly from this point.

    Saves model weights, optimizer state, scheduler state, current epoch,
    best validation loss, loss history, and LR history. Saving the
    optimizer and scheduler states is essential — without them the
    optimizer loses its momentum and adaptive LR history, causing a
    loss spike on resume.

    Args:
        model         : the SpectralTransformer being trained
        optimizer     : AdamW optimizer instance
        scheduler     : ReduceLROnPlateau scheduler instance
        epoch         : epoch just completed (resume will start from epoch+1)
        best_val_loss : best validation loss seen so far for early stopping
        train_losses  : list of training losses recorded so far
        val_losses    : list of validation losses recorded so far
        lr_history    : list of learning rates recorded so far
        path          : save path — defaults to config.BEST_MODEL_PATH
    """
    ensure_dirs()
    path = path or config.BEST_MODEL_PATH
    torch.save({
        "epoch":         epoch,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "lr_history":    lr_history or [],
    }, path)


def load_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    path:      str = None,
) -> dict:
    """
    Restore full training state from a saved checkpoint.

    Loads weights and optimizer/scheduler states directly into the
    provided instances. Returns metadata so the training loop knows
    exactly where to resume from.

    Args:
        model     : an instantiated SpectralTransformer (same architecture)
        optimizer : the AdamW optimizer instance to restore state into
        scheduler : the scheduler instance to restore state into
        path      : checkpoint path — defaults to config.BEST_MODEL_PATH

    Returns:
        dict with keys: epoch, best_val_loss, train_losses, val_losses,
        lr_history

    Raises:
        FileNotFoundError if no checkpoint exists at path
    """
    path = path or config.BEST_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    # map_location='cpu' ensures checkpoints saved on GPU load cleanly
    # on CPU and vice versa — important when moving between environments
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return {
        "epoch":         checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "train_losses":  checkpoint["train_losses"],
        "val_losses":    checkpoint["val_losses"],
        "lr_history":    checkpoint.get("lr_history", []),
    }


def checkpoint_exists(path: str = None) -> bool:
    """Returns True if a saved checkpoint exists at path."""
    path = path or config.BEST_MODEL_PATH
    return os.path.exists(path)


# ── figure saving ──────────────────────────────────────────────────────────


def save_figure(filename: str, dpi: int = 150) -> None:
    """
    Save the current matplotlib figure to results/plots/.
    Call after building a plot, before plt.show().

    Args:
        filename : e.g. 'loss_curve.png' — no path needed
        dpi      : resolution (150 is sharp without being huge)
    """
    ensure_dirs()
    path = os.path.join(config.PLOTS_DIR, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


# ── timer ──────────────────────────────────────────────────────────────────


class Timer:
    """
    Lightweight context manager for timing code blocks.

    Usage:
        with Timer("Epoch 1"):
            train_one_epoch(...)
        # prints: Epoch 1 — 42.3s
    """

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"{self.label} — {elapsed:.1f}s")