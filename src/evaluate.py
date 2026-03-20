import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from src import config
from src.model import build_model, masked_mse_loss
from src.dataset import SpectralDataset, get_dataloaders
from src.utils import (
    get_device,
    ensure_dirs,
    load_checkpoint,
    save_figure,
)


def evaluate(csv_path: str = None) -> dict:
    """
    Full evaluation pipeline for the best saved SpectralTransformer.

    Loads the best checkpoint, computes final MSE on the validation set,
    and generates all diagnostic plots saved to results/plots/.

    Plots generated:
        reconstruction.png         : predicted vs true signal for one sample
        uncertainty_bands.png      : MC Dropout confidence intervals
        multi_instrument_grid.png  : reconstruction across 6 diverse samples
        residual_distribution.png  : histogram of prediction errors
        mse_distribution.png       : per-sample MSE across validation set
        predicted_vs_actual.png    : scatter plot of predicted vs true values

    Args:
        csv_path : path to dataset CSV — defaults to config.TRAIN_FILE

    Returns:
        dict with keys:
            mean_mse     : mean MSE across all validation target points
            median_mse   : median per-sample MSE
            std_mse      : std deviation of per-sample MSE
    """
    device = get_device()
    ensure_dirs()

    # ── load best model ────────────────────────────────────────────────────

    model     = build_model().to(device)
    best_path = config.BEST_MODEL_PATH.replace(".pth", "_best.pth")

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best model not found at {best_path}.\n"
            f"Run train.py first to generate a trained model."
        )

    # load weights only — no optimizer or scheduler needed for evaluation
    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded best model from {best_path}")
    print(model)

    # ── validation dataloader ──────────────────────────────────────────────

    _, val_loader = get_dataloaders(csv_path)

    # ── compute final MSE ──────────────────────────────────────────────────

    all_preds      = []
    all_trues      = []
    all_masks      = []
    per_sample_mse = []

    with torch.no_grad():
        for batch in val_loader:
            x      = batch["x"].to(device)
            y_true = batch["y_true"].to(device)
            mask   = batch["mask"].to(device)

            y_pred = model(x)

            # collect for global metrics
            all_preds.append(y_pred.cpu())
            all_trues.append(y_true.cpu())
            all_masks.append(mask.cpu())

            # per-sample MSE within this batch
            target_mask    = 1.0 - mask
            squared_errors = (y_pred - y_true) ** 2
            # mean over time dimension for each sample individually
            sample_mse = (
                (squared_errors * target_mask).sum(dim=1) /
                (target_mask.sum(dim=1) + 1e-8)
            )
            per_sample_mse.append(sample_mse.cpu())

    # concatenate all batches
    all_preds      = torch.cat(all_preds,      dim=0)  # [N, 100]
    all_trues      = torch.cat(all_trues,      dim=0)  # [N, 100]
    all_masks      = torch.cat(all_masks,      dim=0)  # [N, 100]
    per_sample_mse = torch.cat(per_sample_mse, dim=0)  # [N]

    # global MSE across all target points
    target_mask    = 1.0 - all_masks
    squared_errors = (all_preds - all_trues) ** 2
    mean_mse = (
        (squared_errors * target_mask).sum() /
        (target_mask.sum() + 1e-8)
    ).item()

    print(f"\nFinal evaluation results:")
    print(f"  Mean MSE   : {mean_mse:.6f}")
    print(f"  Median MSE : {per_sample_mse.median().item():.6f}")
    print(f"  Std MSE    : {per_sample_mse.std().item():.6f}")

    # ── generate all plots ─────────────────────────────────────────────────

    # pick a clean sample for the single-sample plots — one where the
    # model performed close to the median so it is representative
    median_val    = per_sample_mse.median().item()
    distances     = (per_sample_mse - median_val).abs()
    sample_idx    = distances.argmin().item()

    _plot_reconstruction(
        all_preds[sample_idx],
        all_trues[sample_idx],
        all_masks[sample_idx],
        sample_idx,
    )

    _plot_uncertainty_bands(model, val_loader, device, sample_idx)

    _plot_multi_instrument_grid(
        all_preds, all_trues, all_masks, per_sample_mse
    )

    _plot_residual_distribution(all_preds, all_trues, all_masks)

    _plot_mse_distribution(per_sample_mse)

    _plot_predicted_vs_actual(all_preds, all_trues, all_masks)

    return {
        "mean_mse":   mean_mse,
        "median_mse": per_sample_mse.median().item(),
        "std_mse":    per_sample_mse.std().item(),
    }


# ── plot functions ─────────────────────────────────────────────────────────


def _plot_reconstruction(
    y_pred:     torch.Tensor,
    y_true:     torch.Tensor,
    mask:       torch.Tensor,
    sample_idx: int,
) -> None:
    """
    Plot predicted signal reconstruction against ground truth for one sample.

    Shows three elements on the same axes:
        - ground truth voltage across all 100 time steps (thin gray line)
        - context points the model was given (teal dots)
        - model's predicted gap values (coral dots)
    """
    time_steps  = np.arange(1, config.SEQ_LEN + 1)
    y_pred_np   = y_pred.numpy()
    y_true_np   = y_true.numpy()
    mask_np     = mask.numpy().astype(bool)
    target_mask = ~mask_np

    fig, ax = plt.subplots(figsize=(12, 4))

    # ground truth as a continuous reference line
    ax.plot(
        time_steps, y_true_np,
        color="lightgray", linewidth=1.2,
        label="Ground truth", zorder=1,
    )

    # context points — what the model was allowed to see
    ax.scatter(
        time_steps[mask_np], y_true_np[mask_np],
        color="#1D9E75", s=18, zorder=3,
        label=f"Context ({mask_np.sum()} points)",
    )

    # predicted gap values — what the model reconstructed
    ax.scatter(
        time_steps[target_mask], y_pred_np[target_mask],
        color="#D85A30", s=18, zorder=3,
        label=f"Predicted ({target_mask.sum()} gaps)",
    )

    sample_mse = (
        ((y_pred - y_true) ** 2 * (1 - mask)).sum() /
        ((1 - mask).sum() + 1e-8)
    ).item()

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (normalized)")
    ax.set_title(
        f"Signal reconstruction — sample {sample_idx} "
        f"(MSE: {sample_mse:.6f})"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("reconstruction.png")
    plt.show()


def _plot_uncertainty_bands(
    model:      torch.nn.Module,
    val_loader: DataLoader,
    device:     torch.device,
    sample_idx: int,
) -> None:
    """
    Plot MC Dropout uncertainty bands for one sample.

    Runs 20 stochastic forward passes with dropout active to estimate
    prediction confidence. Shaded region shows ±1 std deviation —
    wide bands indicate high uncertainty, narrow bands indicate
    high confidence.
    """
    # extract the specific sample from the validation set
    dataset  = val_loader.dataset
    sample   = dataset[sample_idx]
    x        = sample["x"].unsqueeze(0).to(device)   # [1, 100, 2]
    y_true   = sample["y_true"].numpy()
    mask     = sample["mask"].numpy().astype(bool)
    target   = ~mask

    mean, std = model.predict_with_uncertainty(x, n_samples=20)
    mean_np   = mean.squeeze(0).cpu().numpy()
    std_np    = std.squeeze(0).cpu().numpy()

    time_steps = np.arange(1, config.SEQ_LEN + 1)

    fig, ax = plt.subplots(figsize=(12, 4))

    # ground truth reference
    ax.plot(
        time_steps, y_true,
        color="lightgray", linewidth=1.2,
        label="Ground truth", zorder=1,
    )

    # mean prediction line across gap positions
    ax.plot(
        time_steps[target], mean_np[target],
        color="#D85A30", linewidth=1.5,
        label="Mean prediction", zorder=3,
    )

    # ±1 std uncertainty band
    ax.fill_between(
        time_steps[target],
        mean_np[target] - std_np[target],
        mean_np[target] + std_np[target],
        alpha=0.3, color="#D85A30",
        label="±1 std (uncertainty)",
    )

    # context points
    ax.scatter(
        time_steps[mask], y_true[mask],
        color="#1D9E75", s=18, zorder=4,
        label="Context points",
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (normalized)")
    ax.set_title(
        f"Uncertainty estimation via MC Dropout — sample {sample_idx} "
        f"(n=20 passes)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("uncertainty_bands.png")
    plt.show()


def _plot_multi_instrument_grid(
    all_preds:      torch.Tensor,
    all_trues:      torch.Tensor,
    all_masks:      torch.Tensor,
    per_sample_mse: torch.Tensor,
) -> None:
    """
    Plot reconstruction for 6 diverse samples in a 2×3 grid.

    Samples are chosen to cover a range of MSE values — two near the
    best performance, two near the median, two near the worst — showing
    how the model handles different signal types and gap patterns.
    """
    n        = len(per_sample_mse)
    sorted_i = per_sample_mse.argsort()

    # pick two from best, two from middle, two from worst performers
    indices = [
        sorted_i[int(n * 0.05)].item(),
        sorted_i[int(n * 0.15)].item(),
        sorted_i[int(n * 0.45)].item(),
        sorted_i[int(n * 0.55)].item(),
        sorted_i[int(n * 0.85)].item(),
        sorted_i[int(n * 0.95)].item(),
    ]
    labels = ["best 5%", "best 15%", "median-", "median+", "worst 15%", "worst 5%"]

    time_steps = np.arange(1, config.SEQ_LEN + 1)
    fig, axes  = plt.subplots(2, 3, figsize=(15, 7))
    axes       = axes.flatten()

    for ax, idx, lbl in zip(axes, indices, labels):
        y_pred_np   = all_preds[idx].numpy()
        y_true_np   = all_trues[idx].numpy()
        mask_np     = all_masks[idx].numpy().astype(bool)
        target_mask = ~mask_np
        mse         = per_sample_mse[idx].item()

        ax.plot(
            time_steps, y_true_np,
            color="lightgray", linewidth=1, zorder=1,
        )
        ax.scatter(
            time_steps[mask_np], y_true_np[mask_np],
            color="#1D9E75", s=10, zorder=3,
        )
        ax.scatter(
            time_steps[target_mask], y_pred_np[target_mask],
            color="#D85A30", s=10, zorder=3,
        )

        ax.set_title(f"{lbl}  (MSE: {mse:.6f})", fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Voltage", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Reconstruction across performance range",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    save_figure("multi_instrument_grid.png")
    plt.show()


def _plot_residual_distribution(
    all_preds: torch.Tensor,
    all_trues: torch.Tensor,
    all_masks: torch.Tensor,
) -> None:
    """
    Plot histogram of residual errors (y_pred - y_true) at target positions.

    A well-trained model produces residuals centered at zero with an
    approximately Gaussian shape — indicating no systematic bias.
    Skew or heavy tails indicate the model over- or under-predicts
    in certain regions.
    """
    target_mask = (1.0 - all_masks).bool()
    residuals   = (all_preds - all_trues)[target_mask].numpy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(residuals, bins=100, color="#378ADD", edgecolor="none", alpha=0.8)
    ax.axvline(0,               color="black", linewidth=1.2, linestyle="--", label="Zero")
    ax.axvline(residuals.mean(), color="#D85A30", linewidth=1.2,
               linestyle="--", label=f"Mean: {residuals.mean():.4f}")

    ax.set_xlabel("Residual error  (predicted − true)")
    ax.set_ylabel("Count")
    ax.set_title("Residual error distribution across all target points")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("residual_distribution.png")
    plt.show()


def _plot_mse_distribution(per_sample_mse: torch.Tensor) -> None:
    """
    Plot histogram of per-sample MSE across the full validation set.

    Shows whether the model performs consistently or has a long tail of
    catastrophic failures on certain clip types. A tight distribution
    near zero indicates robust generalization.
    """
    mse_np = per_sample_mse.numpy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(mse_np, bins=80, color="#1D9E75", edgecolor="none", alpha=0.8)
    ax.axvline(
        mse_np.mean(), color="#D85A30", linewidth=1.5,
        linestyle="--", label=f"Mean: {mse_np.mean():.6f}",
    )
    ax.axvline(
        np.median(mse_np), color="#534AB7", linewidth=1.5,
        linestyle="--", label=f"Median: {np.median(mse_np):.6f}",
    )

    ax.set_xlabel("Per-sample MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("MSE distribution across validation set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("mse_distribution.png")
    plt.show()


def _plot_predicted_vs_actual(
    all_preds: torch.Tensor,
    all_trues: torch.Tensor,
    all_masks: torch.Tensor,
) -> None:
    """
    Scatter plot of predicted vs true voltage at all target positions.

    A perfect model produces points along the diagonal y=x line.
    Spread around the diagonal indicates prediction error.
    Systematic deviation above or below indicates directional bias.

    Downsamples to 50,000 points for plotting speed — the full
    validation set has millions of target points.
    """
    target_mask = (1.0 - all_masks).bool()
    preds_flat  = all_preds[target_mask].numpy()
    trues_flat  = all_trues[target_mask].numpy()

    # downsample for plotting speed — 50k points is visually sufficient
    if len(preds_flat) > 50_000:
        idx        = np.random.choice(len(preds_flat), 50_000, replace=False)
        preds_flat = preds_flat[idx]
        trues_flat = trues_flat[idx]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        trues_flat, preds_flat,
        alpha=0.15, s=3,
        color="#378ADD", rasterized=True,
    )

    # perfect prediction line
    lims = [
        min(trues_flat.min(), preds_flat.min()),
        max(trues_flat.max(), preds_flat.max()),
    ]
    ax.plot(lims, lims, color="black", linewidth=1.2,
            linestyle="--", label="Perfect prediction (y=x)")

    ax.set_xlabel("True voltage")
    ax.set_ylabel("Predicted voltage")
    ax.set_title("Predicted vs true voltage at all gap positions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    save_figure("predicted_vs_actual.png")
    plt.show()


# ── entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    evaluate()