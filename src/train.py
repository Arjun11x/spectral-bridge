import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.model import SpectralTransformer, masked_mse_loss, build_model
from src.dataset import get_dataloaders
from src.utils import (
    set_seed,
    get_device,
    ensure_dirs,
    save_checkpoint,
    load_checkpoint,
    checkpoint_exists,
    save_figure,
    Timer,
)


def train(cfg: dict = None) -> dict:
    """
    Full training loop for SpectralTransformer with checkpoint resuming.

    Automatically resumes from the last saved checkpoint if one exists,
    making it safe to run on Colab where sessions can disconnect at any time.
    Saves a checkpoint after every epoch and separately tracks the best
    model based on validation loss.

    Generates and saves two plots at the end of training:
        - loss_curve.png     : train vs val MSE across all epochs
        - lr_schedule.png    : learning rate decay across all epochs

    Args:
        cfg : optional ablation config dict — see config.ABLATION_CONFIGS.
              If None, trains the default model from config.py.

    Returns:
        dict with keys:
            train_losses  : list of per-epoch training MSE
            val_losses    : list of per-epoch validation MSE
            lr_history    : list of per-epoch learning rates
            best_val_loss : best validation MSE achieved
            epochs_run    : total number of epochs completed
    """
    # ── setup ──────────────────────────────────────────────────────────────

    set_seed()
    device = get_device()
    ensure_dirs()

    # ── model, optimizer, scheduler ────────────────────────────────────────

    model = build_model(cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # halves LR when val loss stops improving for LR_PATIENCE epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
    )

    # ── data ───────────────────────────────────────────────────────────────

    train_loader, val_loader = get_dataloaders()

    # ── resume or start fresh ──────────────────────────────────────────────

    start_epoch    = 0
    best_val_loss  = float("inf")
    train_losses   = []
    val_losses     = []
    lr_history     = []
    patience_count = 0

    if checkpoint_exists():
        print("Checkpoint found — resuming training...")
        state = load_checkpoint(model, optimizer, scheduler)
        start_epoch    = state["epoch"] + 1
        best_val_loss  = state["best_val_loss"]
        train_losses   = state["train_losses"]
        val_losses     = state["val_losses"]
        lr_history     = state.get("lr_history", [])
        patience_count = _recompute_patience(val_losses)
        print(
            f"Resumed from epoch {start_epoch} | "
            f"best val MSE so far: {best_val_loss:.6f}"
        )
    else:
        print("No checkpoint found — starting fresh...")
        print(model)

    # ── training loop ──────────────────────────────────────────────────────

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        label = f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}]"

        with Timer(label):

            # ── train phase ────────────────────────────────────────────────

            model.train()
            total_train_loss = 0.0

            loop = tqdm(
                train_loader,
                desc=f"{label} train",
                leave=False,
            )

            for batch in loop:
                x      = batch["x"].to(device)
                y_true = batch["y_true"].to(device)
                mask   = batch["mask"].to(device)

                optimizer.zero_grad()

                y_pred = model(x)
                loss   = masked_mse_loss(y_pred, y_true, mask)

                loss.backward()

                # clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP
                )

                optimizer.step()

                total_train_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.6f}")

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # ── validation phase ───────────────────────────────────────────

            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x      = batch["x"].to(device)
                    y_true = batch["y_true"].to(device)
                    mask   = batch["mask"].to(device)

                    y_pred       = model(x)
                    val_loss     = masked_mse_loss(y_pred, y_true, mask)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # ── lr scheduling and tracking ─────────────────────────────────

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)

            # ── epoch summary ──────────────────────────────────────────────

            print(
                f"  train MSE: {avg_train_loss:.6f} | "
                f"val MSE: {avg_val_loss:.6f} | "
                f"lr: {current_lr:.2e}"
            )

            # ── checkpoint every epoch ─────────────────────────────────────

            # saved regardless of whether val loss improved so that any
            # Colab disconnect loses at most one epoch of work
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, best_val_loss,
                train_losses, val_losses,
                lr_history=lr_history,
            )

            # ── best model tracking ────────────────────────────────────────

            if avg_val_loss < best_val_loss:
                best_val_loss  = avg_val_loss
                patience_count = 0
                # separate copy so best weights are never overwritten
                # by a later epoch that performs worse
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, best_val_loss,
                    train_losses, val_losses,
                    lr_history=lr_history,
                    path=config.BEST_MODEL_PATH.replace(
                        ".pth", "_best.pth"
                    ),
                )
                print(f"  new best — saved to best_model_best.pth")
            else:
                patience_count += 1
                print(
                    f"  no improvement — patience "
                    f"{patience_count}/{config.PATIENCE}"
                )

            # ── early stopping ─────────────────────────────────────────────

            if patience_count >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # ── plots ──────────────────────────────────────────────────────────────

    _plot_loss_curve(train_losses, val_losses)
    _plot_lr_schedule(lr_history)

    # ── final summary ──────────────────────────────────────────────────────

    print(f"\nTraining complete.")
    print(f"Best val MSE : {best_val_loss:.6f}")
    print(f"Epochs run   : {len(train_losses)}")

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "lr_history":    lr_history,
        "best_val_loss": best_val_loss,
        "epochs_run":    len(train_losses),
    }


# ── private helpers ────────────────────────────────────────────────────────


def _recompute_patience(val_losses: list) -> int:
    """
    Recomputes patience counter from saved loss history on resume.

    When resuming we recompute patience from val_losses rather than
    storing it directly in the checkpoint — this way patience is always
    consistent with the actual loss history regardless of how training
    was interrupted.

    Args:
        val_losses : complete validation loss history from checkpoint

    Returns:
        patience count to resume early stopping from
    """
    if not val_losses:
        return 0

    best  = min(val_losses)
    count = 0

    for loss in reversed(val_losses):
        if loss > best:
            count += 1
        else:
            break

    return count


def _plot_loss_curve(
    train_losses: list,
    val_losses:   list,
) -> None:
    """
    Plot training and validation MSE across all epochs and save to disk.

    Marks the epoch with the best validation loss with a vertical dashed
    line so it is immediately clear where the model peaked.
    """
    epochs     = range(1, len(train_losses) + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_losses, label="Train MSE",      linewidth=2)
    ax.plot(epochs, val_losses,   label="Val MSE",        linewidth=2)
    ax.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch ({best_epoch})",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # log scale on y axis makes early large losses and later small
    # losses both visible on the same plot
    ax.set_yscale("log")

    plt.tight_layout()
    save_figure("loss_curve.png")
    plt.show()


def _plot_lr_schedule(lr_history: list) -> None:
    """
    Plot learning rate across all epochs and save to disk.

    Step-downs from ReduceLROnPlateau appear as sudden drops —
    visually confirms the scheduler fired at the right moments.
    """
    epochs = range(1, len(lr_history) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(epochs, lr_history, linewidth=2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("lr_schedule.png")
    plt.show()


# ── entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train()