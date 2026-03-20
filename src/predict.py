import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.model import build_model
from src.dataset import SpectralDataset
from src.utils import get_device, ensure_dirs


def predict(test_csv_path: str, output_path: str = None) -> str:
    """
    Generate gap predictions for the test set and save as submission.csv.

    Loads the best trained model, runs inference on the test CSV, and
    writes predictions only for gap positions (Is_Context=0) in the
    format required for submission.

    Args:
        test_csv_path : path to test_features_spectral.csv
        output_path   : where to save submission.csv.
                        Defaults to results/submission.csv.

    Returns:
        path to the saved submission CSV
    """
    device     = get_device()
    ensure_dirs()
    output_path = output_path or os.path.join(
        config.RESULTS_DIR, "submission.csv"
    )

    # ── load best model ────────────────────────────────────────────────────

    model     = build_model().to(device)
    best_path = config.BEST_MODEL_PATH.replace(".pth", "_best.pth")

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best model not found at {best_path}.\n"
            f"Run train.py first to generate a trained model."
        )

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded best model from {best_path}")
    print(model)

    # ── load test data ─────────────────────────────────────────────────────

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            f"Test CSV not found at {test_csv_path}."
        )

    # read raw CSV to preserve Sample_ID and Time_ms for submission
    test_df  = pd.read_csv(test_csv_path)
    dataset  = SpectralDataset(test_csv_path)
    loader   = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE * 4,  # larger batch — no gradients needed
        shuffle=False,                      # must stay in original order
        num_workers=0,                      # safe default for inference
        pin_memory=torch.cuda.is_available(),
    )

    # ── run inference ──────────────────────────────────────────────────────

    all_predictions = []

    print(f"Running inference on {len(dataset):,} test samples...")

    with torch.no_grad():
        for batch in loader:
            x      = batch["x"].to(device)
            y_pred = model(x)                   # [batch, 100]
            all_predictions.append(y_pred.cpu().numpy())

    # shape: [n_samples, 100]
    all_predictions = np.concatenate(all_predictions, axis=0)

    # ── build submission dataframe ─────────────────────────────────────────

    # flatten predictions back to match the original CSV row order.
    # dataset._build_samples sorts each group by Time_ms so we must
    # reconstruct in the same order.
    records = []

    for sample_pos, (sample_id, group) in enumerate(
        test_df.groupby("Sample_ID")
    ):
        group      = group.sort_values("Time_ms").reset_index(drop=True)
        preds      = all_predictions[sample_pos]  # [100]

        for row_idx, row in group.iterrows():
            if row["Is_Context"] == 0:
                # clip predictions to observed value range to avoid
                # out-of-distribution outputs
                pred_value = float(np.clip(
                    preds[int(row["Time_ms"]) - 1],
                    test_df["Value"].min(),
                    test_df["Value"].max(),
                ))
                records.append({
                    "Sample_ID":       int(row["Sample_ID"]),
                    "Time_ms":         int(row["Time_ms"]),
                    "Predicted_Value": round(pred_value, 4),
                })

    submission_df = pd.DataFrame(records)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to {output_path}")
    print(f"Total gap predictions : {len(submission_df):,}")
    print(f"Unique samples        : {submission_df['Sample_ID'].nunique():,}")
    print(f"\nSample preview:")
    print(submission_df.head(10).to_string(index=False))

    return output_path


# ── entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_test_csv>")
        print("Example: python -m src.predict data/test_features_spectral.csv")
        sys.exit(1)

    predict(test_csv_path=sys.argv[1])
