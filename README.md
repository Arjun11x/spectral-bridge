# Spectral Bridge — Few-Shot Audio Signal In-Painting

Reconstructing missing voltage readings from degraded magnetic tape recordings using a masked Transformer encoder.

**Best result: Val MSE 0.002024 — 2.67× better than cubic spline, 154× better than zero predictor.**

---

## Problem Statement

Each audio clip is 100ms at 1kHz (100 time steps). Only 20 context points are observed — the remaining 80 gap positions (80% of the signal) must be reconstructed. The model sees surviving context points and predicts all missing values in a single forward pass.

- **Dataset:** 80,000 training clips, 10,000 test clips
- **Input:** 20 context (time, value) pairs per clip
- **Output:** 80 predicted voltage values per clip
- **Metric:** MSE on gap positions only

---

## Results

| Method | Val MSE | vs Zero |
|---|---|---|
| Zero predictor | 0.311986 | 1.0× |
| Mean predictor | 0.062171 | 5.0× |
| Linear interpolation | 0.007958 | 39.2× |
| Cubic spline | 0.005403 | 57.7× |
| Simple MLP | 0.007928 | 39.4× |
| **SpectralTransformer (ours)** | **0.002024** | **154.1×** |

---

## Architecture

**Masked Transformer Encoder** — bidirectional self-attention over all 100 time steps simultaneously.

- Input: `[batch, 100, 2]` — (masked_value, is_context) per time step
- Learnable positional encoding
- 3 Transformer encoder layers, d_model=64, 4 attention heads
- Output: `[batch, 100]` — predicted voltage at every time step
- Loss: masked MSE on gap positions only (`mask == 0`)
- 156,609 trainable parameters

Bidirectional attention allows every gap position to attend to all 20 context points regardless of temporal distance — unlike interpolation methods which only use nearby points.

---

## Ablation Study

| Config | Params | Val MSE | Beats spline? |
|---|---|---|---|
| tiny (d_model=32) | 45,377 | 0.004691 | ✓ |
| **small (d_model=64)** | **156,609** | **0.002346** | **✓** |
| medium (d_model=128) | 543,233 | 0.061862 | ✗ |
| large (d_model=256) | 2,401,281 | 0.061862 | ✗ |

Small is the sweet spot. Medium and large failed to converge in 15 epochs — over-parameterized for this dataset size.

---

## Key Findings

- Interpolation baselines are strong because signals are smooth — cubic spline achieves 57.7× over zero predictor
- The Transformer beats cubic spline by 2.67× by leveraging global context rather than local neighborhood
- Bigger models are not better — small (156K params) outperforms large (2.4M params)
- MC Dropout uncertainty bands are extremely narrow — the model is highly confident in its predictions
- Edge positions (t=0, t=99ms) are consistently harder — fewer context points on one side

---

## Project Structure
```
spectral-bridge/
├── notebooks/
│   ├── 01_EDA.ipynb           — dataset exploration and quality checks
│   ├── 02_baselines.ipynb     — zero, mean, interpolation, spline, MLP baselines
│   ├── 03_transformer.ipynb   — training, evaluation, diagnostic plots
│   ├── 04_ablations.ipynb     — architecture size comparison
│   └── 05_predict.ipynb       — test set inference and submission
├── src/
│   ├── __init__.py            — exposes core symbols from src package
│   ├── config.py              — all hyperparameters and paths
│   ├── dataset.py             — SpectralDataset and DataLoader
│   ├── evaluate.py            — diagnostic plots and metrics
│   ├── model.py               — SpectralTransformer and masked MSE loss
│   ├── predict.py             — test set inference and submission generation
│   ├── train.py               — training loop with checkpointing
│   └── utils.py               — seed, device, checkpoint, plotting utilities
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run

**1. Clone and install**
```bash
git clone https://github.com/Arjun11x/spectral-bridge.git
cd spectral-bridge
pip install -r requirements.txt
```

**2. Add data**
Place `spectral_graffiti.csv` and `test_features_spectral.csv` in `data/`.

**3. Train**
```bash
python -m src.train
```
Saves best model to `results/checkpoints/best_model_best.pth`.

**4. Evaluate**
```bash
python -m src.evaluate
```
Requires trained model from step 3. Saves 6 diagnostic plots to `results/plots/`.

**5. Generate test predictions**
```bash
python -m src.predict data/test_features_spectral.csv
```
Saves `results/submission.csv` with 800,000 gap predictions.a/test_features_spectral.csv


---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pandas, numpy, matplotlib, scipy, tqdm

See `requirements.txt` for full list.

---

## Notebooks

All notebooks are designed to run on Google Colab with a T4 GPU (notebooks 03, 04, 05) or CPU (notebooks 01, 02). Each notebook mounts Google Drive for data access and clones this repo automatically.