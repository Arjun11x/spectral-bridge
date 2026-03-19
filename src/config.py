import os

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
TRAIN_FILE      = os.path.join(DATA_DIR, "spectral_graffiti.csv")
RESULTS_DIR     = os.path.join(ROOT_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

# ── Data ───────────────────────────────────────────────────────────────────

SEQ_LEN     = 100  # 100ms clips at 1kHz = 100 time steps per sample
FEATURE_DIM = 2    # (masked_value, is_context) — tells model what to trust vs predict
VAL_SPLIT   = 0.2  # 80/20 train-val split across 80,000 samples

# ── Model ──────────────────────────────────────────────────────────────────

# D_MODEL must be divisible by N_HEADS
# Default is "small" config — see ABLATION_CONFIGS for full search space
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 3
D_FF       = D_MODEL * 4  # standard Transformer convention
DROPOUT    = 0.1
ACTIVATION = "gelu"       # smoother than relu for continuous-valued signals

# ── Training ───────────────────────────────────────────────────────────────

BATCH_SIZE    = 32    # fits comfortably in Colab T4 VRAM
NUM_EPOCHS    = 20    # upper ceiling — early stopping determines actual stop
LEARNING_RATE = 1e-3  # standard AdamW starting point (Vaswani et al. 2017)
WEIGHT_DECAY  = 1e-4  # light regularization suits regression over classification
GRAD_CLIP     = 1.0   # prevents exploding gradients common in Transformer training
PATIENCE      = 5     # early stopping patience in epochs
LR_FACTOR     = 0.5   # LR multiplier on plateau
LR_PATIENCE   = 2     # epochs to wait before reducing LR

# ── Reproducibility ────────────────────────────────────────────────────────

SEED = 42

# ── Ablation search space ──────────────────────────────────────────────────

# Systematically scale d_model, n_heads, n_layers together.
# Results documented in notebooks/04_ablations.ipynb.
ABLATION_CONFIGS = [
    {"name": "tiny",   "d_model": 32,  "n_heads": 2, "n_layers": 2},
    {"name": "small",  "d_model": 64,  "n_heads": 4, "n_layers": 3},  # default
    {"name": "medium", "d_model": 128, "n_heads": 8, "n_layers": 4},
    {"name": "large",  "d_model": 256, "n_heads": 8, "n_layers": 6},
]