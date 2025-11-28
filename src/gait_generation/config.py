import torch

def set_seed(seed: int = 42):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

GAIT_BASE_DIR = "BB-MAS_Dataset/BB-MAS_Dataset"
SENSOR_TYPES = ("Accelerometer", "Gyroscope")
RARE_THRESH = 10
TEST_SIZE = 0.15
VAL_SIZE = 0.15
NUM_WORKERS = 2
COND_COLS = ()

WINDOW_SIZE = 128
WINDOW_STRIDE = 64

LATENT_DIM = 128
HIDDEN_DIM = 1024
N_LAYERS = 4
CAT_EMBED_DIM = 16
DROPOUT = 0.05

VAE_EPOCHS = 50
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3
KL_MAX_BETA = 1.0
KL_WARMUP_EPOCHS = 20
VAE_PATIENCE = 8
KL_BETA = 1.0

USE_SMOOTHNESS_LOSS = True
USE_DISTRIBUTION_LOSS = True
LAMBDA_SMOOTH = 1e-3
LAMBDA_PHYS = 5e-3
LAMBDA_SPECTRAL = 1e-2

T = 400
BETA_SCHEDULE = "cosine"
DDPM_EPOCHS = 200
DDPM_BATCH_SIZE = 512
DDPM_LR = 2e-4
DDPM_PATIENCE = 60
DDPM_USE_EMA = True
DDPM_EMA_DECAY = 0.999
SAMPLE_STEPS = 300
NUM_COND_BUCKETS = 128
DDPM_USE_MU_ONLY = True

N_SAMPLES_DEMO = 2048
OUT_CSV_PREFIX = "synth_gait"

SEED = 42
SAVE_DIR = "checkpoints/gait"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PIN_MEMORY = (DEVICE == "cuda")

