import os
import torch

# Configuration for LVLM-Seg

# Model Settings
# Using Qwen2-VL-7B-Instruct for stability as Qwen2.5/3 proved problematic with current environment/weights mismatch.
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct" 

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMAGE_MAX_PIXELS = 1024 * 1024
MIN_PIXELS = 256 * 256

# Dataset
DATASET_NAME = "lmms-lab/RefCOCO"
SPLIT = "val"

# Paths
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
