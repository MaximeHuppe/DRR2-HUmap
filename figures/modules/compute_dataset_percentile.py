import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import pickle

# === Setup ===
TRAIN_CT_DIR = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2"  # Replace with your actual CT path

def collect_foreground_voxels(data_dirs, ct_subfolder="CT", filename="CT_reg.npy", hu_threshold=-1000):
    """Collect all foreground voxels (HU > threshold) from the training set."""
    hu_values = []
    for path in tqdm(data_dirs, desc="Collecting foreground HU values"):
        ct_path = os.path.join(path, ct_subfolder, filename)
        if not os.path.isfile(ct_path):
            continue
        vol = np.load(ct_path).astype(np.float32)
        mask = vol > hu_threshold
        hu_values.append(vol[mask])
    if hu_values:
        return np.concatenate(hu_values)
    return np.array([])

train_dirs_path = os.path.join(TRAIN_CT_DIR, "data_splitting_dir.pkl")
splits = pickle.load(open(train_dirs_path, "rb"))
train_dirs = splits["train"]

# Collect HU values from training set
foreground_voxels = collect_foreground_voxels(train_dirs)
HU_min = np.percentile(foreground_voxels, 1.0)
hu_max_999 = np.percentile(foreground_voxels, 99.9) if foreground_voxels.size > 0 else None

print(f"HU_min: {HU_min}, hu_max_999: {hu_max_999}")