#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Define the training routine for the PCA regression pipeline

# Description : 
"""
    compute_metrics:
        Compute MAE and R² metrics between true and predicted values.
    train_pc1_sequential:
        Train a PC1Regressor model for a specific PC index.
        The model is trained sequentially, one PC at a time.
        The model is trained using the raw DRR channels (AP, ML).
        The model is trained using the Adam optimizer.
        The model is trained for a fixed number of epochs.
        The model is trained using a dropout rate that is annealed from DROPOUT_START to DROPOUT_END.
        The model is trained using a learning rate that is reduced if the validation loss does not improve.
        The model is trained using a patience parameter.
        The model is trained using a max_grad_norm parameter.
"""
#%% ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pickle
from tqdm import tqdm

from config import *
from model import *
from dataset import *


def compute_metrics(y_true, y_pred):
    """
    Compute MAE and R² metrics between true and predicted values.
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        mae: mean absolute error
        r2: R-squared score
    """
    mae = np.mean(np.abs(y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
    return mae, r2

def train_pc1_sequential(pca, scaler, idx, pc_idx):
    """
    Train a PC1Regressor model for a specific PC index.
    Args:
        pca: PCA model
        scaler: StandardScaler
        idx: voxel indices
        pc_idx: which PC to train (0-based index)
    """
    model_dir = os.path.join(OUT_ROOT,"PC_models")
    os.makedirs(model_dir, exist_ok=True); print(f"Model directory created: {model_dir}")
    splits = pickle.load(open(SPLIT_PATH,"rb"))
    train = [d.split("/CT")[0] for d in splits["train"]]
    val   = [d.split("/CT")[0] for d in splits["val"]]

    print("Training length : ", len(train))
    print("Validation length : ", len(val))

    ds_tr = TibiaMultiPCDataset(train, pca, scaler, idx, pc_idx=pc_idx, augment=True)
    ds_va = TibiaMultiPCDataset(val,   pca, scaler, idx, pc_idx=pc_idx, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False)

    model = PC1Regressor().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_PC1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, mode='min')

    best_mae, patience = float("inf"), 0
    best_r2 = float("-inf")
    for ep in range(1, EPOCHS_PC1+1):
        # anneal dropout
        p = DROPOUT_START - (DROPOUT_START-DROPOUT_END)*min(ep/EPOCHS_PC1,1)
        for m in model.modules():
            if isinstance(m, MCDropout): m.p = p

        # Training phase
        model.train(); tl=0
        for drr, tgt, _ in tqdm(dl_tr, desc="Training ..."):
            inp = drr[:,:2].to(DEVICE)
            tgt = tgt.to(DEVICE)
            pred = model(inp)
            loss = F.smooth_l1_loss(pred, tgt)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step(); tl+=loss.item()
        tl/=len(dl_tr)

        # Validation phase with metrics
        model.eval()
        all_true = []
        all_pred = []
        vl = 0
        with torch.no_grad():
            for drr, tgt, _ in tqdm(dl_va, desc="Validation..."):
                inp = drr[:,:2].to(DEVICE)
                tgt = tgt.to(DEVICE)
                draws = torch.stack([model(inp) for _ in range(20)])
                mean_pred = draws.mean(0)
                vl += F.smooth_l1_loss(mean_pred, tgt).item()
                
                # Store predictions and true values for metrics
                all_true.append(tgt.cpu().numpy())
                all_pred.append(mean_pred.cpu().numpy())
        
        # Compute validation metrics
        vl /= len(dl_va)
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        mae, r2 = compute_metrics(all_true, all_pred)
        
        # Update learning rate based on validation loss
        #sched.step(vl)
        sched.step(r2)

        # Save model if MAE improved
        if r2 > best_r2:
            best_r2, patience = r2, 0
            best_mae = mae
            print(f"New best model saved with MAE: {mae:.4f} (R²: {r2:.4f})")
            torch.save(model.state_dict(), os.path.join(model_dir,f"model_pc{pc_idx+1}_sequential.pt"))
        
        elif [ep > 10 and r2 < best_r2]:
            patience+=1
            if patience>10:
                print(f"\n\n ✅ PC{pc_idx+1} early stop @ epoch {ep} (best MAE: {best_mae:.4f}, best r^2 : {best_r2:.4f})"); break

        print(f"\n\n ✅ [PC{pc_idx+1}] Ep{ep:03d} tr {tl:.4f} vl {vl:.4f} MAE {mae:.4f} R² {r2:.4f}")
    
    return model