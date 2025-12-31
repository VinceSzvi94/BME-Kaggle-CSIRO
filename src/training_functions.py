import torch
import torch.nn as nn
from torchvision import models, utils
import torch.nn.functional as F
import numpy as np
import math
import tqdm
import os, time
import wandb

from src.model_wrapper import ModelWrapper
from src.dataloader import TARGET_WEIGHTS

# =========================================================
# METRICS
# =========================================================
def csiro_r2(preds, targets, weights=None):
    if weights is None:
        weights = torch.tensor(TARGET_WEIGHTS, device=preds.device)
    else:
        weights = torch.tensor(weights, device=preds.device)
    SS_res = torch.sum(weights * (targets - preds)**2)
    global_weighted_mean = torch.sum(weights * targets) / torch.sum(weights)
    SS_tot = torch.sum(weights * (targets - global_weighted_mean)**2)
    r2_scores = 1 - (SS_res / (SS_tot + 1e-8))  # Add epsilon to avoid division by zero
    return r2_scores

def csiro_r2_loss(preds, targets, weights=None):
    return 1 - csiro_r2(preds, targets, weights)

# =========================================================
# TRAINING
# =========================================================
def train_predictor(
        train_loader, val_loader, 
        wrapped_model: ModelWrapper, 
        num_epochs=10, 
        lr=1e-4, 
        weigh_decay=1e-4, # use 0 to turn off regularization 
        patience=3, 
        use_wandb=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            name=f"csiro_training_{time.strftime('%Y%m%d-%H%M%S')}",
            project="kaggle-csiro", 
            config={
            "num_epochs": num_epochs,
            "device": device,
            "lr": lr,
            "architecture": f"{wrapped_model.get_architecture_name()}",
            "patience": patience,
        })
        wandb.watch(wrapped_model, log="all", log_freq=10)

    # train
    print(f"Training {wrapped_model.get_architecture_name()}...")

    optim = torch.optim.Adam(wrapped_model.parameters(), lr=lr, weight_decay=weigh_decay)
    mse = nn.MSELoss()
    mse.to(device)

    # Track best model
    best_val_r2_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        wrapped_model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        epoch_r2_loss = 0
        epoch_mse = 0
        num_batches = 0
        
        for input_img, targets in loop:
            input_img = input_img.to(device)
            targets = targets.to(device)

            optim.zero_grad()
            preds = wrapped_model(input_img)

            # loss (calc mse too only for logging)
            r2_loss = csiro_r2_loss(preds, targets)
            mse_loss = mse(preds, targets)

            # backpropagation
            r2_loss.backward()
            optim.step()

            epoch_r2_loss += r2_loss.item()
            epoch_mse += mse_loss.item()
            num_batches += 1

            loop.set_postfix({
                "R2_loss": f"{r2_loss.item():.4f}",
                "mse": f"{mse_loss.item():.2f}",
            })

            # Log batch metrics
            if use_wandb:
                wandb.log({"batch_r2_loss": r2_loss.item(), "batch_mse": mse_loss.item()})

        # Log epoch metrics
        avg_r2_loss = epoch_r2_loss / num_batches
        avg_mse = epoch_mse / num_batches
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "avg_r2_loss": avg_r2_loss,
                "avg_mse": avg_mse
            })

        # Do validation
        if val_loader is not None:
            wrapped_model.eval()
            with torch.no_grad():
                r2_loss_vals = []
                mse_vals = []
                for input_img, targets in val_loader:
                    input_img = input_img.to(device)
                    targets = targets.to(device)
                    preds = wrapped_model(input_img)
                    r2_loss = csiro_r2_loss(preds, targets)
                    mse_loss = mse(preds, targets)
                    r2_loss_vals.append(r2_loss.item())
                    mse_vals.append(mse_loss.item())

                mean_r2_loss = sum(r2_loss_vals) / len(r2_loss_vals)
                mean_r2 = 1 - mean_r2_loss
                mean_mse = sum(mse_vals) / len(mse_vals)
                print(f"Validation after epoch {epoch}: R2={mean_r2:.2f}, R_loss={mean_r2_loss:.2f}, mse={mean_mse:.2f}")

                # Save best model
                if mean_r2_loss < best_val_r2_loss:
                    best_val_r2_loss = mean_r2_loss
                    best_model_state = wrapped_model.state_dict().copy()
                    epochs_without_improvement = 0
                    print(f"✓ New best model found! R² loss: {best_val_r2_loss:.4f}")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epoch(s)")
                
                if use_wandb:
                    wandb.log({
                        "val/epoch": epoch,
                        "val/r2": mean_r2,
                        "val/r2_loss": mean_r2_loss,
                        "val/mse": mean_mse,
                        "val/best_r2_loss": best_val_r2_loss,
                        "val/epochs_without_improvement": epochs_without_improvement,
                    })
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
                    break
        else:
            print(f"Epoch {epoch}: validation skipped (no pairs).")
    
    # Load best model state before finishing
    if best_model_state is not None:
        wrapped_model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation R² loss: {best_val_r2_loss:.4f}")

    if use_wandb:
        wandb.finish()
    
    return wrapped_model

def save_models(model: ModelWrapper):
    os.makedirs("models", exist_ok=True)
    save_name = f"models/{model.get_architecture_name()}_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Models saved as {save_name}")

