#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for pixel-wise water body extraction
using a DenseNet-style fully convolutional neural network (CNN).

This script implements:
- Dataset loading from raster files listed in CSV tables
- Model training and validation
- Best-model checkpoint saving based on validation loss
- Logging and visualization of training dynamics

The model outputs raw logits; sigmoid activation is applied
only during inference.
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import rasterio

# Suppress rasterio warnings for non-georeferenced rasters
warnings.filterwarnings(
    "ignore",
    category=rasterio.errors.NotGeoreferencedWarning
)

# Avoid OpenMP duplicated library issue on some platforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ======================
# Dataset definition
# ======================
class Dataset_Loader(Dataset):
    """
    Custom PyTorch Dataset for loading raster-based
    image–label pairs.

    The dataset is defined by a CSV file containing:
    - scene_file: path to multi-band input raster
    - truth_file: path to corresponding binary mask
    """
    def __init__(self, data_path):
        self.scene_files = pd.read_csv(data_path)["scene_file"].tolist()
        self.truth_files = pd.read_csv(data_path)["truth_file"].tolist()

    def __len__(self):
        """Return the number of samples."""
        return len(self.scene_files)

    def __getitem__(self, idx):
        """
        Load one sample consisting of:
        - scene: multi-band raster (H, W, C)
        - truth: binary label raster (H, W)
        """
        scene_path = self.scene_files[idx]
        truth_path = self.truth_files[idx]

        try:
            # Read multi-band input image
            with rasterio.open(scene_path) as src:
                scene = src.read().transpose((1, 2, 0)).astype("float32")

            # Read corresponding ground-truth mask
            with rasterio.open(truth_path) as src:
                truth = src.read(1).astype("float32")

        except Exception as e:
            raise RuntimeError(
                f"Error reading file at index {idx}: {e}"
            )

        return {"scene": scene, "truth": truth}


# ======================
# Model architecture
# ======================
class DenseNetConvNet(nn.Module):
    """
    A DenseNet-style fully convolutional network
    for pixel-wise classification.
    """
    def __init__(self, n_channels, n_classes):
        super(DenseNetConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16 + n_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32 + 16 + n_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64 + 32 + 16 + n_channels,
            out_channels=n_classes,
            kernel_size=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1_cat = torch.cat((x, x1), dim=1)

        x2 = self.relu(self.conv2(x1_cat))
        x2_cat = torch.cat((x1_cat, x2), dim=1)

        x3 = self.relu(self.conv3(x2_cat))
        x3_cat = torch.cat((x2_cat, x3), dim=1)

        x4 = self.conv4(x3_cat)
        return x4

# ======================
# Main training workflow
# ======================
if __name__ == "__main__":

    # -------- Basic settings --------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    lr = 1e-5                 # Learning rate
    num_epochs = 3            # Number of training epochs
    scale = 10000             # Input normalization factor
    bs = 64                   # Batch size

    # -------- Dataset & DataLoader --------
    train_dataset = Dataset_Loader(
        data_path="./data/train_files.csv"
    )
    valid_dataset = Dataset_Loader(
        data_path="./data/val_files.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False
    )

    # -------- Model initialization --------
    net = DenseNetConvNet(
        n_channels=6,
        n_classes=1
    ).to(device)

    # Binary classification loss (logits-based)
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr
    )

    best_loss = float("inf")
    metrics_data = []

    # -------- Training & Validation loop --------
    for epoch in range(num_epochs):

        # ---- Training phase ----
        net.train()
        train_loss = 0.0
        print("Training...")

        for batch in train_loader:
            image = batch["scene"].to(device).float()
            label = batch["truth"].to(device).float()

            # Convert to (N, C, H, W) and normalize
            image = image.permute(0, 3, 1, 2) / scale
            label = label.unsqueeze(1)

            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation phase ----
        net.eval()
        valid_loss = 0.0
        print("Validating...")

        with torch.no_grad():
            for batch in valid_loader:
                image = batch["scene"].to(device).float()
                label = batch["truth"].to(device).float()

                image = image.permute(0, 3, 1, 2) / scale
                label = label.unsqueeze(1)

                output = net(image)
                loss = criterion(output, label)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        # ---- Save best model checkpoint ----
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {"state_dict": net.state_dict()},
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            print(
                f"Best model saved "
                f"(Val Loss = {best_loss:.4f})"
            )

        # ---- Logging ----
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {valid_loss:.4f}"
        )

        metrics_data.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": valid_loss
        })

    # -------- Save metrics & visualization --------
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(
        "training_metrics.xlsx",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Train Loss"],
        label="Train Loss"
    )
    plt.plot(
        metrics_df["Epoch"],
        metrics_df["Val Loss"],
        label="Val Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Training complete.")