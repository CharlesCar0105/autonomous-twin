"""
cnn_drive_arch.py -- Architecture du CNN de conduite (behavioral cloning).

Module separe pour partager l'archi entre le notebook d'entrainement
et l'inference (pilot.control.cnn_policy). Evite la duplication.

Input :
    mask  (B, 1, 64, 64)  float [0, 1]   masque route U-Net ou GT
    lidar (B, 5)          float [0, 1]   normalise (px / 300)

Output :
    (B, 2)  [steering_normalized [-1, 1], throttle [0, 1]]
            steering_deg = steering_normalized * 45
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DriveCNN(nn.Module):
    """CNN leger (~300 k params) behavioral cloning."""

    def __init__(self, lidar_dim: int = 5, mask_feat_dim: int = 128):
        super().__init__()
        # Extracteur spatial sur le masque : 64 -> 32 -> 16 -> 8.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8
        )
        self.conv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, mask_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Encodeur lidar leger.
        self.lidar_head = nn.Sequential(
            nn.Linear(lidar_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
        )

        # Fusion + tete de regression.
        self.regressor = nn.Sequential(
            nn.Linear(mask_feat_dim + 16, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, mask: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        feat_mask = self.conv_head(self.conv(mask))
        feat_lidar = self.lidar_head(lidar)
        z = torch.cat([feat_mask, feat_lidar], dim=1)
        out = self.regressor(z)
        steering = torch.tanh(out[:, 0:1])         # [-1, 1]
        throttle = torch.sigmoid(out[:, 1:2])      # [0, 1]
        return torch.cat([steering, throttle], dim=1)
