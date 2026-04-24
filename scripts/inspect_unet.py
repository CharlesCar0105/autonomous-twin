"""
inspect_unet.py -- Sanity check du U-Net apres training.

A lancer DANS le notebook Colab avec :
    %run scripts/inspect_unet.py

Charge le best `.pth` (models/unet_road.pth), prend quelques samples val,
affiche (camera, GT, prediction, proba) + stats de fraction de pixels
"predits = 1". Aide a detecter si le modele a appris ou s'il triche
(tout-blanc).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT = Path.cwd() if Path.cwd().name == "autonomous-twin" else Path.cwd() / "autonomous-twin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ROOT / "models" / "unet_road.pth"
VAL_DIR = ROOT / "data" / "val"


# --- Meme archi que le notebook ------------------------------------------

def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(3, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# --- Inspection ----------------------------------------------------------

def main() -> None:
    print(f"[inspect] device = {DEVICE}")
    print(f"[inspect] weights = {WEIGHTS}")

    model = UNet(base=32).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()

    files = sorted(VAL_DIR.rglob("*.npz"))
    if not files:
        print("[inspect] AUCUN fichier val trouve.")
        return

    # Echantillonne 1 frame par circuit val (diversifie).
    per_circuit: dict[str, Path] = {}
    for f in files:
        per_circuit.setdefault(f.parent.name, f)
    samples = list(per_circuit.values())[:6]

    print(f"[inspect] {len(samples)} samples val ({len(per_circuit)} circuits disponibles)")

    # Stats globales sur le val complet : mean(GT) et mean(pred)
    mean_gt_all = 0.0
    mean_pred_all = 0.0
    n_total = 0
    for f in files:
        d = np.load(f)
        mean_gt_all += d["mask"].mean()
        with torch.no_grad():
            t = torch.from_numpy(d["camera"].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            probs = torch.sigmoid(model(t)).cpu().squeeze().numpy()
        mean_pred_all += (probs > 0.5).mean()
        n_total += 1
    mean_gt_all /= n_total
    mean_pred_all /= n_total
    print(f"[inspect] val complet : GT mean = {mean_gt_all:.4f}  |  pred mean = {mean_pred_all:.4f}")
    print(f"[inspect] diff = {abs(mean_gt_all - mean_pred_all):.4f}  "
          f"({'OK' if abs(mean_gt_all - mean_pred_all) < 0.05 else 'WARN: predit peut-etre tout blanc'})")

    # Grille visuelle.
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i, p in enumerate(samples):
        d = np.load(p)
        img = d["camera"].astype(np.float32) / 255.0
        mask = d["mask"]
        with torch.no_grad():
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            probs = torch.sigmoid(model(t)).cpu().squeeze().numpy()
            pred = (probs > 0.5).astype(np.uint8)

        frac_gt = mask.mean()
        frac_pr = pred.mean()
        diff = np.abs(pred.astype(int) - mask.astype(int)).mean()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{p.parent.name}  (camera)")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"GT mean={frac_gt:.3f}")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"pred mean={frac_pr:.3f}  err={diff:.3f}")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(probs, cmap="viridis", vmin=0, vmax=1)
        axes[i, 3].set_title("prob sigmoid")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
else:
    # Mode `%run` : execute quand meme.
    main()
