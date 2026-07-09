"""
train_signs.py -- Entrainement de la tete du classifieur panneaux (CPU-ok).

Backbone gele (D5) -> on precalcule les features MobileNetV2 (1280-d) UNE
FOIS pour toutes les images (~2-3 min CPU), puis on entraine uniquement la
tete (MLP) sur ce cache (quelques secondes par epoch). Leçon audit Q8 :
l'historique complet (loss/acc train+val par epoch) est persiste en JSON a
cote du modele.

Sorties :
    models/signs_cls.pth           state_dict COMPLET (features ImageNet + tete)
    models/signs_cls_history.json  historique + matrice de confusion val

Usage : python scripts/train_signs.py [--epochs 100] [--lr 1e-3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pilot.signs_arch import (FEAT_INPUT_SIZE, SIGN_CLASSES, build_signs_net,
                              preprocess_batch)

DATA = ROOT / "data" / "signs"
MODELS = ROOT / "models"


def _load_split_tensors(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge toutes les images d'un split -> (X (N,3,96,96) brut [0,1], y (N,)).

    Pas de normalisation ici : le preprocessing canonique (normalisation
    ImageNet + upscale) est centralise dans pilot.signs_arch.preprocess_batch."""
    xs, ys = [], []
    for ci, cls in enumerate(SIGN_CLASSES):
        for f in sorted((DATA / split / cls).glob("*.png")):
            xs.append(np.asarray(Image.open(f), dtype=np.float32) / 255.0)
            ys.append(ci)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
    return x, torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def _extract_features(net: nn.Module, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """Features 1280-d (avant classifier), backbone en eval.

    Preprocessing canonique (normalisation ImageNet + upscale FEAT_INPUT_SIZE)
    applique par batch a la volee via preprocess_batch (jamais un tenseur 224
    complet en RAM, on ne garde que les tenseurs 96px bruts)."""
    net.eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = preprocess_batch(x[i:i + batch])
        f = net.features(xb)
        outs.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
        if i % (batch * 20) == 0:
            print(f"    features {i}/{len(x)}")
    return torch.cat(outs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print("[train_signs] chargement images...")
    x_train, y_train = _load_split_tensors("train")
    x_val, y_val = _load_split_tensors("val")
    print(f"  train {tuple(x_train.shape)}  val {tuple(x_val.shape)}")

    net = build_signs_net(pretrained=True)

    print("[train_signs] extraction features (backbone gele, 1 seule fois)...")
    t0 = time.time()
    f_train = _extract_features(net, x_train)
    f_val = _extract_features(net, x_val)
    print(f"  features en {time.time()-t0:.0f}s : train {tuple(f_train.shape)}")

    head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(f_train.shape[1], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, len(SIGN_CLASSES)),
    )
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    history, best_acc, best_state = [], 0.0, None

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(f_train))
        tot_loss = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(head(f_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            logits = head(f_val)
            val_loss = F.cross_entropy(logits, y_val).item()
            val_acc = (logits.argmax(1) == y_val).float().mean().item()
            train_acc = (head(f_train).argmax(1) == y_train).float().mean().item()
        history.append({"epoch": epoch, "train_loss": tot_loss / len(f_train),
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc})
        star = ""
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.clone() for k, v in head.state_dict().items()}
            star = "  *best"
        print(f"  epoch {epoch:2d}  train_acc {train_acc:.4f}  val_acc {val_acc:.4f}{star}")

    head.load_state_dict(best_state)

    # Matrice de confusion val (listes int, ordre SIGN_CLASSES).
    with torch.no_grad():
        pred = head(f_val).argmax(1)
    conf = [[int(((y_val == i) & (pred == j)).sum()) for j in range(len(SIGN_CLASSES))]
            for i in range(len(SIGN_CLASSES))]

    # Injecte la tete dans le reseau complet et sauve le state_dict entier.
    net.classifier.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), MODELS / "signs_cls.pth")
    (MODELS / "signs_cls_history.json").write_text(json.dumps({
        "best_val_acc": best_acc, "classes": SIGN_CLASSES,
        "confusion_val": conf, "history": history,
        "args": vars(args), "feat_input_size": FEAT_INPUT_SIZE,
    }, indent=2), encoding="utf-8")

    print(f"[train_signs] best val_acc = {best_acc:.4f}")
    print(f"  confusion (lignes=verite, colonnes=pred, ordre {SIGN_CLASSES}) :")
    for cls, row in zip(SIGN_CLASSES, conf):
        print(f"    {cls:6s} {row}")
    print(f"  -> {MODELS/'signs_cls.pth'}")
    assert best_acc > 0.95, f"cible spec >95% non atteinte : {best_acc:.4f}"


if __name__ == "__main__":
    main()
