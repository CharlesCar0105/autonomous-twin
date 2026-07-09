"""
signs_arch.py -- Architecture du classifieur de panneaux (D5).

MobileNetV2 pretrained ImageNet, features GELEES, tete de classification
remplacee (MLP : Dropout + Linear 1280 -> 256 + ReLU + Dropout + Linear
256 -> 5). Contrat d'entree : crop RGB float [0, 1] -> preprocess_batch
(normalise ImageNet + upscale FEAT_INPUT_SIZE) -> net(x).

Partage entre scripts/train_signs.py (training) et pilot/signs.py (inference),
comme cnn_drive_arch.py pour le CNN conduite.
"""

import torch.nn as nn
from torchvision import models

# Ordre canonique = tri alphabetique des dossiers dataset. NE PAS reordonner.
SIGN_CLASSES = ["30", "50", "90", "aucun", "stop"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Resolution d'entree pour l'extraction de features. 96px donne une carte
# 3x3 avant le GAP de MobileNetV2 : trop grossier pour discriminer les
# chiffres 30/50/90 (constat empirique : val_acc 0.83, confusion exclusive
# entre les 3 limites). A 224 (natif ImageNet) la carte fait 7x7.
FEAT_INPUT_SIZE = 224


def preprocess_batch(x):
    """Pipeline canonique d'entree du reseau : (B, 3, H, W) float [0, 1]
    -> normalisation ImageNet PUIS upscale bilineaire FEAT_INPUT_SIZE.
    Partage entre training (scripts/train_signs.py) et inference
    (pilot/signs.py) : toute evolution passe ICI, jamais en local.
    H, W typiques : 96 (dataset) ou taille du crop detecteur."""
    import torch
    import torch.nn.functional as F
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return F.interpolate(x, size=(FEAT_INPUT_SIZE, FEAT_INPUT_SIZE),
                         mode="bilinear", align_corners=False)


def build_signs_net(pretrained: bool = True) -> nn.Module:
    """Construit le reseau. pretrained=True telecharge les poids ImageNet
    (1ere fois seulement, cache torch) ; False pour recharger un .pth."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    for p in net.features.parameters():
        p.requires_grad = False
    net.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, len(SIGN_CLASSES)),
    )
    return net
