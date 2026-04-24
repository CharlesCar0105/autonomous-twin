"""
perception.py — U-Net segmentation de la route.

Responsabilités :
    - Définir l'architecture U-Net (PyTorch)
    - Inférence : image RGB → masque binaire (route / non-route)
    - Pré-processing et post-processing de l'image
"""

# TODO: Implémenter U-Net léger (3-4 niveaux encodage)
# Entrée : image RGB 128×128
# Sortie : masque binaire (route=1, non-route=0)
# Loss : Dice Loss + BCE
# Métrique cible : IoU > 0.90
