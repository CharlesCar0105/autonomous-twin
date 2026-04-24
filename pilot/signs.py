"""
signs.py — Détection et classification des panneaux routiers.

Responsabilités :
    - Localiser les panneaux dans l'image caméra (contours / sliding window)
    - Classifier : Stop, 30 km/h, 50 km/h, 90 km/h, Aucun
    - Retourner la décision associée
"""

# TODO: Implémenter ResNet18/MobileNetV2 (transfer learning ImageNet)
# TODO: Pipeline : détection → crop → classification → décision
# Classes : Stop, 30, 50, 90, Aucun
# Métrique cible : Accuracy > 95%
