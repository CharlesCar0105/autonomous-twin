"""
control.py — Contrôle du véhicule (CNN conduite + PID sécurité).

Responsabilités :
    - CNN conduite : image 64×64 + 5 lidar → angle + accélération
    - PID de sécurité en backup
    - Calcul des commandes finales (steering, throttle, brake)
"""

# TODO: Implémenter CNN conduite (Behavioral Cloning)
# TODO: Implémenter PID backup (lidar uniquement)
# Entrée CNN : image 64×64 (ou masque segmenté) + 5 valeurs lidar
# Sortie CNN : angle volant (float) + accélération (float)
