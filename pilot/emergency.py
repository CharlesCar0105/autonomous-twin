"""
emergency.py — Freinage d'urgence.

Responsabilités :
    - Analyser les données lidar en continu
    - Si obstacle détecté à distance critique → frein max
    - Priorité absolue sur les autres commandes
"""


# --- Constantes -----------------------------------------------------------

# Seuil en pixels. Desactive en pratique pour Sprint 1 (valeur tres basse) :
# le freinage d'urgence est concu pour le MUR qui apparait (touche Espace
# cote simu, pas encore implemente). Il ne doit PAS se declencher juste
# parce qu'un rayon voit une bordure de piste en virage serre -- dans ce
# cas c'est au PID de gerer le steering. Relever quand le mur sera en place.
EMERGENCY_DISTANCE = 10.0


def check_emergency_brake(lidar: list[float]) -> bool:
    """
    Vérifie si un freinage d'urgence est nécessaire.

    Args:
        lidar: Liste de 5 distances lidar (pixels).

    Returns:
        True si un obstacle est trop proche (frein immédiat).
    """
    if not lidar:
        return False

    # Si UN des rayons détecte un obstacle sous le seuil → urgence
    min_distance = min(lidar)
    return min_distance < EMERGENCY_DISTANCE


def get_emergency_commands() -> dict:
    """Retourne les commandes de freinage d'urgence."""
    return {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
