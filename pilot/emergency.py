"""
emergency.py — Freinage d'urgence.

Responsabilités :
    - Analyser les données lidar en continu
    - Si obstacle détecté à distance critique → frein max
    - Priorité absolue sur les autres commandes
"""


# --- Constantes -----------------------------------------------------------

# Seuil de declenchement (pixels). Le Mur spawn a ~200 px ; a vitesse de
# croisiere (~110 px/s) la distance d'arret est ~30 px, donc 70 px laisse
# une marge nette avant impact.
EMERGENCY_DISTANCE = 70.0

# Indices des rayons frontaux dans lidar = [-60, -30, 0, +30, +60].
# On regarde les trois du milieu (-30, 0, +30).
FRONT_RAY_INDICES = (1, 2, 3)

# Nombre minimal de rayons frontaux sous le seuil pour declencher.
MIN_RAYS_TRIGGER = 2


def check_emergency_brake(lidar: list[float]) -> bool:
    """
    Vérifie si un freinage d'urgence est nécessaire.

    On exige qu'au moins MIN_RAYS_TRIGGER des trois rayons frontaux
    (-30, 0, +30) soient courts simultanement. C'est la signature d'un
    obstacle *large* barrant la route (le Mur, ~130 px de large) et non
    d'une simple bordure frolee par un seul rayon en virage serre -- dans
    ce dernier cas c'est au controleur lateral (PID/CNN) de gerer le
    braquage, pas au frein d'urgence.

    Choix de 2/3 (et non 3/3) : de loin, un mur etroit vu de face ne barre
    que le rayon central ; les rayons ±30° divergent et ne le touchent qu'a
    courte portee. Exiger 3/3 ferait freiner trop tard. 2/3 declenche des
    que le mur occupe le centre + un cote, avec une marge d'arret confortable.

    Args:
        lidar: Liste de 5 distances lidar (pixels).

    Returns:
        True si un obstacle large est trop proche devant (frein immédiat).
    """
    if not lidar or len(lidar) < 5:
        return False

    n_close = sum(1 for i in FRONT_RAY_INDICES if lidar[i] < EMERGENCY_DISTANCE)
    return n_close >= MIN_RAYS_TRIGGER


def get_emergency_commands() -> dict:
    """Retourne les commandes de freinage d'urgence."""
    return {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
