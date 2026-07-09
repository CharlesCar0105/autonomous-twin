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

# Indices des rayons dans lidar = [-60, -30, 0, +30, +60].
CENTER_RAY = 2          # rayon central (droit devant)
SIDE_RAYS = (1, 3)      # rayons -30 / +30


def check_emergency_brake(lidar: list[float]) -> bool:
    """
    Vérifie si un freinage d'urgence est nécessaire.

    Signature d'un mur barrant la route : le **rayon central** (droit devant)
    devient court, confirme par au moins un rayon lateral (-30/+30). Le mur
    fait ~130 px de large : a moins de EMERGENCY_DISTANCE il bouche forcement
    le centre ET les cotes.

    Pourquoi exiger le rayon CENTRAL court (et pas juste 2 rayons sur 3) :
    sur une voie etroite (~45 px), les rayons ±30° tapent en permanence les
    bords de la voie a courte distance. Un critere "2 rayons avant courts"
    se declencherait donc tout le temps en ligne droite et bloquerait la
    voiture. Le rayon central, lui, voit loin le long de la voie tant qu'il
    n'y a pas d'obstacle reel devant -> c'est le bon discriminant.

    Une bordure frolee en virage serre ne raccourcit que les rayons
    lateraux, pas le central : c'est alors au controleur (PID/CNN) de gerer
    le braquage, pas au frein d'urgence.

    Args:
        lidar: Liste de 5 distances lidar (pixels).

    Returns:
        True si un obstacle est trop proche droit devant (frein immédiat).
    """
    if not lidar or len(lidar) < 5:
        return False

    if lidar[CENTER_RAY] >= EMERGENCY_DISTANCE:
        return False  # voie devant degagee -> pas de mur
    return any(lidar[i] < EMERGENCY_DISTANCE for i in SIDE_RAYS)


def get_emergency_commands() -> dict:
    """Retourne les commandes de freinage d'urgence."""
    return {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
