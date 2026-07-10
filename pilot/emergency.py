"""
emergency.py — Freinage d'urgence.

Responsabilités :
    - Analyser les données lidar en continu
    - Si obstacle détecté à distance critique → frein max
    - Priorité absolue sur les autres commandes
"""


# --- Constantes -----------------------------------------------------------

# Indices des rayons frontaux dans lidar = [-60, -30, 0, +30, +60].
# On regarde les trois du milieu (-30, 0, +30).
FRONT_RAY_INDICES = (1, 2, 3)

# Nombre minimal de rayons frontaux sous le seuil pour declencher.
MIN_RAYS_TRIGGER = 2

# --- Distance dynamique -----------------------------------------------------
# Seuil = distance d'arret physique (freinage plein pot depuis la vitesse
# courante) + marge de securite, avec un plancher a l'arret/tres basse
# vitesse. BRAKE_FORCE et SPEED_SCALE sont dupliques depuis
# simulator/physics.py (le pilote ne doit PAS dependre du simulateur en
# production -- il recoit ses capteurs par ZMQ, cf pilot/network.py -- donc
# pas d'import cross-package ; GARDER CES DEUX VALEURS SYNCHRONISEES si
# physics.py change).
_BRAKE_FORCE_PX_S2 = 200.0   # cf simulator/physics.py::BRAKE_FORCE
_SPEED_SCALE = 0.72          # cf simulator/physics.py::SPEED_SCALE (px/s -> km/h)

# Marge ajoutee a la distance d'arret theorique (px). Calibree empiriquement
# (rapport optimisation, agent C) : plus petite marge encore compatible avec
# 0 faux positif sur 40 571 frames de conduite reelle en virage.
EMERGENCY_MARGIN_PX = 38.0

# Plancher de distance (px) a vitesse quasi nulle -- la distance d'arret
# theorique tend vers 0, on garde une marge de securite minimale absolue.
EMERGENCY_FLOOR_PX = 20.0

# Filtre de planeite : un MUR (rectangle plat perpendiculaire au cap) donne
# des rayons frontaux quasi egaux ; une bordure frolee en virage serre donne
# des distances asymetriques (interieur court, exterieur nettement plus
# long). On exige un ecart max entre le rayon le plus court et le plus long
# parmi les 3 rayons frontaux -- c'est ce qui elimine les faux positifs de
# virage la ou la seule distance dynamique ne suffit pas (spread observe
# jusqu'a 12 px sur un vrai virage gen_014).
FLATNESS_MAX_SPREAD_PX = 28.0


def _dynamic_threshold(speed_kmh: float) -> float:
    """Distance de declenchement (px) = distance d'arret physique (v^2 /
    (2*BRAKE_FORCE), freinage plein pot depuis speed_kmh) + marge, avec un
    plancher basse vitesse."""
    v_px = speed_kmh / _SPEED_SCALE
    stop_dist = (v_px * v_px) / (2.0 * _BRAKE_FORCE_PX_S2)
    return max(EMERGENCY_FLOOR_PX, stop_dist + EMERGENCY_MARGIN_PX)


def check_emergency_brake(lidar: list[float], speed_kmh: float = 0.0) -> bool:
    """
    Vérifie si un freinage d'urgence est nécessaire.

    Deux conditions cumulatives sur les trois rayons frontaux (-30, 0, +30) :

    1. Au moins MIN_RAYS_TRIGGER d'entre eux sont plus courts que la distance
       d'arret dynamique (fonction de la vitesse courante) + marge. Remplace
       l'ancien seuil fixe 70 px : a 20-25 km/h (virage serre) la distance
       d'arret reelle n'est que de quelques px, un seuil fixe pense pour la
       vitesse de croisiere declenchait ~30x trop tot sur une simple bordure.

    2. L'ecart entre le rayon le plus court et le plus long des trois ne
       depasse pas FLATNESS_MAX_SPREAD_PX. Un mur (surface plate, 130 px de
       large, perpendiculaire au cap) donne des rayons quasi egaux ; une
       bordure frolee obliquement en virage donne un rayon interieur court
       et les deux autres nettement plus longs -- c'est ce qui distingue
       "obstacle large barrant la route" de "la piste tourne".

    Choix de 2/3 (et non 3/3) : de loin, un mur etroit vu de face ne barre
    que le rayon central ; les rayons ±30° divergent et ne le touchent qu'a
    courte portee -- exiger 3/3 ferait freiner trop tard.

    Args:
        lidar: Liste de 5 distances lidar (pixels).
        speed_kmh: Vitesse courante (km/h). A fournir SYSTEMATIQUEMENT par
            l'appelant : le defaut 0.0 correspond au plus petit seuil
            possible -- un appelant qui omet la vitesse sous-estime la
            distance d'arret requise (moins sur, pas plus).

    Returns:
        True si un obstacle large est trop proche devant (frein immédiat).
    """
    if not lidar or len(lidar) < 5:
        return False

    front = [lidar[i] for i in FRONT_RAY_INDICES]
    threshold = _dynamic_threshold(speed_kmh)

    n_close = sum(1 for r in front if r < threshold)
    if n_close < MIN_RAYS_TRIGGER:
        return False

    spread = max(front) - min(front)
    return spread <= FLATNESS_MAX_SPREAD_PX


def get_emergency_commands() -> dict:
    """Retourne les commandes de freinage d'urgence."""
    return {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
