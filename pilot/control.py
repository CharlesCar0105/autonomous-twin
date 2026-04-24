"""
control.py — Controle du vehicule.

Politique PID baseline (proportionnelle pure sur lidar) + hooks pour
brancher un CNN de conduite (Sprint 2+). Signature commune :
    steering, throttle, brake = policy(lidar, speed, mask=None)

Le PID est utilise seul en Sprint 1 (D9 : fonctionnel avant ML) puis reste
comme backup quand le CNN deraille (D3).
"""

from typing import Optional
import numpy as np


# --- PID baseline ---------------------------------------------------------

# Gains du controleur lateral (angles -60, -30, 0, 30, 60 degres)
# steering = KP_LAT * (lidar_droite_proche - lidar_gauche_proche)
KP_LAT = 0.18  # degres volant par pixel de difference lidar

# Throttle : reduit quand un obstacle est proche devant
THROTTLE_MAX = 0.85        # plafond a vitesse de croisiere
THROTTLE_MIN = 0.25        # plancher en approche virage
LIDAR_FRONT_SAFE = 250.0   # pixels : throttle max au-dessus de ce seuil
LIDAR_FRONT_SLOW = 80.0    # pixels : throttle min en-dessous

# Vitesse cible (km/h) — utile quand on aura la classif panneaux
SPEED_TARGET_DEFAULT = 80.0


def pid_policy(
    lidar: list[float],
    speed: float,
    mask: Optional[np.ndarray] = None,
    speed_target: float = SPEED_TARGET_DEFAULT,
) -> tuple[float, float, float]:
    """Controleur PID baseline.

    Args:
        lidar: [gauche_loin, gauche_proche, avant, droite_proche, droite_loin].
        speed: vitesse courante en km/h.
        mask: masque U-Net (ignore ici, garde pour compat API future).
        speed_target: vitesse de croisiere cible en km/h.

    Returns:
        (steering_deg, throttle [0,1], brake [0,1]).
    """
    if len(lidar) != 5:
        return 0.0, 0.0, 1.0

    left_close = lidar[1]
    front = lidar[2]
    right_close = lidar[3]

    # Lateral : on tourne du cote ou il y a le plus de marge.
    error = right_close - left_close
    steering = float(np.clip(KP_LAT * error, -45.0, 45.0))

    # Longitudinal : throttle interpole sur la distance frontale.
    t = (front - LIDAR_FRONT_SLOW) / (LIDAR_FRONT_SAFE - LIDAR_FRONT_SLOW)
    t = float(np.clip(t, 0.0, 1.0))
    throttle = THROTTLE_MIN + t * (THROTTLE_MAX - THROTTLE_MIN)

    # Respect d'une vitesse cible molle (pour futur couplage panneaux).
    if speed > speed_target:
        throttle = min(throttle, 0.4)

    return steering, throttle, 0.0


# --- Hook CNN (a brancher Sprint 3) ---------------------------------------

def cnn_policy(
    lidar: list[float],
    speed: float,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Placeholder pour le CNN de conduite (behavioral cloning).

    Sera implemente en Sprint 3 une fois le dataset collecte. Meme signature
    que pid_policy pour pouvoir swap les deux sans toucher la boucle pilote.
    """
    raise NotImplementedError("CNN conduite : Sprint 3 (cf. wiki/projets/autonomous-twin-ml.md)")
