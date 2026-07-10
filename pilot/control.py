"""
control.py — Controle du vehicule.

Politique PID baseline (proportionnelle pure sur lidar) + CNN behavioral
cloning (Sprint 3). Signature commune :
    steering, throttle, brake = policy(lidar, speed, mask=None)

- Le PID (`pid_policy`) sert de baseline et de backup si le CNN deraille.
- Le CNN (`cnn_policy`) prend (mask 64x64, lidar 5) et predit
  (steering, throttle). L'image brute n'est pas utilisee -> modularite +
  generalisation (D4 du wiki).
"""

from pathlib import Path
from typing import Optional
import numpy as np


# --- PID baseline ---------------------------------------------------------

# Gains du controleur lateral (angles -60, -30, 0, 30, 60 degres)
KP_LAT_CLOSE = 0.25   # gain sur lidar +/-30 (reactif)
KP_LAT_FAR = 0.12     # gain sur lidar +/-60 (anticipation)

# Seuil au-dessous duquel on considere le front "court" et on force le
# steering vers le cote qui a le plus de marge (somme close+far).
FRONT_TURN_THRESHOLD = 120.0
FRONT_TURN_GAIN = 0.8       # degres par px de difference de marge laterale

# Boost anti-collision laterale : quand un cote est tres proche, on fuit.
TIGHT_THRESHOLD = 40.0
TIGHT_BOOST = 35.0

# Throttle : reduit quand un obstacle est proche devant.
# Retune 2026-07-10 (phase optimisation, mesures agent B au banc LapTimer) :
# - THROTTLE_MAX 0.85 -> 1.0 : la friction (2%/frame) plafonne de toute
#   facon l'equilibre pleine accel a ~98 px/s (~70 km/h) ; brider en plus
#   le throttle ne protegeait rien (pas de grip lateral dans ce modele).
# - LIDAR_FRONT_SAFE 250 -> 70 : avec 250, le PID n'etait a fond que 0.7%
#   des frames sur ces pistes sinueuses -- c'etait LE frein cache.
# Mesure : best lap gen_000 32.0s -> 16.45s, 0% offtrack sur 7 circuits
# (a 1-4% de l'oracle theorique). NE PAS retirer TIGHT_BOOST : porteur
# (son retrait seul = DNF gen_014, teste).
THROTTLE_MAX = 1.0         # plafond a vitesse de croisiere
THROTTLE_MIN = 0.25        # plancher en approche virage
LIDAR_FRONT_SAFE = 70.0    # pixels : throttle max au-dessus de ce seuil
LIDAR_FRONT_SLOW = 40.0    # pixels : throttle min en-dessous

# Vitesse cible (km/h) — utile quand on aura la classif panneaux
SPEED_TARGET_DEFAULT = 80.0

# Amortissement de la commande de steering (filtre passe-bas 1er ordre :
# steering_t = kd*prev + (1-kd)*brut). Mesure agent B (10/07) : kd=0.5
# reduit l'oscillation frame-a-frame (11% d'inversions de signe en
# baseline) et gagne ~0.5-1% au chrono, sans contrepartie sur 7 circuits.
# ATTENTION : la derivee classique sur l'ERREUR a ete testee et est
# DANGEREUSE ici (bruit de quantification lidar amplifie -> 86% offtrack
# a kd=1.0) -- ne pas "ameliorer" dans ce sens.
STEER_DAMPING = 0.5

# Etat du filtre (module-level, comme _cnn_model). Pas de reset entre runs :
# converge en ~5 frames et le spawn se fait a steering ~0.
_prev_steering = 0.0


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

    left_far, left_close = lidar[0], lidar[1]
    front = lidar[2]
    right_close, right_far = lidar[3], lidar[4]

    # Steering : combinaison rayons proches (reactif) + lointains (anticipation).
    err_close = right_close - left_close
    err_far = right_far - left_far
    steering = KP_LAT_CLOSE * err_close + KP_LAT_FAR * err_far

    # Virage imminent : si le front est court, on fonce vers le cote qui
    # totalise la plus grosse marge. Sans ce terme, un PID purement base sur
    # les ecarts gauche/droite ne tourne pas assez quand on arrive sur une
    # epingle (les deux cotes deviennent courts simultanement).
    if front < FRONT_TURN_THRESHOLD:
        margin_right = right_close + right_far
        margin_left = left_close + left_far
        urgency = 1.0 - (front / FRONT_TURN_THRESHOLD)  # 0 -> 1 quand front tombe
        steering += FRONT_TURN_GAIN * (margin_right - margin_left) * urgency / 10.0

    # Boost anti-collision laterale : un cote tres proche -> fuir l'autre cote.
    if left_close < TIGHT_THRESHOLD and left_close < right_close:
        steering += TIGHT_BOOST
    elif right_close < TIGHT_THRESHOLD and right_close < left_close:
        steering -= TIGHT_BOOST

    steering = float(np.clip(steering, -45.0, 45.0))

    # Amortissement de sortie (cf STEER_DAMPING) : moyenne de deux valeurs
    # dans [-45, 45], donc pas de re-clip necessaire.
    global _prev_steering
    steering = STEER_DAMPING * _prev_steering + (1.0 - STEER_DAMPING) * steering
    _prev_steering = steering

    # Longitudinal : throttle interpole sur la distance frontale.
    t = (front - LIDAR_FRONT_SLOW) / (LIDAR_FRONT_SAFE - LIDAR_FRONT_SLOW)
    t = float(np.clip(t, 0.0, 1.0))
    throttle = THROTTLE_MIN + t * (THROTTLE_MAX - THROTTLE_MIN)

    # Respect d'une vitesse cible molle (pour futur couplage panneaux).
    if speed > speed_target:
        throttle = min(throttle, 0.4)

    return steering, throttle, 0.0


# --- CNN conduite (behavioral cloning, Sprint 3) --------------------------

# Normalisation cote training : steering stocke en degres [-45, +45] -> [-1, 1],
# throttle deja en [0, 1], lidar en pixels 0-300 -> [0, 1] via division 300.
STEER_MAX_DEG = 45.0
LIDAR_MAX_PX = 300.0

# Lazy import de torch : ne plante pas si seul le PID est utilise.
_cnn_model = None
_cnn_device = None


def _load_cnn(weights_path: Optional[str] = None):
    """Charge le modele CNN (lazy). Utilise models/cnn_drive.pth par defaut."""
    global _cnn_model, _cnn_device
    if _cnn_model is not None:
        return
    import torch
    from pilot.cnn_drive_arch import DriveCNN  # archi partagee avec le notebook

    if weights_path is None:
        weights_path = str(Path(__file__).resolve().parent.parent / "models" / "cnn_drive.pth")
    _cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _cnn_model = DriveCNN().to(_cnn_device)
    _cnn_model.load_state_dict(torch.load(weights_path, map_location=_cnn_device))
    _cnn_model.eval()


def cnn_policy(
    lidar: list[float], speed: float, mask: np.ndarray,
    weights_path: Optional[str] = None,
) -> tuple[float, float, float]:
    """Inference CNN : (masque U-Net 64x64, lidar 5) -> (steering, throttle, brake).

    Args:
        lidar : 5 distances (meme ordre que pid_policy).
        speed : vitesse km/h (pas utilisee par le CNN actuel mais gardee
                pour compat d'API et usage futur).
        mask  : masque binaire 64x64 (float ou uint8) issu de U-Net ou GT.
        weights_path : chemin vers cnn_drive.pth. None = defaut.

    Returns:
        (steering deg, throttle [0,1], brake [0,1])
    """
    import torch
    _load_cnn(weights_path)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    if m.max() > 1.0:
        m = m / 255.0
    lid = np.asarray(lidar, dtype=np.float32) / LIDAR_MAX_PX
    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(_cnn_device)  # (1,1,64,64)
    lidar_t = torch.from_numpy(lid).unsqueeze(0).to(_cnn_device)             # (1, 5)
    with torch.no_grad():
        steer_norm, throttle = _cnn_model(mask_t, lidar_t).cpu().numpy().squeeze()
    steering = float(np.clip(steer_norm, -1.0, 1.0) * STEER_MAX_DEG)
    throttle = float(np.clip(throttle, 0.0, 1.0))
    return steering, throttle, 0.0
