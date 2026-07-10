"""
physics.py — Modèle cinématique Bicycle.

Implémente les 4 équations du sujet pour mettre à jour
la position (x, y) et l'orientation (θ) du véhicule.

Équations :
    β  = arctan( Lr / (Lf + Lr) * tan(δ) )
    x' = x + v * cos(θ + β) * dt
    y' = y + v * sin(θ + β) * dt
    θ' = θ + (v / Lr) * sin(β) * dt

Paramètres :
    Lf, Lr = 1.5  (distances centre de gravité → essieux)
    dt     = 1/60 (60 FPS)
"""

import math


# --- Constantes du modèle -------------------------------------------------

LF = 8.5   # distance CG → essieu avant (augmenté = virages plus stables)
LR = 8.5   # distance CG → essieu arrière

# Paramètres de dynamique
# MAX_ACCELERATION 120 -> 170 (proposition equipe 2026-07-10, a valider par
# Charles) : la friction par frame (2%) fixe l'equilibre pleine charge a
# v* = (1-f)/f * MAX_ACC * dt, soit 70.6 km/h avec 120 -- les panneaux 90
# n'etaient jamais contraignants. Avec 170 : v* = 138.9 px/s = 100.0 km/h,
# donc limite 90 = ecart visible de 10 km/h. Freinage/friction inchanges.
MAX_ACCELERATION = 170.0   # accélération max (pixels/s²)
BRAKE_FORCE = 200.0        # force de freinage (pixels/s²)
FRICTION = 0.02            # coefficient de frottement (augmenté)
MAX_SPEED_PX = 300.0       # vitesse max en pixels/s

# Conversion vitesse
SPEED_SCALE = 0.72         # pixels/s → km/h (à calibrer)


def update(car, dt: float) -> None:
    """
    Met à jour la position et la vitesse de la voiture
    selon le modèle cinématique Bicycle.

    Args:
        car: Instance de Car (modifiée in-place).
        dt: Delta time en secondes.
    """
    # --- Mise à jour de la vitesse ----------------------------------------
    # Accélération
    if car.throttle > 0:
        car.speed += car.throttle * MAX_ACCELERATION * dt

    # Freinage
    if car.brake > 0:
        car.speed -= car.brake * BRAKE_FORCE * dt

    # Frottement naturel (décélération passive)
    car.speed -= car.speed * FRICTION

    # Bornes de vitesse
    car.speed = max(0.0, min(MAX_SPEED_PX, car.speed))

    # --- Modèle cinématique Bicycle ---------------------------------------
    delta = math.radians(car.steering)  # angle volant → radians

    # Angle de dérive (slip angle)
    beta = math.atan2(LR * math.tan(delta), LF + LR)

    # Mise à jour de la position
    car.x += car.speed * math.cos(car.angle + beta) * dt
    car.y += car.speed * math.sin(car.angle + beta) * dt

    # Mise à jour de l'orientation
    car.angle += (car.speed / LR) * math.sin(beta) * dt


def speed_kmh(car) -> float:
    """Retourne la vitesse actuelle en km/h virtuels."""
    return car.speed * SPEED_SCALE
