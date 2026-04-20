"""
sensors.py — Capteurs simulés (caméra, lidar, vitesse).

Responsabilités :
    - Caméra : capture un rectangle de pixels devant la voiture
    - Lidar : 5 rayons raycasting retournant la distance au bord de piste
    - Vitesse : lecture du compteur en km/h virtuels
"""

import pygame
import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.car import Car
    from simulator.track import Track


# --- Constantes -----------------------------------------------------------

# Caméra
CAMERA_WIDTH = 128     # largeur de l'image caméra (pixels)
CAMERA_HEIGHT = 128    # hauteur de l'image caméra (pixels)
CAMERA_DISTANCE = 60   # distance devant la voiture (pixels)

# Lidar
LIDAR_MAX_RANGE = 300  # portée maximale des rayons (pixels)
LIDAR_ANGLES = [-60, -30, 0, 30, 60]  # angles des 5 rayons (degrés)


def get_camera_view(screen: pygame.Surface, car: "Car") -> np.ndarray:
    """
    Capture un crop rectangulaire de pixels devant la voiture.

    Args:
        screen: Surface Pygame (écran complet).
        car: Instance de Car.

    Returns:
        Image numpy (H, W, 3) en RGB.
    """
    # Centre de la capture : devant la voiture
    cx = int(car.x + CAMERA_DISTANCE * math.cos(car.angle))
    cy = int(car.y + CAMERA_DISTANCE * math.sin(car.angle))

    # Rectangle de capture
    x = cx - CAMERA_WIDTH // 2
    y = cy - CAMERA_HEIGHT // 2

    # Clamp aux limites de l'écran
    sw, sh = screen.get_size()
    x = max(0, min(x, sw - CAMERA_WIDTH))
    y = max(0, min(y, sh - CAMERA_HEIGHT))

    # Crop
    rect = pygame.Rect(x, y, CAMERA_WIDTH, CAMERA_HEIGHT)
    sub = screen.subsurface(rect)

    # Conversion en numpy array (RGB)
    arr = pygame.surfarray.array3d(sub)       # shape (W, H, 3)
    arr = np.transpose(arr, (1, 0, 2))        # shape (H, W, 3)
    return arr


def get_lidar(track: "Track", car: "Car") -> list[float]:
    """
    Lance 5 rayons depuis la voiture et retourne les distances
    au bord de piste (pixel non-blanc).

    Args:
        track: Instance de Track.
        car: Instance de Car.

    Returns:
        Liste de 5 distances en pixels.
    """
    distances = []

    for angle_offset in LIDAR_ANGLES:
        ray_angle = car.angle + math.radians(angle_offset)
        dist = _cast_ray(track, car.x, car.y, ray_angle)
        distances.append(dist)

    return distances


def _cast_ray(track: "Track", x: float, y: float, angle: float) -> float:
    """
    Lance un rayon pixel par pixel jusqu'au bord de piste.

    Returns:
        Distance en pixels jusqu'au premier pixel non-route.
    """
    for d in range(1, LIDAR_MAX_RANGE + 1):
        px = int(x + d * math.cos(angle))
        py = int(y + d * math.sin(angle))

        if not track.is_on_road(px, py):
            return float(d)

    return float(LIDAR_MAX_RANGE)
