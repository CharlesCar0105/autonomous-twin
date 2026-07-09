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


def get_camera_view_from_track(track: "Track", car: "Car", wall=None,
                               signs=None) -> np.ndarray:
    """Capture identique a get_camera_view mais directement depuis l'image
    de la piste (track.pixels) -- donc sans la voiture, le HUD, les rayons
    lidar superposes, etc. C'est la vraie "vue caméra" que le pilote doit
    analyser, pas une capture d'écran bruitée. Utilise aussi pour générer
    le dataset U-Net : image propre + masque ground-truth aligné.

    Args:
        track: Instance de Track (pour accéder à track.pixels).
        car: Instance de Car (pour position + orientation).
        wall: Mur optionnel a compositer dans la vue (obstacle visible par
              le pilote). None = piste seule (comportement d'origine, dataset
              U-Net inchange).
        signs: Liste de RoadSign a compositer dans la vue (panneaux visibles
               par le pilote). None/[] = piste seule. Les panneaux sont
               peints AVANT le mur : un mur qui spawn devant un panneau
               l'occlut (obstacle au premier plan).

    Returns:
        Image numpy (H, W, 3) RGB.
    """
    cx = int(car.x + CAMERA_DISTANCE * math.cos(car.angle))
    cy = int(car.y + CAMERA_DISTANCE * math.sin(car.angle))
    x = cx - CAMERA_WIDTH // 2
    y = cy - CAMERA_HEIGHT // 2
    x = max(0, min(x, track.width - CAMERA_WIDTH))
    y = max(0, min(y, track.height - CAMERA_HEIGHT))

    # track.pixels est (W, H, 3). On copie le crop (pour ne pas modifier la
    # piste), on y peint le mur, puis on transpose en (H, W, 3).
    crop = track.pixels[x:x + CAMERA_WIDTH, y:y + CAMERA_HEIGHT].copy()
    if signs:
        for sign in signs:
            sign.paste_into_camera(crop, x, y)
    if wall is not None:
        wmask = wall.mask_for_region(x, y, CAMERA_WIDTH, CAMERA_HEIGHT)
        crop[wmask] = wall_camera_color()
    return np.transpose(crop, (1, 0, 2))


def wall_camera_color() -> tuple[int, int, int]:
    """Couleur du mur dans la vue camera. Volontairement sombre (< seuil
    route a 200) pour que la segmentation le traite comme non-roulable."""
    return (150, 45, 35)


def get_ground_truth_mask(track: "Track", car: "Car") -> np.ndarray:
    """Masque ground-truth binaire (1 = pixel blanc "drivable",
    0 = bordure noire) aligné sur get_camera_view_from_track.

    Sert de label pour l'entrainement du U-Net : pas d'annotation
    manuelle, le simulateur connaît la verite terrain.

    Returns:
        Masque numpy (H, W) dtype=uint8, valeurs dans {0, 1}.
    """
    camera = get_camera_view_from_track(track, car)
    # Pixel "drivable" = luminosite > 200 sur chaque canal.
    r, g, b = camera[..., 0], camera[..., 1], camera[..., 2]
    return ((r > 200) & (g > 200) & (b > 200)).astype(np.uint8)


def get_lidar(track: "Track", car: "Car", wall=None) -> list[float]:
    """
    Lance 5 rayons depuis la voiture et retourne les distances
    au bord de piste (pixel non-blanc) OU au mur s'il est plus proche.

    Args:
        track: Instance de Track.
        car: Instance de Car.
        wall: Mur optionnel. Les rayons s'arretent aussi dessus.

    Returns:
        Liste de 5 distances en pixels.
    """
    distances = []

    for angle_offset in LIDAR_ANGLES:
        ray_angle = car.angle + math.radians(angle_offset)
        dist = _cast_ray(track, car.x, car.y, ray_angle, wall)
        distances.append(dist)

    return distances


def _cast_ray(track: "Track", x: float, y: float, angle: float, wall=None) -> float:
    """
    Lance un rayon pixel par pixel jusqu'au bord de piste ou au mur.

    Returns:
        Distance en pixels jusqu'au premier pixel non-route (ou mur).
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for d in range(1, LIDAR_MAX_RANGE + 1):
        px = int(x + d * cos_a)
        py = int(y + d * sin_a)

        if wall is not None and wall.contains(px, py):
            return float(d)
        if not track.is_on_road(px, py):
            return float(d)

    return float(LIDAR_MAX_RANGE)
