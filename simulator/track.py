"""
track.py — Gestion et chargement de la piste.

Responsabilités :
    - Charger une image PNG de circuit et la redimensionner à la taille de la fenêtre
    - Pré-calculer un tableau de pixels numpy pour accès rapide
    - Fournir is_border() pour le raycasting lidar
    - Dessiner la piste à l'écran
"""

import pygame
import numpy as np
import os

# --- Constantes -----------------------------------------------------------

TRACKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "tracks")

# Seuil de luminosité pour considérer un pixel comme bordure (noir)
BORDER_THRESHOLD = 80


class Track:
    """Représente un circuit chargé depuis une image PNG."""

    def __init__(self, track_name: str, screen_w: int = 1280, screen_h: int = 720) -> None:
        """
        Charge et redimensionne le circuit.

        Args:
            track_name: Nom du fichier circuit (sans extension).
            screen_w: Largeur cible (pixels).
            screen_h: Hauteur cible (pixels).
        """
        path = os.path.join(TRACKS_DIR, f"{track_name}.png")
        if not os.path.exists(path):
            abs_dir = os.path.abspath(TRACKS_DIR)
            raise FileNotFoundError(
                f"Circuit introuvable : {path}\n"
                f"Place ton image PNG dans : {abs_dir}"
            )

        # Charger et redimensionner à la taille de la fenêtre
        raw = pygame.image.load(path).convert()
        self.image = pygame.transform.scale(raw, (screen_w, screen_h))
        self.width = screen_w
        self.height = screen_h

        # Pré-calcul du tableau de pixels (W, H, 3) pour accès rapide
        self.pixels = pygame.surfarray.array3d(self.image)

        # Position de départ par défaut (bas du circuit, sens → droite)
        self.start_x = screen_w // 2
        self.start_y = int(screen_h * 0.86)
        self.start_angle = 0.0  # degrés, 0 = vers la droite

        # Essayer de trouver un pixel blanc proche de la position par défaut
        self._adjust_start_position()

    def _adjust_start_position(self) -> None:
        """Cherche un pixel blanc (route) autour de la position de départ."""
        for dy in range(-50, 51, 5):
            for dx in range(-80, 81, 10):
                tx = self.start_x + dx
                ty = self.start_y + dy
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    if self.is_on_road(tx, ty):
                        self.start_x = tx
                        self.start_y = ty
                        return

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """
        Retourne la couleur RGB du pixel (x, y).
        Hors limites → noir (bordure).
        """
        ix, iy = int(x), int(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return tuple(self.pixels[ix, iy])
        return (0, 0, 0)

    def is_on_road(self, x: int, y: int) -> bool:
        """
        Vérifie si (x, y) est sur une zone blanche (route OU herbe).
        Utile pour le raycasting lidar : le rayon s'arrête sur les pixels sombres.
        """
        r, g, b = self.get_pixel(int(x), int(y))
        return r > 200 and g > 200 and b > 200

    def is_border(self, x: int, y: int) -> bool:
        """
        Vérifie si (x, y) est un pixel de bordure noire.
        """
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return True  # hors écran = bordure
        r, g, b = self.pixels[ix, iy]
        luminance = (int(r) + int(g) + int(b)) / 3
        return luminance < BORDER_THRESHOLD

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine la piste sur l'écran."""
        screen.blit(self.image, (0, 0))
