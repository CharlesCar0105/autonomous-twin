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
        """Place la voiture au *centre* de la ligne droite du bas du circuit.

        On scanne uniquement une fine bande horizontale autour de la position
        theorique (screen_h * 0.86), pour ne pas tomber dans les zones
        fermees du centre du circuit (boucles internes). Score = distance
        minimale aux bords haut et bas -> position la plus centree dans la
        piste.
        """
        best = None
        best_score = -1
        # Recherche large verticalement (la piste peut etre haut ou bas) ;
        # le filtre de largeur dans _track_centering_score se charge de
        # rejeter les zones trop larges (exterieur) ou trop etroites.
        for dy in range(-200, 151, 4):
            for dx in range(-500, 501, 10):
                tx = self.start_x + dx
                ty = self.start_y + dy
                if not (0 <= tx < self.width and 0 <= ty < self.height):
                    continue
                score = self._track_centering_score(tx, ty)
                if score > best_score:
                    best = (tx, ty)
                    best_score = score
        if best is not None:
            self.start_x, self.start_y = best

    # Largeur de piste attendue (pixels) : une piste de course fait ~40-80 px.
    # En dehors de cette fenetre c'est l'exterieur (herbe) ou une boucle
    # interne non pilotable.
    TRACK_MIN_WIDTH = 30
    TRACK_MAX_WIDTH = 100

    def _track_centering_score(self, x: int, y: int, max_scan: int = 200) -> int:
        """Score -1 si (x,y) n'est pas dans un *couloir de piste* (largeur
        verticale dans la fenetre attendue), sinon min(up, down) : plus le
        pixel est centre et la piste est large, plus le score est eleve.

        L'idee : scanner le haut et le bas jusqu'aux premieres bordures
        noires, mesurer la largeur totale (up + down). Si cette largeur
        est hors de la plage piste attendue, on rejette (zone exterieure
        ou zone interne fermee).
        """
        if not self.is_on_road(x, y):
            return -1
        up = 0
        for d in range(1, max_scan + 1):
            if y - d < 0 or self.is_border(x, y - d):
                up = d
                break
        if up == 0:
            return -1
        down = 0
        for d in range(1, max_scan + 1):
            if y + d >= self.height or self.is_border(x, y + d):
                down = d
                break
        if down == 0:
            return -1
        width = up + down
        if width < self.TRACK_MIN_WIDTH or width > self.TRACK_MAX_WIDTH:
            return -1
        return min(up, down)

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
