"""
wall.py — L'Aléa : le Mur de briques (obstacle dynamique).

Le Mur apparait sur commande (touche Espace cote simu) a une distance
critique devant la voiture. C'est un rectangle oriente (aligne sur le cap
de la voiture au moment du spawn) qui sert a trois choses, avec une
*seule* source de verite geometrique :

    - contains(x, y)          -> collision voiture + arret du raycast lidar
    - draw(screen)            -> rendu brique a l'ecran
    - mask_for_region(...)    -> compositing dans la vue camera du pilote

La geometrie est un simple test "point dans rectangle oriente" : on
projette le point dans le repere local du mur (axe u = cap, axe v =
perpendiculaire) et on compare aux demi-dimensions.
"""

import math
import pygame
import numpy as np


# --- Constantes ----------------------------------------------------------

# Distance de spawn devant la voiture. Plus c'est court, plus l'aléa est
# soudain. PLANCHER DE SECURITE : doit rester > EMERGENCY_DISTANCE (70 px)
# + distance d'arret (~30 px a 80 km/h), sinon la voiture n'a plus le temps
# de freiner avant l'impact. 130 px laisse ~30 px de marge apres l'arret.
SPAWN_DISTANCE = 130.0

# Duree de vie du mur (secondes) : il disparait tout seul apres ce delai.
WALL_LIFETIME = 6.0

# Dimensions du mur (pixels).
WALL_LENGTH = 130.0     # etendue perpendiculaire au cap (barre la piste)
WALL_THICKNESS = 26.0   # profondeur le long du cap

# Couleurs briques.
COL_BRICK = (150, 45, 35)
COL_MORTAR = (60, 20, 15)
COL_BRICK_HI = (180, 70, 55)


class Wall:
    """Rectangle oriente representant le mur de briques."""

    def __init__(self, cx: float, cy: float, angle_rad: float,
                 length: float = WALL_LENGTH, thickness: float = WALL_THICKNESS,
                 spawn_time: float = 0.0, lifetime: float = WALL_LIFETIME) -> None:
        """
        Args:
            cx, cy:     Centre du mur (pixels monde).
            angle_rad:  Orientation du cap de la voiture (radians). Le mur
                        barre perpendiculairement a cet axe.
            length:     Etendue perpendiculaire au cap.
            thickness:  Profondeur le long du cap.
            spawn_time: Horodatage de creation (s), pour l'auto-disparition.
            lifetime:   Duree de vie (s) avant disparition automatique.
        """
        self.cx = float(cx)
        self.cy = float(cy)
        self.angle = float(angle_rad)
        self.length = float(length)
        self.thickness = float(thickness)
        self.spawn_time = float(spawn_time)
        self.lifetime = float(lifetime)

        # Repere local : u = cap (profondeur), v = perpendiculaire (etendue).
        self._cos = math.cos(self.angle)
        self._sin = math.sin(self.angle)

    # --- Fabrique ---------------------------------------------------------

    @classmethod
    def spawn_ahead(cls, car, now: float = 0.0,
                    distance: float = SPAWN_DISTANCE) -> "Wall":
        """Cree un mur `distance` px devant la voiture, aligne sur son cap.

        `now` = horodatage courant (s), pour l'auto-disparition apres
        WALL_LIFETIME secondes.
        """
        cx = car.x + distance * math.cos(car.angle)
        cy = car.y + distance * math.sin(car.angle)
        return cls(cx, cy, car.angle, spawn_time=now)

    def is_expired(self, now: float) -> bool:
        """True si le mur a depasse sa duree de vie."""
        return (now - self.spawn_time) >= self.lifetime

    def time_left(self, now: float) -> float:
        """Secondes restantes avant disparition (>= 0)."""
        return max(0.0, self.lifetime - (now - self.spawn_time))

    # --- Geometrie (source de verite unique) ------------------------------

    def contains(self, x: float, y: float) -> bool:
        """True si le point (x, y) est a l'interieur du mur."""
        dx = x - self.cx
        dy = y - self.cy
        proj_u = dx * self._cos + dy * self._sin        # le long du cap
        proj_v = -dx * self._sin + dy * self._cos       # perpendiculaire
        return (abs(proj_u) <= self.thickness / 2.0 and
                abs(proj_v) <= self.length / 2.0)

    def mask_for_region(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """Masque booleen (w, h) des pixels couverts par le mur sur la
        region monde [x0, x0+w) x [y0, y0+h).

        Indexation (dx, dy) alignee sur track.pixels[x0:x0+w, y0:y0+h],
        pour compositer directement dans la vue camera du pilote.
        """
        xs = np.arange(x0, x0 + w, dtype=np.float32) - self.cx   # (w,)
        ys = np.arange(y0, y0 + h, dtype=np.float32) - self.cy   # (h,)
        dx = xs[:, None]   # (w, 1)
        dy = ys[None, :]   # (1, h)
        proj_u = dx * self._cos + dy * self._sin
        proj_v = -dx * self._sin + dy * self._cos
        return ((np.abs(proj_u) <= self.thickness / 2.0) &
                (np.abs(proj_v) <= self.length / 2.0))

    def corners(self) -> list[tuple[float, float]]:
        """Les 4 coins du rectangle oriente (monde), pour le rendu."""
        hu_x = (self.thickness / 2.0) * self._cos
        hu_y = (self.thickness / 2.0) * self._sin
        hv_x = -(self.length / 2.0) * self._sin
        hv_y = (self.length / 2.0) * self._cos
        return [
            (self.cx - hu_x - hv_x, self.cy - hu_y - hv_y),
            (self.cx + hu_x - hv_x, self.cy + hu_y - hv_y),
            (self.cx + hu_x + hv_x, self.cy + hu_y + hv_y),
            (self.cx - hu_x + hv_x, self.cy - hu_y + hv_y),
        ]

    # --- Rendu ------------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine le mur en briques (rectangle oriente + joints de mortier)."""
        corners = self.corners()
        pygame.draw.polygon(screen, COL_BRICK, corners)

        # Joints de mortier : lignes perpendiculaires au cap, reparties le
        # long de l'etendue (v), pour donner l'aspect "rangees de briques".
        n_rows = 6
        for i in range(1, n_rows):
            t = i / n_rows                     # 0..1 le long de v
            offset = (t - 0.5) * self.length
            ox = -offset * self._sin
            oy = offset * self._cos
            # extremites de la ligne = bord a bord le long de u
            hu_x = (self.thickness / 2.0) * self._cos
            hu_y = (self.thickness / 2.0) * self._sin
            p1 = (self.cx + ox - hu_x, self.cy + oy - hu_y)
            p2 = (self.cx + ox + hu_x, self.cy + oy + hu_y)
            pygame.draw.line(screen, COL_MORTAR, p1, p2, 2)

        # Liseré clair sur le contour pour le detacher du fond.
        pygame.draw.polygon(screen, COL_BRICK_HI, corners, 2)
