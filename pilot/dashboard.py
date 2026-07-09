"""
dashboard.py — Tableau de bord temps réel côté pilote.

Fenetre Pygame *separee* (cote pilote) qui affiche ce que la voiture
"voit" et decide, en n'utilisant QUE les donnees dont dispose le pilote :
caméra, lidar, vitesse, + le masque de segmentation qu'il calcule et ses
propres commandes. Aucune donnee "carte globale" -> respecte la contrainte
anti-triche du sujet.

Contenu :
    - Vue caméra brute (ce que recoit le pilote)
    - Masque de segmentation route (overlay vert = roulable)
    - Barres lidar (5 rayons, couleur selon proximite)
    - Vitesse (compteur + jauge)
    - Courbe de vitesse temps reel
    - Panneau detecte (emplacement, rempli quand signs.py sera pret)
    - Etat pilote : CONDUITE / FREINAGE URGENCE

Usage (branche depuis pilot/main.py) :
    dash = Dashboard()
    ...
    alive = dash.update(camera, mask, lidar, speed, commands, state)
    if not alive:   # fenetre fermee par l'utilisateur
        break
    ...
    dash.close()
"""

from collections import deque

import numpy as np
import pygame

from simulator import sensors  # pour LIDAR_ANGLES / LIDAR_MAX_RANGE (constantes)


# --- Constantes -----------------------------------------------------------

WIN_W, WIN_H = 960, 600

COL_BG = (18, 18, 22)
COL_PANEL = (30, 30, 38)
COL_GREEN = (0, 230, 64)
COL_RED = (255, 60, 60)
COL_YELLOW = (255, 220, 50)
COL_CYAN = (0, 255, 255)
COL_WHITE = (240, 240, 240)
COL_GRAY = (150, 150, 150)

CAM_SIZE = 240                 # cote d'affichage caméra / masque
SPEED_HIST = 240               # points de la courbe de vitesse
SPEED_AXIS_MAX = 160.0         # plafond de l'axe vitesse (km/h)

# Ordre des rayons lidar pour l'affichage (index -> libelle).
LIDAR_LABELS = ["G loin", "G pres", "Avant", "D pres", "D loin"]


class Dashboard:
    """Fenetre de telemetrie temps reel du pilote."""

    def __init__(self, title: str = "Pilote — Tableau de bord") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 15)
        self.font_sm = pygame.font.SysFont("Consolas", 12)
        self.font_big = pygame.font.SysFont("Consolas", 40, bold=True)
        self.font_lbl = pygame.font.SysFont("Consolas", 14, bold=True)
        self.speed_hist: deque[float] = deque(maxlen=SPEED_HIST)

    # --- API publique -----------------------------------------------------

    def update(self, camera: np.ndarray, mask: np.ndarray, lidar: list[float],
               speed: float, commands: dict, state: str,
               sign_label: str = "—") -> bool:
        """Redessine le tableau de bord. Retourne False si l'utilisateur a
        ferme la fenetre (le pilote doit alors s'arreter)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.speed_hist.append(float(speed))
        self.screen.fill(COL_BG)

        # En-tete
        self._blit(self.font_lbl, "PILOTE — TABLEAU DE BORD (vue capteurs)",
                   14, 12, COL_WHITE)

        # Vue caméra + segmentation
        self._draw_image(camera, 20, 44, "CAMERA (brute)")
        self._draw_segmentation(camera, mask, 275, 44, "SEGMENTATION (vert = route)")

        # Panneau telemetrie a droite
        self._draw_telemetry(speed, commands, state, sign_label, 535, 44)

        # Barres lidar
        self._draw_lidar_bars(lidar, 535, 300)

        # Courbe de vitesse
        self._draw_speed_curve(20, 360, WIN_W - 40, 220)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self) -> None:
        pygame.quit()

    # --- Helpers de rendu -------------------------------------------------

    def _blit(self, font, text, x, y, col) -> None:
        self.screen.blit(font.render(text, True, col), (x, y))

    def _panel(self, x, y, w, h) -> None:
        pygame.draw.rect(self.screen, COL_PANEL, (x, y, w, h), border_radius=6)

    @staticmethod
    def _to_surface(rgb_hw3: np.ndarray) -> pygame.Surface:
        """numpy (H, W, 3) RGB -> Surface Pygame (transpose en (W, H, 3))."""
        arr = np.ascontiguousarray(np.transpose(rgb_hw3, (1, 0, 2)))
        return pygame.surfarray.make_surface(arr)

    def _draw_image(self, camera: np.ndarray, x: int, y: int, label: str) -> None:
        surf = pygame.transform.scale(self._to_surface(camera), (CAM_SIZE, CAM_SIZE))
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, COL_GRAY, (x, y, CAM_SIZE, CAM_SIZE), 1)
        self._blit(self.font_sm, label, x, y + CAM_SIZE + 4, COL_GRAY)

    def _draw_segmentation(self, camera: np.ndarray, mask: np.ndarray,
                           x: int, y: int, label: str) -> None:
        """Overlay : route (mask=1) teintee vert, hors-piste teinte rouge,
        sur la caméra assombrie -> montre ce que le pilote considere roulable."""
        cam = np.asarray(camera, dtype=np.float32)
        m = np.asarray(mask, dtype=np.float32)
        if m.shape[:2] != cam.shape[:2]:      # aligner le masque sur la caméra
            m = self._resize_nearest(m, cam.shape[0], cam.shape[1])
        m3 = m[..., None]
        base = cam * 0.45
        road = np.array([0, 230, 64], dtype=np.float32)
        off = np.array([120, 30, 30], dtype=np.float32)
        overlay = base + m3 * road * 0.55 + (1.0 - m3) * off * 0.35
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        surf = pygame.transform.scale(self._to_surface(overlay), (CAM_SIZE, CAM_SIZE))
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, COL_GRAY, (x, y, CAM_SIZE, CAM_SIZE), 1)
        self._blit(self.font_sm, label, x, y + CAM_SIZE + 4, COL_GRAY)

    @staticmethod
    def _resize_nearest(a: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        h, w = a.shape[:2]
        ys = (np.arange(new_h) * h / new_h).astype(np.int64)
        xs = (np.arange(new_w) * w / new_w).astype(np.int64)
        return a[ys[:, None], xs[None, :]]

    def _draw_telemetry(self, speed, commands, state, sign_label, x, y) -> None:
        w = WIN_W - x - 20
        self._panel(x, y, w, 240)
        pad = x + 16

        # Vitesse (gros chiffre + jauge)
        self._blit(self.font_sm, "VITESSE", pad, y + 12, COL_GRAY)
        self._blit(self.font_big, f"{speed:5.1f}", pad, y + 26, COL_WHITE)
        self._blit(self.font, "km/h", pad + 150, y + 46, COL_GRAY)
        gauge_w = w - 32
        pygame.draw.rect(self.screen, (60, 60, 68), (pad, y + 78, gauge_w, 10), border_radius=4)
        ratio = min(1.0, speed / SPEED_AXIS_MAX)
        gcol = COL_RED if speed > 140 else COL_GREEN
        pygame.draw.rect(self.screen, gcol, (pad, y + 78, int(gauge_w * ratio), 10), border_radius=4)

        # Commandes (volant / gaz / frein)
        cy = y + 100
        self._blit(self.font_sm,
                   f"Volant {commands['steering']:+5.1f}   "
                   f"Gaz {commands['throttle']:.2f}   Frein {commands['brake']:.2f}",
                   pad, cy, COL_CYAN)

        # Panneau detecte (emplacement)
        self._blit(self.font_sm, f"Panneau : {sign_label}", pad, cy + 22, COL_GRAY)

        # Banniere d'etat
        by = y + 158
        urgent = "URGENCE" in state.upper()
        bcol = COL_RED if urgent else COL_GREEN
        pygame.draw.rect(self.screen, bcol, (pad, by, w - 32, 48), border_radius=6)
        st = self.font_lbl.render(state, True, (10, 10, 10))
        self.screen.blit(st, (pad + (w - 32) // 2 - st.get_width() // 2, by + 15))

    def _draw_lidar_bars(self, lidar: list[float], x: int, y: int) -> None:
        w = WIN_W - x - 20
        self._panel(x, y, w, 200)
        self._blit(self.font_sm, "LIDAR (distance aux bords / obstacle)",
                   x + 16, y + 10, COL_GRAY)
        max_range = float(sensors.LIDAR_MAX_RANGE)
        bar_x = x + 90
        bar_w = w - 110
        for i in range(5):
            dist = lidar[i] if i < len(lidar) else max_range
            by = y + 36 + i * 30
            self._blit(self.font_sm, LIDAR_LABELS[i], x + 16, by + 1, COL_WHITE)
            pygame.draw.rect(self.screen, (60, 60, 68), (bar_x, by, bar_w, 14), border_radius=3)
            ratio = max(0.0, min(1.0, dist / max_range))
            # proche = rouge, loin = vert
            col = (int(255 * (1 - ratio)), int(230 * ratio), 40)
            pygame.draw.rect(self.screen, col, (bar_x, by, int(bar_w * ratio), 14), border_radius=3)
            self._blit(self.font_sm, f"{dist:4.0f}px", bar_x + bar_w - 52, by + 1, COL_WHITE)

    def _draw_speed_curve(self, x: int, y: int, w: int, h: int) -> None:
        self._panel(x, y, w, h)
        self._blit(self.font_sm, "COURBE DE VITESSE (km/h)", x + 12, y + 8, COL_GRAY)
        plot_x, plot_y = x + 12, y + 28
        plot_w, plot_h = w - 24, h - 44

        # Grille + graduations (0, moitie, max)
        for frac, lbl in ((0.0, "0"), (0.5, f"{SPEED_AXIS_MAX/2:.0f}"),
                          (1.0, f"{SPEED_AXIS_MAX:.0f}")):
            gy = plot_y + plot_h - int(plot_h * frac)
            pygame.draw.line(self.screen, (55, 55, 62), (plot_x, gy),
                             (plot_x + plot_w, gy), 1)
            self._blit(self.font_sm, lbl, plot_x + plot_w + 2, gy - 7, COL_GRAY)

        if len(self.speed_hist) >= 2:
            n = len(self.speed_hist)
            pts = []
            for i, s in enumerate(self.speed_hist):
                px = plot_x + int(plot_w * i / (SPEED_HIST - 1))
                ratio = max(0.0, min(1.0, s / SPEED_AXIS_MAX))
                py = plot_y + plot_h - int(plot_h * ratio)
                pts.append((px, py))
            pygame.draw.lines(self.screen, COL_CYAN, False, pts, 2)
