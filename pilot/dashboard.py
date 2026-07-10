"""
dashboard.py — Tableau de bord temps réel côté pilote.

Fenêtre Pygame séparée : le pilote est un process distinct du simulateur,
donc il peut ouvrir SA PROPRE fenêtre (un display par process). Rendu
purement passif : la classe ne touche jamais au réseau ni aux commandes,
elle affiche ce que le pilote perçoit et décide (critère CDC "interface
temps-réel côté pilote affichant ce que la voiture voit").

Panneaux affichés :
    1. Vue caméra brute (128x128 reçue par ZMQ, affichée x2)
    2. Masque perception (compute_mask, "image segmentée" du CDC)
    3. Lidar : 5 barres verticales (-60/-30/0/+30/+60°)
    4. Vitesse : valeur + courbe temps réel (axe 0-80 km/h)
    5. Panneau détecté : sprite + dernière détection + limite + zone restante
    6. État pilote : bannière CONDUITE / LIMITE XX / ARRET STOP / URGENCE

Câblage : pilot/main.py --dashboard instancie Dashboard() une fois puis
appelle dashboard.update(...) en fin de boucle. Opt-in : import et
instanciation restent dans le bloc `if args.dashboard`, donc zero cout
quand le flag est absent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collections import deque
from typing import Optional

import numpy as np
import pygame

from pilot.perception import compute_mask
from pilot.signs import LIMIT_ZONE_PX, SignTracker


# --- Couleurs (memes tons que simulator/main.py, coherence visuelle) ------

COL_GREEN = (0, 230, 64)
COL_RED = (255, 60, 60)
COL_ORANGE = (255, 150, 30)
COL_CYAN = (0, 255, 255)
COL_WHITE = (240, 240, 240)
COL_GRAY = (160, 160, 160)
COL_PANEL_BG = (28, 28, 34)
COL_BORDER = (70, 70, 78)
COL_TEXT_ON_BANNER = (10, 10, 10)

ASSETS_SIGNS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "signs"
)

# cf simulator/sensors.py::LIDAR_MAX_RANGE / LIDAR_ANGLES -- dupliquees ici
# (meme regle que pilot/emergency.py : le pilote ne depend pas de simulator/
# en production, il ne recoit que ses capteurs par ZMQ).
LIDAR_MAX_RANGE_PX = 300.0
LIDAR_ANGLES = (-60, -30, 0, 30, 60)
LIDAR_DANGER_PX = 70.0          # barre rouge sous ce seuil (spec)

SPEED_AXIS_MAX_KMH = 80.0       # echelle fixe de la courbe vitesse
SPEED_HISTORY_LEN = 300         # ~ dernieres valeurs affichees (deque)


class Dashboard:
    """Fenetre Pygame secondaire : miroir visuel de la perception + etat pilote."""

    def __init__(self, width: int = 420, height: int = 580) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Twin — Dashboard Pilote")

        self.font_sm = pygame.font.SysFont("Consolas", 12)
        self.font = pygame.font.SysFont("Consolas", 15)
        self.font_big = pygame.font.SysFont("Consolas", 22, bold=True)

        self._speed_history: deque = deque(maxlen=SPEED_HISTORY_LEN)
        self._sign_cache: dict = {}

        # --- Rects des panneaux (calcules une fois pour toutes) ---
        m = 8
        banner_h = 40
        cam_size = 256   # 128 * 2 (spec : camera affichee x2)

        self._rect_banner = pygame.Rect(0, 0, width, banner_h)
        self._rect_cam = pygame.Rect(m, banner_h + m, cam_size, cam_size)

        right_x = self._rect_cam.right + m
        right_w = width - right_x - m
        self._rect_mask = pygame.Rect(right_x, self._rect_cam.y, right_w, 140)
        self._rect_sign = pygame.Rect(
            right_x, self._rect_mask.bottom + m, right_w,
            self._rect_cam.bottom - (self._rect_mask.bottom + m),
        )

        self._rect_lidar = pygame.Rect(
            m, self._rect_cam.bottom + m, width - 2 * m, 100
        )
        self._rect_speed = pygame.Rect(
            m, self._rect_lidar.bottom + m, width - 2 * m, 120
        )
        self._rect_footer = pygame.Rect(
            m, self._rect_speed.bottom + m, width - 2 * m,
            height - (self._rect_speed.bottom + m) - 6,
        )

    # --- API publique -------------------------------------------------

    def update(
        self,
        camera: np.ndarray,
        lidar: list,
        speed_kmh: float,
        commands: dict,
        tracker: Optional[SignTracker],
        policy_name: str,
        emergency_active: bool,
    ) -> bool:
        """Dessine une frame complete et l'affiche (a appeler 1x/boucle).

        Retourne False si l'utilisateur a ferme la fenetre (croix ou Echap)
        -- l'appelant doit alors arreter proprement (contrat repris de la
        version Charles du dashboard)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        self._speed_history.append(float(speed_kmh))

        self.screen.fill((14, 14, 18))
        self._draw_camera_panel(camera)
        self._draw_mask_panel(camera)
        self._draw_sign_panel(tracker)
        self._draw_lidar_panel(lidar)
        self._draw_speed_panel(speed_kmh, tracker)
        self._draw_footer(commands)
        self._draw_status_banner(tracker, policy_name, emergency_active)

        pygame.display.flip()
        return True

    def close(self) -> None:
        """Ferme la fenetre du dashboard (le pilote n'utilise pygame que
        pour elle)."""
        pygame.quit()

    # --- Panneaux -------------------------------------------------------

    def _draw_camera_panel(self, camera: np.ndarray) -> None:
        rect = self._rect_cam
        cam = np.asarray(camera, dtype=np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(cam, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (rect.w, rect.h))
        self.screen.blit(surf, rect.topleft)
        pygame.draw.rect(self.screen, COL_CYAN, rect, width=2, border_radius=2)
        self._label_chip(rect.x + 4, rect.y + 4, "CAMERA")

    def _draw_mask_panel(self, camera: np.ndarray) -> None:
        rect = self._rect_mask
        mask = compute_mask(camera, size=64)          # (64, 64) float32 {0,1}
        mask_rgb = np.full((64, 64, 3), (35, 35, 42), dtype=np.uint8)
        mask_rgb[mask > 0.5] = (0, 220, 90)            # route detectee
        surf = pygame.surfarray.make_surface(np.transpose(mask_rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (rect.w, rect.h))
        self.screen.blit(surf, rect.topleft)
        pygame.draw.rect(self.screen, COL_CYAN, rect, width=2, border_radius=2)
        self._label_chip(rect.x + 4, rect.y + 4, "MASQUE")

    def _draw_sign_panel(self, tracker: Optional[SignTracker]) -> None:
        rect = self._rect_sign
        self._panel_bg(rect)
        self._label_chip(rect.x + 4, rect.y + 4, "PANNEAU")

        if tracker is None:
            txt = self.font_sm.render("desactive (--no-signs)", True, COL_GRAY)
            self.screen.blit(txt, (rect.x + 6, rect.y + 24))
            return

        kind = None
        if tracker.stop_active:
            kind = "stop"
        elif tracker.speed_limit is not None:
            kind = f"{int(tracker.speed_limit)}"

        sprite_size = 48
        sx = rect.x + (rect.w - sprite_size) // 2
        sy = rect.y + 20
        if kind is not None:
            sprite = self._get_sign_sprite(kind, sprite_size)
            self.screen.blit(sprite, (sx, sy))
        else:
            center = (sx + sprite_size // 2, sy + sprite_size // 2)
            pygame.draw.circle(self.screen, (50, 50, 58), center, sprite_size // 2)
            dash = self.font.render("--", True, COL_GRAY)
            self.screen.blit(dash, (center[0] - dash.get_width() // 2,
                                     center[1] - dash.get_height() // 2))

        ty = sy + sprite_size + 4
        det = tracker.last_detection or "--"
        self.screen.blit(self.font_sm.render(f"det: {det}", True, COL_WHITE),
                          (rect.x + 6, ty))

        has_limit = tracker.speed_limit is not None
        lim_txt = f"{tracker.speed_limit:.0f} km/h" if has_limit else "--"
        lim_col = COL_ORANGE if has_limit else COL_GRAY
        self.screen.blit(self.font_sm.render(f"limite: {lim_txt}", True, lim_col),
                          (rect.x + 6, ty + 14))

        if has_limit:
            ratio = max(0.0, min(1.0, tracker.limit_zone_left_px / LIMIT_ZONE_PX))
            bar = pygame.Rect(rect.x + 6, ty + 30, rect.w - 12, 6)
            pygame.draw.rect(self.screen, (60, 60, 66), bar, border_radius=3)
            fill = pygame.Rect(bar.x, bar.y, int(bar.w * ratio), bar.h)
            pygame.draw.rect(self.screen, COL_ORANGE, fill, border_radius=3)

    def _draw_lidar_panel(self, lidar: list) -> None:
        rect = self._rect_lidar
        self._panel_bg(rect)
        self._label_chip(rect.right - 50, rect.y + 4, "LIDAR")

        n = len(LIDAR_ANGLES)
        slot_w = rect.w / n
        bar_w = slot_w * 0.42
        top_pad, bottom_pad = 16, 18
        bar_area_h = rect.h - top_pad - bottom_pad
        baseline_y = rect.y + top_pad + bar_area_h

        for i, angle in enumerate(LIDAR_ANGLES):
            dist = lidar[i] if i < len(lidar) else 0.0
            ratio = max(0.0, min(1.0, dist / LIDAR_MAX_RANGE_PX))
            bar_h = ratio * bar_area_h
            bx = rect.x + i * slot_w + (slot_w - bar_w) / 2
            by = baseline_y - bar_h
            color = COL_RED if dist < LIDAR_DANGER_PX else COL_GREEN
            pygame.draw.rect(self.screen, color,
                              (int(bx), int(by), int(bar_w), int(bar_h)),
                              border_radius=2)

            val_txt = self.font_sm.render(f"{dist:.0f}", True, COL_WHITE)
            self.screen.blit(val_txt, (int(bx + bar_w / 2 - val_txt.get_width() / 2),
                                        int(by) - 14))

            ang_str = f"{angle:+d}°" if angle != 0 else "0°"
            ang_txt = self.font_sm.render(ang_str, True, COL_GRAY)
            ang_x = rect.x + i * slot_w + slot_w / 2 - ang_txt.get_width() / 2
            self.screen.blit(ang_txt, (int(ang_x), int(baseline_y) + 3))

        pygame.draw.line(self.screen, COL_BORDER,
                          (rect.x, int(baseline_y)), (rect.right, int(baseline_y)), 1)

    def _draw_speed_panel(self, speed_kmh: float, tracker: Optional[SignTracker]) -> None:
        rect = self._rect_speed
        self._panel_bg(rect)
        self._label_chip(rect.x + 4, rect.y + 4, "VITESSE")

        limit = tracker.speed_limit if tracker is not None else None
        val_col = COL_RED if (limit is not None and speed_kmh > limit) else COL_GREEN
        val_txt = self.font_big.render(f"{speed_kmh:.0f} km/h", True, val_col)
        self.screen.blit(val_txt, (rect.x + 8, rect.y + 20))

        chart_top = rect.y + 20 + val_txt.get_height() + 4
        chart = pygame.Rect(rect.x + 8, chart_top, rect.w - 16,
                             rect.bottom - chart_top - 6)
        pygame.draw.rect(self.screen, (18, 18, 22), chart)
        pygame.draw.rect(self.screen, COL_BORDER, chart, width=1)

        hist = list(self._speed_history)
        if len(hist) >= 2:
            denom = max(len(hist) - 1, 1)
            pts = []
            for i, v in enumerate(hist):
                x = chart.x + chart.w * i / denom
                ratio = max(0.0, min(1.0, v / SPEED_AXIS_MAX_KMH))
                y = chart.bottom - ratio * chart.h
                pts.append((int(x), int(y)))
            pygame.draw.lines(self.screen, COL_CYAN, False, pts, 2)

        if limit is not None:
            ratio = max(0.0, min(1.0, limit / SPEED_AXIS_MAX_KMH))
            ly = int(chart.bottom - ratio * chart.h)
            pygame.draw.line(self.screen, COL_ORANGE, (chart.x, ly), (chart.right, ly), 1)
            lim_txt = self.font_sm.render(f"limite {limit:.0f}", True, COL_ORANGE)
            self.screen.blit(lim_txt, (chart.right - lim_txt.get_width() - 2,
                                        max(chart.y, ly - 13)))

    def _draw_footer(self, commands: dict) -> None:
        rect = self._rect_footer
        txt = (f"steer {commands.get('steering', 0.0):+6.1f}°   "
               f"thr {commands.get('throttle', 0.0):.2f}   "
               f"brk {commands.get('brake', 0.0):.2f}")
        surf = self.font_sm.render(txt, True, COL_GRAY)
        self.screen.blit(surf, (rect.x, rect.y))

    def _draw_status_banner(self, tracker: Optional[SignTracker], policy_name: str,
                             emergency_active: bool) -> None:
        rect = self._rect_banner
        stop_active = tracker.stop_active if tracker is not None else False
        limit = tracker.speed_limit if tracker is not None else None

        if emergency_active:
            # Clignote entre deux rouges CLAIRS (jamais sombre) pour garder le
            # texte sombre lisible sur les deux phases du clignotement.
            blink_on = (pygame.time.get_ticks() // 250) % 2 == 0
            color = COL_RED if blink_on else (255, 120, 90)
            text = "FREINAGE URGENCE"
        elif stop_active:
            color, text = COL_RED, "ARRET STOP"
        elif limit is not None:
            color, text = COL_ORANGE, f"LIMITE {limit:.0f}"
        else:
            color, text = COL_GREEN, "CONDUITE"

        pygame.draw.rect(self.screen, color, rect)
        label = self.font_big.render(text, True, COL_TEXT_ON_BANNER)
        self.screen.blit(label, (10, rect.centery - label.get_height() // 2))

        policy_txt = self.font.render(f"[{policy_name.upper()}]", True, COL_TEXT_ON_BANNER)
        self.screen.blit(policy_txt, (rect.right - policy_txt.get_width() - 10,
                                       rect.centery - policy_txt.get_height() // 2))

    # --- Aides ------------------------------------------------------------

    def _panel_bg(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, COL_PANEL_BG, rect, border_radius=4)
        pygame.draw.rect(self.screen, COL_BORDER, rect, width=1, border_radius=4)

    def _label_chip(self, x: int, y: int, text: str) -> None:
        surf = self.font_sm.render(text, True, COL_CYAN)
        bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        self.screen.blit(bg, (x, y))
        self.screen.blit(surf, (x + 3, y + 1))

    def _get_sign_sprite(self, kind: str, size: int) -> pygame.Surface:
        key = (kind, size)
        cached = self._sign_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(ASSETS_SIGNS_DIR, f"{kind}.png")
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (size, size))
        self._sign_cache[key] = img
        return img


# --- Preview headless (verification visuelle hors simulateur) -------------

if __name__ == "__main__":
    import glob
    import math

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    dash = Dashboard()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "val", "gen_025")
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    cameras = [np.load(f)["camera"] for f in npz_files[::40][:8]]

    class _FakeTracker:
        """Double minimal de SignTracker pour le preview : etats pilotes poses
        a la main (limite/stop/zone), pas de vraie inference camera-only --
        seule la vue camera injectee est un vrai npz (cf spec)."""

        def __init__(self) -> None:
            self.speed_limit: Optional[float] = None
            self.stop_active = False
            self.last_detection: Optional[str] = None
            self.limit_zone_left_px = 0.0

    tracker = _FakeTracker()

    for i in range(40):
        camera = cameras[i % len(cameras)]

        # --- vitesse synthetique : rampe, freinage stop, reprise, urgence ---
        if i < 15:
            speed = i * 4.5
        elif i < 20:
            speed = 63.0 + (i % 2) * 3.0
        elif i < 26:
            speed = max(0.0, 60.0 - (i - 20) * 12.0)
        elif i < 30:
            speed = (i - 25) * 10.0
        elif i < 34:
            speed = max(20.0, 55.0 - (i - 30) * 8.0)
        else:
            speed = 48.0 + (i % 3) * 2.0

        # --- lidar synthetique : jitter, puis rapproche pendant l'urgence ---
        base = [220.0, 180.0, 260.0, 190.0, 225.0]
        if 30 <= i < 34:
            lidar = [60.0, 45.0, 55.0, 50.0, 210.0]
        elif 5 <= i < 8:
            lidar = [200.0, 65.0, 90.0, 170.0, 150.0]
        else:
            lidar = [v + 10.0 * math.sin(i * 0.5 + k) for k, v in enumerate(base)]

        # --- panneaux / etat pilote synthetiques ---
        emergency_active = 30 <= i < 34
        if 10 <= i < 20:
            tracker.speed_limit = 50.0
            tracker.last_detection = "50(0.94)"
            tracker.stop_active = False
            tracker.limit_zone_left_px = max(0.0, 600.0 - (i - 10) * 60.0)
        elif 20 <= i < 26:
            tracker.stop_active = True
            tracker.speed_limit = None
            tracker.last_detection = None
            tracker.limit_zone_left_px = 0.0
        elif i >= 34:
            tracker.stop_active = False
            tracker.speed_limit = 50.0
            tracker.last_detection = "50(0.91)"
            tracker.limit_zone_left_px = max(0.0, 600.0 - (i - 34) * 60.0)
        else:
            tracker.stop_active = False
            tracker.speed_limit = None
            tracker.last_detection = None
            tracker.limit_zone_left_px = 0.0

        commands = {
            "steering": 30.0 * math.sin(i * 0.3),
            "throttle": 0.0 if (emergency_active or tracker.stop_active) else 0.6,
            "brake": 1.0 if (emergency_active or tracker.stop_active) else 0.0,
        }

        dash.update(camera, lidar, speed, commands, tracker, "pid", emergency_active)

    out_path = r"C:\Users\lorenzo\Desktop\captures\dashboard_preview.png"
    pygame.image.save(dash.screen, out_path)
    print(f"[Dashboard] Preview sauvegardee -> {out_path}")
