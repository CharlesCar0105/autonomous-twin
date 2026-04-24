"""
main.py — Boucle de jeu principale du simulateur.

Contrôles :
    ↑ / ↓       Accélérer / Freiner
    ← / →       Tourner
    R            Reset position
    L            Afficher/masquer Lidar
    F3           Afficher/masquer debug
    Clic droit   Repositionner la voiture
    Échap        Quitter
"""

import pygame
import sys
import os
import math

# Ajouter la racine du projet au path pour les imports absolus
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simulator.track import Track
from simulator.car import Car
from simulator import physics
from simulator import sensors
import numpy as np


# --- Constantes -----------------------------------------------------------

WINDOW_TITLE = "Autonomous Twin — Simulateur"
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Sensibilité du volant
STEERING_SPEED = 150.0          # degrés/s quand on appuie
STEERING_RETURN_SPEED = 300.0   # degrés/s retour au centre

# Couleurs HUD
COL_GREEN = (0, 230, 64)
COL_RED = (255, 60, 60)
COL_YELLOW = (255, 220, 50)
COL_CYAN = (0, 255, 255)
COL_WHITE = (240, 240, 240)
COL_GRAY = (160, 160, 160)


# --- HUD -----------------------------------------------------------------

def draw_hud(screen, car, track, clock, font, font_big, show_debug):
    """Affiche le tableau de bord en surimpression."""
    speed = physics.speed_kmh(car)
    on_border = track.is_border(int(car.x), int(car.y))

    # --- Fond semi-transparent pour le HUD gauche ---
    hud_bg = pygame.Surface((200, 130), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 120))
    screen.blit(hud_bg, (5, 5))

    # Vitesse
    speed_col = COL_RED if speed > 150 else COL_GREEN
    txt = font_big.render(f"{speed:.0f} km/h", True, speed_col)
    screen.blit(txt, (15, 12))

    # Angle volant (texte)
    txt = font.render(f"Volant: {car.steering:+6.1f}", True, COL_WHITE)
    screen.blit(txt, (15, 45))

    # Barre de direction visuelle
    bar_x, bar_y, bar_w = 15, 70, 175
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, 8), border_radius=4)
    center_x = bar_x + bar_w // 2
    indicator_x = center_x + int((car.steering / 45.0) * (bar_w // 2))
    pygame.draw.circle(screen, COL_YELLOW, (indicator_x, bar_y + 4), 6)
    pygame.draw.line(screen, (120, 120, 120), (center_x, bar_y), (center_x, bar_y + 8), 1)

    # Barre gaz (vert)
    pygame.draw.rect(screen, (60, 60, 60), (15, 88, 80, 10), border_radius=3)
    if car.throttle > 0:
        tw = int(car.throttle * 80)
        pygame.draw.rect(screen, COL_GREEN, (15, 88, tw, 10), border_radius=3)
    txt = font.render("GAZ", True, COL_GRAY)
    screen.blit(txt, (100, 86))

    # Barre frein (rouge)
    pygame.draw.rect(screen, (60, 60, 60), (15, 104, 80, 10), border_radius=3)
    if car.brake > 0:
        bw = int(car.brake * 80)
        pygame.draw.rect(screen, COL_RED, (15, 104, bw, 10), border_radius=3)
    txt = font.render("FREIN", True, COL_GRAY)
    screen.blit(txt, (100, 102))

    # Avertissement bordure
    if on_border:
        txt = font_big.render("BORDURE !", True, COL_RED)
        cx = SCREEN_WIDTH // 2 - txt.get_width() // 2
        screen.blit(txt, (cx, 15))

    # Debug
    if show_debug:
        lines = [
            f"Pos: ({car.x:.0f}, {car.y:.0f})  Cap: {math.degrees(car.angle):.1f} deg",
            f"FPS: {clock.get_fps():.0f}   V_px: {car.speed:.0f} px/s",
        ]
        for i, line in enumerate(lines):
            txt = font.render(line, True, COL_GRAY)
            screen.blit(txt, (15, SCREEN_HEIGHT - 48 + i * 20))

    # Aide contrôles
    help_str = "Fleches: Conduire | R: Reset | L: Lidar | C: Camera | F3: Debug | Clic droit: Placer | Echap: Quitter"
    txt = font.render(help_str, True, (100, 100, 100))
    screen.blit(txt, (SCREEN_WIDTH - txt.get_width() - 10, SCREEN_HEIGHT - 20))


def draw_lidar(screen, car, lidar_distances):
    """Dessine les rayons lidar en surimpression."""
    for i, angle_offset in enumerate(sensors.LIDAR_ANGLES):
        ray_angle = car.angle + math.radians(angle_offset)
        dist = lidar_distances[i]
        end_x = car.x + dist * math.cos(ray_angle)
        end_y = car.y + dist * math.sin(ray_angle)

        # Couleur selon distance (vert = loin, rouge = proche)
        ratio = min(1.0, dist / sensors.LIDAR_MAX_RANGE)
        col = (int(255 * (1 - ratio)), int(255 * ratio), 0)

        pygame.draw.line(screen, col, (int(car.x), int(car.y)),
                         (int(end_x), int(end_y)), 2)
        pygame.draw.circle(screen, COL_RED, (int(end_x), int(end_y)), 4)

        # Afficher la distance
        mid_x = int(car.x + dist * 0.6 * math.cos(ray_angle))
        mid_y = int(car.y + dist * 0.6 * math.sin(ray_angle))
        font_sm = pygame.font.SysFont("Consolas", 11)
        dtxt = font_sm.render(f"{dist:.0f}", True, COL_CYAN)
        screen.blit(dtxt, (mid_x, mid_y))


# --- Boucle principale ---------------------------------------------------

def main() -> None:
    """Point d'entrée du simulateur."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 15)
    font_big = pygame.font.SysFont("Consolas", 24, bold=True)

    # --- Chargement du circuit ---
    try:
        track = Track("circuit_01", SCREEN_WIDTH, SCREEN_HEIGHT)
    except FileNotFoundError as e:
        print(f"\n{'=' * 60}")
        print(f"ERREUR : {e}")
        print(f"{'=' * 60}\n")
        pygame.quit()
        sys.exit(1)

    # --- Création de la voiture ---
    car = Car(track.start_x, track.start_y, track.start_angle)

    # --- État ---
    show_lidar = False
    show_debug = True
    show_camera = False
    font_sm = pygame.font.SysFont("Consolas", 12)

    print(f"[Simulateur] Circuit charge. Voiture en ({car.x:.0f}, {car.y:.0f})")
    print(f"[Simulateur] Fleches=Conduire | R=Reset | L=Lidar | Echap=Quitter")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)  # cap pour éviter explosion physique

        # ── Événements ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset(track.start_x, track.start_y, track.start_angle)
                elif event.key == pygame.K_l:
                    show_lidar = not show_lidar
                elif event.key == pygame.K_F3:
                    show_debug = not show_debug
                elif event.key == pygame.K_c:
                    show_camera = not show_camera
                    print(f"[Simulateur] Camera {'ON' if show_camera else 'OFF'}")
                # TODO: K_SPACE → apparition du mur

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # clic droit → repositionner
                    mx, my = event.pos
                    car.reset(mx, my, math.degrees(car.angle))
                    print(f"[Simulateur] Voiture repositionnee en ({mx}, {my})")

        # ── Contrôles clavier (continu) ───────────────────────────────
        keys = pygame.key.get_pressed()

        # Gaz / Frein
        car.throttle = 1.0 if keys[pygame.K_UP] else 0.0
        car.brake = 1.0 if keys[pygame.K_DOWN] else 0.0

        # Direction (progressive avec retour au centre)
        if keys[pygame.K_LEFT]:
            car.steering -= STEERING_SPEED * dt
        elif keys[pygame.K_RIGHT]:
            car.steering += STEERING_SPEED * dt
        else:
            # Retour doux au centre
            if abs(car.steering) < STEERING_RETURN_SPEED * dt:
                car.steering = 0.0
            elif car.steering > 0:
                car.steering -= STEERING_RETURN_SPEED * dt
            else:
                car.steering += STEERING_RETURN_SPEED * dt
        car.steering = max(-45.0, min(45.0, car.steering))

        # ── Physique ──────────────────────────────────────────────────
        physics.update(car, dt)

        # ── Lidar (debug) ────────────────────────────────────────────
        lidar_distances = sensors.get_lidar(track, car) if show_lidar else []

        # ── Rendu ─────────────────────────────────────────────────────
        track.draw(screen)

        if show_lidar and lidar_distances:
            draw_lidar(screen, car, lidar_distances)

        car.draw(screen)
        draw_hud(screen, car, track, clock, font, font_big, show_debug)

        # ── Aperçu caméra (ce que le pilote verra) ────────────────────
        if show_camera:
            cam_img = sensors.get_camera_view(screen, car)
            # Convertir numpy → Surface Pygame
            cam_surf = pygame.surfarray.make_surface(
                np.transpose(cam_img, (1, 0, 2))
            )
            # Agrandir pour visibilité (×2) 
            cam_display = pygame.transform.scale(cam_surf, (256, 256))
            # Position en haut à droite
            cam_x = SCREEN_WIDTH - 266
            cam_y = 10
            # Cadre
            pygame.draw.rect(screen, COL_CYAN, (cam_x - 2, cam_y - 2, 260, 260), 2)
            screen.blit(cam_display, (cam_x, cam_y))
            # Label
            lbl = font_sm.render("CAMERA (vue pilote)", True, COL_CYAN)
            screen.blit(lbl, (cam_x, cam_y + 258))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
