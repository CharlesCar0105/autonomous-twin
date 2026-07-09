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
import time
import argparse

# Ajouter la racine du projet au path pour les imports absolus
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simulator.track import Track
from simulator.car import Car
from simulator import physics
from simulator import sensors
from simulator.network import SimulatorServer
from simulator.wall import Wall
from simulator.timing import LapTimer, format_time
from simulator.signs import load_signs
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
    help_str = ("Fleches: Conduire | Espace: Mur | X: Retirer mur | T: Chrono | "
                "R: Reset | L: Lidar | C: Camera | F3: Debug | Echap: Quitter")
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


def draw_finish_line(screen, lap_timer):
    """Dessine la ligne d'arrivee (damier) sur la piste."""
    p1 = lap_timer.line_p1
    p2 = lap_timer.line_p2
    # Segment epais pointille noir/blanc facon damier.
    n = 12
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        a = (p1[0] + (p2[0] - p1[0]) * t0, p1[1] + (p2[1] - p1[1]) * t0)
        b = (p1[0] + (p2[0] - p1[0]) * t1, p1[1] + (p2[1] - p1[1]) * t1)
        col = (20, 20, 20) if i % 2 == 0 else (200, 200, 200)
        pygame.draw.line(screen, col, a, b, 6)


def draw_chrono(screen, lap_timer, font, font_big):
    """Affiche le panneau chrono (tours, temps, best lap) en haut au centre."""
    panel_w, panel_h = 260, 96
    px = SCREEN_WIDTH // 2 - panel_w // 2
    py = 6
    bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 130))
    screen.blit(bg, (px, py))

    # Ligne 1 : tour X/3 + statut
    if lap_timer.finished:
        head = f"TERMINE  -  Total {format_time(lap_timer.total_time)}"
        head_col = COL_YELLOW
    elif not lap_timer.started:
        head = "CHRONO : pret (avance pour demarrer)"
        head_col = COL_GRAY
    else:
        head = f"TOUR {lap_timer.current_lap_number}/3"
        head_col = COL_GREEN
    screen.blit(font.render(head, True, head_col), (px + 10, py + 6))

    # Ligne 2 : temps du tour en cours (gros)
    cur = format_time(lap_timer.current_lap_time)
    screen.blit(font_big.render(cur, True, COL_WHITE), (px + 10, py + 26))

    # Ligne 3 : dernier tour + meilleur tour
    last = format_time(lap_timer.last_lap)
    best = format_time(lap_timer.best_lap)
    screen.blit(font.render(f"Dernier: {last}", True, COL_CYAN), (px + 10, py + 58))
    best_col = COL_YELLOW if lap_timer._new_record else COL_GRAY
    screen.blit(font.render(f"Best: {best}", True, best_col), (px + 10, py + 76))


# --- Boucle principale ---------------------------------------------------

def main() -> None:
    """Point d'entrée du simulateur."""
    parser = argparse.ArgumentParser(description="Simulateur — Autonomous Twin")
    parser.add_argument("--server", action="store_true",
                        help="Active le serveur ZMQ : les commandes viennent du pilote IA.")
    parser.add_argument("--address", default="tcp://*:5555",
                        help="Adresse de bind du serveur ZMQ.")
    parser.add_argument("--circuit", default="circuit_02",
                        help="Nom du circuit a charger (fichier PNG dans assets/tracks/).")
    parser.add_argument("--start-pos", nargs=3, type=float, metavar=("X", "Y", "ANGLE_DEG"),
                        default=None,
                        help="Force la position/angle de depart (en pixels, degres). "
                             "Bypass la detection auto. Ex: --start-pos 640 615 0")
    parser.add_argument("--no-signs", action="store_true",
                        help="Ne charge pas les panneaux du circuit (sidecar .signs.json).")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 15)
    font_big = pygame.font.SysFont("Consolas", 24, bold=True)

    # --- Chargement du circuit ---
    try:
        track = Track(args.circuit, SCREEN_WIDTH, SCREEN_HEIGHT)
    except FileNotFoundError as e:
        print(f"\n{'=' * 60}")
        print(f"ERREUR : {e}")
        print(f"{'=' * 60}\n")
        pygame.quit()
        sys.exit(1)

    # --- Création de la voiture ---
    if args.start_pos is not None:
        sx, sy, sa = args.start_pos
        print(f"[Simulateur] Start pos force par CLI : ({sx:.0f}, {sy:.0f}) angle={sa:.1f} deg")
        car = Car(sx, sy, sa)
    else:
        car = Car(track.start_x, track.start_y, track.start_angle)

    # --- État ---
    show_lidar = False
    show_debug = True
    show_camera = False
    font_sm = pygame.font.SysFont("Consolas", 12)

    # Mur (aléa) : None tant qu'on n'a pas appuye sur Espace.
    wall = None

    # Chrono 3 tours : ligne d'arrivee a la position de depart du circuit.
    lap_timer = LapTimer(track.start_x, track.start_y, track.start_angle, args.circuit)

    # Panneaux statiques du circuit (sidecar JSON, absent = aucun).
    signs = [] if args.no_signs else load_signs(args.circuit)
    if signs:
        print(f"[Simulateur] {len(signs)} panneau(x) charges : "
              + ", ".join(s.kind for s in signs))

    # Position de la frame precedente (pour collision mur + detection de
    # franchissement de la ligne d'arrivee).
    prev_x, prev_y = car.x, car.y

    # --- Serveur ZMQ (mode pilote IA) ---
    server = SimulatorServer(args.address) if args.server else None

    print(f"[Simulateur] Circuit charge. Voiture en ({car.x:.0f}, {car.y:.0f})")
    mode = "PILOTE IA (ZMQ)" if args.server else "CLAVIER"
    print(f"[Simulateur] Mode : {mode}")
    print("[Simulateur] Fleches=Conduire | ESPACE=Mur | X=Retirer mur | "
          "T=Reset chrono | R=Reset | L=Lidar | Echap=Quitter")

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
                elif event.key == pygame.K_SPACE:
                    # L'Aléa : faire apparaitre le mur devant la voiture.
                    # Il disparaitra tout seul apres WALL_LIFETIME secondes.
                    wall = Wall.spawn_ahead(car, now=time.time())
                    print(f"[Simulateur] MUR apparu en ({wall.cx:.0f}, {wall.cy:.0f}) "
                          f"(disparait dans {wall.lifetime:.0f}s)")
                elif event.key == pygame.K_x:
                    wall = None
                    print("[Simulateur] Mur retire.")
                elif event.key == pygame.K_t:
                    lap_timer.reset()
                    car.reset(track.start_x, track.start_y, track.start_angle)
                    prev_x, prev_y = car.x, car.y
                    print("[Simulateur] Chrono remis a zero.")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # clic droit → repositionner
                    mx, my = event.pos
                    car.reset(mx, my, math.degrees(car.angle))
                    print(f"[Simulateur] Voiture repositionnee en ({mx}, {my})")

        # ── Auto-disparition du mur apres WALL_LIFETIME secondes ──────
        if wall is not None and wall.is_expired(time.time()):
            wall = None
            print("[Simulateur] Mur disparu (fin de duree de vie).")

        # ── Contrôles : pilote IA (ZMQ) OU clavier ────────────────────
        if server is not None:
            # Prepare capteurs pour le pilote. On utilise la camera tiree
            # directement de track.pixels (sans HUD, sans voiture dessinee,
            # sans rayons lidar) pour coller au dataset d'entrainement U-Net.
            lidar = sensors.get_lidar(track, car, wall)
            camera = sensors.get_camera_view_from_track(track, car, wall, signs=signs)
            speed_kmh = physics.speed_kmh(car)
            commands = server.send_sensors(camera, lidar, speed_kmh)
            if commands is not None:
                car.set_controls(
                    commands.get("steering", 0.0),
                    commands.get("throttle", 0.0),
                    commands.get("brake", 0.0),
                )
            # si timeout (pilote non connecte), on laisse les commandes en l'etat
        else:
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
                if abs(car.steering) < STEERING_RETURN_SPEED * dt:
                    car.steering = 0.0
                elif car.steering > 0:
                    car.steering -= STEERING_RETURN_SPEED * dt
                else:
                    car.steering += STEERING_RETURN_SPEED * dt
            car.steering = max(-45.0, min(45.0, car.steering))

        # ── Physique ──────────────────────────────────────────────────
        prev_x, prev_y = car.x, car.y
        physics.update(car, dt)

        # ── Collision mur : la voiture ne traverse pas le mur ─────────
        # On teste le pare-choc avant (18 px devant le centre). En cas de
        # contact on annule le deplacement de la frame et on stoppe net.
        if wall is not None:
            front_x = car.x + 18.0 * math.cos(car.angle)
            front_y = car.y + 18.0 * math.sin(car.angle)
            if wall.contains(front_x, front_y) or wall.contains(car.x, car.y):
                car.x, car.y = prev_x, prev_y
                car.speed = 0.0

        # ── Chrono : detection de franchissement de la ligne ──────────
        lap_timer.update(car, (prev_x, prev_y), time.time())

        # ── Lidar (debug) ────────────────────────────────────────────
        lidar_distances = sensors.get_lidar(track, car, wall) if show_lidar else []

        # ── Rendu ─────────────────────────────────────────────────────
        track.draw(screen)
        draw_finish_line(screen, lap_timer)

        for sign in signs:
            sign.draw(screen)

        if wall is not None:
            wall.draw(screen)
            # Compte a rebours avant disparition, au-dessus du mur.
            secs_left = wall.time_left(time.time())
            cd = font_big.render(f"{secs_left:.1f}s", True, COL_YELLOW)
            screen.blit(cd, (int(wall.cx) - cd.get_width() // 2, int(wall.cy) - 40))

        if show_lidar and lidar_distances:
            draw_lidar(screen, car, lidar_distances)

        car.draw(screen)
        draw_hud(screen, car, track, clock, font, font_big, show_debug)
        draw_chrono(screen, lap_timer, font, font_big)

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

    if server is not None:
        server.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
