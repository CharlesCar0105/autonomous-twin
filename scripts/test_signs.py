"""
test_signs.py -- Valide la lecture de panneaux en boucle fermee (headless).

Scenarios (in-process, pattern test_cnn.py -- verite terrain issue du
placement, JAMAIS montree au pilote qui ne voit que la camera) :

  1. LIMITE : panneau 30 place sur la trajectoire PID (dry-run prealable,
     meme technique que place_signs.py). Le PID roule a speed_target=80.
     PASS si la vitesse descend sous 35 km/h dans les 6 s apres le passage.
  2. STOP : panneau STOP place pareil. PASS si v < 2 km/h est atteint avant
     t_pass+8 s (apres lancement de la voiture -- le panneau est vu ~2 s
     avant t_pass, portee camera avant), puis v > 20 km/h dans les 10 s
     suivantes (reprise).

Usage : python scripts/test_signs.py [--circuit gen_000]
"""

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.track import Track
from simulator.car import Car
from simulator import physics, sensors
from simulator.signs import RoadSign
from pilot.control import pid_policy
from pilot.signs import SignTracker

FPS = 60

OFFSETS_PX = (55.0, 70.0, 85.0)   # offsets lateraux candidats (cf place_signs)


# copie de place_signs._crosses_border -- garder synchronise
def _crosses_border(track: Track, x1: float, y1: float, x2: float, y2: float) -> bool:
    """True si le segment [p1,p2] traverse au moins un pixel de bordure noire."""
    n = max(2, int(math.hypot(x2 - x1, y2 - y1)))
    black = 0
    for i in range(n + 1):
        t = i / n
        px_, py_ = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
        if 0 <= px_ < track.width and 0 <= py_ < track.height:
            r, g, b = track.pixels[px_, py_]
            if r < 80 and g < 80 and b < 80:
                black += 1
    return black >= 2


def _dry_run_place(track: Track, t_place: float, kind: str,
                   circuit: str = "?") -> tuple[RoadSign, int]:
    """Trajectoire PID sans panneau ; pose `kind` perpendiculairement au
    point atteint a t_place secondes. Retourne (sign, frame_du_passage)."""
    car = Car(track.start_x, track.start_y, track.start_angle)
    dt = 1.0 / FPS
    target_frame = int(t_place * FPS)
    for f in range(target_frame + 1):
        lidar = sensors.get_lidar(track, car)
        s, t, b = pid_policy(lidar, physics.speed_kmh(car), speed_target=80.0)
        car.set_controls(s, t, b)
        physics.update(car, dt)
    perp = (-math.sin(car.angle), math.cos(car.angle))
    for off in OFFSETS_PX:
        sx, sy = car.x + perp[0] * off, car.y + perp[1] * off
        in_frame = 20 < sx < track.width - 20 and 20 < sy < track.height - 20
        # Panneau garanti HORS piste (meme regle que place_signs) : le
        # segment voiture -> panneau doit croiser une bordure noire, herbe
        # et route etant toutes deux blanches.
        if in_frame and _crosses_border(track, car.x, car.y, sx, sy):
            return RoadSign(sx, sy, kind), target_frame
    raise RuntimeError(
        f"placement impossible : circuit={circuit!r} kind={kind!r} "
        f"t_place={t_place}s pos_voiture=({car.x:.1f}, {car.y:.1f}) "
        f"offsets essayes={OFFSETS_PX} (bord de frame ou bordure jamais croisee)"
    )


def _closed_loop(track: Track, sign: RoadSign, seconds: float) -> list[dict]:
    """Roule le PID avec le panneau composite + SignTracker actif.
    Retourne l'historique [{t, speed, limit, stop_active}]."""
    car = Car(track.start_x, track.start_y, track.start_angle)
    tracker = SignTracker()
    dt = 1.0 / FPS
    hist = []
    for f in range(int(seconds * FPS)):
        now = f * dt
        lidar = sensors.get_lidar(track, car)
        camera = sensors.get_camera_view_from_track(track, car, signs=[sign])
        speed = physics.speed_kmh(car)
        steering, throttle, brake = pid_policy(lidar, speed, speed_target=80.0)
        tracker.update(camera, speed, now)
        # copie de pilot/main.py:109-120 (governor) -- garder synchronise
        if tracker.speed_limit is not None and speed > tracker.speed_limit:
            throttle = min(throttle, 0.15)
            if speed > tracker.speed_limit + 15.0:
                brake = max(brake, 0.3)
        if tracker.stop_active:
            throttle, brake = 0.0, 1.0
        car.set_controls(steering, throttle, brake)
        physics.update(car, dt)
        hist.append({"t": now, "speed": speed,
                     "limit": tracker.speed_limit,
                     "stop": tracker.stop_active})
    return hist


def scenario_limite(circuit: str) -> bool:
    track = Track(circuit, 1280, 720)
    sign, f_pass = _dry_run_place(track, t_place=6.0, kind="30", circuit=circuit)
    hist = _closed_loop(track, sign, seconds=6.0 + 6.0 + 2.0)
    t_pass = f_pass / FPS
    window = [h for h in hist if t_pass <= h["t"] <= t_pass + 6.0]
    ok_detect = any(h["limit"] == 30.0 for h in hist)
    ok_slow = any(h["speed"] <= 35.0 for h in window)
    print(f"  [limite30] detecte={ok_detect}  v<=35 dans fenetre={ok_slow}  "
          f"-> {'PASS' if ok_detect and ok_slow else 'FAIL'}")
    return ok_detect and ok_slow


def scenario_stop(circuit: str) -> bool:
    track = Track(circuit, 1280, 720)
    sign, f_pass = _dry_run_place(track, t_place=6.0, kind="stop", circuit=circuit)
    hist = _closed_loop(track, sign, seconds=6.0 + 8.0 + 10.0)
    t_pass = f_pass / FPS
    # Detection anticipee : le panneau entre dans le champ camera ~2 s AVANT
    # t_pass (la camera regarde CAMERA_DISTANCE px devant la voiture), donc
    # l'arret complet peut se produire entierement avant [t_pass, t_pass+8].
    # Pas de borne basse sur la fenetre d'arret : le harness ne composite
    # que NOTRE panneau STOP (signs=[sign]), seule cause d'arret possible.
    # On exclut juste l'immobilite initiale au spawn en exigeant que la
    # voiture ait d'abord ete lancee (v > 20 km/h).
    t_launch = next((h["t"] for h in hist if h["speed"] > 20.0), None)
    stopped = []
    if t_launch is not None:
        stopped = [h["t"] for h in hist
                   if t_launch < h["t"] <= t_pass + 8.0 and h["speed"] < 2.0]
    ok_stop = bool(stopped)
    ok_resume = False
    if ok_stop:
        t_stop = stopped[-1]
        ok_resume = any(h["speed"] > 20.0 for h in hist
                        if t_stop < h["t"] <= t_stop + 10.0)
    print(f"  [stop] arret={ok_stop}  reprise={ok_resume}  "
          f"-> {'PASS' if ok_stop and ok_resume else 'FAIL'}")
    return ok_stop and ok_resume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default="gen_000")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    print(f"[test_signs] circuit {args.circuit}")
    results = [scenario_limite(args.circuit), scenario_stop(args.circuit)]
    pygame.quit()
    n = sum(results)
    print(f"[test_signs] {n}/2 PASS")
    sys.exit(0 if n == 2 else 1)


if __name__ == "__main__":
    main()
