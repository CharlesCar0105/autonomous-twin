"""
test_pid.py -- Valide que le PID baseline passe sur un set de circuits.

Simule la boucle simulateur+pilote IN-PROCESS (pas de ZMQ, pas de GUI)
pour chaque circuit et collecte :
  - frames off-track (voiture sur pixel non-blanc)
  - distance totale parcourue
  - vitesse moyenne
  - temps avant premier off-track (proxy "quand est-ce que ca casse")

Verdict par circuit : PASS si off_track_frac < 0.05 et distance > 200 px.

Usage :
    python scripts/test_pid.py                       # test tous les gen_*
    python scripts/test_pid.py --pattern 'gen_*'     # glob custom
    python scripts/test_pid.py --seconds 20          # duree par circuit
    python scripts/test_pid.py --verbose             # details par frame
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
from pilot.control import pid_policy


def run_circuit(
    track_name: str, seconds: float, fps: int, speed_target: float, verbose: bool
) -> dict:
    pygame.init()
    pygame.display.set_mode((1, 1))  # minimal surface pour que Track charge
    track = Track(track_name, 1280, 720)
    car = Car(track.start_x, track.start_y, track.start_angle)

    dt = 1.0 / fps
    n_frames = int(seconds * fps)
    off_track_frames = 0
    total_dist = 0.0
    speeds = []
    first_offtrack_frame = -1

    prev_x, prev_y = car.x, car.y
    for f in range(n_frames):
        lidar = sensors.get_lidar(track, car)
        speed_kmh = physics.speed_kmh(car)
        steering, throttle, brake = pid_policy(lidar, speed_kmh, speed_target=speed_target)
        car.set_controls(steering, throttle, brake)
        physics.update(car, dt)

        total_dist += math.hypot(car.x - prev_x, car.y - prev_y)
        speeds.append(car.speed)
        prev_x, prev_y = car.x, car.y

        if not track.is_on_road(int(car.x), int(car.y)):
            off_track_frames += 1
            if first_offtrack_frame < 0:
                first_offtrack_frame = f

        if verbose and f % 60 == 0:
            print(f"    f={f:4d}  ({car.x:4.0f}, {car.y:4.0f})  "
                  f"lidar_front={lidar[2]:4.0f}  steer={steering:+5.1f}  "
                  f"offtrack={off_track_frames}")

    pygame.quit()

    frac = off_track_frames / n_frames if n_frames > 0 else 1.0
    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
    return {
        "track": track_name,
        "frames": n_frames,
        "off_track_frames": off_track_frames,
        "off_track_frac": frac,
        "distance_px": total_dist,
        "avg_speed_px_s": avg_speed,
        "first_offtrack_frame": first_offtrack_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PID headless sur un set de circuits")
    parser.add_argument("--pattern", default="gen_*",
                        help="Glob des circuits (sans extension). Defaut: gen_*")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed-target", type=float, default=35.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max", type=int, default=None,
                        help="Limite le nombre de circuits testes.")
    args = parser.parse_args()

    tracks_dir = ROOT / "assets" / "tracks"
    candidates = sorted(p.stem for p in tracks_dir.glob(f"{args.pattern}.png"))
    if args.max:
        candidates = candidates[: args.max]

    if not candidates:
        print(f"[test_pid] aucun circuit trouve pour pattern {args.pattern}")
        return

    print(f"[test_pid] {len(candidates)} circuits, {args.seconds}s chacun, "
          f"speed_target={args.speed_target} km/h")
    print(f"  {'track':<12}  {'offtrack%':>9}  {'dist_px':>8}  {'avgV':>6}  verdict")
    print(f"  {'-' * 12}  {'-' * 9}  {'-' * 8}  {'-' * 6}  {'-' * 8}")

    n_pass = 0
    fails = []
    for name in candidates:
        if args.verbose:
            print(f"\n[test_pid] === {name} ===")
        try:
            r = run_circuit(name, args.seconds, args.fps,
                            args.speed_target, args.verbose)
        except Exception as e:
            print(f"  {name:<12}  ERREUR : {e}")
            fails.append(name)
            continue

        # PASS si < 5 % de frames off-track ET distance > 200 px (a avance).
        ok = (r["off_track_frac"] < 0.05 and r["distance_px"] > 200.0)
        verdict = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            fails.append(name)
        print(f"  {name:<12}  {r['off_track_frac']*100:8.1f}%  "
              f"{r['distance_px']:8.0f}  {r['avg_speed_px_s']:6.1f}  {verdict}")

    print(f"\n[test_pid] {n_pass}/{len(candidates)} PASS")
    if fails:
        print(f"[test_pid] FAIL: {', '.join(fails)}")


if __name__ == "__main__":
    main()
