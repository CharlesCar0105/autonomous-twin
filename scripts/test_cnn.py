"""
test_cnn.py -- Valide le CNN conduite en boucle fermee (headless).

Charge models/cnn_drive.pth et roule sur chaque circuit gen_*.png N
secondes, metrique : fraction de frames off-track et distance parcourue.

Verdict PASS si off-track < 5 % et distance > 200 px.
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
from simulator.signs import load_signs
from pilot.control import cnn_policy
from pilot.perception import compute_mask


def run_circuit(name: str, seconds: float, fps: int, weights: str,
                 with_signs: bool = False) -> dict:
    pygame.init()
    pygame.display.set_mode((1, 1))
    track = Track(name, 1280, 720)
    car = Car(track.start_x, track.start_y, track.start_angle)
    signs = load_signs(name) if with_signs else []

    dt = 1.0 / fps
    n_frames = int(seconds * fps)
    off_track = 0
    total_dist = 0.0
    prev_x, prev_y = car.x, car.y

    for _ in range(n_frames):
        lidar = sensors.get_lidar(track, car)
        camera = sensors.get_camera_view_from_track(track, car, signs=signs)
        mask = compute_mask(camera, size=64)
        speed = physics.speed_kmh(car)
        steering, throttle, brake = cnn_policy(lidar, speed, mask, weights)
        car.set_controls(steering, throttle, brake)
        physics.update(car, dt)
        total_dist += math.hypot(car.x - prev_x, car.y - prev_y)
        prev_x, prev_y = car.x, car.y
        if not track.is_on_road(int(car.x), int(car.y)):
            off_track += 1

    pygame.quit()
    return {
        "track": name, "frames": n_frames,
        "off_track_frac": off_track / n_frames,
        "distance_px": total_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="gen_*")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--weights", default=str(ROOT / "models" / "cnn_drive.pth"))
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--with-signs", action="store_true",
                        help="Composite les panneaux du sidecar dans la camera (non-regression).")
    args = parser.parse_args()

    tracks_dir = ROOT / "assets" / "tracks"
    candidates = sorted(p.stem for p in tracks_dir.glob(f"{args.pattern}.png"))
    if args.max:
        candidates = candidates[: args.max]
    if not candidates:
        print("[test_cnn] aucun circuit"); return

    print(f"[test_cnn] {len(candidates)} circuits x {args.seconds}s  weights={args.weights}")
    print(f"  {'track':<12}  {'offtrack%':>9}  {'dist_px':>8}  verdict")
    print(f"  {'-'*12}  {'-'*9}  {'-'*8}  -------")

    n_pass = 0; fails = []
    for name in candidates:
        try:
            r = run_circuit(name, args.seconds, args.fps, args.weights, args.with_signs)
        except Exception as e:
            print(f"  {name:<12}  ERREUR : {e}"); fails.append(name); continue
        ok = r["off_track_frac"] < 0.05 and r["distance_px"] > 200.0
        if ok:
            n_pass += 1
        else:
            fails.append(name)
        print(f"  {name:<12}  {r['off_track_frac']*100:8.1f}%  "
              f"{r['distance_px']:8.0f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n[test_cnn] {n_pass}/{len(candidates)} PASS")
    if fails:
        print(f"FAIL : {', '.join(fails)}")


if __name__ == "__main__":
    main()
