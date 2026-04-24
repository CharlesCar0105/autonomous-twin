"""
record_dataset.py -- Collecte dataset (image, masque, lidar, commandes)
pour l'entrainement U-Net + CNN conduite.

Tourne le PID baseline headless sur un set de circuits, dump 1 frame
sur N dans data/{split}/{circuit}/{frame}.npz. Chaque .npz contient :
    camera  (128, 128, 3) uint8   -- vue devant la voiture (depuis track.pixels)
    mask    (128, 128)    uint8   -- ground-truth route binaire
    lidar   (5,)          float32 -- distances -60/-30/0/30/60 deg
    speed   ()            float32 -- km/h virtuels
    steering () float32 throttle () float32 brake () float32

Split par circuit (D10 : generalisation) : les circuits val ne sont
jamais vus par le modele en training.

Usage par defaut :
    python scripts/record_dataset.py   # 25 train, 5 val, 30s par circuit

Override possible :
    python scripts/record_dataset.py --seconds 60 --frame-stride 3
    python scripts/record_dataset.py --val-count 0          # dataset complet pour test
    python scripts/record_dataset.py --tracks gen_000 gen_001  # circuits specifiques
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.track import Track
from simulator.car import Car
from simulator import physics, sensors
from pilot.control import pid_policy


def record_track(
    track_name: str, out_dir: Path, seconds: float, fps: int,
    speed_target: float, frame_stride: int,
) -> dict:
    """Enregistre une session PID headless sur un circuit. Retourne
    un dict de stats (frames gardees, offtrack, etc.)."""
    pygame.init()
    pygame.display.set_mode((1, 1))
    track = Track(track_name, 1280, 720)
    car = Car(track.start_x, track.start_y, track.start_angle)

    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / fps
    n_frames = int(seconds * fps)

    saved = 0
    off_track = 0
    total_dist = 0.0
    prev_x, prev_y = car.x, car.y

    for f in range(n_frames):
        lidar = sensors.get_lidar(track, car)
        camera = sensors.get_camera_view_from_track(track, car)
        mask = sensors.get_ground_truth_mask(track, car)
        speed_kmh = physics.speed_kmh(car)

        steering, throttle, brake = pid_policy(lidar, speed_kmh, speed_target=speed_target)
        car.set_controls(steering, throttle, brake)

        if f % frame_stride == 0:
            path = out_dir / f"{f:05d}.npz"
            np.savez_compressed(
                path,
                camera=camera,
                mask=mask,
                lidar=np.array(lidar, dtype=np.float32),
                speed=np.float32(speed_kmh),
                steering=np.float32(steering),
                throttle=np.float32(throttle),
                brake=np.float32(brake),
            )
            saved += 1

        physics.update(car, dt)
        total_dist += math.hypot(car.x - prev_x, car.y - prev_y)
        prev_x, prev_y = car.x, car.y
        if not track.is_on_road(int(car.x), int(car.y)):
            off_track += 1

    pygame.quit()
    return {
        "track": track_name, "frames": n_frames, "saved": saved,
        "off_track": off_track, "distance_px": total_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collecte dataset U-Net / CNN")
    parser.add_argument("--pattern", default="gen_*",
                        help="Glob des circuits (defaut: gen_*)")
    parser.add_argument("--tracks", nargs="+", default=None,
                        help="Liste explicite de circuits (prioritaire sur --pattern)")
    parser.add_argument("--val-count", type=int, default=5,
                        help="Nombre de circuits reserves pour val (defaut 5)")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=3,
                        help="1 frame sur N sauvee (defaut 3 = 20 frames/s)")
    parser.add_argument("--speed-target", type=float, default=35.0)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    tracks_dir = ROOT / "assets" / "tracks"
    if args.tracks:
        candidates = args.tracks
    else:
        candidates = sorted(p.stem for p in tracks_dir.glob(f"{args.pattern}.png"))
    if not candidates:
        print(f"[record_dataset] aucun circuit trouve")
        return

    # Split train/val : les N derniers (seed plus eleve) -> val.
    if args.val_count > 0 and len(candidates) > args.val_count:
        val_tracks = candidates[-args.val_count:]
        train_tracks = candidates[:-args.val_count]
    else:
        val_tracks = []
        train_tracks = candidates

    data_dir = Path(args.data_dir)
    print(f"[record_dataset] {len(train_tracks)} train + {len(val_tracks)} val, "
          f"{args.seconds}s par circuit, stride {args.frame_stride}")
    print(f"[record_dataset] sortie : {data_dir}")

    all_stats = []
    for split, tracks in [("train", train_tracks), ("val", val_tracks)]:
        for t in tracks:
            out = data_dir / split / t
            r = record_track(
                t, out, args.seconds, args.fps,
                args.speed_target, args.frame_stride,
            )
            r["split"] = split
            all_stats.append(r)
            print(f"  {split}/{t:<10}  saved={r['saved']:3d}  "
                  f"dist={r['distance_px']:5.0f}  offtrack={r['off_track']}")

    total_saved = sum(r["saved"] for r in all_stats)
    total_train = sum(r["saved"] for r in all_stats if r["split"] == "train")
    total_val = sum(r["saved"] for r in all_stats if r["split"] == "val")
    print(f"\n[record_dataset] total {total_saved} frames "
          f"({total_train} train, {total_val} val)")


if __name__ == "__main__":
    main()
