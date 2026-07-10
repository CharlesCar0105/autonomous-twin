"""
bench_laps.py -- Banc de mesure des temps de tour (phase optimisation).

Branche l'eval boucle fermee sur le LapTimer de simulator/timing.py
(chrono 3 tours, detection de franchissement de ligne) -- resout la
question Q7 du wiki : on mesure enfin de VRAIS tours, pas une distance
odometre.

Chaque run = 1 circuit x 1 config, headless, dt fixe 1/60 (deterministe,
independant du CPU : les temps sont en secondes SIMULEES). On collecte,
en plus des temps au tour, les compteurs qui expliquent ou part le temps :
frames de freinage d'urgence, frames quasi-arret, arrets STOP, offtrack.

Configs de base :
    regles : PID + panneaux + freinage d'urgence  (conditions CDC completes)
    libre  : PID seul, sans panneaux ni urgence   (potentiel chassis/PID)

Usage :
    python scripts/bench_laps.py                                   # baseline 3 circuits
    python scripts/bench_laps.py --tracks gen_000 gen_014 --laps 3
    python scripts/bench_laps.py --configs regles --out bench/xxx.json

Sortie : tableau console + JSON date dans bench/ (committe : c'est la
trace de progression pour la soutenance).
"""

import argparse
import json
import os
import sys
import time as wallclock
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
from simulator.timing import LapTimer
from pilot.control import pid_policy, cnn_policy
from pilot.perception import compute_mask
from pilot.emergency import check_emergency_brake
from pilot.signs import SignTracker

FPS = 60
DT = 1.0 / FPS
STOPPED_KMH = 2.0            # seuil "voiture quasi arretee" pour les stats

CONFIGS = {
    #            policy   signs  emergency
    "regles":   ("pid",   True,  True),    # conditions CDC completes
    "libre":    ("pid",   False, False),   # potentiel chassis/PID pur
    "cnn":      ("cnn",   False, False),   # CNN conduite pur (lent : inference/frame)
    "cnn-regles": ("cnn", True,  True),    # CNN + regles completes
}


def run_bench(circuit: str, config: str, laps_target: int,
              timeout_s: float, speed_target: float) -> dict:
    """Un run complet. Retourne le dict de stats (temps en s simulees)."""
    policy, with_signs, with_emergency = CONFIGS[config]

    pygame.init()
    pygame.display.set_mode((1, 1))
    track = Track(circuit, 1280, 720)
    car = Car(track.start_x, track.start_y, track.start_angle)
    signs = load_signs(circuit) if with_signs else []
    tracker = SignTracker() if with_signs else None

    lap_timer = LapTimer(track.start_x, track.start_y, track.start_angle, circuit)
    lap_timer.laps_target = laps_target  # attribut informel, LAPS_TARGET reste 3
    lap_timer._save_best_lap = lambda: None  # ne pas polluer best_laps.json

    n_frames_max = int(timeout_s * FPS)
    frames = 0
    speed_sum = 0.0
    frames_stopped = 0
    frames_emergency = 0
    frames_offtrack = 0
    n_stops = 0
    prev_stop_active = False
    limit_history = []
    prev_x, prev_y = car.x, car.y

    for f in range(n_frames_max):
        now = f * DT
        lidar = sensors.get_lidar(track, car)
        speed = physics.speed_kmh(car)

        if policy == "cnn":
            camera = sensors.get_camera_view_from_track(track, car, signs=signs)
            mask = compute_mask(camera, size=64)
            steering, throttle, brake = cnn_policy(lidar, speed, mask)
        else:
            steering, throttle, brake = pid_policy(lidar, speed,
                                                   speed_target=speed_target)
            camera = (sensors.get_camera_view_from_track(track, car, signs=signs)
                      if with_signs else None)

        # copie de pilot/main.py:109-131 (panneaux + emergency) -- garder synchronise
        if tracker is not None:
            tracker.update(camera, speed, now)
            limit = tracker.speed_limit
            if limit is not None and speed > limit:
                throttle = min(throttle, 0.15)
                if speed > limit + 15.0:
                    brake = max(brake, 0.3)
            if tracker.stop_active:
                throttle, brake = 0.0, 1.0
            if tracker.stop_active and not prev_stop_active:
                n_stops += 1
            prev_stop_active = tracker.stop_active
            if not limit_history or limit_history[-1] != limit:
                limit_history.append(limit)
        if with_emergency and speed > 5.0 and check_emergency_brake(lidar, speed):
            throttle, brake = 0.0, 1.0
            frames_emergency += 1

        car.set_controls(steering, throttle, brake)
        prev_pos = (car.x, car.y)
        physics.update(car, DT)
        lap_timer.update(car, prev_pos, now)

        frames += 1
        speed_sum += speed
        if speed < STOPPED_KMH:
            frames_stopped += 1
        if not track.is_on_road(int(car.x), int(car.y)):
            frames_offtrack += 1
        if lap_timer.laps_done >= laps_target:
            break

    pygame.quit()
    sim_time = frames * DT
    return {
        "circuit": circuit,
        "config": config,
        "laps_done": lap_timer.laps_done,
        "laps_target": laps_target,
        "finished": lap_timer.laps_done >= laps_target,
        "lap_times_s": [round(t, 2) for t in lap_timer.lap_times],
        "best_lap_s": round(min(lap_timer.lap_times), 2) if lap_timer.lap_times else None,
        "total_time_s": round(sum(lap_timer.lap_times), 2) if lap_timer.lap_times else None,
        "sim_time_s": round(sim_time, 1),
        "mean_kmh": round(speed_sum / max(frames, 1), 1),
        "pct_stopped": round(100.0 * frames_stopped / max(frames, 1), 1),
        "pct_emergency": round(100.0 * frames_emergency / max(frames, 1), 1),
        "pct_offtrack": round(100.0 * frames_offtrack / max(frames, 1), 1),
        "n_stops_panneau": n_stops,
        "limites_vues": [l for l in limit_history if l is not None],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Banc temps de tour")
    parser.add_argument("--tracks", nargs="*", default=["gen_000", "gen_003", "gen_014"])
    parser.add_argument("--configs", nargs="*", default=["regles", "libre"],
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Timeout par run en secondes SIMULEES.")
    parser.add_argument("--speed-target", type=float, default=80.0)
    parser.add_argument("--out", default=None,
                        help="Chemin JSON de sortie (defaut: bench/bench_<date>.json).")
    parser.add_argument("--label", default="",
                        help="Etiquette de l'etat mesure (ex: 'baseline', 'fix-emergency').")
    args = parser.parse_args()

    results = []
    t0 = wallclock.time()
    print(f"[bench_laps] {len(args.tracks)} circuits x {len(args.configs)} configs, "
          f"{args.laps} tours, timeout {args.timeout:.0f}s sim/run")
    print(f"  {'circuit':<10} {'config':<10} {'tours':>5} {'best':>7} {'total':>8} "
          f"{'v_moy':>6} {'arret%':>6} {'urg%':>5} {'off%':>5}")
    print(f"  {'-'*10} {'-'*10} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")

    for circuit in args.tracks:
        for config in args.configs:
            r = run_bench(circuit, config, args.laps, args.timeout, args.speed_target)
            results.append(r)
            best = f"{r['best_lap_s']:.1f}s" if r["best_lap_s"] else "--"
            total = f"{r['total_time_s']:.1f}s" if r["finished"] else f"DNF({r['laps_done']})"
            print(f"  {circuit:<10} {config:<10} {r['laps_done']}/{args.laps:<3} "
                  f"{best:>7} {total:>8} {r['mean_kmh']:>5.1f} "
                  f"{r['pct_stopped']:>5.1f}% {r['pct_emergency']:>4.1f}% "
                  f"{r['pct_offtrack']:>4.1f}%")

    out = Path(args.out) if args.out else (
        ROOT / "bench" / f"bench_{wallclock.strftime('%Y-%m-%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "date": wallclock.strftime("%Y-%m-%d %H:%M"),
        "speed_target": args.speed_target,
        "laps": args.laps,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[bench_laps] {len(results)} runs en {wallclock.time()-t0:.0f}s reel -> {out}")


if __name__ == "__main__":
    main()
