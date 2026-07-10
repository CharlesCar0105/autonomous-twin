"""
place_signs.py -- Genere les sidecars <circuit>.signs.json par dry-run PID.

Principe : le PID n'utilise QUE le lidar, sa trajectoire est donc identique
avec ou sans panneaux -> on le fait rouler headless, on echantillonne des
points le long de la trajectoire (espacement arclength >= 300 px : le crop
camera fait 128 px, jamais 2 panneaux dans la meme vue), et on pose chaque
panneau perpendiculairement, au premier offset (55/70/85 px) dont le segment
vers le point trajectoire croise une bordure noire -> panneau garanti HORS
piste (l'herbe et la route sont toutes deux blanches, seule la bordure les
separe).

Un seul chemin de code pour tous les circuits (gen_* et historiques).
Deterministe : PID + spawn fixes ; le rng (ordre des candidats + type de
panneau) est seede par NOM de circuit (crc32) -> regenerer un circuit seul
ou en lot produit le meme sidecar.

Usage :
    python scripts/place_signs.py --pattern "gen_*"                  # les 30
    python scripts/place_signs.py --tracks circuit_01 circuit_02
"""

import argparse
import json
import math
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.track import Track
from simulator.car import Car
from simulator import physics, sensors
from simulator.signs import SIGN_KINDS, SIGN_SIZE_DEFAULT
from pilot.control import pid_policy

SPACING_PX = 300.0          # arclength mini entre 2 panneaux (ligne de conduite)
MIN_DIST_PX = 200.0         # >= 200 px entre centres : garantit <=1 panneau
                            # SIGNIFICATIF par vue camera (les slivers de coin
                            # residuels sont rejetes par le filtre densite du
                            # detecteur).
                            # L'arclength seul ne suffit pas : un circuit qui
                            # boucle sur lui-meme (epingle, multi-tours) peut
                            # rapprocher physiquement deux points eloignes en
                            # arclength cumule.
# Offset lateral UNIQUE : 46 px = demi-crop camera (64) - demi-panneau (18) :
# panneau entierement visible au passage (le detecteur tolere ~4 px de
# rognage residuel, chiffres centres non touches).
# Au-dela, le panneau depasse du champ camera au passage -> le premier
# chiffre (discriminant 30/50/90) peut etre rogne -> misclassification
# STRUCTURELLE (un "x0" ampute est reellement ambigu ; constat 10/07 :
# un '30' lu 50/90 a conf 0.94-1.00). Les anciens offsets 55/70/85
# placaient des panneaux partiellement hors champ. Un candidat trajectoire
# dont l'offset 46 ne croise pas de bordure est simplement rejete (le scan
# trouve d'autres points ; demi-largeur piste max ~41 px < 46).
OFFSETS_PX = (46.0,)
SIGN_HALF = SIGN_SIZE_DEFAULT / 2.0            # demi-taille panneau (36 px)


def _record_trajectory(track_name: str, seconds: float, fps: int = 60) -> tuple[list[dict], Track]:
    """Roule le PID headless.

    Retourne (traj, track) : traj = [{x, y, angle, arclen}] par frame,
    track = l'instance Track utilisee pour l'enregistrement (reutilisee par
    l'appelant pour la verification hors-piste, pas besoin de la recharger).
    """
    pygame.init()
    pygame.display.set_mode((1, 1))
    track = Track(track_name, 1280, 720)
    car = Car(track.start_x, track.start_y, track.start_angle)
    dt = 1.0 / fps
    traj, arclen, px, py = [], 0.0, car.x, car.y
    for _ in range(int(seconds * fps)):
        lidar = sensors.get_lidar(track, car)
        steering, throttle, brake = pid_policy(lidar, physics.speed_kmh(car),
                                               speed_target=40.0)
        car.set_controls(steering, throttle, brake)
        physics.update(car, dt)
        arclen += math.hypot(car.x - px, car.y - py)
        px, py = car.x, car.y
        traj.append({"x": car.x, "y": car.y, "angle": car.angle,
                     "arclen": arclen})
    pygame.quit()
    return traj, track


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


def _place_for_track(track_name: str, per_track: int, seconds: float,
                     rng: np.random.Generator) -> list[dict]:
    traj, track = _record_trajectory(track_name, seconds)
    if not traj:
        return []
    signs, used_arclen = [], []
    # Candidats : frames triees pour couvrir le tour, tirage rng sur l'ordre.
    order = rng.permutation(len(traj))
    for idx in order:
        if len(signs) >= per_track:
            break
        f = traj[idx]
        if f["arclen"] < 150.0:          # pas de panneau sur la ligne de depart
            continue
        if any(abs(f["arclen"] - u) < SPACING_PX for u in used_arclen):
            continue
        perp = (-math.sin(f["angle"]), math.cos(f["angle"]))
        for off in OFFSETS_PX:
            sx = f["x"] + perp[0] * off
            sy = f["y"] + perp[1] * off
            in_frame = (SIGN_HALF < sx < track.width - SIGN_HALF
                        and SIGN_HALF < sy < track.height - SIGN_HALF)
            too_close = any(math.hypot(sx - s["x"], sy - s["y"]) < MIN_DIST_PX
                            for s in signs)
            if (in_frame and not too_close
                    and _crosses_border(track, f["x"], f["y"], sx, sy)):
                kind = str(rng.choice(SIGN_KINDS))
                signs.append({"x": round(sx, 1), "y": round(sy, 1),
                              "kind": kind, "arclen": round(f["arclen"], 1)})
                used_arclen.append(f["arclen"])
                break
    for s in signs:
        s.pop("arclen")
    return signs


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere les sidecars panneaux")
    parser.add_argument("--tracks", nargs="*", default=None)
    parser.add_argument("--pattern", default=None, help='ex: "gen_*"')
    parser.add_argument("--per-track", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tracks_dir = ROOT / "assets" / "tracks"
    if args.tracks:
        names = args.tracks
    elif args.pattern:
        names = sorted(p.stem for p in tracks_dir.glob(f"{args.pattern}.png"))
    else:
        parser.error("--tracks ou --pattern requis")

    for i, name in enumerate(names):
        # Seed derive du NOM du circuit (pas de l'index dans la liste) :
        # regenerer un circuit seul ou en lot donne le meme sidecar.
        rng = np.random.default_rng(args.seed + zlib.crc32(name.encode()))
        signs = _place_for_track(name, args.per_track, args.seconds, rng)
        out = tracks_dir / f"{name}.signs.json"
        out.write_text(json.dumps({"signs": signs}, indent=2), encoding="utf-8")
        kinds = ", ".join(s["kind"] for s in signs) or "AUCUN"
        print(f"  [{i+1:2d}/{len(names)}] {out.name:28s} {len(signs)} panneaux ({kinds})")


if __name__ == "__main__":
    main()
