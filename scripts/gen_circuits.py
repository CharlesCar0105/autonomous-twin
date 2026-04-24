"""
gen_circuits.py -- Generateur procedural de circuits PNG.

Produit N circuits de course au format attendu par track.py (route/herbe
blanche, bordures noires 1-3 px) a partir d'une centerline polaire
perturbee par un bruit smooth multi-frequence. Chaque circuit est
accompagne d'un JSON contenant la position/angle de depart calcules
au moment de la generation (plus besoin de cliquer manuellement).

Utilise uniquement numpy + Pillow (deja dans requirements.txt).
Reproductible via --seed.

Usage :
    python scripts/gen_circuits.py --n 50 --prefix gen_
    python scripts/gen_circuits.py --n 5 --seed 42 --preview    # ouvre un PNG

Sortie :
    assets/tracks/{prefix}{id:03d}.png
    assets/tracks/{prefix}{id:03d}.json   {start_x, start_y, start_angle_deg}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# --- Constantes ----------------------------------------------------------

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Largeur de piste : cible ~60 px (dans la fenetre [30, 100] du filtre
# _track_centering_score cote simulator/track.py).
TRACK_WIDTH_MIN = 55
TRACK_WIDTH_MAX = 75

# Epaisseur du trait des bordures noires.
BORDER_THICKNESS = 3

# Nombre de points sur la centerline (plus = plus lisse).
N_WAYPOINTS = 400


# --- Generation de la centerline -----------------------------------------

def _smooth_noise(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bruit periodique smooth = somme de sinusoides a frequences basses.

    Garantit la continuite (le circuit se referme parfaitement, le point a
    theta=0 et theta=2*pi a exactement le meme rayon).
    """
    noise = np.zeros_like(angles)
    for freq in (2, 3, 4, 5, 7):
        amp = rng.uniform(0.06, 0.22)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        noise += amp * np.sin(freq * angles + phase)
    return noise


def _build_centerline(
    width: int, height: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cree une centerline fermee par perturbation radiale d'un cercle.

    Returns:
        (xs, ys) en pixels, N_WAYPOINTS points regulierement espaces en
        angle autour du centre de l'image.
    """
    rng = np.random.default_rng(seed)
    cx, cy = width / 2.0, height / 2.0
    # Rayon de base : couvre environ 70% du plus petit cote pour garder
    # une marge autour du circuit.
    base_r = 0.35 * min(width, height)

    angles = np.linspace(0.0, 2.0 * np.pi, N_WAYPOINTS, endpoint=False)
    noise = _smooth_noise(angles, rng)

    # Perturbation radiale : +/- 25% du rayon de base.
    radii = base_r * (1.0 + 0.25 * noise)
    # Clip pour eviter les radii trop petits (piste qui se replie) ou
    # trop grands (qui sort de l'image).
    max_r = 0.48 * min(width, height)
    radii = np.clip(radii, 0.55 * base_r, max_r)

    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return xs, ys


def _normals(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vecteurs normaux unitaires (perpendiculaires a la tangente) le long
    de la centerline. Utilise gradient numpy avec edge wrap pour preserver
    la continuite aux extremites (centerline fermee).
    """
    dx = np.gradient(np.concatenate([xs, xs[:1]]))[:-1]
    dy = np.gradient(np.concatenate([ys, ys[:1]]))[:-1]
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    tx, ty = dx / norm, dy / norm
    # Normale = tangente tournee de 90 deg (orientation vers l'exterieur).
    nx, ny = -ty, tx
    return nx, ny


# --- Dessin du circuit ---------------------------------------------------

def generate_circuit(
    out_png: Path,
    out_json: Path,
    seed: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict:
    """Genere un PNG de circuit + un JSON de metadata. Retourne la metadata."""
    rng = np.random.default_rng(seed)
    xs, ys = _build_centerline(width, height, seed)
    nx, ny = _normals(xs, ys)

    track_w = float(rng.uniform(TRACK_WIDTH_MIN, TRACK_WIDTH_MAX))
    half = track_w / 2.0
    inner_x = xs - nx * half
    inner_y = ys - ny * half
    outer_x = xs + nx * half
    outer_y = ys + ny * half

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def _draw_closed(points_x, points_y):
        pts = list(zip(points_x.astype(int).tolist(), points_y.astype(int).tolist()))
        pts.append(pts[0])
        draw.line(pts, fill=(0, 0, 0), width=BORDER_THICKNESS)

    _draw_closed(inner_x, inner_y)
    _draw_closed(outer_x, outer_y)

    # Position de depart : premier waypoint (angle 0 autour du centre).
    # Angle de cap = direction tangente au circuit en ce point, convertie
    # en degres et cote pygame (y vers le bas -> sens horaire = positif).
    start_idx = 0
    sx = float(xs[start_idx])
    sy = float(ys[start_idx])
    tangent_x = -ny[start_idx]  # tangente = rotation -90 de la normale
    tangent_y = nx[start_idx]
    start_angle_deg = float(np.degrees(np.arctan2(tangent_y, tangent_x)))

    img.save(out_png)
    meta = {
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "track_width_px": float(track_w),
        "start_x": sx,
        "start_y": sy,
        "start_angle_deg": start_angle_deg,
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --- CLI -----------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here.parent / "assets" / "tracks"

    parser = argparse.ArgumentParser(description="Generateur de circuits PNG procedural")
    parser.add_argument("--n", type=int, default=30,
                        help="Nombre de circuits a generer (defaut 30).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed de base. Le circuit i utilise seed+i.")
    parser.add_argument("--prefix", default="gen_",
                        help="Prefixe des noms de fichiers (defaut 'gen_').")
    parser.add_argument("--out", default=str(default_out),
                        help="Dossier de sortie.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--preview", action="store_true",
                        help="Ouvre le premier PNG genere a la fin.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_circuits] {args.n} circuits -> {out_dir}")
    for i in range(args.n):
        seed = args.seed + i
        png = out_dir / f"{args.prefix}{i:03d}.png"
        js = out_dir / f"{args.prefix}{i:03d}.json"
        meta = generate_circuit(png, js, seed, args.width, args.height)
        print(f"  [{i+1}/{args.n}] {png.name}  seed={seed}  "
              f"start=({meta['start_x']:.0f}, {meta['start_y']:.0f})  "
              f"angle={meta['start_angle_deg']:+.1f}deg  "
              f"width={meta['track_width_px']:.0f}px")

    if args.preview and args.n > 0:
        first = out_dir / f"{args.prefix}000.png"
        try:
            os.startfile(first)  # Windows
        except AttributeError:
            print(f"[gen_circuits] preview : ouvre manuellement {first}")


if __name__ == "__main__":
    main()
