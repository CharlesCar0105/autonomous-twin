"""
gen_circuits.py -- Generateur procedural de circuits PNG (v2).

Genere N circuits fermes varies au format attendu par track.py (route
blanche, bordures noires 1-3 px). Chaque PNG est accompagne d'un JSON
avec position / angle / largeur de depart (lus automatiquement par
simulator/track.py).

V2 : les circuits ne sont plus des "patates rondes" uniformes. L'algo
injecte explicitement des FEATURES dans la centerline polaire :
  - epingles (creux profonds et etroits)
  - lignes droites (rayon fige sur un secteur)
  - chicanes (oscillation haute frequence localisee)
  - protuberances (bosses)
Plus des variations globales : decentrage, allongement horizontal /
vertical, echelle, largeur de piste modulee le long du tracé.

Dependencies : numpy + Pillow (deja dans requirements.txt).
Reproductible via --seed.

Usage :
    python scripts/gen_circuits.py --n 30 --seed 100
    python scripts/gen_circuits.py --n 1 --seed 7 --preview

Sortie :
    assets/tracks/{prefix}{id:03d}.png
    assets/tracks/{prefix}{id:03d}.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# --- Constantes ----------------------------------------------------------

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Largeur de piste nominale (moyenne). Une modulation douce +/- 20 %
# est appliquee ensuite le long du tracé pour creer des goulots.
TRACK_WIDTH_MEAN_MIN = 55.0
TRACK_WIDTH_MEAN_MAX = 75.0

# Epaisseur du trait des bordures noires.
BORDER_THICKNESS = 3

# Nombre de points sur la centerline.
N_WAYPOINTS = 600


# --- Helpers bruit -------------------------------------------------------

def _ang_distance(a: np.ndarray, center: float) -> np.ndarray:
    """Distance angulaire (signed) entre chaque angle et `center`,
    wrappee dans [-pi, pi] pour etre correcte sur la boucle fermee."""
    return (a - center + np.pi) % (2.0 * np.pi) - np.pi


def _smooth_noise(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bruit periodique = somme de sinusoides, frequences basses a
    moyennes. Amplitude totale normalisee a +/- 0.35."""
    noise = np.zeros_like(angles)
    for freq in (2, 3, 4, 5, 6, 8, 10):
        amp = rng.uniform(0.05, 0.25)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        noise += amp * np.sin(freq * angles + phase)
    peak = np.abs(noise).max()
    if peak > 0:
        noise = noise / peak * 0.35
    return noise


# --- Features injectees --------------------------------------------------

def _add_pinch(noise: np.ndarray, angles: np.ndarray, rng: np.random.Generator) -> None:
    """Epingle : creux sur un secteur angulaire. Amplitude limitee pour
    eviter que le rayon de courbure local tombe sous la demi-largeur de
    piste (auto-intersection de la bordure interieure)."""
    center = rng.uniform(0.0, 2.0 * np.pi)
    spread = rng.uniform(0.12, 0.22)
    depth = rng.uniform(0.18, 0.30)
    d = _ang_distance(angles, center)
    noise -= depth * np.exp(-(d * d) / (2.0 * spread * spread))


def _add_bump(noise: np.ndarray, angles: np.ndarray, rng: np.random.Generator) -> None:
    """Bosse : rayon qui s'eloigne sur un secteur (partie rapide)."""
    center = rng.uniform(0.0, 2.0 * np.pi)
    spread = rng.uniform(0.18, 0.35)
    amp = rng.uniform(0.10, 0.22)
    d = _ang_distance(angles, center)
    noise += amp * np.exp(-(d * d) / (2.0 * spread * spread))


def _add_chicane(noise: np.ndarray, angles: np.ndarray, rng: np.random.Generator) -> None:
    """Chicane : oscillation gauche-droite sur un secteur. Frequence et
    amplitude moderees pour rester drivable."""
    center = rng.uniform(0.0, 2.0 * np.pi)
    spread = rng.uniform(0.18, 0.28)
    amp = rng.uniform(0.06, 0.12)
    freq = rng.integers(6, 10)
    d = _ang_distance(angles, center)
    modulator = np.exp(-(d * d) / (2.0 * spread * spread))
    noise += amp * np.sin(freq * angles) * modulator


def _force_straight(
    radii: np.ndarray, angles: np.ndarray, rng: np.random.Generator
) -> None:
    """Fige le rayon a une constante sur un secteur angulaire pour creer
    une ligne droite apparente (arc de courbure quasi nulle)."""
    center = rng.uniform(0.0, 2.0 * np.pi)
    spread = rng.uniform(0.20, 0.45)
    d = _ang_distance(angles, center)
    mask = np.abs(d) < spread
    if not mask.any():
        return
    # Interpolation douce aux bords pour eviter un saut brutal.
    blend = np.exp(-(d * d) / (2.0 * (spread * 0.5) ** 2))
    target = np.mean(radii[mask])
    radii[:] = blend * target + (1.0 - blend) * radii


# --- Centerline ----------------------------------------------------------

def _smooth_closed(values: np.ndarray, window: int) -> np.ndarray:
    """Moyenne glissante periodique (circular) pour adoucir les transitions
    abruptes entre features. Preserve la continuite fermeeneeded."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    extended = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(extended, kernel, mode="same")
    return smoothed[window:-window]


def _min_curvature_radius(xs: np.ndarray, ys: np.ndarray) -> float:
    """Rayon de courbure minimum pertinent (5e percentile) le long de la
    centerline fermee. Le 5e percentile plutot que min absolu pour ignorer
    le bruit de discretisation aux points individuels."""
    x = np.concatenate([xs[-3:], xs, xs[:3]])
    y = np.concatenate([ys[-3:], ys, ys[:3]])
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5
    den[den < 1e-6] = 1e-6
    curvature = num / den
    radius = 1.0 / np.maximum(curvature, 1e-6)
    # 5e percentile : ignore les outliers bruiteux de la discretisation.
    return float(np.percentile(radius[3:-3], 5))


def _build_centerline_once(
    width: int, height: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Un seul essai de generation. Peut produire un circuit degenerere
    (self-intersection) -- le caller gere le retry."""
    rng = np.random.default_rng(seed)

    jitter = 0.10 * min(width, height)
    cx = width / 2.0 + rng.uniform(-jitter, jitter)
    cy = height / 2.0 + rng.uniform(-jitter, jitter)

    base_scale = rng.uniform(0.28, 0.40)
    aspect_h = rng.uniform(0.75, 1.30)
    aspect_v = rng.uniform(0.75, 1.30)
    base_r = base_scale * min(width, height)

    angles = np.linspace(0.0, 2.0 * np.pi, N_WAYPOINTS, endpoint=False)
    noise = _smooth_noise(angles, rng)

    n_pinches = int(rng.integers(1, 4))
    n_bumps = int(rng.integers(0, 3))
    n_chicanes = int(rng.integers(0, 2))
    n_straights = int(rng.integers(0, 3))

    for _ in range(n_pinches):
        _add_pinch(noise, angles, rng)
    for _ in range(n_bumps):
        _add_bump(noise, angles, rng)
    for _ in range(n_chicanes):
        _add_chicane(noise, angles, rng)

    radii = base_r * (1.0 + noise)

    for _ in range(n_straights):
        _force_straight(radii, angles, rng)

    min_r = 0.35 * base_r
    # Marge pour la bordure exterieure : on garde une zone tampon
    # egale a la largeur de piste max + 15 px pour etre sur que la
    # bordure ne sort pas du cadre.
    max_track_half = 60.0  # superieur a track_width_mean_max / 2
    margin = max_track_half + 15.0
    max_r_h = ((width / 2.0) - margin) / max(aspect_h, 1e-6)
    max_r_v = ((height / 2.0) - margin) / max(aspect_v, 1e-6)
    radii = np.clip(radii, min_r, min(max_r_h, max_r_v))

    xs = cx + aspect_h * radii * np.cos(angles)
    ys = cy + aspect_v * radii * np.sin(angles)

    # Lissage final pour adoucir les jointures entre features.
    xs = _smooth_closed(xs, window=15)
    ys = _smooth_closed(ys, window=15)

    meta = {
        "center_xy": (float(cx), float(cy)),
        "aspect": (float(aspect_h), float(aspect_v)),
        "base_radius": float(base_r),
        "features": {
            "pinches": n_pinches, "bumps": n_bumps,
            "chicanes": n_chicanes, "straights": n_straights,
        },
    }
    return xs, ys, meta


def _min_self_distance(xs: np.ndarray, ys: np.ndarray, min_angle_sep: int = 40) -> float:
    """Distance min entre 2 points non-adjacents de la centerline.
    Si cette distance est < track_width, la piste est ecrasee entre
    2 virages proches (bordures qui se touchent / se croisent).

    Ne compare pas les paires d'indices voisins (|i-j| < min_angle_sep
    ou wrap equivalent) car c'est trivialement petit pour des points
    consecutifs de la centerline."""
    n = len(xs)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy
    # Masque : ignorer les paires proches sur la boucle (incluant wrap).
    idx = np.arange(n)
    ang_dist = np.abs(idx[:, None] - idx[None, :])
    ang_dist = np.minimum(ang_dist, n - ang_dist)
    d2[ang_dist < min_angle_sep] = np.inf
    return float(np.sqrt(d2.min()))


def _fits_in_frame(
    xs: np.ndarray, ys: np.ndarray,
    track_width_max: float, width: int, height: int, margin: int = 10,
) -> bool:
    """True si tous les points BORDURE (centerline +/- track_width/2 dans
    une marge conservatrice) restent dans le cadre image."""
    half = track_width_max / 2.0 + margin
    return bool(
        (xs.min() - half >= 0) and (xs.max() + half <= width) and
        (ys.min() - half >= 0) and (ys.max() + half <= height)
    )


def _border_self_distance(
    xs: np.ndarray, ys: np.ndarray, widths: np.ndarray, min_angle_sep: int = 40
) -> float:
    """Distance min entre 2 points non-adjacents des bordures interieure
    et exterieure de la piste. Detecte les self-intersections de bordure
    (2 virages si proches que la bordure se replie sur elle-meme).
    Plus fiable que la centerline seule."""
    x = np.concatenate([xs[-1:], xs, xs[:1]])
    y = np.concatenate([ys[-1:], ys, ys[:1]])
    dx = np.gradient(x)[1:-1]
    dy = np.gradient(y)[1:-1]
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    tx, ty = dx / norm, dy / norm
    nx, ny = -ty, tx
    half = widths / 2.0
    ix = xs - nx * half
    iy = ys - ny * half
    ox = xs + nx * half
    oy = ys + ny * half
    return min(_min_self_distance(ix, iy, min_angle_sep),
               _min_self_distance(ox, oy, min_angle_sep))


def _build_centerline(
    width: int, height: int, seed: int, widths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """Genere une centerline valide : in-frame, courbure raisonnable,
    centerline auto-distance OK, bordures auto-distance OK (evite les
    self-intersections de bordure)."""
    track_width_max = float(widths.max())
    min_required_r = track_width_max * 0.30
    min_required_center_d = track_width_max * 1.15
    # La distance min entre 2 points de la MEME bordure doit etre
    # strictement positive (sinon self-intersection). Marge de 5 px.
    min_required_border_d = 5.0

    best = None
    best_score = -np.inf
    for attempt in range(25):
        s = seed + attempt * 100_003
        xs, ys, meta = _build_centerline_once(width, height, s)
        r_min = _min_curvature_radius(xs, ys)
        center_d = _min_self_distance(xs, ys)
        border_d = _border_self_distance(xs, ys, widths)
        in_frame = _fits_in_frame(xs, ys, track_width_max, width, height)
        meta["min_curvature_radius"] = float(r_min)
        meta["min_self_distance"] = float(center_d)
        meta["min_border_distance"] = float(border_d)
        meta["in_frame"] = bool(in_frame)

        # Score pour fallback : bord le plus sensible = border_d.
        score = min(
            r_min / min_required_r,
            center_d / min_required_center_d,
            border_d / max(min_required_border_d, 1e-6),
        )
        if not in_frame:
            score -= 10.0
        if score > best_score:
            best = (xs, ys, meta)
            best_score = score
        if (in_frame and r_min >= min_required_r
                and center_d >= min_required_center_d
                and border_d >= min_required_border_d):
            return xs, ys, meta, attempt
    xs, ys, meta = best
    return xs, ys, meta, 25


def _normals(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normales unitaires orientees vers l'exterieur (rotation +90 de la
    tangente). Wrap aux extremites pour preserver la continuite."""
    dx = np.gradient(np.concatenate([xs, xs[:1]]))[:-1]
    dy = np.gradient(np.concatenate([ys, ys[:1]]))[:-1]
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    tx, ty = dx / norm, dy / norm
    return -ty, tx


def _variable_track_width(n: int, rng: np.random.Generator) -> np.ndarray:
    """Largeur de piste qui module doucement le long du tracé (goulots)."""
    mean = float(rng.uniform(TRACK_WIDTH_MEAN_MIN, TRACK_WIDTH_MEAN_MAX))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    variation = np.zeros(n)
    for freq in (1, 2, 3):
        amp = rng.uniform(0.0, 0.14)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        variation += amp * np.sin(freq * angles + phase)
    widths = mean * (1.0 + variation)
    return np.clip(widths, 35.0, 100.0)


# --- Rendu ---------------------------------------------------------------

def generate_circuit(
    out_png: Path, out_json: Path, seed: int,
    width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
) -> dict:
    rng_width = np.random.default_rng(seed + 10_000)
    widths = _variable_track_width(N_WAYPOINTS, rng_width)
    xs, ys, meta, n_retries = _build_centerline(width, height, seed, widths)
    nx, ny = _normals(xs, ys)

    half = widths / 2.0
    inner_x = xs - nx * half
    inner_y = ys - ny * half
    outer_x = xs + nx * half
    outer_y = ys + ny * half

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def _draw_closed(px: np.ndarray, py: np.ndarray) -> None:
        pts = list(zip(px.astype(int).tolist(), py.astype(int).tolist()))
        pts.append(pts[0])
        draw.line(pts, fill=(0, 0, 0), width=BORDER_THICKNESS)

    _draw_closed(inner_x, inner_y)
    _draw_closed(outer_x, outer_y)

    # Position de depart : premier point de la centerline, tangent a la
    # courbe (sens positif = angle polaire qui augmente).
    start_idx = 0
    sx = float(xs[start_idx])
    sy = float(ys[start_idx])
    tangent_x = -ny[start_idx]
    tangent_y = nx[start_idx]
    start_angle_deg = float(np.degrees(np.arctan2(tangent_y, tangent_x)))

    img.save(out_png)
    data = {
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "track_width_mean_px": float(np.mean(widths)),
        "track_width_range_px": [float(widths.min()), float(widths.max())],
        "start_x": sx,
        "start_y": sy,
        "start_angle_deg": start_angle_deg,
        "features": meta["features"],
        "min_curvature_radius_px": float(meta.get("min_curvature_radius", 0.0)),
        "min_self_distance_px": float(meta.get("min_self_distance", 0.0)),
        "min_border_distance_px": float(meta.get("min_border_distance", 0.0)),
        "in_frame": bool(meta.get("in_frame", True)),
        "retries": int(n_retries),
    }
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# --- CLI -----------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here.parent / "assets" / "tracks"

    parser = argparse.ArgumentParser(description="Generateur procedural v2")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="gen_")
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_circuits v2] {args.n} circuits -> {out_dir}")
    for i in range(args.n):
        seed = args.seed + i
        png = out_dir / f"{args.prefix}{i:03d}.png"
        js = out_dir / f"{args.prefix}{i:03d}.json"
        data = generate_circuit(png, js, seed, args.width, args.height)
        feats = data["features"]
        print(
            f"  [{i+1:2d}/{args.n}] {png.name}  "
            f"seed={seed:<4d}  "
            f"w={data['track_width_mean_px']:4.1f}  "
            f"r={data['min_curvature_radius_px']:3.0f}  "
            f"cd={data['min_self_distance_px']:5.1f}  "
            f"bd={data['min_border_distance_px']:5.1f}  "
            f"frame={'Y' if data['in_frame'] else 'N'}  "
            f"retr={data['retries']:2d}  "
            f"p{feats['pinches']}b{feats['bumps']}c{feats['chicanes']}s{feats['straights']}"
        )

    if args.preview and args.n > 0:
        first = out_dir / f"{args.prefix}000.png"
        try:
            os.startfile(first)
        except AttributeError:
            print(f"[gen_circuits] preview : ouvre manuellement {first}")


if __name__ == "__main__":
    main()
