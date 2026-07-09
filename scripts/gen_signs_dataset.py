"""
gen_signs_dataset.py -- Dataset synthetique classif panneaux (D7).

Fonds : crops camera 128x128 REELS tires des .npz du dataset conduite
(train <- data/train = gen_000..024, val <- data/val = gen_025..029 ; D10 :
les fonds du val viennent de circuits jamais vus par le train).

Positifs : sprite colle sur fond avec augmentations (echelle, rotation +/-20,
luminosite, bruit, flou, troncature partielle 20%) puis crop 96x96 centre
sur le panneau (simule la sortie du detecteur bbox+marge).

"aucun" : moitie fonds nus, moitie crops de MUR brique (memes couleurs que
simulator/wall.py) -> le classifieur apprend a rejeter la brique.

Usage : python scripts/gen_signs_dataset.py   # ~9k images, seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.wall import COL_BRICK, COL_MORTAR, COL_BRICK_HI  # source unique

CLASSES = ("30", "50", "90", "aucun", "stop")
OUT_SIZE = 96
MARGIN = 1.4                     # bbox + 20% de chaque cote (facteur 1.4)


def _load_backgrounds(split_dir: Path, rng: np.random.Generator, k: int) -> list[np.ndarray]:
    files = sorted(split_dir.glob("*/*.npz"))
    assert files, f"aucun npz dans {split_dir} -- lancer record_dataset.py d'abord"
    picks = rng.choice(len(files), size=k, replace=True)
    return [np.load(files[i])["camera"] for i in picks]


def _augment_sprite(kind: str, rng: np.random.Generator) -> Image.Image:
    img = Image.open(ROOT / "assets" / "signs" / f"{kind}.png").convert("RGBA")
    size = int(rng.uniform(25, 50))                       # 36 px nominal en camera
    img = img.resize((size, size), Image.LANCZOS)
    img = img.rotate(float(rng.uniform(-20, 20)), expand=True,
                     resample=Image.BICUBIC)
    return img


def _paste_and_crop(bg: np.ndarray, sprite: Image.Image,
                    rng: np.random.Generator, truncate: bool) -> Image.Image:
    canvas = Image.fromarray(bg).convert("RGBA")
    sw, sh = sprite.size
    if truncate:   # panneau partiellement hors champ (bord du crop camera)
        # Bord droit : plage de cx miroir de la plage bord gauche (cx+sw doit
        # deborder de 128 comme cx deborde de 0 a gauche). La forme naive
        # "128 - sw*0.7 + sw" echantillonne (cx+sw) et l'assigne a cx sans
        # retrancher sw -> peut placer tout le sprite hors canvas (cx > 128)
        # et faire planter le crop (right < left) a l'echelle de ~9000 tirages.
        cx = int(rng.choice([rng.uniform(-sw * 0.3, sw * 0.2),
                             rng.uniform(128 - sw * 1.2, 128 - sw * 0.7)]))
        cy = int(rng.uniform(0, 128 - sh))
    else:
        cx = int(rng.uniform(0, max(1, 128 - sw)))
        cy = int(rng.uniform(0, max(1, 128 - sh)))
    canvas.alpha_composite(sprite, (cx, cy))
    canvas = canvas.convert("RGB")

    # Crop carre centre sur le panneau, bbox * MARGIN, clamp aux bords.
    ccx, ccy = cx + sw / 2, cy + sh / 2
    half = max(sw, sh) * MARGIN / 2
    x1, y1 = int(max(0, ccx - half)), int(max(0, ccy - half))
    x2, y2 = int(min(128, ccx + half)), int(min(128, ccy + half))
    crop = canvas.crop((x1, y1, x2, y2)).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)

    # Photometrie : luminosite, flou leger, bruit gaussien.
    crop = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.5:
        crop = crop.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
    arr = np.asarray(crop).astype(np.int16)
    arr += rng.normal(0.0, rng.uniform(0, 8), arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _wall_patch(bg: np.ndarray, rng: np.random.Generator) -> Image.Image:
    """Faux positif potentiel : morceau de mur brique sur fond reel."""
    canvas = Image.fromarray(bg).convert("RGB")
    d = ImageDraw.Draw(canvas)
    w = int(rng.uniform(40, 110)); h = int(rng.uniform(16, 34))
    x = int(rng.uniform(0, 128 - w)); y = int(rng.uniform(0, 128 - h))
    d.rectangle([x, y, x + w, y + h], fill=COL_BRICK, outline=COL_BRICK_HI, width=2)
    for i in range(1, 5):
        yy = y + i * h // 5
        d.line([x, yy, x + w, yy], fill=COL_MORTAR, width=2)
    half = int(max(w, h) * MARGIN / 2); ccx, ccy = x + w // 2, y + h // 2
    x1, y1 = max(0, ccx - half), max(0, ccy - half)
    x2, y2 = min(128, ccx + half), min(128, ccy + half)
    return canvas.crop((x1, y1, x2, y2)).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)


def _empty_patch(bg: np.ndarray, rng: np.random.Generator) -> Image.Image:
    s = int(rng.uniform(30, 70))
    x = int(rng.uniform(0, 128 - s)); y = int(rng.uniform(0, 128 - s))
    return Image.fromarray(bg).convert("RGB").crop((x, y, x + s, y + s)).resize(
        (OUT_SIZE, OUT_SIZE), Image.BILINEAR)


def generate_split(split: str, n_per_class: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bg_dir = ROOT / "data" / split
    out_root = ROOT / "data" / "signs" / split
    backgrounds = _load_backgrounds(bg_dir, rng, k=n_per_class * len(CLASSES))
    bg_iter = iter(backgrounds)

    for cls in CLASSES:
        out = out_root / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            bg = next(bg_iter)
            if cls == "aucun":
                img = _wall_patch(bg, rng) if i % 2 == 0 else _empty_patch(bg, rng)
            else:
                sprite = _augment_sprite(cls, rng)
                img = _paste_and_crop(bg, sprite, rng, truncate=(rng.random() < 0.2))
            img.save(out / f"{i:05d}.png")
        print(f"  [{split}] {cls:6s} : {n_per_class} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--val-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_split("train", args.train_per_class, args.seed)
    generate_split("val", args.val_per_class, args.seed + 1)
    print("[gen_signs_dataset] termine.")


if __name__ == "__main__":
    main()
