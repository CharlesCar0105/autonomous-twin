"""
gen_signs.py -- Dessine les 4 sprites panneaux (masters 256x256 RGBA).

STOP  : octogone rouge vif + texte STOP blanc.
30/50/90 : disque blanc, anneau rouge vif, chiffres noirs (style FR).

Rouge choisi >= 200 pour rester discriminable du mur brique de Charles
(COL_BRICK = (150,45,35), lisere (180,70,55)) par le seuil detecteur r>=190.

Usage : python scripts/gen_signs.py   # ecrit dans assets/signs/
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "signs"

SIZE = 256                     # master haute resolution (downscale a la pose)
RED = (220, 30, 30, 255)       # rouge vif panneau (r >= 200, cf spec)
WHITE = (255, 255, 255, 255)
BLACK = (10, 10, 10, 255)
RING_WIDTH = 30                # epaisseur anneau des limites de vitesse


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Police bold si dispo (Windows), sinon fallback PIL."""
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _center_text(draw: ImageDraw.ImageDraw, text: str, font, fill) -> None:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((SIZE - (r - l)) / 2 - l, (SIZE - (b - t)) / 2 - t),
              text, font=font, fill=fill)


def draw_stop() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = SIZE / 2, SIZE / 2 - 4
    pts = [(c + r * math.cos(math.radians(22.5 + 45 * i)),
            c + r * math.sin(math.radians(22.5 + 45 * i))) for i in range(8)]
    d.polygon(pts, fill=RED, outline=WHITE, width=10)
    _center_text(d, "STOP", _font(78), WHITE)
    return img


def draw_limit(number: str) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, SIZE - 4, SIZE - 4], fill=WHITE)
    d.ellipse([4, 4, SIZE - 4, SIZE - 4], outline=RED, width=RING_WIDTH)
    _center_text(d, number, _font(110), BLACK)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sprites = {"stop": draw_stop(), "30": draw_limit("30"),
               "50": draw_limit("50"), "90": draw_limit("90")}
    for name, img in sprites.items():
        path = OUT_DIR / f"{name}.png"
        img.save(path)
        print(f"[gen_signs] {path.name}  {img.size}  OK")


if __name__ == "__main__":
    main()
