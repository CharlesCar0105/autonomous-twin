"""
signs.py -- Panneaux routiers statiques (CDC : "images statiques (PNG) sur
les bords de la piste").

Pattern Wall (une source de verite, track.pixels jamais modifie), en plus
simple : un panneau n'a AUCUN role physique (pas de collision, pas de lidar
-- il est hors piste, au-dela de la bordure ou les rayons s'arretent deja).
Il n'existe que pour la camera (paste_into_camera) et l'ecran (draw).

Placement : sidecar assets/tracks/<circuit>.signs.json (genere par
scripts/place_signs.py). Le pilote ne lit JAMAIS ces fichiers (anti-triche) :
ils ne servent qu'au simulateur et aux harness d'eval.
"""

import json
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SIGNS_DIR = ROOT / "assets" / "signs"
TRACKS_DIR = ROOT / "assets" / "tracks"

SIGN_KINDS = ("stop", "30", "50", "90")
SIGN_SIZE_DEFAULT = 36   # diametre en px monde (crop camera = 128 px)

_SPRITE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _load_sprite(kind: str, size: int) -> np.ndarray:
    """Sprite (W, H, 4) uint8, axe 0 = x monde (layout track.pixels)."""
    key = (kind, size)
    if key not in _SPRITE_CACHE:
        img = Image.open(SIGNS_DIR / f"{kind}.png").convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img)                      # (H, W, 4), axe 0 = y
        _SPRITE_CACHE[key] = np.transpose(arr, (1, 0, 2)).copy()
    return _SPRITE_CACHE[key]


class RoadSign:
    """Panneau statique pose en (x, y) monde (centre du sprite)."""

    def __init__(self, x: float, y: float, kind: str,
                 size: int = SIGN_SIZE_DEFAULT) -> None:
        if kind not in SIGN_KINDS:
            raise ValueError(f"kind inconnu : {kind!r} (attendu {SIGN_KINDS})")
        self.x = float(x)
        self.y = float(y)
        self.kind = kind
        self.size = int(size)
        self._sprite = _load_sprite(kind, self.size)   # (W, H, 4)
        self._surf: pygame.Surface | None = None       # lazy (display requis)

    # --- Camera (vue pilote) ----------------------------------------------

    def paste_into_camera(self, crop: np.ndarray, x0: int, y0: int) -> None:
        """Colle le sprite (alpha) dans un crop (W, H, 3) NON transpose,
        region monde [x0, x0+W) x [y0, y0+H). Meme convention d'indexation
        que Wall.mask_for_region. In-place, no-op si hors region."""
        s = self.size
        left = int(round(self.x - s / 2.0))
        top = int(round(self.y - s / 2.0))
        w, h = crop.shape[0], crop.shape[1]

        ox1, oy1 = max(left, x0), max(top, y0)
        ox2, oy2 = min(left + s, x0 + w), min(top + s, y0 + h)
        if ox1 >= ox2 or oy1 >= oy2:
            return

        sub = self._sprite[ox1 - left:ox2 - left, oy1 - top:oy2 - top]
        alpha = sub[..., 3:4].astype(np.float32) / 255.0
        region = crop[ox1 - x0:ox2 - x0, oy1 - y0:oy2 - y0].astype(np.float32)
        blended = alpha * sub[..., :3].astype(np.float32) + (1.0 - alpha) * region
        crop[ox1 - x0:ox2 - x0, oy1 - y0:oy2 - y0] = blended.astype(np.uint8)

    # --- Ecran (rendu simulateur) ------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        if self._surf is None:
            arr = np.transpose(self._sprite, (1, 0, 2))  # (H, W, 4)
            self._surf = pygame.image.frombytes(
                arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGBA"
            ).convert_alpha()
        screen.blit(self._surf,
                    (int(self.x - self.size / 2), int(self.y - self.size / 2)))


def load_signs(circuit_name: str) -> list[RoadSign]:
    """Charge le sidecar du circuit. Fichier absent = pas de panneaux."""
    path = TRACKS_DIR / f"{circuit_name}.signs.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RoadSign(s["x"], s["y"], s["kind"]) for s in data.get("signs", [])]


if __name__ == "__main__":
    # Self-check : collage alpha correct, coins transparents preserves.
    crop = np.full((128, 128, 3), 255, dtype=np.uint8)      # fond blanc
    sign = RoadSign(600.0 + 64, 400.0 + 64, "stop")          # centre du crop
    sign.paste_into_camera(crop, 600, 400)
    r, g, b = crop[..., 0].astype(int), crop[..., 1].astype(int), crop[..., 2].astype(int)
    red = (r >= 190) & (g < 90) & (b < 90)
    assert red.sum() > 200, f"rouge insuffisant : {red.sum()}"
    assert (crop[0, 0] == 255).all(), "coin du crop modifie hors sprite"
    corner = crop[64 - 18, 64 - 18]                          # coin du carre sprite (octogone -> transparent)
    assert (corner == 255).all(), f"alpha ignore : {corner}"
    before = crop.copy()
    RoadSign(0.0, 0.0, "30").paste_into_camera(crop, 600, 400)  # tres loin
    assert (crop == before).all(), "no-op hors region viole"
    assert load_signs("circuit_inexistant_xyz") == []
    print("simulator/signs.py self-check OK")
