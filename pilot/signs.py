"""
signs.py -- Detection et classification des panneaux routiers (M5).

Pipeline camera-only (anti-triche CDC : le pilote ne lit jamais les
sidecars .signs.json, il n'a que ses capteurs) :

    1. detect_sign_bbox : masque couleur rouge vif -> bbox.
       Seuil r>=190 & g<90 & b<90 : le rouge sprite est (220,30,30), le mur
       brique (150,45,35) et son lisere (180,70,55) restent SOUS le seuil.
       # ponytail: bbox globale des pixels rouges (pas de composantes
       # connexes) -- l'espacement >=300px des panneaux garantit <=1 panneau
       # par vue ; passer a du connected-components si ca change.
    2. classify_crop : crop bbox -> preprocess_batch (224) -> MobileNetV2
       (models/signs_cls.pth, tete MLP — cf signs_arch).
    3. SignTracker : hysteresis 3 frames (conf >= 0.8) + cooldown 4 s PAR
       TYPE (spec : ne pas re-declencher LE MEME panneau ; un panneau d'un
       autre type reste detectable immediatement).
       Limites 30/50/90 -> speed_limit persistante (regle routiere reelle).
       STOP -> BRAKING -> STOPPED (v < 2 km/h, maintien 2 s) -> RESUME.

Metrique cible : accuracy > 95% (atteinte au training, cf
models/signs_cls_history.json). Decision integree dans pilot/main.py.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from pilot.signs_arch import SIGN_CLASSES, preprocess_batch

# --- Detection (couleur) ---------------------------------------------------

RED_MIN_R = 190
RED_MAX_GB = 90
MIN_RED_PIXELS = 60          # aire mini du blob rouge (px)
BBOX_MARGIN = 0.2            # elargissement bbox de 20 %
ASPECT_MIN, ASPECT_MAX = 0.5, 2.0

# --- Tracker ----------------------------------------------------------------

CONF_THRESHOLD = 0.8
HYSTERESIS_FRAMES = 3        # frames consecutives meme classe pour valider
COOLDOWN_S = 4.0             # apres application : ignore le MEME type de panneau (cf docstring)
STOP_HOLD_S = 2.0            # arret complet maintenu (CDC "arret si Stop")
STOP_SPEED_KMH = 2.0         # seuil "voiture arretee"

_model = None


def _load_model(weights_path: Optional[str] = None):
    global _model
    if _model is not None:
        return
    import torch
    from pilot.signs_arch import build_signs_net
    if weights_path is None:
        weights_path = str(Path(__file__).resolve().parent.parent
                           / "models" / "signs_cls.pth")
    _model = build_signs_net(pretrained=False)
    _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    _model.eval()


def detect_sign_bbox(camera: np.ndarray):
    """Bbox (x1, y1, x2, y2) du blob rouge vif, ou None. camera (H, W, 3)."""
    r = camera[..., 0].astype(np.int16)
    g = camera[..., 1].astype(np.int16)
    b = camera[..., 2].astype(np.int16)
    red = (r >= RED_MIN_R) & (g < RED_MAX_GB) & (b < RED_MAX_GB)
    n = int(red.sum())
    if n < MIN_RED_PIXELS:
        return None
    ys, xs = np.nonzero(red)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    w, h = x2 - x1, y2 - y1
    if not (ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX):
        return None
    if n / max(w * h, 1) < 0.10:      # blob trop diffus = pas un panneau
        return None
    mx, my = int(w * BBOX_MARGIN), int(h * BBOX_MARGIN)
    H, W = camera.shape[:2]
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def classify_crop(camera: np.ndarray, bbox, weights_path: Optional[str] = None):
    """Classifie le crop bbox. Retourne (classe, confiance softmax)."""
    import torch
    import torch.nn.functional as F
    _load_model(weights_path)
    x1, y1, x2, y2 = bbox
    crop = camera[y1:y2, x1:x2].astype(np.float32) / 255.0
    t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
    # preprocess_batch = pipeline canonique (normalise ImageNet + upscale
    # FEAT_INPUT_SIZE=224) partage avec le training — ne JAMAIS re-inliner.
    with torch.no_grad():
        probs = F.softmax(_model(preprocess_batch(t)), dim=1).squeeze(0)
    idx = int(probs.argmax())
    return SIGN_CLASSES[idx], float(probs[idx])


class SignTracker:
    """Hysteresis + cooldown + decision (limite persistante, machine STOP)."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self._weights = weights_path
        self.speed_limit: Optional[float] = None
        self.last_detection: Optional[str] = None
        self._pending_kind: Optional[str] = None
        self._pending_count = 0
        self._cooldown_until = 0.0
        self._last_applied_kind: Optional[str] = None
        self._stop_state = "NONE"          # NONE | BRAKING | STOPPED
        self._stop_until = 0.0

    @property
    def stop_active(self) -> bool:
        return self._stop_state in ("BRAKING", "STOPPED")

    def _apply(self, kind: str, now: float) -> None:
        if kind in ("30", "50", "90"):
            self.speed_limit = float(kind)
        elif kind == "stop":
            self._stop_state = "BRAKING"
        self._last_applied_kind = kind
        self._cooldown_until = now + COOLDOWN_S
        self._pending_kind, self._pending_count = None, 0

    def update(self, camera: np.ndarray, speed_kmh: float, now: float) -> None:
        # Machine STOP prioritaire (independante des nouvelles detections).
        if self._stop_state == "BRAKING" and speed_kmh < STOP_SPEED_KMH:
            self._stop_state = "STOPPED"
            self._stop_until = now + STOP_HOLD_S
        elif self._stop_state == "STOPPED" and now >= self._stop_until:
            self._stop_state = "NONE"
            self._cooldown_until = now + COOLDOWN_S

        self.last_detection = None
        if self.stop_active:
            return

        bbox = detect_sign_bbox(camera)
        if bbox is None:
            self._pending_kind, self._pending_count = None, 0
            return
        kind, conf = classify_crop(camera, bbox, self._weights)
        self.last_detection = f"{kind}({conf:.2f})"
        if kind == "aucun" or conf < CONF_THRESHOLD:
            self._pending_kind, self._pending_count = None, 0
            return
        # Cooldown PAR TYPE (spec : "ne pas re-declencher le meme panneau") :
        # seul le type qui vient d'etre applique est ignore pendant la
        # fenetre ; un panneau d'un autre type reste detectable immediatement.
        if now < self._cooldown_until and kind == self._last_applied_kind:
            self._pending_kind, self._pending_count = None, 0
            return
        if kind == self._pending_kind:
            self._pending_count += 1
        else:
            self._pending_kind, self._pending_count = kind, 1
        if self._pending_count >= HYSTERESIS_FRAMES:
            self._apply(kind, now)


if __name__ == "__main__":
    # Self-check sans modele : detecteur seul (le classif exige le .pth).
    cam = np.full((128, 128, 3), 255, dtype=np.uint8)
    assert detect_sign_bbox(cam) is None, "blanc pur ne doit rien detecter"
    cam[40:70, 50:80] = (220, 30, 30)                      # pseudo panneau
    bbox = detect_sign_bbox(cam)
    assert bbox is not None and bbox[0] <= 50 and bbox[2] >= 80, bbox
    cam2 = np.full((128, 128, 3), 255, dtype=np.uint8)
    cam2[40:70, 30:110] = (150, 45, 35)                    # mur brique
    cam2[40:42, 30:110] = (180, 70, 55)                    # lisere
    assert detect_sign_bbox(cam2) is None, "la brique ne doit PAS declencher"
    # Tracker : hysteresis + cooldown par type + machine STOP.
    # classify_crop MOCKEE (rebind du global, resolu a l'appel par update) :
    # le vrai classif exigerait models/signs_cls.pth.
    _mock_kind = ["30"]

    def classify_crop(camera, bbox, weights_path=None):    # noqa: F811 (mock)
        return _mock_kind[0], 0.99

    tr = SignTracker()
    tr._apply("30", now=100.0)
    assert tr.speed_limit == 30.0 and tr._cooldown_until == 104.0
    # Cooldown PAR TYPE : re-voir "30" pendant la fenetre -> jete, pas
    # de re-application (cooldown_until inchange, pending reste a zero).
    for t in (101.0, 101.5, 102.0):
        tr.update(cam, speed_kmh=50.0, now=t)
    assert tr.speed_limit == 30.0 and tr._cooldown_until == 104.0
    assert tr._pending_count == 0, "les '30' en cooldown doivent etre jetes"
    # ... mais un panneau d'un AUTRE type latche normalement malgre le
    # cooldown (hysteresis 3 frames, puis application).
    _mock_kind[0] = "50"
    tr.update(cam, speed_kmh=50.0, now=102.2)
    tr.update(cam, speed_kmh=50.0, now=102.6)
    assert tr.speed_limit == 30.0, "pas d'application avant 3 frames"
    tr.update(cam, speed_kmh=50.0, now=103.0)              # 3e frame -> apply
    assert tr.speed_limit == 50.0, "autre type doit passer malgre cooldown"
    assert tr._last_applied_kind == "50" and tr._cooldown_until == 107.0
    # Machine STOP.
    tr._apply("stop", now=200.0)
    assert tr.stop_active
    tr.update(cam, speed_kmh=1.0, now=201.0)               # v<2 -> STOPPED
    assert tr._stop_state == "STOPPED"
    tr.update(cam, speed_kmh=0.0, now=203.5)               # 2s ecoulees -> NONE
    assert tr._stop_state == "NONE" and not tr.stop_active
    print("pilot/signs.py self-check OK")
