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
HYSTERESIS_FRAMES = 3        # classifications consecutives pour la 1ere pose
# Remplacer une limite DEJA active par une autre exige plus de preuves que
# la premiere pose : pare le flip transitoire 30<->90 observe quand un
# panneau se degrade en sortant du champ (mesures agent E, 10/07 : la
# moyenne des softmax est PIRE ; l'hysteresis durcie est gratuite et sure).
REPLACE_HYSTERESIS_FRAMES = 5
# Classification 1 frame rouge sur 2 : ~-50% de cout CPU quand un panneau
# est visible, sans perdre le latch (stride 2 valide par test_signs ET
# simulation temporelle ; stride 3 ECHOUE test_signs -- ne pas augmenter).
CLASSIFY_STRIDE = 2
COOLDOWN_S = 4.0             # apres application : ignore le MEME type de panneau (cf docstring)
STOP_HOLD_S = 0.8            # arret marque (decision 10/07 : le CDC ne fixe
                             # aucune duree ; 0.8s reste demontrable. Etait 2.0)
STOP_SPEED_KMH = 2.0         # seuil "voiture arretee"
# Portee d'une limite : zone de validite en distance parcourue apres
# application (decision 10/07). Le CDC ne fixe aucune duree ; la persistance
# infinie bridait des tours entiers sur circuits pauvres en panneaux
# (cf circuit_02 : un seul '50', aucun relachement). ~600 px = une zone.
LIMIT_ZONE_PX = 600.0
_SPEED_SCALE = 0.72          # px/s -> km/h, cf simulator/physics.py (garder synchronise)

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
    # Reglages CPU (mesures agent E, 10/07, machine hybride P/E-cores) :
    # 12 threads (defaut torch) est le PIRE reglage mesure ; 6 = -36% stable.
    # ATTENTION : set_num_threads est process-global (affecte aussi
    # cnn_policy -- 12 etant le pire la aussi, c'est un gain partout).
    # channels_last : ~25% supplementaires gratuits sur MobileNetV2 CPU.
    torch.set_num_threads(6)
    _model = _model.to(memory_format=torch.channels_last)


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
        x = preprocess_batch(t).contiguous(memory_format=torch.channels_last)
        probs = F.softmax(_model(x), dim=1).squeeze(0)
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
        self._red_frames = 0               # compteur frames rouges (stride)
        self._limit_travel_px = 0.0        # distance depuis l'application
        self._last_now: Optional[float] = None

    @property
    def stop_active(self) -> bool:
        return self._stop_state in ("BRAKING", "STOPPED")

    @property
    def limit_zone_left_px(self) -> float:
        """Distance restante (px) avant expiration de la limite active
        (lecture publique de la zone de validite -- cf LIMIT_ZONE_PX --
        pour affichage tableau de bord, sans exposer _limit_travel_px).
        0.0 si aucune limite active."""
        if self.speed_limit is None:
            return 0.0
        return max(0.0, LIMIT_ZONE_PX - self._limit_travel_px)

    def _apply(self, kind: str, now: float) -> None:
        if kind in ("30", "50", "90"):
            self.speed_limit = float(kind)
            self._limit_travel_px = 0.0    # nouvelle zone de validite
        elif kind == "stop":
            self._stop_state = "BRAKING"
        self._last_applied_kind = kind
        self._cooldown_until = now + COOLDOWN_S
        self._pending_kind, self._pending_count = None, 0

    def update(self, camera: np.ndarray, speed_kmh: float, now: float) -> None:
        # Zone de validite : la limite expire apres LIMIT_ZONE_PX parcourus
        # depuis son application (integration de la vitesse entre updates).
        if self.speed_limit is not None and self._last_now is not None:
            dt_s = max(0.0, now - self._last_now)
            self._limit_travel_px += (speed_kmh / _SPEED_SCALE) * dt_s
            if self._limit_travel_px >= LIMIT_ZONE_PX:
                self.speed_limit = None
        self._last_now = now

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
            self._red_frames = 0
            return
        # Stride : une classification par CLASSIFY_STRIDE frames rouges.
        # L'hysteresis compte des CLASSIFICATIONS ; les frames sautees ne
        # cassent pas le streak (valide agent E : stride 2, test_signs 2/2).
        self._red_frames += 1
        if (self._red_frames - 1) % CLASSIFY_STRIDE != 0:
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
        # Remplacer une limite active DIFFERENTE exige plus de preuves que
        # la premiere pose (anti-flip 30<->90, cf constantes).
        needed = HYSTERESIS_FRAMES
        if (kind in ("30", "50", "90") and self.speed_limit is not None
                and self.speed_limit != float(kind)):
            needed = REPLACE_HYSTERESIS_FRAMES
        if self._pending_count >= needed:
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
    # NB stride : seules les frames rouges 1, 1+CLASSIFY_STRIDE, ... classifient.
    for t in (101.0, 101.2, 101.4, 101.6):
        tr.update(cam, speed_kmh=50.0, now=t)
    assert tr.speed_limit == 30.0 and tr._cooldown_until == 104.0
    assert tr._pending_count == 0, "les '30' en cooldown doivent etre jetes"
    # ... mais un panneau d'un AUTRE type latche malgre le cooldown.
    # Remplacer la limite 30 ACTIVE exige REPLACE_HYSTERESIS_FRAMES (5)
    # classifications, a 1 frame rouge sur CLASSIFY_STRIDE (2) -> pas
    # instantane, mais doit aboutir en < 2*stride*5 updates.
    _mock_kind[0] = "50"
    t0, n_updates = 102.0, 0
    while tr.speed_limit == 30.0 and n_updates < 2 * CLASSIFY_STRIDE * REPLACE_HYSTERESIS_FRAMES:
        tr.update(cam, speed_kmh=50.0, now=t0)
        t0 += 0.02
        n_updates += 1
    assert tr.speed_limit == 50.0, "autre type doit passer malgre cooldown"
    assert n_updates >= REPLACE_HYSTERESIS_FRAMES, "remplacement trop facile (anti-flip vide)"
    assert tr._last_applied_kind == "50"
    # Machine STOP (STOP_HOLD_S=0.8 : NONE des que le maintien est ecoule).
    tr._apply("stop", now=200.0)
    assert tr.stop_active
    tr.update(cam, speed_kmh=1.0, now=201.0)               # v<2 -> STOPPED
    assert tr._stop_state == "STOPPED"
    tr.update(cam, speed_kmh=0.0, now=201.0 + STOP_HOLD_S + 0.1)
    assert tr._stop_state == "NONE" and not tr.stop_active
    # Zone de validite : la limite expire apres LIMIT_ZONE_PX parcourus.
    tr2 = SignTracker()
    blank = np.full((128, 128, 3), 255, dtype=np.uint8)
    tr2._apply("90", now=300.0)
    tr2._last_now = 300.0
    tr2.update(blank, speed_kmh=72.0, now=302.0)   # 100 px/s x 2 s = 200 px
    assert tr2.speed_limit == 90.0, "la limite doit tenir dans la zone"
    tr2.update(blank, speed_kmh=72.0, now=306.5)   # +450 px -> 650 > 600
    assert tr2.speed_limit is None, "la limite doit expirer apres la zone"
    print("pilot/signs.py self-check OK")
