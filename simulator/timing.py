"""
timing.py — Chronometrage sur 3 tours + sauvegarde des best laps.

Une ligne d'arrivee (segment) est placee a la position de depart du
circuit, perpendiculaire au cap de depart. Le chrono :

    - demarre automatiquement au premier mouvement de la voiture ;
    - compte un tour a chaque franchissement *dans le bon sens* de la ligne ;
    - se protege des faux positifs (voiture qui oscille sur la ligne) via un
      systeme d'armement : il faut s'etre eloigne de la ligne (ARM_DISTANCE)
      avant qu'un nouveau franchissement soit comptabilise ;
    - s'arrete apres LAPS_TARGET tours et persiste le meilleur tour par
      circuit dans best_laps.json.

Detection de franchissement : intersection du segment [pos_precedente ->
pos_courante] avec le segment de la ligne d'arrivee.
"""

import json
import math
import os


# --- Constantes ----------------------------------------------------------

LAPS_TARGET = 3                 # nombre de tours chronometres
FINISH_HALF_LEN = 70.0          # demi-longueur de la ligne d'arrivee (px)
ARM_DISTANCE = 90.0             # distance a parcourir avant de ré-armer (px)
START_SPEED_THRESHOLD = 5.0     # vitesse (px/s) au-dela de laquelle le chrono demarre

# Fichier de persistance des meilleurs tours (racine du projet).
BEST_LAPS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "best_laps.json"
)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True si le segment [p1,p2] croise le segment [p3,p4]."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


class LapTimer:
    """Chronometre 3 tours avec detection de franchissement de ligne."""

    def __init__(self, start_x: float, start_y: float, start_angle_deg: float,
                 circuit_name: str) -> None:
        """
        Args:
            start_x, start_y:   Position de la ligne d'arrivee (= depart).
            start_angle_deg:    Cap de depart en degres (sens du tour).
            circuit_name:       Nom du circuit (cle de persistance best lap).
        """
        self.circuit_name = circuit_name

        a = math.radians(start_angle_deg)
        self.forward = (math.cos(a), math.sin(a))
        # Ligne d'arrivee : perpendiculaire au cap, centree sur le depart.
        perp = (-math.sin(a), math.cos(a))
        self.line_p1 = (start_x - FINISH_HALF_LEN * perp[0],
                        start_y - FINISH_HALF_LEN * perp[1])
        self.line_p2 = (start_x + FINISH_HALF_LEN * perp[0],
                        start_y + FINISH_HALF_LEN * perp[1])
        self.line_center = (start_x, start_y)

        self.best_lap = self._load_best_lap()
        self.reset()

    # --- Cycle de vie -----------------------------------------------------

    def reset(self) -> None:
        """Remet le chrono a zero (touche T cote simu)."""
        self.started = False
        self.finished = False
        self.armed = False              # True quand on peut compter un tour
        self.laps_done = 0
        self.lap_times: list[float] = []
        self._t_lap_start = 0.0
        self._t_now = 0.0
        self.last_lap = None
        self._new_record = False        # flash "record !" transitoire

    def update(self, car, prev_pos, now: float) -> None:
        """A appeler chaque frame.

        Args:
            car:      voiture (position, vitesse).
            prev_pos: (x, y) de la voiture a la frame precedente.
            now:      horodatage (secondes, ex: time.time()).
        """
        self._t_now = now

        if self.finished:
            return

        # Demarrage automatique au premier mouvement.
        if not self.started:
            if car.speed > START_SPEED_THRESHOLD:
                self.started = True
                self._t_lap_start = now
            else:
                return

        cur = (car.x, car.y)

        # Armement : on doit s'eloigner de la ligne avant de compter un tour
        # (evite de compter le depart et les oscillations sur la ligne).
        dist_to_line = math.hypot(cur[0] - self.line_center[0],
                                  cur[1] - self.line_center[1])
        if not self.armed and dist_to_line > ARM_DISTANCE:
            self.armed = True

        if not self.armed:
            return

        # Franchissement dans le bon sens ?
        crossed = _segments_intersect(prev_pos, cur, self.line_p1, self.line_p2)
        if not crossed:
            return
        heading = (cur[0] - prev_pos[0], cur[1] - prev_pos[1])
        forward_dot = heading[0] * self.forward[0] + heading[1] * self.forward[1]
        if forward_dot <= 0:
            return  # franchissement a contresens : ignore

        # Tour valide.
        lap_time = now - self._t_lap_start
        self.lap_times.append(lap_time)
        self.last_lap = lap_time
        self.laps_done += 1
        self._t_lap_start = now
        self.armed = False

        if self.best_lap is None or lap_time < self.best_lap:
            self.best_lap = lap_time
            self._new_record = True
            self._save_best_lap()

        if self.laps_done >= LAPS_TARGET:
            self.finished = True

    # --- Accesseurs pour le HUD -------------------------------------------

    @property
    def current_lap_time(self) -> float:
        """Temps ecoule sur le tour en cours (0 si pas demarre)."""
        if not self.started or self.finished:
            return self.lap_times[-1] if (self.finished and self.lap_times) else 0.0
        return self._t_now - self._t_lap_start

    @property
    def total_time(self) -> float:
        return sum(self.lap_times)

    @property
    def current_lap_number(self) -> int:
        """Numero du tour en cours (1..LAPS_TARGET), plafonne a LAPS_TARGET."""
        return min(self.laps_done + 1, LAPS_TARGET)

    # --- Persistance best lap ---------------------------------------------

    def _load_best_lap(self):
        if not os.path.exists(BEST_LAPS_FILE):
            return None
        try:
            with open(BEST_LAPS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            val = data.get(self.circuit_name)
            return float(val) if val is not None else None
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def _save_best_lap(self) -> None:
        data = {}
        if os.path.exists(BEST_LAPS_FILE):
            try:
                with open(BEST_LAPS_FILE, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data[self.circuit_name] = round(self.best_lap, 3)
        try:
            with open(BEST_LAPS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"[Chrono] Impossible d'ecrire {BEST_LAPS_FILE} : {e}")


def format_time(seconds: float) -> str:
    """Formate un temps en M:SS.mmm (ou --:--.--- si None)."""
    if seconds is None:
        return "--:--.---"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"
