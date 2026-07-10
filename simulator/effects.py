"""
effects.py — Effets visuels de course : traces de pneus + poussiere.

IMPORTANT : tout est dessine sur des surfaces ECRAN uniquement. Rien
n'ecrit dans track.pixels -> aucun impact sur les capteurs du pilote
(camera / lidar / masque) ni sur l'entrainement de l'IA.

- Les traces de pneus s'accumulent sur une couche transparente persistante
  (marks_layer), affichee SOUS la voiture.
- La poussiere est un petit systeme de particules ephemeres, affichee
  au-dessus du sol lors des freinages appuyes.
"""

import math
import random
import pygame


# Declenchement des traces de virage : base sur l'INTENSITE de virage
# = vitesse (px/s) x |braquage| (deg), un proxy de la force laterale qui
# fait "gratter" les pneus. Ainsi meme la conduite douce de l'IA (qui
# braque peu mais roule) laisse des traces des qu'elle tourne a vitesse,
# alors qu'une ligne droite (braquage ~0) n'en laisse pas. Mesure sur un
# tour PID : ~35% du temps sur circuit sinueux, corners sur l'ovale.
SKID_TURN_INTENSITY = 250.0
SKID_MIN_SPEED = 40.0      # vitesse mini pour tracer (evite le sur-place)
BRAKE_SPEED_PX = 30.0      # vitesse mini pour une trace de freinage (px/s)

REAR_OFFSET = 11.0         # distance centre -> essieu arriere (px)
HALF_TRACK = 8.0           # demi-voie (ecart lateral des roues, px)

MARK_COLOR = (20, 20, 24, 170)   # plus foncé et opaque -> bien visible
MARK_WIDTH = 5

# Estompage : l'alpha des traces decroit de ~45/s. Une trace a 170 disparait
# donc en ~3.8 s ("au bout de quelques secondes"). Independant du framerate.
MARK_FADE_PER_SEC = 45.0


class Effects:
    """Traces de pneus (persistantes) + particules de poussiere."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.marks_layer = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        self.particles: list[dict] = []
        self._prev_wheels = None   # (rl, rr) de la frame precedente
        self._fade_accum = 0.0     # accumulateur pour l'estompage fractionnaire

    def reset(self) -> None:
        """Efface traces + particules (au reset de la voiture)."""
        self.marks_layer.fill((0, 0, 0, 0))
        self.particles.clear()
        self._prev_wheels = None

    def _rear_wheels(self, car):
        cos_a, sin_a = math.cos(car.angle), math.sin(car.angle)
        bx = car.x - REAR_OFFSET * cos_a
        by = car.y - REAR_OFFSET * sin_a
        perp = (-sin_a, cos_a)
        rl = (bx + HALF_TRACK * perp[0], by + HALF_TRACK * perp[1])
        rr = (bx - HALF_TRACK * perp[0], by - HALF_TRACK * perp[1])
        return rl, rr

    def update(self, car, dt: float, braking: bool) -> None:
        """A appeler chaque frame apres la physique."""
        # Estompage progressif : on retranche de l'alpha sur toute la couche.
        # Accumulateur pour gerer les fractions (fill n'accepte que des int).
        self._fade_accum += MARK_FADE_PER_SEC * dt
        sub = int(self._fade_accum)
        if sub > 0:
            self._fade_accum -= sub
            self.marks_layer.fill((0, 0, 0, sub), special_flags=pygame.BLEND_RGBA_SUB)

        speed = car.speed
        rl, rr = self._rear_wheels(car)

        turn_intensity = speed * abs(car.steering)     # px/s * deg
        hard_turn = speed > SKID_MIN_SPEED and turn_intensity > SKID_TURN_INTENSITY
        hard_brake = braking and speed > BRAKE_SPEED_PX
        skidding = hard_turn or hard_brake

        if skidding and self._prev_wheels is not None:
            prl, prr = self._prev_wheels
            pygame.draw.line(self.marks_layer, MARK_COLOR, prl, rl, MARK_WIDTH)
            pygame.draw.line(self.marks_layer, MARK_COLOR, prr, rr, MARK_WIDTH)
        self._prev_wheels = (rl, rr)

        # Poussiere : petit jet au freinage appuye.
        if hard_brake:
            for wheel in (rl, rr):
                self._spawn_dust(wheel, car.angle)

        # Mise a jour des particules.
        alive = []
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vx"] *= 0.90
            p["vy"] *= 0.90
            p["life"] -= dt
            if p["life"] > 0:
                alive.append(p)
        self.particles = alive

    def _spawn_dust(self, pos, angle) -> None:
        for _ in range(2):
            spread = random.uniform(-0.8, 0.8)
            back = angle + math.pi + spread
            speed = random.uniform(20, 60)
            self.particles.append({
                "x": pos[0], "y": pos[1],
                "vx": speed * math.cos(back), "vy": speed * math.sin(back),
                "life": random.uniform(0.25, 0.5),
                "max_life": 0.5,
                "r": random.uniform(2, 4),
            })

    def draw_marks(self, screen: pygame.Surface) -> None:
        """Traces au sol (a appeler juste apres track.draw, sous la voiture)."""
        screen.blit(self.marks_layer, (0, 0))

    def draw_particles(self, screen: pygame.Surface) -> None:
        """Poussiere (a appeler apres la voiture)."""
        for p in self.particles:
            a = max(0, min(180, int(180 * p["life"] / p["max_life"])))
            surf = pygame.Surface((int(p["r"] * 2) + 2, int(p["r"] * 2) + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (170, 160, 150, a),
                               (int(p["r"]) + 1, int(p["r"]) + 1), int(p["r"]))
            screen.blit(surf, (p["x"] - p["r"], p["y"] - p["r"]))
