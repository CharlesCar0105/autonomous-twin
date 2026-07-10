"""
car.py — Sprite et état du véhicule.

Responsabilités :
    - Charger le sprite de la voiture (ou placeholder)
    - Stocker l'état (position, angle, vitesse, commandes)
    - Dessiner la voiture avec rotation
    - Reset de la position
"""

import pygame
import os
import math


# --- Constantes -----------------------------------------------------------

SPRITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "sprites")

# Limites des actionneurs
MAX_STEERING_ANGLE = 45.0  # degrés


class Car:
    """Représente le véhicule sur le circuit."""

    def __init__(self, x: float, y: float, angle: float = 0.0) -> None:
        """
        Initialise la voiture.

        Args:
            x: Position x initiale (pixels).
            y: Position y initiale (pixels).
            angle: Orientation initiale en degrés (0 = droite).
        """
        self.x = x
        self.y = y
        self.angle = math.radians(angle)  # stocké en radians
        self.speed = 0.0                  # vitesse en pixels/s

        # Commandes courantes
        self.steering = 0.0    # angle volant en degrés [-45, +45]
        self.throttle = 0.0    # accélérateur [0.0, 1.0]
        self.brake = 0.0       # frein [0.0, 1.0]

        # Sprite + ombre portee
        self._load_sprite()
        self.shadow = self._create_shadow()

    def _load_sprite(self) -> None:
        """Charge le sprite de la voiture ou crée un placeholder."""
        sprite_path = os.path.join(SPRITES_DIR, "car.png")
        if os.path.exists(sprite_path):
            self.original_image = pygame.image.load(sprite_path).convert_alpha()
            # Redimensionner si trop grand
            w, h = self.original_image.get_size()
            if w > 60 or h > 60:
                scale = 60 / max(w, h)
                self.original_image = pygame.transform.smoothscale(
                    self.original_image, (int(w * scale), int(h * scale))
                )
        else:
            self.original_image = self._create_placeholder()

    @staticmethod
    def _create_placeholder() -> pygame.Surface:
        """Cree un sprite de voiture de course stylise (vue de dessus).

        Convention : l'AVANT est vers +x (cote droit du sprite), coherent
        avec car.angle = 0 -> cap vers la droite.
        """
        w, h = 46, 22
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        red = (210, 40, 40)
        red_dark = (150, 22, 22)
        tyre = (28, 28, 32)

        # Roues (dessinees avant la carrosserie pour depasser legerement).
        for wx in (7, w - 15):
            pygame.draw.rect(surf, tyre, (wx, -1, 9, 5), border_radius=2)      # cote haut
            pygame.draw.rect(surf, tyre, (wx, h - 4, 9, 5), border_radius=2)   # cote bas

        # Carrosserie profilee : museau avant plus etroit.
        body = [
            (4, 5), (w - 12, 3), (w - 3, h // 2), (w - 12, h - 3), (4, h - 5),
        ]
        pygame.draw.polygon(surf, red, body)
        pygame.draw.polygon(surf, red_dark, body, 2)

        # Bande de course claire au centre (longitudinale).
        pygame.draw.rect(surf, (245, 245, 245), (8, h // 2 - 2, w - 22, 4), border_radius=2)

        # Cockpit / canopy (vitres teintees).
        pygame.draw.ellipse(surf, (40, 48, 70), (w // 2 - 6, 5, 14, h - 10))

        # Aileron arriere (a gauche du sprite).
        pygame.draw.rect(surf, red_dark, (1, 3, 4, h - 6), border_radius=1)

        # Phares avant (jaunes) et feux arriere (rouges).
        pygame.draw.circle(surf, (255, 230, 120), (w - 5, 7), 2)
        pygame.draw.circle(surf, (255, 230, 120), (w - 5, h - 7), 2)
        pygame.draw.circle(surf, (255, 70, 70), (5, 7), 2)
        pygame.draw.circle(surf, (255, 70, 70), (5, h - 7), 2)

        return surf

    def _create_shadow(self) -> pygame.Surface:
        """Ombre portee : ellipse sombre semi-transparente a la taille du sprite."""
        w, h = self.original_image.get_size()
        shadow = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), (2, 3, w - 4, h - 4))
        return shadow

    def reset(self, x: float, y: float, angle_deg: float = 0.0) -> None:
        """Remet la voiture à la position donnée, vitesse à zéro."""
        self.x = x
        self.y = y
        self.angle = math.radians(angle_deg)
        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0

    def set_controls(self, steering: float, throttle: float, brake: float) -> None:
        """
        Applique les commandes (utilisé par le pilote IA via réseau).

        Args:
            steering: Angle volant en degrés [-45, +45].
            throttle: Accélérateur [0.0, 1.0].
            brake: Frein [0.0, 1.0].
        """
        self.steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering))
        self.throttle = max(0.0, min(1.0, throttle))
        self.brake = max(0.0, min(1.0, brake))

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine la voiture sur l'écran avec rotation (+ ombre portee)."""
        angle_deg = -math.degrees(self.angle)  # Pygame tourne dans le sens inverse

        # Ombre : legerement decalee vers le bas-droite, sous la voiture.
        rot_shadow = pygame.transform.rotate(self.shadow, angle_deg)
        sh_rect = rot_shadow.get_rect(center=(int(self.x) + 3, int(self.y) + 4))
        screen.blit(rot_shadow, sh_rect.topleft)

        rotated = pygame.transform.rotate(self.original_image, angle_deg)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)
