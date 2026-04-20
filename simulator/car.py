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

        # Sprite
        self._load_sprite()

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
        """Crée un sprite placeholder en forme de voiture."""
        w, h = 36, 18
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        # Carrosserie
        pygame.draw.rect(surf, (220, 40, 40), (2, 2, w - 4, h - 4), border_radius=3)

        # Pare-brise (avant = côté droit du sprite)
        pygame.draw.rect(surf, (60, 60, 180), (w - 12, 4, 8, h - 8), border_radius=2)

        # Indicateur avant (triangle jaune)
        pygame.draw.polygon(surf, (255, 220, 50), [
            (w - 2, h // 2 - 3),
            (w, h // 2),
            (w - 2, h // 2 + 3),
        ])

        # Roues
        pygame.draw.rect(surf, (40, 40, 40), (4, 0, 8, 3))       # arrière gauche
        pygame.draw.rect(surf, (40, 40, 40), (4, h - 3, 8, 3))   # arrière droite
        pygame.draw.rect(surf, (40, 40, 40), (w - 14, 0, 8, 3))  # avant gauche
        pygame.draw.rect(surf, (40, 40, 40), (w - 14, h - 3, 8, 3))  # avant droite

        return surf

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
        """Dessine la voiture sur l'écran avec rotation."""
        angle_deg = -math.degrees(self.angle)  # Pygame tourne dans le sens inverse
        rotated = pygame.transform.rotate(self.original_image, angle_deg)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)
