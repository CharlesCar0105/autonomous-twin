"""
network.py — Serveur ZeroMQ du simulateur.

Responsabilités :
    - Créer un socket ZMQ REP (request-reply)
    - Envoyer les données capteurs au pilote (caméra JPEG, lidar, vitesse)
    - Recevoir les commandes du pilote (volant, gaz, frein)

Protocole :
    Simulateur → Pilote (capteurs) :
        {
            "camera": "<bytes JPEG base64>",
            "lidar": [120.5, 85.3, 200.0, 90.1, 110.7],
            "speed": 45.2,
            "timestamp": 1234567890.123
        }

    Pilote → Simulateur (commandes) :
        {
            "steering": 12.5,
            "throttle": 0.7,
            "brake": 0.0
        }
"""

import zmq
import json
import base64
import time
import cv2
import numpy as np


# --- Constantes -----------------------------------------------------------

DEFAULT_PORT = 5555
DEFAULT_ADDRESS = f"tcp://*:{DEFAULT_PORT}"


class SimulatorServer:
    """Serveur ZMQ du simulateur."""

    def __init__(self, address: str = DEFAULT_ADDRESS) -> None:
        """
        Initialise le serveur ZMQ.

        Args:
            address: Adresse de bind (ex: "tcp://*:5555").
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(address)
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # timeout 100ms
        print(f"[Simulateur] Serveur ZMQ démarré sur {address}")

    def send_sensors(self, camera: np.ndarray, lidar: list[float], speed: float) -> dict | None:
        """
        Envoie les capteurs et attend les commandes du pilote.

        Args:
            camera: Image caméra numpy (H, W, 3) RGB.
            lidar:  Liste de 5 distances lidar.
            speed:  Vitesse actuelle en km/h.

        Returns:
            Dict des commandes reçues, ou None si timeout.
        """
        # Encoder l'image en JPEG
        bgr = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
        _, jpeg_bytes = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        camera_b64 = base64.b64encode(jpeg_bytes.tobytes()).decode("ascii")

        # Construire le message capteurs
        sensor_data = {
            "camera": camera_b64,
            "lidar": lidar,
            "speed": round(speed, 2),
            "timestamp": time.time(),
        }

        try:
            # Attendre une requête du pilote
            request = self.socket.recv_string()
            # Envoyer les capteurs en réponse
            self.socket.send_string(json.dumps(sensor_data))

            # La prochaine requête contiendra les commandes
            cmd_str = self.socket.recv_string()
            commands = json.loads(cmd_str)
            self.socket.send_string("OK")
            return commands

        except zmq.Again:
            return None  # timeout — pas de pilote connecté

    def close(self) -> None:
        """Ferme le socket et le contexte ZMQ."""
        self.socket.close()
        self.context.term()
        print("[Simulateur] Serveur ZMQ fermé.")
