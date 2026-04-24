"""
network.py — Client ZeroMQ du pilote.

Responsabilités :
    - Se connecter au simulateur via ZMQ REQ
    - Recevoir les capteurs (caméra JPEG, lidar, vitesse)
    - Envoyer les commandes (volant, gaz, frein)
"""

import zmq
import json
import base64
import cv2
import numpy as np


# --- Constantes -----------------------------------------------------------

DEFAULT_PORT = 5555
DEFAULT_ADDRESS = f"tcp://localhost:{DEFAULT_PORT}"


class PilotClient:
    """Client ZMQ du pilote IA."""

    def __init__(self, address: str = DEFAULT_ADDRESS) -> None:
        """
        Se connecte au simulateur.

        Args:
            address: Adresse du simulateur (ex: "tcp://localhost:5555").
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)
        print(f"[Pilote] Connecté au simulateur sur {address}")

    def request_sensors(self) -> dict:
        """
        Demande les capteurs au simulateur.

        Returns:
            Dict contenant :
                - "camera": np.ndarray (H, W, 3) RGB
                - "lidar": list[float] (5 distances)
                - "speed": float (km/h)
                - "timestamp": float
        """
        # Envoyer requête
        self.socket.send_string("GET_SENSORS")
        response = self.socket.recv_string()
        data = json.loads(response)

        # Décoder l'image caméra
        jpeg_bytes = base64.b64decode(data["camera"])
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return {
            "camera": rgb,
            "lidar": data["lidar"],
            "speed": data["speed"],
            "timestamp": data["timestamp"],
        }

    def send_commands(self, steering: float, throttle: float, brake: float) -> None:
        """
        Envoie les commandes au simulateur.

        Args:
            steering: Angle volant [-45, +45] degrés.
            throttle: Accélérateur [0.0, 1.0].
            brake: Frein [0.0, 1.0].
        """
        commands = {
            "steering": round(steering, 2),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
        }
        self.socket.send_string(json.dumps(commands))
        self.socket.recv_string()  # attendre "OK"

    def close(self) -> None:
        """Ferme la connexion."""
        self.socket.close()
        self.context.term()
        print("[Pilote] Connexion fermée.")
