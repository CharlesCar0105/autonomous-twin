"""
main.py — Boucle principale du pilote IA.

Pipeline par frame :
    1. Recevoir capteurs (camera, lidar, speed) via ZMQ
    2. Emergency check (lidar < seuil -> frein max, priorite absolue)
    3. Politique de conduite (PID en Sprint 1, CNN en Sprint 3)
    4. Envoyer commandes (steering, throttle, brake) au simulateur
    5. [Optionnel] Dump (capteurs, commandes) sur disque pour dataset

Mode record (D8) : `python -m pilot.main --record session_nom` intercepte
tout ce qui transite sur ZMQ et le sauve dans data/records/{session}/.

Usage :
    python -m pilot.main                         # conduite autonome PID
    python -m pilot.main --record run001         # conduite + dump dataset
    python -m pilot.main --policy none           # passif : no throttle, no steering
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pilot.network import PilotClient
from pilot.control import pid_policy
from pilot.emergency import check_emergency_brake, get_emergency_commands


RECORDS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "records"
)


def _dump_frame(session_dir: str, frame_id: int, sensors: dict, commands: dict) -> None:
    """Sauve une frame (capteurs + commandes) dans un .npz."""
    path = os.path.join(session_dir, f"{frame_id:06d}.npz")
    np.savez_compressed(
        path,
        camera=sensors["camera"],
        lidar=np.array(sensors["lidar"], dtype=np.float32),
        speed=np.float32(sensors["speed"]),
        steering=np.float32(commands["steering"]),
        throttle=np.float32(commands["throttle"]),
        brake=np.float32(commands["brake"]),
        timestamp=np.float64(sensors["timestamp"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilote IA — Autonomous Twin")
    parser.add_argument("--address", default="tcp://localhost:5555",
                        help="Adresse du serveur ZMQ du simulateur.")
    parser.add_argument("--policy", choices=["pid", "none"], default="pid",
                        help="Politique de conduite. 'none' = voiture passive (tests).")
    parser.add_argument("--record", metavar="SESSION", default=None,
                        help="Active le dump dataset dans data/records/SESSION/.")
    parser.add_argument("--speed-target", type=float, default=80.0,
                        help="Vitesse cible (km/h) pour le PID.")
    args = parser.parse_args()

    session_dir = None
    if args.record:
        session_dir = os.path.join(RECORDS_DIR, args.record)
        os.makedirs(session_dir, exist_ok=True)
        print(f"[Pilote] Enregistrement dataset -> {session_dir}")

    client = PilotClient(args.address)
    print(f"[Pilote] Politique : {args.policy.upper()}  |  speed_target = {args.speed_target} km/h")
    print("[Pilote] Boucle demarree (Ctrl+C pour quitter).")

    frame_id = 0
    t_start = time.time()
    try:
        while True:
            sensors = client.request_sensors()
            lidar = sensors["lidar"]
            speed = sensors["speed"]

            if check_emergency_brake(lidar):
                commands = get_emergency_commands()
            elif args.policy == "pid":
                steering, throttle, brake = pid_policy(
                    lidar, speed, speed_target=args.speed_target
                )
                commands = {"steering": steering, "throttle": throttle, "brake": brake}
            else:  # "none"
                commands = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

            client.send_commands(**commands)

            if session_dir is not None:
                _dump_frame(session_dir, frame_id, sensors, commands)

            if frame_id % 60 == 0:
                fps = frame_id / max(time.time() - t_start, 1e-6)
                print(
                    f"[Pilote] f={frame_id:06d}  v={speed:5.1f} km/h  "
                    f"lidar_front={lidar[2]:5.0f}px  steer={commands['steering']:+6.1f}  "
                    f"thr={commands['throttle']:.2f}  brk={commands['brake']:.2f}  "
                    f"FPS={fps:4.1f}"
                )
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[Pilote] Interrompu par l'utilisateur.")
    finally:
        if session_dir is not None:
            print(f"[Pilote] {frame_id} frames sauvegardees dans {session_dir}")
        client.close()


if __name__ == "__main__":
    main()
