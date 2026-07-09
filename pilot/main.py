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
from pilot.control import pid_policy, cnn_policy
from pilot.emergency import check_emergency_brake, get_emergency_commands
from pilot.perception import compute_mask


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
    parser.add_argument("--policy", choices=["pid", "cnn", "none"], default="pid",
                        help="Politique de conduite. 'none' = voiture passive (tests).")
    parser.add_argument("--cnn-weights", default=None,
                        help="Path du .pth du CNN conduite (utile si --policy cnn).")
    parser.add_argument("--record", metavar="SESSION", default=None,
                        help="Active le dump dataset dans data/records/SESSION/.")
    parser.add_argument("--speed-target", type=float, default=80.0,
                        help="Vitesse cible (km/h) pour le PID.")
    parser.add_argument("--no-emergency", action="store_true",
                        help="Desactive le freinage d'urgence (utile tant que "
                             "le mur dynamique n'est pas cote simu : evite les "
                             "blocages quand la voiture frole une bordure en virage).")
    parser.add_argument("--dashboard", action="store_true",
                        help="Ouvre le tableau de bord temps reel (fenetre Pygame "
                             "cote pilote : camera, segmentation, lidar, vitesse).")
    args = parser.parse_args()

    session_dir = None
    if args.record:
        session_dir = os.path.join(RECORDS_DIR, args.record)
        os.makedirs(session_dir, exist_ok=True)
        print(f"[Pilote] Enregistrement dataset -> {session_dir}")

    client = PilotClient(args.address)
    print(f"[Pilote] Politique : {args.policy.upper()}  |  speed_target = {args.speed_target} km/h")

    dashboard = None
    if args.dashboard:
        from pilot.dashboard import Dashboard  # import tardif : pygame requis seulement ici
        dashboard = Dashboard()
        print("[Pilote] Tableau de bord ouvert (Echap ou fermeture = quitter).")

    print("[Pilote] Boucle demarree (Ctrl+C pour quitter).")

    frame_id = 0
    t_start = time.time()
    try:
        while True:
            sensors = client.request_sensors()
            lidar = sensors["lidar"]
            speed = sensors["speed"]

            # Commandes de base selon la politique
            if args.policy == "pid":
                steering, throttle, brake = pid_policy(
                    lidar, speed, speed_target=args.speed_target
                )
                commands = {"steering": steering, "throttle": throttle, "brake": brake}
            elif args.policy == "cnn":
                mask = compute_mask(sensors["camera"], size=64)
                steering, throttle, brake = cnn_policy(
                    lidar, speed, mask, weights_path=args.cnn_weights
                )
                commands = {"steering": steering, "throttle": throttle, "brake": brake}
            else:  # "none"
                commands = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

            # Emergency : coupe le throttle et freine, mais garde le steering
            # du PID pour que la voiture puisse fuir la bordure proche.
            emergency_active = (not args.no_emergency) and check_emergency_brake(lidar)
            if emergency_active:
                commands["throttle"] = 0.0
                commands["brake"] = 1.0

            client.send_commands(**commands)

            # Tableau de bord temps reel (ce que le pilote voit + decide).
            if dashboard is not None:
                mask_disp = compute_mask(sensors["camera"])
                state = "FREINAGE URGENCE" if emergency_active else "CONDUITE"
                if not dashboard.update(sensors["camera"], mask_disp, lidar,
                                        speed, commands, state):
                    print("[Pilote] Tableau de bord ferme -> arret.")
                    break

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
        if dashboard is not None:
            dashboard.close()
        client.close()


if __name__ == "__main__":
    main()
