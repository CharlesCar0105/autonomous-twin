"""
main.py — Boucle principale du pilote IA.

Responsabilités :
    - Connexion au simulateur via ZMQ
    - Réception des capteurs (caméra, lidar, vitesse)
    - Pipeline : perception → contrôle → commandes
    - Envoi des commandes (volant, gaz, frein)
"""

import time


def main() -> None:
    """Point d'entrée du pilote."""

    # TODO: Initialiser client ZMQ (network.py)
    # TODO: Charger modèles IA (U-Net, CNN conduite, CNN panneaux)

    print("[Pilote] Démarrage du pilote IA...")

    running = True
    while running:
        try:
            # TODO: Recevoir capteurs du simulateur
            # TODO: Pipeline perception (U-Net segmentation)
            # TODO: Détection panneaux (signs.py)
            # TODO: Freinage d'urgence (emergency.py)
            # TODO: Calcul des commandes (control.py)
            # TODO: Envoyer commandes au simulateur

            time.sleep(1 / 60)  # placeholder

        except KeyboardInterrupt:
            running = False

    print("[Pilote] Arrêt du pilote.")


if __name__ == "__main__":
    main()
