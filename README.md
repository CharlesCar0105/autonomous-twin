# Autonomous Twin

Jumeau numérique automobile : un simulateur 2D (vérité terrain) et un pilote
IA qui ne voit que ses capteurs (caméra 128 px, lidar 5 rayons, vitesse),
reliés par ZeroMQ. Projet annuel M1 ESGI (Architecture Logicielle + IA Big
Data). Équipe : Charles, Nohlan, Lorenzo.

## Installation

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

## Lancer

Le plus simple sous Windows : double-cliquer `demo-pid.bat` (pilote
réglementaire, circuit gen_014, panneaux 30/50/90/STOP) ou `demo-cnn.bat`
(le réseau de conduite par imitation, circuit gen_003).

À la main, dans deux terminaux :

```bash
.venv/Scripts/python.exe -m simulator.main --server --circuit gen_014
.venv/Scripts/python.exe -m pilot.main --dashboard
```

Options utiles : `--policy cnn` (réseau au lieu du PID), `--no-signs`,
`--no-emergency`, `--circuit gen_XXX` (30 circuits dans `assets/tracks/`).

## Tests et mesures

```bash
.venv/Scripts/python.exe scripts/test_cnn.py --with-signs   # conduite, 30 circuits
.venv/Scripts/python.exe scripts/test_signs.py              # scénarios panneaux
.venv/Scripts/python.exe scripts/bench_laps.py              # banc de mesure (3 tours chronométrés)
```

Les résultats du banc sont archivés dans `bench/` (JSON datés + graphiques).

## Structure

- `simulator/` : monde, physique bicycle, capteurs, panneaux, mur, chrono, serveur ZMQ
- `pilot/` : client ZMQ, PID, CNN de conduite, perception, lecture de panneaux, freinage d'urgence, dashboard
- `models/` : poids entraînés (U-Net, CNN conduite, classifieur de panneaux)
- `scripts/` : génération de circuits et de datasets, entraînements, tests, banc
- `docs/soutenance/` : support de présentation (ouvrir `index.html`)
- `notebooks/` : entraînements (Colab/local)
