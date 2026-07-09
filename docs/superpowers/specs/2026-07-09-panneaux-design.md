# Spec — Feature Panneaux (M5) : détection, classification, décision

**Date** : 2026-07-09 · **Branche** : `dev/panneaux-lorenzo` · **Owner** : Lorenzo
**Barème visé** : +3 pts « Lecture de Panneaux » (détection → segmentation/crop →
classification → décision). Contribue aussi à « IA avancée » (+2 pts, déjà couvert
par le CNN conduite) et prépare le dashboard (panneau détecté affiché, Phase 5).

## 1. Contexte et décisions amont (non renégociées ici)

- **D5** (wiki ml.md) : classifieur = transfer learning, backbone pretrained
  ImageNet **gelé**, tête réentraînée. On part sur **MobileNetV2** (3.4M params,
  rapide à l'inférence) ; ResNet18 en plan B si souci.
- **D7** : dataset **synthétique** (sprites propres + augmentations), 1-2k
  images/classe. Fallback GTSRB uniquement si accuracy < 90 % après tuning.
- **D10** (contrainte prof) : généralisation. Appliquée ici : les fonds du val
  set proviennent des circuits held-out (gen_025-029) jamais utilisés en train.
- **CDC** : « Le simulateur doit pouvoir placer des images statiques (PNG) sur
  les bords de la piste (ex : Panneau STOP, Limites de vitesse) ». Décision :
  « ralentir à 30 km/h si panneau 30, arrêt si Stop ».
- Classes : **Stop, 30, 50, 90, Aucun** (5).

## 2. Architecture d'ensemble

```
gen_signs.py ──> assets/signs/{stop,30,50,90}.png   (source de vérité unique)
                      │                    │
        ┌─────────────┘                    └──────────────┐
        v                                                 v
SIMULATEUR                                          DATASET (offline)
simulator/signs.py::RoadSign                        scripts/gen_signs_dataset.py
  - chargé depuis <circuit>.signs.json                - fonds = crops caméra réels (npz data/)
  - draw(screen)          [rendu écran]               - paste sprite + augmentations
  - paste_into_camera(...) [vue pilote]               - classe "Aucun" = fonds nus + crops MUR
        │                                                 v
        v                                           notebooks/train_signs.ipynb
   camera ZMQ ──> PILOTE pilot/signs.py             pilot/signs_arch.py (MobileNetV2 gelé + tête 5)
  détection couleur → crop → classif → hystérésis         v
        │                                           models/signs_cls.pth (+ history JSON)
        v
  décision : speed_limit / STOP state machine ──> governor dans pilot/main.py
```

Principe directeur : même pattern d'intégration que le `Wall` de Charles —
objet passé explicitement aux fonctions capteurs, `track.pixels` jamais modifié,
une seule source de vérité géométrique. Différences assumées : pas de
`contains()` (aucun rôle physique/lidar — le panneau est hors piste, au-delà de
la bordure où le lidar s'arrête déjà), et compositing **sprite RGBA** (pixels
réels du PNG) au lieu d'une couleur pleine.

## 3. Composants

### 3.1 Sprites — `scripts/gen_signs.py` → `assets/signs/*.png`

- 4 PNG dessinés par script (PIL), master 256×256 avec alpha, downscale à la pose.
- STOP : octogone rouge vif `(220, 30, 30)` + texte « STOP » blanc.
- 30/50/90 : cercle fond blanc, anneau rouge vif, chiffres noirs (style FR).
- **Contrainte couleur** : rouge panneau `r ≥ 200` pour rester discriminable du
  mur brique de Charles (`COL_BRICK = (150,45,35)`, liseré `(180,70,55)`).
- Pas de sprite « Aucun » : cette classe = fonds sans panneau.
- Committés dans git (4 petits fichiers canoniques, pas des artefacts régénérés
  en masse comme les gen_*.png).

### 3.2 Placement simulateur — `simulator/signs.py` + sidecars JSON

- `RoadSign(x, y, kind, size=36)` : sprite chargé une fois (cache classe),
  `draw(screen)` (blit) et `paste_into_camera(crop, x0, y0)` (collage numpy
  alpha-aware dans le crop caméra, même convention d'indexation que
  `Wall.mask_for_region`).
- `load_signs(circuit_name) -> list[RoadSign]` : lit
  `assets/tracks/<circuit>.signs.json` (format :
  `{"signs": [{"x":.., "y":.., "kind":"stop|30|50|90"}]}`). Fichier absent =
  aucun panneau (rétro-compatible, aucun impact sur l'existant).
- **Génération des sidecars** (amendé au planning) : `scripts/place_signs.py`,
  un seul chemin de code pour TOUS les circuits (gen_* ET historiques) —
  dry-run PID headless (le PID n'utilise que le lidar : trajectoire identique
  avec/sans panneaux), échantillonnage de 2-4 positions le long de la
  trajectoire avec espacement arclength ≥ 300 px (crop caméra 128 px : jamais
  2 panneaux dans la même vue), offset perpendiculaire (55→85 px, premier
  offset dont le segment vers le centre croise la bordure noire = panneau
  garanti hors piste), types tirés par rng seedé. Avantages vs l'option
  « étendre gen_circuits.py » initialement envisagée : zéro régénération des
  PNG (l'invariant géométrie devient un non-sujet), couvre circuit_01/02 sans
  cas particulier, et panneaux garantis dans le champ caméra (placés relatifs
  à la ligne réellement conduite). Les sidecars `gen_*.signs.json` tombent
  sous le gitignore existant (`gen_*.json`) : régénérables, seedés.
- `circuit_01/02` : sidecars générés par le même script, committés.
- Câblage `simulator/main.py` : `signs = load_signs(args.circuit)` au boot ;
  `get_camera_view_from_track(track, car, wall, signs=signs)` composite les
  sprites dans la vue pilote (après le mur) ; `draw` à l'écran dans la boucle
  rendu. Flag `--no-signs` pour désactiver.

### 3.3 Dataset — `scripts/gen_signs_dataset.py` → `data/signs/`

- **Fonds** : crops caméra 128×128 réels tirés des `.npz` existants
  (`data/train`, `data/val` du dataset conduite). Train ← fonds des circuits
  gen_000-024, val ← fonds de gen_025-029 (D10).
- **Positifs** : sprite collé sur fond, augmentations : échelle 0.5-1.5,
  rotation ±20°, luminosité 0.7-1.3, bruit gaussien, motion blur léger,
  position aléatoire, troncature partielle au bord (20 % des cas) — puis crop
  96×96 centré sur le panneau (simule la sortie du détecteur).
- **« Aucun »** : crops 96×96 de fonds nus (route/bordure/herbe) + **crops du
  mur brique** (rendu via la classe `Wall`) — anti-confusion mur/panneau.
- Volume : ~1 500/classe train, ~300/classe val. Format :
  `data/signs/{train,val}/{stop,30,50,90,aucun}/*.png`. Gitignoré, régénérable
  (seed fixe).

### 3.4 Classifieur — `pilot/signs_arch.py` + `notebooks/train_signs.ipynb`

- `SignsNet` : `torchvision.models.mobilenet_v2(weights=IMAGENET1K_V1)`,
  `features` gelées, classifier remplacé par `Linear(1280, 5)`. Input 96×96
  normalisé ImageNet. (torchvision 0.26.0 déjà dans le venv, vérifié.)
- Entraînement dans `scripts/train_signs.py` (amendé : script plutôt que
  notebook — exécutable/vérifiable localement sans jupyter, mêmes principes
  que `train_cnn_drive.ipynb` : archi importée du fichier `pilot/`, Adam 1e-3
  sur la tête seule, ~15 epochs, save best). Artefacts soutenance : history
  JSON + matrice de confusion sauvegardés à côté du modèle. Un notebook
  Colab de démonstration pourra être ajouté plus tard si utile.
- **Entraînement 100 % local CPU** (machine dev : Iris Xe, torch 2.11 build
  cpu — vérifié 2026-07-09, pas de GPU NVIDIA). Rendu possible par le backbone
  gelé : les features MobileNetV2 (1280-d) sont **précalculées une fois** sur
  les ~9k images (~2-3 min CPU) puis la tête s'entraîne sur ce cache (secondes
  par epoch). Colab uniquement si fallback GTSRB ou dégel du backbone.
- **Leçon audit Q8 appliquée** : l'historique train/val loss+accuracy est dumpé
  en JSON à côté du modèle (`models/signs_cls_history.json`) et les outputs du
  notebook sont conservés au commit.
- Sortie : `models/signs_cls.pth`. Cible : **accuracy val > 95 %** (attendu
  ≈ 99 % sur synthétique propre) + matrice de confusion sans confusion
  30↔90 notable. Si < 90 % : bascule GTSRB (D7) — improbable.

### 3.5 Pilote — `pilot/signs.py` + câblage `pilot/main.py`

- `detect_sign_bbox(camera) -> bbox | None` : masque couleur rouge vif
  (`r ≥ 190 & g < 90 & b < 90` — exclut la brique à 150/180), composantes
  connexes, filtres aire (≥ 60 px²) et aspect (0.5-2.0), plus grande composante.
  Retourne le carré englobant élargi de 20 %.
- `classify(camera, bbox) -> (kind, conf)` : crop → resize 96×96 → `SignsNet`
  (lazy-load, pattern `_load_cnn` de control.py), softmax.
- `SignTracker` (état interne du pilote, aucune donnée simu) :
  - **Hystérésis** : même classe détectée avec conf ≥ 0.8 sur **3 frames
    consécutives** → événement validé (immunise contre les frames partielles).
  - **Cooldown** 4 s après application (ne pas re-déclencher le même panneau).
  - Limites 30/50/90 : `speed_limit` mise à jour, **persiste** jusqu'au
    prochain panneau de limite (règle routière réelle).
  - STOP : machine à états `BRAKING` (throttle 0, brake 1) → `STOPPED` quand
    v < 2 km/h, maintien **2.0 s** → `RESUME` (+ cooldown).
- Câblage `pilot/main.py`, dans la boucle, **après** la policy, **avant**
  l'emergency (l'emergency reste prioritaire absolu) :
  1. `event = tracker.update(camera)` ;
  2. governor vitesse : si `speed > limit` → `throttle = min(throttle, 0.15)` ;
     si `speed > limit + 15` → `brake = max(brake, 0.3)` (un seul mécanisme,
     valable pour PID **et** CNN) ;
  3. override STOP selon l'état du tracker.
  - Flag `--no-signs` (symétrique de `--no-emergency`).
  - Log frame : classe détectée + limite courante (préparation dashboard).
- **Anti-triche (CDC : « prouver l'indépendance du véhicule »)** : le pilote ne
  lit **jamais** les `.signs.json`. Caméra uniquement. Les JSON ne servent
  qu'à l'éval côté harness.

### 3.6 Éval — `scripts/test_signs.py`

1. **Offline** : accuracy + matrice de confusion sur `data/signs/val`.
2. **Boucle fermée headless** (pattern `test_cnn.py`, simulateur in-process) :
   - scénario « limite » : circuit avec panneau 30, PID lancé à 80 km/h →
     PASS si vitesse ≤ 35 km/h dans les 6 s suivant le passage du panneau ;
   - scénario « stop » : panneau STOP → PASS si v < 2 km/h atteint puis
     reprise (v > 20 km/h) dans les 10 s.
   - Vérité terrain lue depuis les `.signs.json` (côté harness seulement).
3. **Non-régression conduite** : re-run `scripts/test_cnn.py` avec panneaux
   composités dans la caméra → 30/30 PASS attendu. Si régression (le masque
   drive voit un blob sombre là où le sprite couvre l'herbe) : re-record du
   dataset conduite avec panneaux + retrain rapide (~5 min Colab, contingence
   déjà éprouvée en avril).

## 4. Risques et mitigations

| Risque | Mitigation |
|---|---|
| Confusion mur brique (rouge sombre) / panneau | Seuil `r ≥ 190` (brique = 150-180) **et** crops de mur dans la classe « Aucun » |
| Régression CNN conduite (sprite dans le masque) | Test non-régression 3.6.3 ; contingence retrain documentée |
| Panneau partiellement visible au bord du crop | Troncatures dans le dataset + hystérésis 3 frames |
| 30 vs 90 (mêmes formes, chiffres proches) | Matrice de confusion surveillée ; augmentations sans rotation excessive (±20° max) |
| Fallback GTSRB : pas de classe « 90 » exacte | Classes 30/50 existent ; « 90 » via 100 km/h retagué ou génération custom — traité seulement si le fallback est déclenché |
| Sidecars gen_* gitignorés (perte de placements custom) | Placements seedés régénérables ; placements manuels réservés aux circuit_01/02 committés |

## 5. Hors scope (explicitement)

- Dashboard temps réel (Phase 5 du barème, chantier séparé — signs.py expose
  déjà l'état nécessaire via le tracker).
- Détecteur appris (YOLO & co) : sur-engineering pour ce monde synthétique.
- Placement dynamique de panneaux au clavier (le mur est un aléa, pas le panneau).
- Météo/luminosité variable du simulateur (optionnel CDC, non traité ici).

## 6. Ordre de réalisation (phases avec critère de sortie)

1. **Sprites** — `gen_signs.py` + 4 PNG. Sortie : PNG relus visuellement.
2. **Placement simu** — `RoadSign` + sidecars + câblage caméra/rendu.
   Sortie : panneau visible à l'écran ET dans l'aperçu caméra (touche C).
3. **Dataset** — `gen_signs_dataset.py`. Sortie : ~9 000 images, échantillon
   relu, distribution par classe équilibrée.
4. **Training** — `signs_arch.py` + notebook. Sortie : val acc > 95 %,
   history JSON committé.
5. **Pilote** — `signs.py` + câblage main.py. Sortie : démo manuelle — la
   voiture ralentit au 30, s'arrête au STOP, repart.
6. **Éval** — `test_signs.py` + non-régression `test_cnn.py`. Sortie : tous
   scénarios PASS, documenté dans le wiki (ml.md, décisions D12+).

## 7. Fichiers créés / modifiés

| Fichier | Statut |
|---|---|
| `scripts/gen_signs.py` | nouveau |
| `assets/signs/{stop,30,50,90}.png` | nouveaux (committés) |
| `simulator/signs.py` | nouveau |
| `scripts/place_signs.py` | nouveau (sidecars par dry-run PID, tous circuits) |
| `assets/tracks/circuit_01.signs.json`, `circuit_02.signs.json` | nouveaux |
| `simulator/main.py`, `simulator/sensors.py` | modifiés (param `signs`) |
| `scripts/gen_signs_dataset.py` | nouveau |
| `pilot/signs_arch.py` | nouveau |
| `scripts/train_signs.py` | nouveau |
| `models/signs_cls.pth` + `signs_cls_history.json` | nouveaux |
| `pilot/signs.py` | réécrit (stub 13 lignes → implémentation) |
| `pilot/main.py` | modifié (tracker + governor + `--no-signs`) |
| `scripts/test_signs.py` | nouveau |
