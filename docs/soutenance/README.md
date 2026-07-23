# Soutenance M1 — support de présentation

Deck HTML autonome (Reveal.js 5.2 + anime.js 4.3, vendorés dans `vendor/`).
**Aucun réseau ni serveur requis** : double-cliquer `index.html`.

## Préparer (à lire avant la répétition)

- **`TRAME.md`** — script oral minuté (14:00 cible), quasi mot à mot par slide
  et par orateur, transitions, réserves conditionnelles, points de contrôle
  chrono (4:00 / 7:30 / 11:30).
- **`QR.md`** — 22 questions probables du jury avec réponses recommandées,
  porteur et renvoi annexe. Document interne : à réviser, **ne pas projeter**.

## Présenter

| Touche | Action |
|---|---|
| `→` / `Espace` | Slide suivante |
| `S` | Vue présentateur (notes + chrono + slide suivante) |
| `F` | Plein écran |
| `Échap` | Vue d'ensemble |
| `↓` | Entrer dans les annexes (dernière slide) |

- 13 slides principales (~15 min à 3 voix : Charles 2-4, Nohlan 5-7, Lorenzo 8-13).
- Les **annexes A1-A6** (dernière slide, flèche bas) couvrent les questions
  prévisibles — ne pas les présenter, les dégainer en Q&R.
- Les animations se rejouent à chaque entrée de slide (pratique en répétition).

## Démo live (slide 12)

Lancer AVANT la soutenance, laisser en pause en arrière-plan :

```bash
.venv/Scripts/python.exe -m simulator.main --server --circuit gen_014
.venv/Scripts/python.exe -m pilot.main --dashboard
```

Pas de backup enregistré (décision 21/07) : en cas de plantage, relancer
les deux process (~10 s).

Panneaux de la démo : le sidecar `assets/tracks/gen_014.signs.json` est
**versionné** (tirage seed 7, les 4 types : 30, 50, 90, STOP) pour que toute
l'équipe ait le même circuit. Ne pas relancer `place_signs.py` sur gen_014 ;
pour le regénérer à l'identique :
`python scripts/place_signs.py --tracks gen_014 --per-track 4 --seed 7`

## Export PDF

Ouvrir `index.html?print-pdf` dans Chrome → Imprimer → « Enregistrer en PDF »
(destination A4 paysage, marges par défaut).
