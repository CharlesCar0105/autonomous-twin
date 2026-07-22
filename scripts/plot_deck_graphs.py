"""Graphiques de comparaison pour le deck de soutenance (style dark racing).

Produit 6 PNG transparents dans docs/soutenance/assets/graphs/, assortis au
theme du deck (fond bleu nuit). Donnees : bench/*.json committes +
models/signs_cls_history.json. Les temps oracle viennent du rapport A
(escouade du 10/07) : tour parfait theorique par circuit.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
OUT = ROOT / "docs" / "soutenance" / "assets" / "graphs"
CIRCUITS = ["gen_000", "gen_003", "gen_014"]

# Palette claire du deck (thème pro clair, 22/07). Teintes identitaires
# validées par le validateur dataviz (lightness/chroma/CVD/contraste, mode
# light) ; le gris est un neutre sémantique « avant/baseline » assumé,
# lisible grâce aux labels directs sur chaque barre.
TXT, GRID = "#1e293b", "#cbd5e1"
GRIS, BLEU, VERT, ROUGE, ORANGE = "#64748b", "#2563eb", "#16a34a", "#dc2626", "#d97706"

# Oracle : tour parfait theorique (rapport A, .superpowers/optim/)
ORACLE = {"gen_000": 16.35, "gen_003": 16.62, "gen_014": 20.57}

matplotlib.rcParams.update({
    "text.color": TXT, "axes.labelcolor": TXT, "axes.edgecolor": GRID,
    "xtick.color": TXT, "ytick.color": TXT, "axes.facecolor": "none",
    "figure.facecolor": "none", "savefig.facecolor": "none",
    "grid.color": GRID, "font.family": "sans-serif", "font.size": 13,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
})


def regles(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["circuit"]: r for r in data["results"] if r["config"] == "regles"}


def libre(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["circuit"]: r for r in data["results"] if r["config"] == "libre"}


def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(OUT / nom, dpi=200, transparent=True)
    plt.close(fig)
    print(nom)


def g1_temps(base, post):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(CIRCUITS))
    avant = [base[c]["total_time_s"] if base[c]["finished"] else base[c]["sim_time_s"] for c in CIRCUITS]
    apres = [post[c]["total_time_s"] for c in CIRCUITS]
    b1 = ax.bar(x - 0.21, avant, 0.4, color=GRIS, label="Avant optimisation",
                hatch=["" if base[c]["finished"] else "//" for c in CIRCUITS], edgecolor="#0b1220")
    b2 = ax.bar(x + 0.21, apres, 0.4, color=BLEU, label="Après (OPT-1..4)", edgecolor="#0b1220")
    for i, c in enumerate(CIRCUITS):
        lab = f"-{100 * (1 - apres[i] / avant[i]):.0f} %" if base[c]["finished"] else "DNF → 3/3"
        # +28 : au-dessus du bar_label « NNN s » pour éviter la collision
        ax.text(i, max(avant[i], apres[i]) + 28, lab, ha="center", fontweight="bold", color=VERT, fontsize=13)
    ax.bar_label(b1, fmt="%.0f s", padding=2, fontsize=10)
    ax.bar_label(b2, fmt="%.1f s", padding=2, fontsize=10)
    ax.set_xticks(x, CIRCUITS)
    ax.set_ylim(0, 345)
    ax.set_ylabel("3 tours, config règles (s)")
    ax.legend(frameon=False)
    sauver(fig, "temps-avant-apres.png")


def g2_urgence(base, post):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(CIRCUITS))
    avant = [base[c]["pct_emergency"] for c in CIRCUITS]
    apres = [post[c]["pct_emergency"] for c in CIRCUITS]
    b1 = ax.bar(x - 0.21, avant, 0.4, color=ROUGE, label="Avant", edgecolor="#0b1220")
    b2 = ax.bar(x + 0.21, apres, 0.4, color=VERT, label="Après (seuil dynamique + planéité)", edgecolor="#0b1220")
    ax.bar_label(b1, fmt="%.1f %%", padding=2, fontsize=10)
    ax.bar_label(b2, fmt="%.1f %%", padding=2, fontsize=10)
    ax.set_xticks(x, CIRCUITS)
    ax.set_ylabel("% frames en freinage d'urgence")
    ax.legend(frameon=False)
    sauver(fig, "urgence-avant-apres.png")


def g3_oracle(post_libre, post):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(CIRCUITS))
    oracle = [ORACLE[c] for c in CIRCUITS]
    lib = [post_libre[c]["best_lap_s"] for c in CIRCUITS]
    reg = [min(post[c]["lap_times_s"]) for c in CIRCUITS]
    ax.bar(x - 0.27, oracle, 0.25, color=ORANGE, label="Oracle (tour parfait théorique)", edgecolor="#0b1220")
    ax.bar(x, lib, 0.25, color=BLEU, label="PID libre (best lap)", edgecolor="#0b1220")
    ax.bar(x + 0.27, reg, 0.25, color=GRIS, label="PID règles (best lap)", edgecolor="#0b1220")
    for i in range(len(CIRCUITS)):
        ax.text(i - 0.02, lib[i] + 0.4, f"+{100 * (lib[i] / oracle[i] - 1):.0f} %",
                ha="center", fontsize=10, color=BLEU)
    ax.set_xticks(x, CIRCUITS)
    ax.set_ylabel("Temps au tour (s)")
    ax.set_title("Le PID libre roule à 1-4 % de l'oracle — l'écart règles est le coût du code de la route", fontsize=12)
    ax.legend(frameon=False)
    sauver(fig, "oracle.png")


def g4_accuracy(hist):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ep = [h["epoch"] for h in hist["history"]]
    ax.plot(ep, [h["train_acc"] for h in hist["history"]], color=GRIS, lw=2, label="Train")
    ax.plot(ep, [h["val_acc"] for h in hist["history"]], color=VERT, lw=2.5, label="Validation (circuits jamais vus)")
    best = hist["best_val_acc"]
    ax.axhline(best, color=VERT, ls="--", lw=1, alpha=0.6)
    ax.text(ep[-1], best + 0.006, f"best {best:.4f}", ha="right", color=VERT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.legend(frameon=False, loc="lower right")
    sauver(fig, "accuracy-panneaux.png")


def g5_confusion(hist):
    m = np.array(hist["confusion_val"], dtype=float)
    pct = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    classes = hist["classes"]
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = int(m[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="#0b1220" if pct[i, j] > 0.5 else TXT)
    fig.colorbar(im, ax=ax, shrink=0.85)
    sauver(fig, "confusion-panneaux.png")


def g6_physique(post_libre, p100_libre):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(CIRCUITS))
    v70 = [post_libre[c]["best_lap_s"] for c in CIRCUITS]
    v100 = [p100_libre[c]["best_lap_s"] for c in CIRCUITS]
    b1 = ax.bar(x - 0.21, v70, 0.4, color=GRIS, label="Plafond 70,6 km/h (physique d'origine)", edgecolor="#0b1220")
    b2 = ax.bar(x + 0.21, v100, 0.4, color=VERT, label="Plafond 100 km/h (mur déplacé)", edgecolor="#0b1220")
    ax.bar_label(b1, fmt="%.1f s", padding=2, fontsize=10)
    ax.bar_label(b2, fmt="%.1f s", padding=2, fontsize=10)
    ax.set_xticks(x, CIRCUITS)
    ax.set_ylabel("Best lap libre (s)")
    ax.legend(frameon=False)
    sauver(fig, "physique-100.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = regles(BENCH / "2026-07-10_baseline.json")
    post = regles(BENCH / "2026-07-10_post-optim.json")
    post_l = libre(BENCH / "2026-07-10_post-optim.json")
    p100_l = libre(BENCH / "2026-07-10_physics100.json")
    hist = json.loads((ROOT / "models" / "signs_cls_history.json").read_text(encoding="utf-8"))
    g1_temps(base, post)
    g2_urgence(base, post)
    g3_oracle(post_l, post)
    g4_accuracy(hist)
    g5_confusion(hist)
    g6_physique(post_l, p100_l)
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
