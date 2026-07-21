"""Genere les graphiques avant/apres optimisation pour la soutenance.

Lit bench/2026-07-10_baseline.json et bench/2026-07-10_post-optim.json,
produit deux PNG dans bench/graphs/ : temps de tour (config "regles") et
taux de freinage d'urgence, avant/après, par circuit.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
OUT = BENCH / "graphs"
CIRCUITS = ["gen_000", "gen_003", "gen_014"]
SIM_TIMEOUT_S = 300.0  # cf bench_laps.py : plafond simulateur si DNF

sns.set_theme(style="whitegrid", palette="deep")
COL_BASELINE, COL_OPTIM = "#94a3b8", "#2563eb"


def load_regles(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    by_circuit = {r["circuit"]: r for r in data["results"] if r["config"] == "regles"}
    return {c: by_circuit[c] for c in CIRCUITS if c in by_circuit}


def plot_temps_tour(baseline, optim):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(CIRCUITS))
    width = 0.35

    heights_before, dnf = [], [False] * len(CIRCUITS)
    for i, c in enumerate(CIRCUITS):
        r = baseline[c]
        if r["finished"]:
            heights_before.append(r["total_time_s"])
        else:
            heights_before.append(r["sim_time_s"])
            dnf[i] = True
    heights_after = [optim[c]["total_time_s"] for c in CIRCUITS]

    bars_before = ax.bar(
        [i - width / 2 for i in x], heights_before, width,
        label="Avant optimisation", color=COL_BASELINE,
        hatch=["//" if d else "" for d in dnf], edgecolor="white",
    )
    bars_after = ax.bar(
        [i + width / 2 for i in x], heights_after, width,
        label="Après optimisation (OPT-1..4)", color=COL_OPTIM, edgecolor="white",
    )

    ax.set_ylim(0, 345)
    for i, c in enumerate(CIRCUITS):
        if dnf[i]:
            ax.text(i - width / 2, heights_before[i] / 2, "DNF\n(timeout\n300 s)",
                     ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            label = "DNF → 3/3 tours"
        else:
            gain = 100 * (1 - heights_after[i] / heights_before[i])
            label = f"-{gain:.0f} %"
        top = max(heights_before[i], heights_after[i])
        ax.text(i, top + 12, label, ha="center", fontsize=10, fontweight="bold", color="#1e293b")

    ax.bar_label(bars_after, fmt="%.1f s", padding=3, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(CIRCUITS)
    ax.set_ylabel("Temps total 3 tours (s)")
    ax.set_title("Temps de tour avant / après optimisation — config \"règles\" (CDC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "temps_tour_avant_apres.png", dpi=200)
    plt.close(fig)


def plot_urgence(baseline, optim):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(CIRCUITS))
    width = 0.35

    before = [baseline[c]["pct_emergency"] for c in CIRCUITS]
    after = [optim[c]["pct_emergency"] for c in CIRCUITS]

    ax.bar([i - width / 2 for i in x], before, width, label="Avant", color=COL_BASELINE, edgecolor="white")
    bars_after = ax.bar([i + width / 2 for i in x], after, width, label="Après", color=COL_OPTIM, edgecolor="white")

    ax.set_ylim(0, max(before) * 1.15)
    ax.bar_label(ax.containers[0], fmt="%.1f %%", padding=3, fontsize=8)
    ax.bar_label(bars_after, fmt="%.1f %%", padding=3, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(CIRCUITS)
    ax.set_ylabel("% des frames en freinage d'urgence")
    ax.set_title("Freinage d'urgence parasite — corrigé par le seuil dynamique (OPT-1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "urgence_avant_apres.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    baseline = load_regles(BENCH / "2026-07-10_baseline.json")
    optim = load_regles(BENCH / "2026-07-10_post-optim.json")
    plot_temps_tour(baseline, optim)
    plot_urgence(baseline, optim)
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
