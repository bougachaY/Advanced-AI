"""Diagnostic script — analyse l'évolution de MMStar au fil de l'entraînement.

Usage (à lancer sur le cluster, depuis le dossier du projet) :
    python analyze_mmstar_progress.py --results_dir /tmpdir/tpirtnzzdd/eval_results

Ce script :
  1. Charge tous les fichiers mmstar_step*.json présents dans results_dir
  2. Trace l'évolution de l'accuracy globale au fil des steps
  3. Trace la distribution des lettres prédites (A/B/C/D) à chaque step
     pour détecter un mode collapse (le modèle qui répond toujours la même lettre)
  4. Affiche un résumé texte interprétable directement dans le terminal

Ne nécessite que des libs standard + matplotlib (déjà présent dans la plupart
des environnements ML). Si matplotlib n'est pas dispo, le script affiche
quand même le résumé texte avec --no_plot.
"""

import argparse
import glob
import json
import os
from collections import Counter


def load_all_results(results_dir: str):
    """Charge tous les mmstar_step*.json triés par step croissant."""
    pattern = os.path.join(results_dir, "mmstar_step*.json")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(
            os.path.basename(f).replace("mmstar_step", "").replace(".json", "")
        ),
    )
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier mmstar_step*.json trouvé dans {results_dir}"
        )

    runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        step = data["global_step"]
        metrics = data["metrics"]
        runs.append({"step": step, "metrics": metrics, "file": f})
    return runs


def prediction_distribution(metrics: dict) -> Counter:
    """Compte combien de fois chaque lettre (ou None) a été prédite."""
    counter = Counter()
    for p in metrics.get("predictions", []):
        pred = p.get("prediction")
        counter[pred if pred is not None else "invalid"] += 1
    return counter


def gold_distribution(metrics: dict) -> Counter:
    """Distribution des bonnes réponses (pour référence — vérifie le hasard théorique)."""
    counter = Counter()
    for p in metrics.get("predictions", []):
        counter[p.get("gold")] += 1
    return counter


def print_summary(runs):
    print("=" * 78)
    print(f"{'step':>6} | {'accuracy':>9} | {'invalid':>8} | distribution prédictions")
    print("-" * 78)
    for run in runs:
        step = run["step"]
        m = run["metrics"]
        acc = m["accuracy"]
        invalid = m.get("invalid", 0)
        total = m.get("total", 0)
        dist = prediction_distribution(m)
        dist_str = ", ".join(
            f"{k}={v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])
        )
        print(f"{step:>6} | {acc:>9.4f} | {invalid:>4}/{total:<3} | {dist_str}")
    print("=" * 78)

    # Interprétation automatique simple
    if len(runs) >= 2:
        first_acc = runs[0]["metrics"]["accuracy"]
        last_acc = runs[-1]["metrics"]["accuracy"]
        delta = last_acc - first_acc
        print()
        if delta > 0.05:
            print(
                f"-> Tendance: accuracy en hausse (+{delta:.3f} entre step "
                f"{runs[0]['step']} et {runs[-1]['step']}). Le modèle progresse."
            )
        elif delta < -0.02:
            print(
                f"-> Tendance: accuracy en baisse ({delta:.3f}). "
                "Surveiller — possible instabilité ou surapprentissage."
            )
        else:
            print(
                f"-> Tendance: accuracy quasi stable ({delta:+.3f}). "
                "Si le mode collapse persiste sur plusieurs evals, "
                "creuser le modality projector / flux d'info visuelle."
            )

        # Détection mode collapse sur le dernier run
        last_dist = prediction_distribution(runs[-1]["metrics"])
        last_total = sum(last_dist.values())
        if last_total > 0:
            most_common_letter, most_common_count = last_dist.most_common(1)[0]
            ratio = most_common_count / last_total
            print(
                f"-> Au dernier step ({runs[-1]['step']}): la lettre la plus "
                f"prédite est '{most_common_letter}' ({ratio:.0%} des réponses)."
            )
            if ratio > 0.6:
                print(
                    "   ATTENTION: ratio > 60% -> mode collapse probable, "
                    "le modèle ignore probablement l'image."
                )
            elif ratio < 0.35:
                print(
                    "   Bon signe: distribution assez équilibrée entre les "
                    "lettres, le modèle ne semble pas en mode collapse."
                )


def plot_progress(runs, output_path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib non disponible — utilise --no_plot ou installe-le "
            "avec `pip install matplotlib --break-system-packages`."
        )
        return

    steps = [r["step"] for r in runs]
    accs = [r["metrics"]["accuracy"] for r in runs]

    letters = ["A", "B", "C", "D", "invalid"]
    dist_over_time = {letter: [] for letter in letters}
    for run in runs:
        dist = prediction_distribution(run["metrics"])
        total = sum(dist.values()) or 1
        for letter in letters:
            dist_over_time[letter].append(dist.get(letter, 0) / total)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Plot 1 — accuracy
    ax1.plot(steps, accs, marker="o", color="#185FA5", linewidth=2)
    ax1.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="hasard (0.25)")
    ax1.set_ylabel("MMStar accuracy")
    ax1.set_title("Évolution de l'accuracy MMStar au fil de l'entraînement")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2 — distribution des prédictions (stacked area)
    colors = {
        "A": "#378ADD", "B": "#1D9E75", "C": "#BA7517",
        "D": "#D85A30", "invalid": "#888780",
    }
    if len(steps) > 1:
        min_gap = min(b - a for a, b in zip(steps, steps[1:]))
        bar_width = min_gap * 0.6
    else:
        bar_width = max(steps[0] * 0.3, 10)

    bottom = [0] * len(steps)
    for letter in letters:
        values = dist_over_time[letter]
        ax2.bar(
            steps, values, bottom=bottom,
            label=letter, color=colors[letter], width=bar_width,
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax2.axhline(0.25, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Proportion des prédictions")
    ax2.set_xlabel("step")
    ax2.set_xticks(steps)
    ax2.set_title("Distribution des lettres prédites (détection de mode collapse)")
    ax2.legend(loc="upper right", ncol=5, fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nGraphique sauvegardé : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir", type=str, default="/tmpdir/tpirtnzzdd/eval_results",
        help="Dossier contenant les fichiers mmstar_step*.json",
    )
    parser.add_argument(
        "--output", type=str, default="mmstar_progress.png",
        help="Chemin du graphique de sortie",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="N'affiche que le résumé texte, sans générer de graphique",
    )
    args = parser.parse_args()

    runs = load_all_results(args.results_dir)
    print(f"\n{len(runs)} évaluation(s) MMStar trouvée(s) dans {args.results_dir}\n")
    print_summary(runs)

    if not args.no_plot:
        plot_progress(runs, args.output)


if __name__ == "__main__":
    main()