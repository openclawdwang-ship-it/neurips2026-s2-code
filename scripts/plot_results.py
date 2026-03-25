"""
Auto-generate all paper figures from experiment results.

Reads JSON outputs from calibrate.py and rollout_eval.py.
Generates publication-ready matplotlib figures.

Usage:
    python scripts/plot_results.py                          # All figures
    python scripts/plot_results.py --only score_comparison  # Single figure
    python scripts/plot_results.py --results_dir ./results --output_dir ./figures

Figures generated:
  1. score_comparison   — Bar chart: coverage & band width for all scores (C1 hero)
  2. per_freq_coverage  — Per-frequency-band coverage decomposition (C1 key figure)
  3. rollout_coverage   — Coverage vs timestep (C2 hero)
  4. rollout_bandwidth  — Band width vs timestep (C2)
  5. cross_architecture — Heatmap: architecture x PDE x method (C3)
  6. ablation_ncal      — Coverage & width vs calibration set size
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# NeurIPS-friendly style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SCORE_LABELS = {
    "l2": "L2",
    "spectral_uniform": "Spectral (uniform)",
    "spectral_sobolev_1": "Spectral (H\u00b9)",
    "spectral_sobolev_2": "Spectral (H\u00b2)",
    "spectral_inverse": "Spectral (inv. power)",
    "spectral_learned": "Spectral (learned)",
    "cqr_simplified": "CQR-lite",
    "mc_dropout": "MC Dropout",
}

SCORE_COLORS = {
    "l2": "#1f77b4",
    "spectral_uniform": "#aec7e8",
    "spectral_sobolev_1": "#ff7f0e",
    "spectral_sobolev_2": "#ffbb78",
    "spectral_inverse": "#2ca02c",
    "spectral_learned": "#d62728",
    "cqr_simplified": "#9467bd",
    "mc_dropout": "#7f7f7f",
}

METHOD_COLORS = {
    "static_l2": "#1f77b4",
    "static_spectral": "#ff7f0e",
    "spectral_aci": "#d62728",
}

METHOD_LABELS = {
    "static_l2": "Static L2",
    "static_spectral": "Static Spectral (H\u00b9)",
    "spectral_aci": "Spectral ACI (ours)",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig_score_comparison(results_dir, output_dir):
    """Bar chart: coverage & band width for all score functions."""
    # Find calibration results, prefer fno_darcy_64 (hero experiment)
    json_files = sorted(Path(results_dir).glob("*/calibration_results.json"))
    if not json_files:
        print("  [SKIP] No calibration_results.json found")
        return

    preferred = [f for f in json_files if "fno_darcy_64" in str(f)]
    data = load_json(preferred[0] if preferred else json_files[0])
    config_name = json_files[0].parent.name

    scores = [s for s in data if s in SCORE_LABELS]
    coverages = [data[s]["coverage"] for s in scores]
    bandwidths = [data[s]["band_width"] for s in scores]
    labels = [SCORE_LABELS.get(s, s) for s in scores]
    colors = [SCORE_COLORS.get(s, "#333") for s in scores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Coverage
    x = np.arange(len(scores))
    bars1 = ax1.bar(x, coverages, color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(0.9, color="red", linestyle="--", linewidth=1, label="Target (90%)")
    ax1.axhline(0.88, color="red", linestyle=":", linewidth=0.8, label="Tolerance (88%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title(f"Coverage — {config_name}")
    ax1.set_ylim(0.7, 1.02)
    ax1.legend(loc="lower right", fontsize=8)

    # Band width
    bars2 = ax2.bar(x, bandwidths, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("Average Band Width")
    ax2.set_title(f"Band Width — {config_name}")

    # Annotate best (lowest valid band width)
    valid = [(bw, i) for i, (bw, cov) in enumerate(zip(bandwidths, coverages)) if cov >= 0.88]
    if valid:
        best_bw, best_i = min(valid)
        bars2[best_i].set_edgecolor("red")
        bars2[best_i].set_linewidth(2)

    plt.tight_layout()
    out = output_dir / "score_comparison.pdf"
    plt.savefig(out)
    plt.savefig(output_dir / "score_comparison.png")
    plt.close()
    print(f"  [OK] {out}")


def fig_per_freq_coverage(results_dir, output_dir):
    """Per-frequency-band coverage decomposition — the C1 hero figure."""
    json_files = list(Path(results_dir).glob("*/calibration_results.json"))
    if not json_files:
        print("  [SKIP] No calibration results")
        return

    data = load_json(json_files[0])
    config_name = json_files[0].parent.name

    # Find scores that have per_freq_coverage
    scores_with_freq = [
        s for s in data if "per_freq_coverage" in data[s]
    ]
    if not scores_with_freq:
        print("  [SKIP] No per-frequency data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Per-frequency coverage
    ax = axes[0]
    for s in scores_with_freq:
        freq_cov = data[s]["per_freq_coverage"]
        x = np.arange(len(freq_cov))
        label = SCORE_LABELS.get(s, s)
        color = SCORE_COLORS.get(s, "#333")
        ax.plot(x, freq_cov, "o-", label=label, color=color, markersize=4)

    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Frequency Band (low \u2192 high)")
    ax.set_ylabel("Coverage per Band")
    ax.set_title(f"Per-Frequency Coverage — {config_name}")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 1.1)

    # Per-frequency power spectrum
    ax2 = axes[1]
    for s in scores_with_freq:
        if "per_freq_avg_power" in data[s]:
            power = data[s]["per_freq_avg_power"]
            x = np.arange(len(power))
            label = SCORE_LABELS.get(s, s)
            color = SCORE_COLORS.get(s, "#333")
            ax2.semilogy(x, power, "s-", label=label, color=color, markersize=4)

    ax2.set_xlabel("Frequency Band (low \u2192 high)")
    ax2.set_ylabel("Avg Error Power (log)")
    ax2.set_title("Error Power Spectrum by Band")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out = output_dir / "per_freq_coverage.pdf"
    plt.savefig(out)
    plt.savefig(output_dir / "per_freq_coverage.png")
    plt.close()
    print(f"  [OK] {out}")


def fig_rollout_coverage(results_dir, output_dir):
    """Coverage vs timestep — the C2 hero figure."""
    rollout_dirs = list(Path(results_dir).glob("rollout_*/rollout_results.json"))
    if not rollout_dirs:
        print("  [SKIP] No rollout results")
        return

    for rfile in rollout_dirs:
        data = load_json(rfile)
        config_name = rfile.parent.name

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for method, metrics in data.items():
            if "coverage" not in metrics:
                continue
            t = np.array(metrics["timesteps"])
            cov = np.array(metrics["coverage"])
            bw = np.array(metrics["band_width"])
            label = METHOD_LABELS.get(method, method)
            color = METHOD_COLORS.get(method, "#333")

            ax1.plot(t, cov, "o-", label=label, color=color, markersize=3, linewidth=1.5)
            ax2.plot(t, bw, "o-", label=label, color=color, markersize=3, linewidth=1.5)

        # Coverage plot
        ax1.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Target (90%)")
        ax1.set_xlabel("Rollout Timestep")
        ax1.set_ylabel("Empirical Coverage")
        ax1.set_title(f"Coverage vs Timestep — {config_name}")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.1)

        # Band width plot
        ax2.set_xlabel("Rollout Timestep")
        ax2.set_ylabel("Band Width")
        ax2.set_title(f"Band Width vs Timestep — {config_name}")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        out = output_dir / f"rollout_{config_name}.pdf"
        plt.savefig(out)
        plt.savefig(output_dir / f"rollout_{config_name}.png")
        plt.close()
        print(f"  [OK] {out}")


def fig_cross_architecture(results_dir, output_dir):
    """Heatmap: architecture x score — coverage and band width."""
    json_files = list(Path(results_dir).glob("*/calibration_results.json"))
    if len(json_files) < 2:
        print("  [SKIP] Need >=2 calibration results for cross-architecture comparison")
        return

    # Collect all data
    all_data = {}
    for jf in json_files:
        config = jf.parent.name  # e.g., "fno_darcy_64"
        all_data[config] = load_json(jf)

    # Find common scores
    all_scores = set()
    for d in all_data.values():
        all_scores.update(d.keys())
    common_scores = sorted(s for s in all_scores if s in SCORE_LABELS)

    configs = sorted(all_data.keys())

    # Build matrices
    cov_matrix = np.full((len(configs), len(common_scores)), np.nan)
    bw_matrix = np.full((len(configs), len(common_scores)), np.nan)

    for i, cfg in enumerate(configs):
        for j, score in enumerate(common_scores):
            if score in all_data[cfg]:
                cov_matrix[i, j] = all_data[cfg][score].get("coverage", np.nan)
                bw_matrix[i, j] = all_data[cfg][score].get("band_width", np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, len(configs) * 0.8)))

    score_labels = [SCORE_LABELS.get(s, s) for s in common_scores]

    # Coverage heatmap
    im1 = ax1.imshow(cov_matrix, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(len(common_scores)))
    ax1.set_xticklabels(score_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels(configs, fontsize=9)
    ax1.set_title("Empirical Coverage")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(common_scores)):
            v = cov_matrix[i, j]
            if not np.isnan(v):
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    # Band width heatmap
    im2 = ax2.imshow(bw_matrix, cmap="YlOrRd_r", aspect="auto")
    ax2.set_xticks(range(len(common_scores)))
    ax2.set_xticklabels(score_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(configs)))
    ax2.set_yticklabels(configs, fontsize=9)
    ax2.set_title("Average Band Width (lower = better)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for i in range(len(configs)):
        for j in range(len(common_scores)):
            v = bw_matrix[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    out = output_dir / "cross_architecture.pdf"
    plt.savefig(out)
    plt.savefig(output_dir / "cross_architecture.png")
    plt.close()
    print(f"  [OK] {out}")


def fig_ablation_ncal(results_dir, output_dir):
    """Coverage and band width vs calibration set size."""
    ablation_dirs = sorted(Path(results_dir).glob("ablation_ncal_*"))
    if not ablation_dirs:
        print("  [SKIP] No ablation results (ablation_ncal_*)")
        return

    ncals = []
    data_by_ncal = {}
    for d in ablation_dirs:
        jf = list(d.glob("*/calibration_results.json"))
        if jf:
            ncal = int(d.name.split("_")[-1])
            ncals.append(ncal)
            data_by_ncal[ncal] = load_json(jf[0])

    if not ncals:
        print("  [SKIP] No ablation JSON files")
        return

    ncals = sorted(ncals)

    # Find scores present in all ablation runs
    all_scores_sets = [set(data_by_ncal[n].keys()) for n in ncals]
    common = set.intersection(*all_scores_sets) if all_scores_sets else set()
    scores = sorted(s for s in common if s in SCORE_LABELS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for s in scores:
        covs = [data_by_ncal[n][s]["coverage"] for n in ncals]
        bws = [data_by_ncal[n][s]["band_width"] for n in ncals]
        label = SCORE_LABELS.get(s, s)
        color = SCORE_COLORS.get(s, "#333")

        ax1.plot(ncals, covs, "o-", label=label, color=color)
        ax2.plot(ncals, bws, "o-", label=label, color=color)

    ax1.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Calibration Set Size")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage vs Cal Set Size")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Calibration Set Size")
    ax2.set_ylabel("Band Width")
    ax2.set_title("Band Width vs Cal Set Size")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "ablation_ncal.pdf"
    plt.savefig(out)
    plt.savefig(output_dir / "ablation_ncal.png")
    plt.close()
    print(f"  [OK] {out}")


FIGURE_REGISTRY = {
    "score_comparison": fig_score_comparison,
    "per_freq_coverage": fig_per_freq_coverage,
    "rollout_coverage": fig_rollout_coverage,
    "cross_architecture": fig_cross_architecture,
    "ablation_ncal": fig_ablation_ncal,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--only", type=str, default=None,
                        choices=list(FIGURE_REGISTRY.keys()),
                        help="Generate only this figure")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Generating Paper Figures")
    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.output_dir}")
    print("=" * 50)

    figs = {args.only: FIGURE_REGISTRY[args.only]} if args.only else FIGURE_REGISTRY

    for name, func in figs.items():
        print(f"\n[{name}]")
        try:
            func(args.results_dir, output_dir)
        except Exception as e:
            print(f"  [FAIL] {e}")

    print(f"\nDone. Figures in {output_dir}/")
