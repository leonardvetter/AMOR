#!/usr/bin/env python3
"""
Utility to regenerate summary text and figures for stabilization results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "stabilization_results" / "results.json"
OUTPUT_DIR = ROOT / "stabilization_results"


def load_results():
    with RESULTS_PATH.open("r") as fp:
        data = json.load(fp)

    records = []
    for strategy, models in data.items():
        for model_label, info in models.items():
            stats = info["stats"]
            records.append(
                {
                    "strategy": strategy,
                    "model": model_label,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "cv": stats["cv"],
                }
            )
    return records


def write_summary(records):
    by_model = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("STABILIZATION STRATEGIES COMPARISON")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("Source: stabilization_results/results.json")
    summary_lines.append("Metrics: best validation MSE (lower is better)")
    summary_lines.append("")

    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda r: r["mean"])
        best = rows_sorted[0]
        summary_lines.append(f"Model: {model}")
        summary_lines.append(
            f"  Best strategy: {best['strategy']} "
            f"(mean={best['mean']:.4e}, std={best['std']:.4e}, min={best['min']:.4e})"
        )
        summary_lines.append("")
        summary_lines.append(
            "  Strategy                  Mean       Std        Min        Max        CV"
        )
        summary_lines.append("  " + "-" * 66)
        for row in rows_sorted:
            summary_lines.append(
                f"  {row['strategy']:<25} "
                f"{row['mean']:<10.4e} {row['std']:<10.4e} "
                f"{row['min']:<10.4e} {row['max']:<10.4e} {row['cv']:<10.4f}"
            )
        summary_lines.append("")

    summary_lines.append("=" * 70)
    summary_lines.append("Legend:")
    summary_lines.append("  CV = coefficient of variation (std/mean)")
    summary_lines.append("=" * 70)

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))


def plot_strategy_comparison(records):
    strategies = sorted({r["strategy"] for r in records})
    models = sorted({r["model"] for r in records})

    means = {model: [] for model in models}
    for strategy in strategies:
        strategy_rows = [r for r in records if r["strategy"] == strategy]
        for model in models:
            value = next((r["mean"] for r in strategy_rows if r["model"] == model), np.nan)
            means[model].append(value)

    x = np.arange(len(strategies))
    width = 0.35

    plt.figure(figsize=(12, 5))
    for idx, model in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        plt.bar(x + offset, means[model], width, label=model)

    plt.xticks(x, strategies, rotation=45, ha="right")
    plt.ylabel("Mean best val MSE")
    plt.title("Mean validation loss per strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_comparison.png", dpi=200)
    plt.close()


def plot_mean_vs_std(records):
    plt.figure(figsize=(12, 5))
    models = sorted({r["model"] for r in records})
    colors = plt.get_cmap("tab10")

    for idx, model in enumerate(models):
        subset = [r for r in records if r["model"] == model]
        x = [r["mean"] for r in subset]
        y = [r["std"] for r in subset]
        labels = [r["strategy"] for r in subset]
        plt.scatter(x, y, label=model, color=colors(idx))
        for xi, yi, label in zip(x, y, labels):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    plt.xlabel("Mean best val MSE")
    plt.ylabel("Std of best val MSE")
    plt.title("Mean vs Std of validation loss per strategy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(OUTPUT_DIR / "mean_vs_std.png", dpi=200)
    plt.close()


def plot_ranked_means(records):
    plt.figure(figsize=(12, 5))
    models = sorted({r["model"] for r in records})
    colors = plt.get_cmap("tab10")

    for idx, model in enumerate(models):
        subset = sorted([r for r in records if r["model"] == model], key=lambda r: r["mean"])
        x = np.arange(len(subset))
        y = [r["mean"] for r in subset]
        labels = [r["strategy"] for r in subset]
        plt.plot(x, y, marker="o", color=colors(idx), label=model)
        for xi, yi, label in zip(x, y, labels):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    plt.xticks([])
    plt.ylabel("Mean best val MSE")
    plt.title("Ranked strategies by mean validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves_comparison.png", dpi=200)
    plt.close()


def main():
    records = load_results()
    write_summary(records)
    plot_strategy_comparison(records)
    plot_mean_vs_std(records)
    plot_ranked_means(records)
    print("Summary and figures updated.")


if __name__ == "__main__":
    main()


