"""
Autoresearch Experiment Analysis.

Reads results.tsv and produces:
- Console summary of experiment outcomes
- progress.png chart of val_bpb over time
- Ranked list of top improvements

Usage: uv run analysis.py [--output-dir data/]
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path="results.tsv"):
    df = pd.read_csv(path, sep="\t")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()
    return df


def print_summary(df):
    print(f"Total experiments: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    counts = df["status"].value_counts()
    print("Experiment outcomes:")
    print(counts.to_string())

    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_decided = n_keep + n_discard
    if n_decided > 0:
        print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
    print()

    kept = df[df["status"] == "KEEP"].copy()
    print(f"KEPT experiments ({len(kept)} total):")
    for i, row in kept.iterrows():
        bpb = row["val_bpb"]
        desc = row["description"]
        print(f"  #{i:3d}  bpb={bpb:.6f}  mem={row['memory_gb']:.1f}GB  {desc}")
    print()


def print_stats(df):
    kept = df[df["status"] == "KEEP"].copy()
    baseline_bpb = df.iloc[0]["val_bpb"]
    best_bpb = kept["val_bpb"].min()
    best_row = kept.loc[kept["val_bpb"].idxmin()]

    print(f"Baseline val_bpb:  {baseline_bpb:.6f}")
    print(f"Best val_bpb:      {best_bpb:.6f}")
    print(f"Total improvement: {baseline_bpb - best_bpb:.6f} ({(baseline_bpb - best_bpb) / baseline_bpb * 100:.2f}%)")
    print(f"Best experiment:   {best_row['description']}")
    print()

    print("Cumulative effort per improvement:")
    kept_sorted = kept.reset_index()
    for _, row in kept_sorted.iterrows():
        desc = str(row["description"]).strip()
        print(f"  Experiment #{row['index']:3d}: bpb={row['val_bpb']:.6f}  {desc}")
    print()


def print_top_hits(df):
    kept = df[df["status"] == "KEEP"].copy()
    kept["prev_bpb"] = kept["val_bpb"].shift(1)
    kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]

    hits = kept.iloc[1:].copy()
    hits = hits.sort_values("delta", ascending=False)

    print(f"{'Rank':>4}  {'Delta':>8}  {'BPB':>10}  Description")
    print("-" * 80)
    for rank, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"{rank:4d}  {row['delta']:+.6f}  {row['val_bpb']:.6f}  {row['description']}")

    print(f"\n{'':>4}  {hits['delta'].sum():+.6f}  {'':>10}  TOTAL improvement over baseline")
    print()


def plot_progress(df, output_path="progress.png"):
    fig, ax = plt.subplots(figsize=(16, 8))

    valid = df[df["status"] != "CRASH"].copy()
    valid = valid.reset_index(drop=True)

    baseline_bpb = valid.loc[0, "val_bpb"]

    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

    disc = below[below["status"] == "DISCARD"]
    ax.scatter(disc.index, disc["val_bpb"],
               c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    kept_v = below[below["status"] == "KEEP"]
    ax.scatter(kept_v.index, kept_v["val_bpb"],
               c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)

    kept_mask = valid["status"] == "KEEP"
    kept_idx = valid.index[kept_mask]
    kept_bpb = valid.loc[kept_mask, "val_bpb"]
    running_min = kept_bpb.cummin()
    ax.step(kept_idx, running_min, where="post", color="#27ae60",
            linewidth=2, alpha=0.7, zorder=3, label="Running best")

    for idx, bpb in zip(kept_idx, kept_bpb):
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."
        ax.annotate(desc, (idx, bpb),
                    textcoords="offset points",
                    xytext=(6, 6), fontsize=8.0,
                    color="#1a7a3a", alpha=0.9,
                    rotation=30, ha="left", va="bottom")

    best = kept_bpb.min()
    n_total = len(df)
    n_kept = len(df[df["status"] == "KEEP"])
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    margin = (baseline_bpb - best) * 0.15
    ax.set_ylim(best - margin, baseline_bpb + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved progress chart to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze autoresearch experiment results")
    parser.add_argument("--results", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--output-dir", default="data", help="Directory for output files")
    args = parser.parse_args()

    df = load_results(args.results)

    print_summary(df)
    print_stats(df)
    print_top_hits(df)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_progress(df, output_path=os.path.join(args.output_dir, "progress.png"))
