"""
plot_run.py  –  generate learning-curve PNGs for HW3 experiments.

Supports:
  - Single/multi-run overlay (like hw2)
  - Averaging across seeds with std error bars (for DQN/DDQN comparison)

Usage:
    python plot_run.py <logdir1> <logdir2> ...  --out my_figure.png --title "Title"

    # Average seeds by prefix (for Q1 DQN vs DDQN):
    python plot_run.py --avg_seeds --prefix q1_dqn q1_doubledqn --tag Train_AverageReturn \
        --xlabel "Timesteps" --out plots/q1.png --title "DQN vs DDQN on LunarLander-v3"
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tb(logdir, tag="Eval_AverageReturn"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"Tag '{tag}' not found in {logdir}. Available: {available}")
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def exp_label(logdir):
    name = os.path.basename(logdir.rstrip("/"))
    name = re.sub(r"_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$", "", name)
    return name


def find_dirs_by_prefix(data_dir, prefix):
    """Find all log directories matching a prefix (e.g. q1_dqn)."""
    pattern = os.path.join(data_dir, prefix + "_*")
    return sorted(glob.glob(pattern))


def plot_runs(logdirs, tag, out_path, title=None, xlabel="Iteration", ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for logdir in logdirs:
        try:
            steps, values = read_tb(logdir, tag)
            label = exp_label(logdir)
            ax.plot(steps, values, label=label)
        except Exception as exc:
            print(f"  [warn] could not read {logdir}: {exc}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else tag)
    ax.set_title(title or tag)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def plot_averaged_seeds(data_dir, prefixes, tag, out_path, title=None,
                        xlabel="Timesteps", ylabel=None):
    """Plot multiple experiment groups, averaging across seeds with std shading."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for prefix in prefixes:
        dirs = find_dirs_by_prefix(data_dir, prefix)
        if not dirs:
            print(f"  [warn] no dirs found for prefix '{prefix}' in {data_dir}")
            continue

        all_steps = []
        all_values = []
        for d in dirs:
            try:
                steps, values = read_tb(d, tag)
                all_steps.append(steps)
                all_values.append(values)
            except Exception as exc:
                print(f"  [warn] could not read {d}: {exc}")

        if not all_values:
            continue

        # Interpolate all runs to common x-axis
        min_len = min(len(s) for s in all_steps)
        common_steps = all_steps[0][:min_len]
        aligned = np.array([v[:min_len] for v in all_values])

        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)

        label = prefix.replace("_", " ").upper()
        line, = ax.plot(common_steps, mean, label=f"{label} (n={len(all_values)})")
        ax.fill_between(common_steps, mean - std, mean + std,
                        alpha=0.2, color=line.get_color())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else tag)
    ax.set_title(title or tag)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", nargs="*",
                        help="One or more TensorBoard log directories")
    parser.add_argument("--tag", default="Eval_AverageReturn")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--avg_seeds", action="store_true",
                        help="Average across seeds by prefix")
    parser.add_argument("--prefix", nargs="+", default=[],
                        help="Prefixes to group by (used with --avg_seeds)")
    parser.add_argument("--data_dir", default=None,
                        help="Data directory (default: ./data)")
    args = parser.parse_args()

    if args.avg_seeds:
        data_dir = args.data_dir or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data")
        out_path = args.out or "averaged_plot.png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plot_averaged_seeds(data_dir, args.prefix, tag=args.tag,
                            out_path=out_path, title=args.title,
                            xlabel=args.xlabel or "Timesteps",
                            ylabel=args.ylabel)
    else:
        logdirs = args.logdirs
        if not logdirs:
            parser.error("Provide logdirs or use --avg_seeds with --prefix")
        if args.out:
            out_path = args.out
        elif len(logdirs) == 1:
            out_path = os.path.join(logdirs[0], "learning_curve.png")
        else:
            out_path = "combined_learning_curve.png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plot_runs(logdirs, tag=args.tag, out_path=out_path, title=args.title,
                  xlabel=args.xlabel or "Iteration", ylabel=args.ylabel)


if __name__ == "__main__":
    main()
