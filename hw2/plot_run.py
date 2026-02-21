"""
plot_run.py  –  generate a learning-curve PNG for one or more experiment
                log directories.

Usage (single run, saved next to the log):
    python plot_run.py <logdir>

Usage (multi-run overlay, explicit output path):
    python plot_run.py <logdir1> <logdir2> ... --out my_figure.png --title "HalfCheetah"
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── TensorBoard event reader ──────────────────────────────────────────────────

def read_tb(logdir, tag="Eval_AverageReturn"):
    """Return (steps, values) arrays from a TensorBoard event file."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"Tag '{tag}' not found in {logdir}. Available: {available}")
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


# ── helpers ───────────────────────────────────────────────────────────────────

def exp_label(logdir):
    """Best-effort human label from the directory name."""
    name = os.path.basename(logdir.rstrip("/"))
    # strip trailing timestamp  _DD-MM-YYYY_HH-MM-SS
    name = re.sub(r"_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$", "", name)
    return name


def plot_runs(logdirs, tag, out_path, title=None, xlabel="Iteration",
              ylabel="Eval Average Return"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for logdir in logdirs:
        try:
            steps, values = read_tb(logdir, tag)
            label = exp_label(logdir)
            ax.plot(steps, values, label=label)
        except Exception as exc:
            print(f"  [warn] could not read {logdir}: {exc}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or tag)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", nargs="+",
                        help="One or more TensorBoard log directories")
    parser.add_argument("--tag",   default="Eval_AverageReturn")
    parser.add_argument("--out",   default=None,
                        help="Output PNG path (default: <logdir>/learning_curve.png "
                             "for single-run mode)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logdirs = args.logdirs

    if args.out:
        out_path = args.out
    elif len(logdirs) == 1:
        out_path = os.path.join(logdirs[0], "learning_curve.png")
    else:
        out_path = "combined_learning_curve.png"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plot_runs(logdirs, tag=args.tag, out_path=out_path, title=args.title)


if __name__ == "__main__":
    main()
