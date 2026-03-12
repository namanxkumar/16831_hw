#!/usr/bin/env python3
"""
monitor.py  –  live progress table for all hw3 experiments.

Usage:
    python monitor.py              # print once and exit
    python monitor.py --watch 30   # refresh every 30 seconds
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"

# expected total iterations per experiment prefix
# DQN: 300k timesteps logged every 10k → ~30 log entries (use timesteps as progress)
# AC: -n 100 iterations
TOTAL_ITERS = {
    "q1_dqn_1": 300000, "q1_dqn_2": 300000, "q1_dqn_3": 300000,
    "q1_doubledqn_1": 300000, "q1_doubledqn_2": 300000, "q1_doubledqn_3": 300000,
    "q2_10_10": 100,
    "q3_10_10": 100,
}

# For DQN experiments, progress is measured in env steps; for AC in iterations
DQN_EXPS = {"q1_dqn_1", "q1_dqn_2", "q1_dqn_3",
            "q1_doubledqn_1", "q1_doubledqn_2", "q1_doubledqn_3"}

ENV_SUFFIXES = [
    "LunarLander-v3",
    "CartPole-v0",
    "InvertedPendulum-v4",
]
_ENV_PAT = re.compile(
    r"_(" + "|".join(re.escape(e) for e in ENV_SUFFIXES) + r")$"
)
_TS_PAT = re.compile(r"_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$")


def parse_datadir(dirname):
    """Return (exp_name, env_name) from a data directory name."""
    name = _TS_PAT.sub("", dirname)
    m = _ENV_PAT.search(name)
    if m:
        env_name = m.group(1)
        exp_name = name[: m.start()]
    else:
        env_name = ""
        exp_name = name
    return exp_name, env_name


def read_scalars(logdir, tag):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        if tag not in available:
            return []
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except Exception:
        return []


def is_running(exp_name):
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "--", f"exp_name {exp_name}"], text=True
        )
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def tfevents_bytes(logdir):
    total = 0
    for f in Path(logdir).rglob("events.out.tfevents.*"):
        total += f.stat().st_size
    return total


def gather_rows():
    best: dict[str, dict] = {}

    for d in sorted(DATA_DIR.glob("*"), key=lambda p: p.name):
        if not d.is_dir():
            continue
        exp_name, env_name = parse_datadir(d.name)
        if not exp_name or exp_name.startswith("test") or exp_name.startswith("smoke"):
            continue

        is_dqn = exp_name in DQN_EXPS
        tag = "Train_AverageReturn" if is_dqn else "Eval_AverageReturn"
        scalars = read_scalars(d, tag)

        if scalars:
            steps, vals = zip(*scalars)
            if is_dqn:
                cur_iter = steps[-1]  # env timesteps
            else:
                cur_iter = steps[-1]  # iteration number
            cur_ret  = vals[-1]
            best_ret = max(vals)
        else:
            cur_iter = 0
            cur_ret  = None
            best_ret = None

        total = TOTAL_ITERS.get(exp_name, "?")

        best[exp_name] = dict(
            exp_name    = exp_name,
            env_name    = env_name,
            cur_iter    = cur_iter + (1 if not is_dqn else 0),
            total       = total,
            cur_ret     = cur_ret,
            best_ret    = best_ret,
            logdir      = d,
            event_bytes = tfevents_bytes(d),
            is_dqn      = is_dqn,
        )

    rows = []
    for exp_name, r in best.items():
        r["running"] = is_running(exp_name)
        r["plotted"] = any((PLOTS_DIR / f).exists()
                          for f in [f"{exp_name}.png",
                                    "q1_dqn_vs_ddqn.png",
                                    "q2_cartpole.png",
                                    "q3_inverted_pendulum.png"])
        rows.append(r)

    rows.sort(key=lambda r: (not r["running"], r["exp_name"]))
    return rows


def progress_bar(cur, total, width=20):
    if total == "?" or total == 0:
        return " " * width
    frac = min(cur / total, 1.0)
    filled = round(frac * width)
    if cur > 0 and filled == 0:
        filled = 1
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {int(frac*100):3d}%"


def fmt_val(v):
    return f"{v:+8.1f}" if v is not None else "       —"


def fmt_bytes(b):
    if b < 1024:
        return f"{b}B"
    elif b < 1024 ** 2:
        return f"{b//1024}KB"
    else:
        return f"{b//1024**2}MB"


def fmt_progress(cur, total, is_dqn):
    if total == "?":
        return str(cur)
    if is_dqn:
        return f"{cur//1000}k/{total//1000}k"
    return f"{cur}/{total}"


def print_table(rows):
    NAME_W = 36
    sep = "─" * (NAME_W + 80)
    print(sep)
    print(
        f"  {'Experiment':<{NAME_W}} {'Progress':>18}  "
        f"{'Iter/Steps':>10}  {'CurRet':>8}  {'BestRet':>8}  {'Status':^8}  {'Log':>5}"
    )
    print(sep)

    for r in rows:
        cur, total = r["cur_iter"], r["total"]
        prog = progress_bar(cur, total)
        iter_str = fmt_progress(cur, total, r["is_dqn"])

        if r["running"] and cur == 0:
            status = "\033[33mCOLLECT\033[0m"
        elif r["running"]:
            status = "\033[32mRUNNING\033[0m"
        elif cur == 0:
            status = "\033[90mPENDING\033[0m"
        else:
            status = "\033[90m  done \033[0m"

        log_size = fmt_bytes(r["event_bytes"])

        print(
            f"  {r['exp_name']:<{NAME_W}} "
            f"{prog} {iter_str:>10}  "
            f"{fmt_val(r['cur_ret'])}  "
            f"{fmt_val(r['best_ret'])}  "
            f"{status}  {log_size:>5}"
        )

    print(sep)
    n_run  = sum(1 for r in rows if r["running"])
    n_done = sum(1 for r in rows if not r["running"] and r["cur_iter"] > 0)
    print(f"  {n_run} running  |  {n_done} finished\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch", "-w", type=int, default=0, metavar="SECS",
        help="Refresh interval in seconds (0 = print once and exit)",
    )
    args = parser.parse_args()

    while True:
        if args.watch:
            print("\033[2J\033[H", end="")

        print(f"  HW3 Experiment Monitor  —  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rows = gather_rows()
        if rows:
            print_table(rows)
        else:
            print("  No experiment data directories found yet.")

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
