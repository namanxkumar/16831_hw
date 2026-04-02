#!/usr/bin/env python3
"""
monitor.py  –  live progress table for all hw4 experiments.

Usage:
    python monitor.py
    python monitor.py --watch 30
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "rob831" / "data"
PLOTS_DIR = Path(__file__).parent / "plots"

# Expected progress targets (MB runs use iteration count; Q6 uses env-steps)
TOTALS = {
    # Q1
    "q1_cheetah_n500_arch1x16": 1,
    "q1_cheetah_n10_arch2x200": 1,
    "q1_cheetah_n500_arch2x200": 1,
    # Q2
    "q2_obstacles_singleiteration": 1,
    # Q3
    "q3_obstacles": 16,
    "q3_reacher": 16,
    "q3_cheetah": 16,
    # Q4
    "q4_reacher_horizon5": 15,
    "q4_reacher_horizon15": 15,
    "q4_reacher_horizon30": 15,
    "q4_reacher_numseq100": 15,
    "q4_reacher_numseq1000": 15,
    "q4_reacher_ensemble1": 15,
    "q4_reacher_ensemble3": 15,
    "q4_reacher_ensemble5": 15,
    # Q5
    "q5_cheetah_random": 5,
    "q5_cheetah_cem_2": 5,
    "q5_cheetah_cem_4": 5,
    # Q6
    "q6_env1_rnd": 50000,
    "q6_env1_random": 50000,
    "q6_env2_rnd": 50000,
    "q6_env2_random": 50000,
}

Q6_EXPS = {k for k in TOTALS if k.startswith("q6_")}

ENV_SUFFIXES = [
    "cheetah-hw4_part1-v0",
    "reacher-hw4_part1-v0",
    "obstacles-hw4_part1-v0",
    "PointmassEasy-v0",
    "PointmassHard-v0",
]
_ENV_PAT = re.compile(r"_(" + "|".join(re.escape(e) for e in ENV_SUFFIXES) + r")$")
_TS_PAT = re.compile(r"_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$")


def parse_datadir(dirname):
    name = _TS_PAT.sub("", dirname)
    if name.startswith("hw4_"):
        name = name[len("hw4_"):]
    if name.startswith("hw4_part2_expl_"):
        name = name[len("hw4_part2_expl_"):]

    m = _ENV_PAT.search(name)
    if m:
        env_name = m.group(1)
        exp_name = name[:m.start()]
    else:
        env_name = ""
        exp_name = name
    return exp_name, env_name


def read_scalars(logdir, tag):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return []
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]
    except Exception:
        return []


def is_running(exp_name):
    try:
        out = subprocess.check_output(["pgrep", "-f", "--", f"exp_name {exp_name}"], text=True)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def tfevents_bytes(logdir):
    return sum(f.stat().st_size for f in Path(logdir).glob("events.out.tfevents.*"))


def gather_rows():
    best = {}

    for d in sorted(DATA_DIR.glob("*"), key=lambda p: p.name):
        if not d.is_dir():
            continue

        exp_name, env_name = parse_datadir(d.name)
        if not exp_name:
            continue

        is_q6 = exp_name in Q6_EXPS
        progress_tag = "Train_EnvstepsSoFar" if is_q6 else "Eval_AverageReturn"
        metric_tag = "Eval_AverageReturn"

        progress_vals = read_scalars(d, progress_tag)
        metric_vals = read_scalars(d, metric_tag)

        if progress_vals:
            cur_progress = progress_vals[-1][0] if not is_q6 else progress_vals[-1][1]
        else:
            cur_progress = 0

        if metric_vals:
            _, ys = zip(*metric_vals)
            cur_ret = ys[-1]
            best_ret = max(ys)
        else:
            cur_ret = None
            best_ret = None

        best[exp_name] = {
            "exp_name": exp_name,
            "env_name": env_name,
            "cur": int(cur_progress),
            "total": TOTALS.get(exp_name, "?"),
            "cur_ret": cur_ret,
            "best_ret": best_ret,
            "running": is_running(exp_name),
            "event_bytes": tfevents_bytes(d),
            "is_q6": is_q6,
        }

    rows = list(best.values())
    rows.sort(key=lambda r: (not r["running"], r["exp_name"]))
    return rows


def progress_bar(cur, total, width=18):
    if total == "?" or total == 0:
        return " " * width
    frac = min(cur / total, 1.0)
    filled = max(1 if cur > 0 else 0, round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {int(frac * 100):3d}%"


def fmt_ret(v):
    return f"{v:+8.1f}" if v is not None else "       —"


def fmt_bytes(b):
    if b < 1024:
        return f"{b}B"
    if b < 1024 ** 2:
        return f"{b // 1024}KB"
    return f"{b // 1024 ** 2}MB"


def print_table(rows):
    name_w = 34
    sep = "─" * (name_w + 82)
    print(sep)
    print(f"  {'Experiment':<{name_w}} {'Progress':>20}  {'Now':>11}  {'CurRet':>8}  {'BestRet':>8}  {'Status':^8}  {'Log':>5}")
    print(sep)

    for r in rows:
        cur, total = r["cur"], r["total"]
        prog = progress_bar(cur, total)
        now = f"{cur}/{total}" if total != "?" else str(cur)

        if r["running"] and cur == 0:
            status = "\033[33mSTART\033[0m"
        elif r["running"]:
            status = "\033[32mRUN\033[0m"
        elif cur == 0:
            status = "\033[90mPEND\033[0m"
        else:
            status = "\033[90mDONE\033[0m"

        print(
            f"  {r['exp_name']:<{name_w}} {prog}  {now:>11}  {fmt_ret(r['cur_ret'])}  "
            f"{fmt_ret(r['best_ret'])}  {status}  {fmt_bytes(r['event_bytes']):>5}"
        )

    print(sep)
    n_run = sum(1 for r in rows if r["running"])
    n_done = sum(1 for r in rows if (not r["running"]) and r["cur"] > 0)
    n_plot = len(list(PLOTS_DIR.glob("*.png"))) if PLOTS_DIR.exists() else 0
    print(f"  {n_run} running  |  {n_done} finished  |  {n_plot} plots\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", "-w", type=int, default=0, help="Refresh every N seconds")
    args = parser.parse_args()

    while True:
        if args.watch:
            print("\033[2J\033[H", end="")

        print(f"  HW4 Experiment Monitor  —  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rows = gather_rows()
        if rows:
            print_table(rows)
        else:
            print("  No experiment data directories found yet.\n")

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
