import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_latest_run(data_dir, prefix):
    matches = sorted(glob.glob(os.path.join(data_dir, f"hw4_{prefix}_*")))
    if not matches:
        # Exploration runs are stored as hw4_part2_expl_{prefix}_* etc.
        matches = sorted(glob.glob(os.path.join(data_dir, f"hw4_*_{prefix}_*")))
    if not matches:
        return None
    return matches[-1]


def read_tag(logdir, tag):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Tag {tag} not found in {logdir}. Available: {tags}")
    events = ea.Scalars(tag)
    x = np.array([e.step for e in events])
    y = np.array([e.value for e in events])
    return x, y


def plot_overlay(data_dir, prefixes, tag, out_path, title, xlabel="Iteration", ylabel=None):
    plt.figure(figsize=(8, 5))
    for prefix in prefixes:
        logdir = find_latest_run(data_dir, prefix)
        if logdir is None:
            print(f"[warn] missing run for {prefix}")
            continue
        try:
            x, y = read_tag(logdir, tag)
            plt.plot(x, y, label=prefix)
        except Exception as exc:
            print(f"[warn] failed reading {prefix}: {exc}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or tag)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved {out_path}")


def plot_train_eval_single(data_dir, prefix, out_path, title):
    logdir = find_latest_run(data_dir, prefix)
    if logdir is None:
        print(f"[warn] missing run for {prefix}")
        return
    x1, y1 = read_tag(logdir, "Train_AverageReturn")
    x2, y2 = read_tag(logdir, "Eval_AverageReturn")
    plt.figure(figsize=(7, 4.5))
    plt.plot(x1, y1, marker="o", label="Train_AverageReturn")
    plt.plot(x2, y2, marker="o", label="Eval_AverageReturn")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"saved {out_path}")


def copy_prediction_plot(data_dir, prefix, out_path):
    logdir = find_latest_run(data_dir, prefix)
    if logdir is None:
        print(f"[warn] missing run for {prefix}")
        return
    src = os.path.join(logdir, "itr_0_predictions.png")
    if not os.path.exists(src):
        print(f"[warn] missing {src}")
        return
    shutil.copy2(src, out_path)
    print(f"copied {src} -> {out_path}")


def combine_two_images(img_a, img_b, out_path, title_a, title_b):
    if not os.path.exists(img_a) or not os.path.exists(img_b):
        print(f"[warn] missing image(s): {img_a}, {img_b}")
        return
    a = plt.imread(img_a)
    b = plt.imread(img_b)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(a)
    axes[0].set_title(title_a)
    axes[0].axis("off")
    axes[1].imshow(b)
    axes[1].set_title(title_b)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="rob831/data")
    parser.add_argument("--out_dir", default="plots")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Problem 1
    copy_prediction_plot(data_dir, "q1_cheetah_n500_arch1x16", os.path.join(out_dir, "p1_plot1.png"))
    copy_prediction_plot(data_dir, "q1_cheetah_n10_arch2x200", os.path.join(out_dir, "p1_plot2.png"))
    copy_prediction_plot(data_dir, "q1_cheetah_n500_arch2x200", os.path.join(out_dir, "p1_plot3.png"))

    # Problem 2
    plot_train_eval_single(
        data_dir,
        "q2_obstacles_singleiteration",
        os.path.join(out_dir, "p2_plot1.png"),
        "Q2 Obstacles: Train vs Eval Return",
    )

    # Problem 3
    plot_overlay(data_dir, ["q3_obstacles"], "Eval_AverageReturn", os.path.join(out_dir, "p3_plot1.png"),
                 "Q3 Obstacles: Eval Return")
    plot_overlay(data_dir, ["q3_reacher"], "Eval_AverageReturn", os.path.join(out_dir, "p3_plot2.png"),
                 "Q3 Reacher: Eval Return")
    plot_overlay(data_dir, ["q3_cheetah"], "Eval_AverageReturn", os.path.join(out_dir, "p3_plot3.png"),
                 "Q3 Cheetah: Eval Return")

    # Problem 4
    plot_overlay(data_dir, ["q4_reacher_ensemble1", "q4_reacher_ensemble3", "q4_reacher_ensemble5"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p4_plot1.png"),
                 "Q4 Effect of Ensemble Size")
    plot_overlay(data_dir, ["q4_reacher_numseq100", "q4_reacher_numseq1000"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p4_plot2.png"),
                 "Q4 Effect of Number of Action Sequences")
    plot_overlay(data_dir, ["q4_reacher_horizon5", "q4_reacher_horizon15", "q4_reacher_horizon30"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p4_plot3.png"),
                 "Q4 Effect of MPC Horizon")

    # Problem 5
    plot_overlay(data_dir, ["q5_cheetah_random", "q5_cheetah_cem_2", "q5_cheetah_cem_4"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p5_plot1.png"),
                 "Q5 Random Shooting vs CEM")

    # Problem 6 learning curves
    plot_overlay(data_dir, ["q6_env1_random", "q6_env1_rnd"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p6_plot1.png"),
                 "Q6 PointmassEasy: Random vs RND")
    plot_overlay(data_dir, ["q6_env2_random", "q6_env2_rnd"],
                 "Eval_AverageReturn", os.path.join(out_dir, "p6_plot2.png"),
                 "Q6 PointmassHard: Random vs RND")

    # Problem 6 density plots
    env1_rnd = find_latest_run(data_dir, "q6_env1_rnd")
    env1_rand = find_latest_run(data_dir, "q6_env1_random")
    env2_rnd = find_latest_run(data_dir, "q6_env2_rnd")
    env2_rand = find_latest_run(data_dir, "q6_env2_random")

    if env1_rand and env1_rnd:
        combine_two_images(
            os.path.join(env1_rand, "curr_state_density.png"),
            os.path.join(env1_rnd, "curr_state_density.png"),
            os.path.join(out_dir, "p6_plot3.png"),
            "Easy: Random",
            "Easy: RND",
        )

    if env2_rand and env2_rnd:
        combine_two_images(
            os.path.join(env2_rand, "curr_state_density.png"),
            os.path.join(env2_rnd, "curr_state_density.png"),
            os.path.join(out_dir, "p6_plot4.png"),
            "Hard: Random",
            "Hard: RND",
        )


if __name__ == "__main__":
    main()
