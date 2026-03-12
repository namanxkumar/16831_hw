#!/bin/bash
#SBATCH --job-name=hw3
#SBATCH --partition=debug
#SBATCH --output=/home/namankum/16831_hw/hw3/logs/sbatch_%j.log
#SBATCH --error=/home/namankum/16831_hw/hw3/logs/sbatch_%j.log
#SBATCH --account=kfragki2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G

# Run all HW3 experiments in parallel.
# Can be submitted directly to Slurm:  sbatch run_experiments.sh
# Or run interactively:                bash run_experiments.sh

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV_DIR="/home/namankum/16831_hw/hw2/.venv"
PYTHON="${VENV_DIR}/bin/python"
DQN_SCRIPT="rob831/scripts/run_hw3_dqn.py"
AC_SCRIPT="rob831/scripts/run_hw3_actor_critic.py"

export MUJOCO_GL=egl
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1

cd "$WORK_DIR"
mkdir -p logs plots

# ── helpers ───────────────────────────────────────────────────────────────────

run_dqn() {
    local exp_name="$1"; shift
    echo "  Launching DQN: $exp_name"
    "$PYTHON" -u "$DQN_SCRIPT" "$@" --exp_name "$exp_name" \
        > "logs/${exp_name}.log" 2>&1 &
}

run_ac() {
    local exp_name="$1"; shift
    echo "  Launching AC: $exp_name"
    "$PYTHON" -u "$AC_SCRIPT" "$@" --exp_name "$exp_name" \
        > "logs/${exp_name}.log" 2>&1 &
}

# ── Launch all experiments in parallel ────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  Launching all HW3 experiments in parallel"
echo "  Logs: logs/<exp_name>.log"
echo "════════════════════════════════════════════════════════"

# Q1: DQN (3 seeds)
run_dqn q1_dqn_1 --env_name LunarLander-v3 --seed 1
run_dqn q1_dqn_2 --env_name LunarLander-v3 --seed 2
run_dqn q1_dqn_3 --env_name LunarLander-v3 --seed 3

# Q1: Double DQN (3 seeds)
run_dqn q1_doubledqn_1 --env_name LunarLander-v3 --double_q --seed 1
run_dqn q1_doubledqn_2 --env_name LunarLander-v3 --double_q --seed 2
run_dqn q1_doubledqn_3 --env_name LunarLander-v3 --double_q --seed 3

# Q2: Actor-Critic CartPole
run_ac q2_10_10 --env_name CartPole-v0 -n 100 -b 1000 -ntu 10 -ngsptu 10

# Q3: Actor-Critic InvertedPendulum
run_ac q3_10_10 --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.95 \
    -n 100 -l 2 -s 64 -b 5000 -lr 0.01 -ntu 10 -ngsptu 10
echo "  All launched. Waiting for completion..."
wait
echo "  All experiments finished."

# ── Generate plots ────────────────────────────────────────────────────────────

echo "  Generating plots..."

# Q1: DQN vs DDQN averaged across 3 seeds with std error bars
"$PYTHON" plot_run.py --avg_seeds \
    --prefix q1_dqn q1_doubledqn \
    --tag Train_AverageReturn \
    --xlabel "Timesteps" \
    --ylabel "Average Return" \
    --out plots/q1_dqn_vs_ddqn.png \
    --title "DQN vs Double DQN on LunarLander-v3"

# Q2: Actor-Critic CartPole
q2_dir=$(ls -dt "${WORK_DIR}/data/q2_10_10_"* 2>/dev/null | head -1)
if [[ -n "$q2_dir" ]]; then
    "$PYTHON" plot_run.py "$q2_dir" \
        --tag Eval_AverageReturn \
        --xlabel "Iteration" \
        --ylabel "Eval Average Return" \
        --out plots/q2_cartpole.png \
        --title "Actor-Critic on CartPole-v0 (ntu=10, ngsptu=10)"
fi

# Q3: Actor-Critic InvertedPendulum
q3_dir=$(ls -dt "${WORK_DIR}/data/q3_10_10_"* 2>/dev/null | head -1)
if [[ -n "$q3_dir" ]]; then
    "$PYTHON" plot_run.py "$q3_dir" \
        --tag Eval_AverageReturn \
        --xlabel "Iteration" \
        --ylabel "Eval Average Return" \
        --out plots/q3_inverted_pendulum.png \
        --title "Actor-Critic on InvertedPendulum-v4 (ntu=10, ngsptu=10)"
fi

echo "════════════════════════════════════════════════════════"
echo "  All done. Plots saved to plots/"
echo "════════════════════════════════════════════════════════"
