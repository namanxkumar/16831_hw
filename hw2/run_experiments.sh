#!/bin/bash
#SBATCH --job-name=pg
#SBATCH --partition=debug
#SBATCH --output=/home/namankum/16831_hw/hw2/logs/sbatch_%j.log
#SBATCH --error=/home/namankum/16831_hw/hw2/logs/sbatch_%j.log
#SBATCH --account=kfragki2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G

# Run all Q7 and Q8 experiments in parallel.
# Can be submitted directly to Slurm:  sbatch run_experiments.sh
# Or run interactively:                bash run_experiments.sh
#
# Adjust #SBATCH lines to match your cluster's partition/resources.
#

# When run via sbatch, $0 is a temp file; use SLURM_SUBMIT_DIR instead.
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PYTHON="${WORK_DIR}/.venv/bin/python"
SCRIPT="rob831/scripts/run_hw2.py"

export MUJOCO_GL=egl
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export PATH="${WORK_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1

cd "$WORK_DIR"
mkdir -p logs plots

# ── helpers ───────────────────────────────────────────────────────────────────

run_exp() {
    local exp_name="$1"; shift
    echo "  Launching: $exp_name"
    "$PYTHON" -u "$SCRIPT" "$@" --exp_name "$exp_name" \
        > "logs/${exp_name}.log" 2>&1 &
}

plot_exp() {
    local exp_name="$1"
    local data_dir
    data_dir=$(ls -dt "${WORK_DIR}/data/${exp_name}_"* 2>/dev/null | head -1)
    if [[ -n "$data_dir" ]]; then
        "$PYTHON" plot_run.py "$data_dir" \
            --out "plots/${exp_name}.png" \
            --title "$exp_name"
    fi
}

group_plot() {
    local out_name="$1"; shift
    local title="$1";    shift
    local dirs=()
    for exp_name in "$@"; do
        local d
        d=$(ls -dt "${WORK_DIR}/data/${exp_name}_"* 2>/dev/null | head -1)
        [[ -n "$d" ]] && dirs+=("$d")
    done
    if [[ ${#dirs[@]} -gt 0 ]]; then
        echo "  → group plot: plots/${out_name}.png"
        "$PYTHON" plot_run.py "${dirs[@]}" \
            --out "plots/${out_name}.png" \
            --title "$title"
    fi
}

# ── Launch all experiments in parallel ───────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  Launching all HW2 experiments in parallel"
echo "  Logs: logs/<exp_name>.log"
echo "════════════════════════════════════════════════════════"

# ── Bonus: Parallelization timing (CartPole, serial vs 4-worker) ──────────────
# run_exp cartpole_serial \
#     --env_name CartPole-v1 --ep_len 500 \
#     --discount 0.99 -n 50 -l 2 -s 64 -b 5000 -lr 0.01 \
#     --reward_to_go --nn_baseline --n_workers 1

run_exp cartpole_parallel_4 \
    --env_name CartPole-v1 --ep_len 500 \
    --discount 0.99 -n 50 -l 2 -s 64 -b 5000 -lr 0.01 \
    --reward_to_go --nn_baseline --n_workers 4

# ── Bonus: Multi-gradient-steps (CartPole, 1 step vs 10 steps) ───────────────
# run_exp cartpole_1step \
#     --env_name CartPole-v1 --ep_len 500 \
#     --discount 0.99 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
#     --reward_to_go --nn_baseline \
#     --num_agent_train_steps_per_iter 1

# run_exp cartpole_10steps \
#     --env_name CartPole-v1 --ep_len 500 \
#     --discount 0.99 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
#     --reward_to_go --nn_baseline \
#     --num_agent_train_steps_per_iter 10

# # Q7.1: LunarLander
# run_exp q3_b10000_r0.005 \
#     --env_name LunarLanderContinuous-v2 --ep_len 1000 \
#     --discount 0.99 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 \
#     --reward_to_go --nn_baseline

# # Q7.2: HalfCheetah – 4 variants at default b=10000, lr=0.02
# run_exp q4_search_b10000_lr0.02 \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02

# run_exp q4_search_b10000_lr0.02_rtg \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 \
#     -rtg

# run_exp q4_search_b10000_lr0.02_nnbaseline \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 \
#     --nn_baseline

# run_exp q4_search_b10000_lr0.02_rtg_nnbaseline \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 \
#     -rtg --nn_baseline

# # Q7.2: HalfCheetah grid search – rtg+nn_baseline, 3×3 b×r
# for B in 10000 30000 50000; do
#     for R in 0.005 0.01 0.02; do
#         run_exp "q4_search_b${B}_lr${R}_rtg_nnbaseline" \
#             --env_name HalfCheetah-v4 --ep_len 150 \
#             --discount 0.95 -n 100 -l 2 -s 32 -b "$B" -lr "$R" \
#             -rtg --nn_baseline
#     done
# done

# Q7.2: HalfCheetah – 4 variants at optimal b*=50000, r*=0.02
# run_exp q4_b50000_r0.02 \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02

# run_exp q4_b50000_r0.02_rtg \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg

# run_exp q4_b50000_r0.02_nnbaseline \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 \
#     --nn_baseline

# run_exp q4_b50000_r0.02_rtg_nnbaseline \
#     --env_name HalfCheetah-v4 --ep_len 150 \
#     --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 \
#     -rtg --nn_baseline

# Q8: Hopper with GAE-lambda ∈ {0, 0.95, 0.99, 1}
# for LAM in 0 0.95 0.99 1; do
#     run_exp "q5_b2000_r0.001_lambda${LAM}" \
#         --env_name Hopper-v4 --ep_len 1000 \
#         --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
#         --reward_to_go --nn_baseline --action_noise_std 0.5 \
#         --gae_lambda "$LAM"
# done

echo "  All launched. Waiting for completion..."
wait
echo "  All experiments finished."

# ── Per-experiment plots ──────────────────────────────────────────────────────

echo "  Generating plots..."

# plot_exp q3_b10000_r0.005

# for exp in q4_search_b10000_lr0.02 q4_search_b10000_lr0.02_rtg \
#            q4_search_b10000_lr0.02_nnbaseline q4_search_b10000_lr0.02_rtg_nnbaseline; do
#     plot_exp "$exp"
# done

# for B in 10000 30000 50000; do
#     for R in 0.005 0.01 0.02; do
#         plot_exp "q4_search_b${B}_lr${R}_rtg_nnbaseline"
#     done
# done

# for exp in q4_b10000_r0.02 q4_b10000_r0.02_rtg \
#            q4_b10000_r0.02_nnbaseline q4_b10000_r0.02_rtg_nnbaseline; do
#     plot_exp "$exp"
# done

# for LAM in 0 0.95 0.99 1; do
#     plot_exp "q5_b2000_r0.001_lambda${LAM}"
# done

# ── Group overlay plots ───────────────────────────────────────────────────────

# group_plot "Q7.2.2_halfcheetah_default_variants" \
#     "HalfCheetah: 4 variants (b=10000, lr=0.02)" \
#     q4_search_b10000_lr0.02 \
#     q4_search_b10000_lr0.02_rtg \
#     q4_search_b10000_lr0.02_nnbaseline \
#     q4_search_b10000_lr0.02_rtg_nnbaseline

# group_plot "Q7.2.4_halfcheetah_grid_search" \
#     "HalfCheetah grid search (rtg+nn_baseline)" \
#     q4_search_b10000_lr0.005_rtg_nnbaseline \
#     q4_search_b10000_lr0.01_rtg_nnbaseline \
#     q4_search_b10000_lr0.02_rtg_nnbaseline \
#     q4_search_b30000_lr0.005_rtg_nnbaseline \
#     q4_search_b30000_lr0.01_rtg_nnbaseline \
#     q4_search_b30000_lr0.02_rtg_nnbaseline \
#     q4_search_b50000_lr0.005_rtg_nnbaseline \
#     q4_search_b50000_lr0.01_rtg_nnbaseline \
#     q4_search_b50000_lr0.02_rtg_nnbaseline

# group_plot "Q7.2.7_halfcheetah_optimal_variants" \
#     "HalfCheetah: 4 variants at b*=50000, r*=0.02" \
#     q4_b50000_r0.02 \
#     q4_b50000_r0.02_rtg \
#     q4_b50000_r0.02_nnbaseline \
#     q4_b50000_r0.02_rtg_nnbaseline
    
# group_plot "Q8_hopper_gae" \
#     "Hopper: GAE-lambda sweep" \
#     q5_b2000_r0.001_lambda0 \
#     q5_b2000_r0.001_lambda0.95 \
#     q5_b2000_r0.001_lambda0.99 \
#     q5_b2000_r0.001_lambda1

# ── Bonus plots ───────────────────────────────────────────────────────────────

# Parallelization: time per iteration (TimeSinceStart vs iteration)
group_plot "bonus_parallelization_time" \
    "CartPole: Serial vs 4-Worker Parallel (wall-clock time)" \
    cartpole_serial cartpole_parallel_4

# Parallelization: also compare returns to confirm equivalent learning
group_plot "bonus_parallelization_return" \
    "CartPole: Serial vs 4-Worker Parallel (eval return)" \
    cartpole_serial cartpole_parallel_4

# Multi-gradient-steps: return vs iteration
group_plot "bonus_multistep_return" \
    "CartPole: Single vs 10 Gradient Steps per Iter" \
    cartpole_1step cartpole_10steps

# Parallelization timing plot uses TimeSinceStart tag
timing_dirs=()
for exp_name in cartpole_serial cartpole_parallel_4; do
    d=$(ls -dt "${WORK_DIR}/data/${exp_name}_"* 2>/dev/null | head -1)
    [[ -n "$d" ]] && timing_dirs+=("$d")
done
if [[ ${#timing_dirs[@]} -gt 0 ]]; then
    echo "  → timing plot: plots/bonus_parallelization_time.png"
    "$PYTHON" plot_run.py "${timing_dirs[@]}" \
        --tag TimeSinceStart \
        --ylabel "Wall-Clock Time (s)" \
        --out "plots/bonus_parallelization_time.png" \
        --title "CartPole: Wall-Clock Time per Iteration (serial vs 4-worker)"
fi

echo "════════════════════════════════════════════════════════"
echo "  All done. Plots saved to plots/"
echo "════════════════════════════════════════════════════════"
