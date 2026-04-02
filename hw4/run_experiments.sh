#!/bin/bash
#SBATCH --job-name=hw4_all
#SBATCH --partition=debug
#SBATCH --output=/home/namankum/16831_hw/hw4/logs/sbatch_%j.log
#SBATCH --error=/home/namankum/16831_hw/hw4/logs/sbatch_%j.log
#SBATCH --account=kfragki2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV_PYTHON="/home/namankum/16831_hw/hw2/.venv/bin/python"
MB_SCRIPT="rob831/hw4_part1/scripts/run_hw4_mb.py"
EXPL_SCRIPT="rob831/hw4_part2/scripts/run_hw4_expl.py"

export MUJOCO_GL=egl
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$WORK_DIR"
mkdir -p logs plots rob831/data

run_mb() {
  local exp_name="$1"; shift
  echo "Launching MB: ${exp_name}"
  "$VENV_PYTHON" -u "$MB_SCRIPT" "$@" --exp_name "$exp_name" > "logs/${exp_name}.log" 2>&1 &
}

run_expl() {
  local exp_name="$1"; shift
  echo "Launching EXPL: ${exp_name}"
  "$VENV_PYTHON" -u "$EXPL_SCRIPT" "$@" --exp_name "$exp_name" > "logs/${exp_name}.log" 2>&1 &
}

echo "=== Launching all HW4 experiments in parallel ==="

# Problem 1
run_mb q1_cheetah_n500_arch1x16 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 16 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q1_cheetah_n10_arch2x200 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 10 --n_layers 2 --size 200 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q1_cheetah_n500_arch2x200 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 200 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy random

# Problem 2
run_mb q2_obstacles_singleiteration --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 25 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 15 --video_log_freq -1 --mpc_action_sampling_strategy random

# Problem 3
run_mb q3_obstacles --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 25 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 15 --n_iter 16 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q3_reacher --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 16 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q3_cheetah --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 16 --video_log_freq -1 --mpc_action_sampling_strategy random

# Problem 4
run_mb q4_reacher_horizon5 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q4_reacher_horizon15 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q4_reacher_horizon30 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q4_reacher_numseq100 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_num_action_sequences 100 --mpc_action_sampling_strategy random
run_mb q4_reacher_numseq1000 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy random
run_mb q4_reacher_ensemble1 --env_name reacher-hw4_part1-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q4_reacher_ensemble3 --env_name reacher-hw4_part1-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q4_reacher_ensemble5 --env_name reacher-hw4_part1-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy random

# Problem 5 (bonus)
run_mb q5_cheetah_random --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy random
run_mb q5_cheetah_cem_2 --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy cem --cem_iterations 2
run_mb q5_cheetah_cem_4 --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy cem --cem_iterations 4

# Problem 6 (bonus)
run_expl q6_env1_rnd --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration
run_expl q6_env1_random --env_name PointmassEasy-v0 --unsupervised_exploration
run_expl q6_env2_rnd --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration
run_expl q6_env2_random --env_name PointmassHard-v0 --unsupervised_exploration

wait

echo "=== All runs completed. Generating plots... ==="
"$VENV_PYTHON" generate_plots.py --data_dir rob831/data --out_dir plots

echo "=== Done. Plots in $WORK_DIR/plots ==="