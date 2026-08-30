#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting PPO training..."
python3 train_ppo.py --config configs/training_config.yaml

echo "Evaluating baseline..."
python3 eval_baseline.py --config configs/training_config.yaml --episodes 50 --all-seeds


echo "Evaluating PPO..."
python3 eval_ppo.py --all-seeds --episodes 50

echo "Starting Plot training..."
python3 plot_training.py --seed 100
python3 plot_training.py --seed 200
python3 plot_training.py --seed 300

echo "Plotting evaluation results..."
python3 plot_eval.py \
    --all-seeds\
    --eval-dir eval_results \
    --baseline-dir baseline_results \
    --ma-window 5

echo "All tasks completed successfully!"
