#!/bin/bash
# run_gnn_vs_mlp.sh

set -e

EPISODES=50
SEEDS=(100 200)

CONFIGS=(
    # "gnn:configs/training_config.yaml"
    "mlp:configs/training_config_mlp.yaml"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}"

echo "Sweep folder: ${RUN_ROOT}/"
echo "Backbones: ${CONFIGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

COMPARISON_ARGS=()

for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    config="${entry#*:}"
    run_id="${TIMESTAMP}/${name}"

    echo "================================================================"
    echo " Backbone: $name   (config: $config, run_id: $run_id)"
    echo "================================================================"

    echo "[$name] Starting PPO training..."
    python3 train_ppo.py --config "$config" --run-id "$run_id"

    echo "[$name] Evaluating baseline..."
    python3 eval_baseline.py --config "$config" --episodes $EPISODES --all-seeds --run-id "$run_id"

    echo "[$name] Plotting per-seed training diagnostics..."
    for s in "${SEEDS[@]}"; do
        python3 plot_training.py --seed "$s" --run-id "$run_id"
    done

    echo "[$name] Plotting cross-seed training reward (mean +/- std band)..."
    # python3 plot_training.py --run-id "$run_id" --multi-seed

    echo "[$name] Evaluating PPO..."
    python3 eval_ppo.py --all-seeds --episodes $EPISODES --run-id "$run_id"

    echo "[$name] Plotting evaluation results..."
    # python3 plot_eval.py \
    #     --all-seeds \
    #     --run-id "$run_id" \
    #     --baseline-dir baseline_results \
    #     --ma-window 5

    COMPARISON_ARGS+=(--resolver "${name}:${run_id}")

    echo "[$name] Done."
    echo
done

echo "================================================================"
echo " Comparing GNN vs MLP (+ baselines)"
echo "================================================================"

python3 plot_policy_resolver_grid.py \
    "${COMPARISON_ARGS[@]}" \
    --include-ppo \
    --output "${RUN_ROOT}/gnn_vs_mlp_comparison.png" \
    --title "Reward: GNN vs MLP backbone (+ baselines)"

echo
echo "All tasks completed successfully!"
echo "Sweep folder: ${RUN_ROOT}/"
echo "GNN vs MLP comparison: ${RUN_ROOT}/gnn_vs_mlp_comparison.png"
echo "GNN cross-seed training reward: ${RUN_ROOT}/gnn/multi_seed_plots/multi_seed_rewards.png"
echo "MLP cross-seed training reward: ${RUN_ROOT}/mlp/multi_seed_plots/multi_seed_rewards.png"