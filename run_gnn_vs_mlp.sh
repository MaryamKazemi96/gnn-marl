#!/bin/bash
# run_gnn_vs_mlp.sh

set -e

EPISODES=50
SEEDS=(100)

CONFIG="configs/training_config.yaml"
BACKBONES=(gnn:sage mlp:dummy)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}"

echo "Sweep folder: ${RUN_ROOT}/"
echo "Backbones: ${BACKBONES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

COMPARISON_ARGS=()

for entry in "${BACKBONES[@]}"; do
    name="${entry%%:*}"
    backbone="${entry#*:}"
    run_id="${TIMESTAMP}/${name}"

    echo "================================================================"
    echo " Backbone: $name   (--backbone $backbone, run_id: $run_id)"
    echo "================================================================"

    echo "[$name] Starting PPO training..."
    python3 train_ppo.py --config "$CONFIG" --backbone "$backbone" --run-id "$run_id"

    echo "[$name] Evaluating baseline..."
    python3 eval_baseline.py --config "$CONFIG" --episodes $EPISODES --all-seeds --run-id "$run_id"

    echo "[$name] Plotting per-seed training diagnostics..."
    for s in "${SEEDS[@]}"; do
        python3 plot_training.py --seed "$s" --run-id "$run_id"
    done

    echo "[$name] Plotting cross-seed training reward (mean +/- std band)..."
    # only meaningful with >1 seed — SEEDS currently has just one
    # python3 plot_training.py --run-id "$run_id" --multi-seed

    echo "[$name] Evaluating PPO..."
    python3 eval_ppo.py --config "$CONFIG" --all-seeds --episodes $EPISODES --run-id "$run_id"

    echo "[$name] Plotting evaluation results..."
    python3 plot_eval.py \
        --all-seeds \
        --run-id "$run_id" \
        --baseline-dir baseline_results \
        --ma-window 5

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