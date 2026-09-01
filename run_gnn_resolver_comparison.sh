#!/bin/bash
# run_gnn_resolver_comparison.sh
#
# Trains the GNN policy from scratch under each resolver in $RESOLVERS
# (not just re-evaluating baselines under different resolvers, like
# run_policy_resolver_comparison.sh does — this actually retrains PPO
# per resolver, since which resolver is active during TRAINING affects
# what the policy learns, not just how conflicts get settled at eval
# time). Motivated by an earlier finding this session: Hungarian-style
# REASSIGNING resolvers (predicted_reward_joint especially) can corrupt
# PPO's credit assignment during training, while non-reassigning
# resolvers like 'greedy' don't — this script tests that systematically
# instead of relying on one anecdotal comparison.
#
# hungarian_bids is included here (unlike the baselines-only resolver
# script) since it specifically needs a trained policy's own logits as
# bids — meaningless for baselines, but exactly what this script trains.
#
# Default resolver set below is deliberately small (4, not all 7 valid
# choices) since this is 4x full GNN training runs, not 4x a cheap
# baseline eval — expand RESOLVERS if you have the compute budget for
# it, informed by what the first pass shows.

set -e

CONFIG="configs/training_config.yaml"
EPISODES=50
SEEDS=(100)   # keep in sync with configs/training_config.yaml's seeds.train

RESOLVERS=(capacity closest_than_capacity greedy hungarian hungarian_bids predicted_reward predicted_reward_joint)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}_gnn_resolver_cmp"

echo "Sweep folder: ${RUN_ROOT}/"
echo "Resolvers: ${RESOLVERS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

COMPARISON_ARGS=()

for r in "${RESOLVERS[@]}"; do
    run_id="${TIMESTAMP}_gnn_resolver_cmp/${r}"

    echo "================================================================"
    echo " Resolver: $r   (run_id: $run_id)"
    echo "================================================================"

    echo "[$r] Starting PPO training (GNN, resolver=$r)..."
    python3 train_ppo.py \
        --config "$CONFIG" \
        --backbone sage \
        --conflict-resolution "$r" \
        --run-id "$run_id"

    echo "[$r] Evaluating baseline (under the SAME resolver PPO trained with)..."
    python3 eval_baseline.py \
        --config "$CONFIG" \
        --episodes $EPISODES \
        --all-seeds \
        --run-id "$run_id"

    echo "[$r] Plotting per-seed training diagnostics..."
    for s in "${SEEDS[@]}"; do
        python3 plot_training.py --seed "$s" --run-id "$run_id"
    done

    echo "[$r] Evaluating PPO..."
    python3 eval_ppo.py \
        --config "$CONFIG" \
        --all-seeds \
        --episodes $EPISODES \
        --run-id "$run_id"

    echo "[$r] Plotting evaluation results..."
    python3 plot_eval.py \
        --all-seeds \
        --run-id "$run_id" \
        --baseline-dir baseline_results \
        --ma-window 5

    COMPARISON_ARGS+=(--resolver "${r}:${run_id}")

    echo "[$r] Done."
    echo
done

echo "================================================================"
echo " Comparing GNN across resolvers (+ baselines)"
echo "================================================================"

python3 plot_policy_resolver_grid.py \
    "${COMPARISON_ARGS[@]}" \
    --include-ppo \
    --output "${RUN_ROOT}/gnn_resolver_comparison.png" \
    --title "Reward: GNN trained under different resolvers (+ baselines)"

echo
echo "All tasks completed successfully!"
echo "Chart: ${RUN_ROOT}/gnn_resolver_comparison.png"
echo
echo "Reminder: baselines are evaluated once per resolver here too (since"
echo "--conflict-resolution changes what THEY see as well) — so any"
echo "baseline whose score varies meaningfully across the resolver columns"
echo "in the chart is a genuine resolver effect, not noise. Baselines that"
echo "score identically regardless of resolver (e.g. 'random') are"
echo "expected to look flat."