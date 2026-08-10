#!/bin/bash
# Exit immediately if a command fails
set -e

CONFIG="configs/training_config.yaml"
EPISODES=50
MA_WINDOW=5

# One shared run_id (timestamp) for the whole sweep, so every resolver's
# results end up nested under the SAME runs/run_{TIMESTAMP}/ folder:
#
#   runs/run_20260803_140000/
#     greedy/seed_42/  greedy/seed_123/
#     random/seed_42/  random/seed_123/
#     hungarian/...
#     hungarian_bids/...
#
# instead of one top-level runs/run_{resolver}/ per resolver. Every script
# below already builds its path as Path("runs") / f"run_{run_id}", and
# pathlib treats a run_id containing "/" as nested components — so passing
# "TIMESTAMP/resolver" as --run-id is all that's needed; no script changes
# required. Every command explicitly passes --run-id, so nothing here ever
# relies on "find the latest sweep" auto-discovery, which wouldn't handle
# this nested layout the same way.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RESOLVERS=(random hungarian hungarian_bids)

# Seeds to plot individually after each resolver's sweep (matches your
# original script's explicit seed list; training/eval itself still uses
# whatever `seeds:` is set to in $CONFIG for all seeds in the sweep).
PLOT_SEEDS=(42)
# PLOT_SEEDS+=(456)

# resolver_name -> run_id ("TIMESTAMP/resolver"), used everywhere below
# and passed as-is to compare_conflict_resolvers.py at the end.
declare -A RUN_IDS
for r in "${RESOLVERS[@]}"; do
    RUN_IDS[$r]="${TIMESTAMP}/${r}"
done

echo "Sweep folder: runs/run_${TIMESTAMP}/"
echo

for r in "${RESOLVERS[@]}"; do
    run_id="${RUN_IDS[$r]}"

    echo "================================================================"
    echo " Conflict resolution: $r   (run_id: $run_id)"
    echo "================================================================"

    echo "[$r] Starting PPO training..."
    python3 train_ppo.py --config "$CONFIG" --conflict-resolution "$r" --run-id "$run_id"

    echo "[$r] Evaluating baseline..."
    python3 eval_baseline.py --config "$CONFIG" --episodes $EPISODES --all-seeds \
        --conflict-resolution "$r" --run-id "$run_id"

    echo "[$r] Plotting training..."
    for s in "${PLOT_SEEDS[@]}"; do
        python3 plot_training.py --seed "$s" --run-id "$run_id"
    done

    echo "[$r] Evaluating PPO..."
    python3 eval_ppo.py --all-seeds --episodes $EPISODES --run-id "$run_id"

    echo "[$r] Plotting evaluation results..."
    python3 plot_eval.py \
        --all-seeds \
        --run-id "$run_id" \
        --baseline-dir baseline_results \
        --ma-window $MA_WINDOW

    echo "[$r] Done."
    echo
done

echo "================================================================"
echo " Comparing all conflict-resolution mechanisms"
echo "================================================================"

RESOLVER_ARGS=()
for r in "${RESOLVERS[@]}"; do
    RESOLVER_ARGS+=(--resolver "${r}:${RUN_IDS[$r]}")
done

python3 compare_conflict_resolvers.py "${RESOLVER_ARGS[@]}" \
    --output-dir "runs/run_${TIMESTAMP}/conflict_resolver_comparison"

echo "All tasks completed successfully!"
echo "Sweep folder: runs/run_${TIMESTAMP}/"
echo "Cross-resolver comparison: runs/run_${TIMESTAMP}/conflict_resolver_comparison/"