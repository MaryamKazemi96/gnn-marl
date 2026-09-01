#!/bin/bash
# run_policy_resolver_comparison.sh
#
# Reproduces the "Policy/resolver comparison" grouped bar chart (reward by
# policy and resolver) from the reference repo's plot_metrics_wide.py /
# your colleague's slides. Baselines only — no PPO training needed, since
# every baseline policy here is a fixed heuristic, not a trained model.
#
# UPDATED: eval_baseline.py's actual eval-seed diversity now comes
# entirely from configs/training_config.yaml's seeds.eval list, applied
# ONCE per pre-created seed_* directory it finds under --all-seeds. Since
# baselines never touch a trained model, "train seed" identity is
# meaningless for them — pre-creating multiple seed_* directories (as the
# old version of this script did) just re-runs the SAME pooled-over-
# seeds.eval computation redundantly, once per directory, for zero
# statistical benefit. Now pre-creates exactly ONE placeholder directory.
# If you want MORE eval-seed diversity, add values to seeds.eval in the
# config instead — that's the actual lever now.

set -e

CONFIG="configs/training_config.yaml"
EPISODES=50


RESOLVERS=(capacity closest_than_capacity random hungarian predicted_reward predicted_reward_joint)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}_policy_resolver_cmp"

echo "Sweep folder: ${RUN_ROOT}/"
echo "Resolvers: ${RESOLVERS[*]}"
echo "Eval seeds used: whatever configs/training_config.yaml's seeds.eval currently lists"
echo

RESOLVER_PLOT_ARGS=()

for r in "${RESOLVERS[@]}"; do
    run_id="${TIMESTAMP}_policy_resolver_cmp/${r}"

    echo "================================================================"
    echo " Resolver: $r"
    echo "================================================================"

    mkdir -p "runs/run_${run_id}/seed_1"

    echo "[$r] Evaluating baselines..."
    python3 eval_baseline.py --config "$CONFIG" --episodes $EPISODES --all-seeds \
        --conflict-resolution "$r" --run-id "$run_id"

    RESOLVER_PLOT_ARGS+=(--resolver "${r}:${run_id}")

    echo "[$r] Done."
    echo
done

echo "================================================================"
echo " Plotting policy/resolver comparison"
echo "================================================================"

python3 plot_policy_resolver_grid.py \
    "${RESOLVER_PLOT_ARGS[@]}" \
    --output "${RUN_ROOT}/policy_resolver_comparison.png" \
    --title "Reward by policy and resolver"

echo
echo "All tasks completed successfully!"
echo "Chart: ${RUN_ROOT}/policy_resolver_comparison.png"