#!/bin/bash
# run_policy_resolver_comparison.sh
#
# Reproduces the "Policy/resolver comparison" grouped bar chart (reward by
# policy and resolver) from the reference repo's plot_metrics_wide.py /
# your colleague's slides. Baselines only — no PPO training needed, since
# every baseline policy here is a fixed heuristic, not a trained model.
#
# For each resolver, evaluates ALL baseline policies currently implemented
# (random, greedy, unique, pickup_deadline, pickup_deadline_distance,
# predicted_reward, predicted_reward_joint — eval_baseline.py's POLICIES
# list, no need to enumerate them here) across every seed in $SEEDS, then
# plots them all together with plot_policy_resolver_grid.py.

set -e

CONFIG="configs/training_config.yaml"
EPISODES=50

# Must match (or be a subset of) configs/training_config.yaml's seeds: list
# — eval_baseline.py's env construction reads other settings from $CONFIG,
# but the actual seed values used here are these, independent of training.
SEEDS=(42 123)
# SEEDS+=(456)

# Every resolver baselines can meaningfully run under.
# 'greedy' is deliberately excluded even though your environment supports
# it — it's an alias for the exact same code path as 'closest_than_capacity'
# (see environment.py's _resolve_conflicts()), so running both would just
# duplicate compute for identical results.
# 'hungarian_bids' is excluded too — it requires a trained policy's logits
# as bids, which baselines don't have (see eval_baseline.py's automatic
# fallback to 'hungarian' if you ever pass it here anyway).
RESOLVERS=(capacity closest_than_capacity random hungarian predicted_reward predicted_reward_joint)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}_policy_resolver_cmp"

echo "Sweep folder: ${RUN_ROOT}/"
echo "Resolvers: ${RESOLVERS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

RESOLVER_PLOT_ARGS=()

for r in "${RESOLVERS[@]}"; do
    run_id="${TIMESTAMP}_policy_resolver_cmp/${r}"

    echo "================================================================"
    echo " Resolver: $r"
    echo "================================================================"

    # eval_baseline.py's --all-seeds discovers seed_* directories that
    # already exist under runs/run_{run_id}/ (normally created by
    # train_ppo.py) — pre-create them here since baselines don't need a
    # trained model at all.
    for s in "${SEEDS[@]}"; do
        mkdir -p "runs/run_${run_id}/seed_${s}"
    done

    echo "[$r] Evaluating baselines (${SEEDS[*]})..."
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