#!/bin/bash
# reevaluate_all_scenarios.sh
#
# Re-runs ONLY evaluation + plotting for an EXISTING run_all_scenarios.sh
# sweep — training is NOT touched. Fixes the missing --data-dir bug that
# caused eval_baseline.py (and would have caused eval_ppo.py, had it been
# enabled) to silently read from the default data/ folder instead of the
# scenario-specific dataset, regardless of what data_dir was set to in
# the --config file.
#
# Requires: the sweep already exists under runs/run_{RUN_TIMESTAMP}/, with
# each scenario/backbone's training already completed (models/best_model.zip
# and ppo_final.zip present).
#
# Single seed only (100) — matches the current run; not attempting
# multi-seed here.
#
# Usage:
#   ./reevaluate_all_scenarios.sh 20260831_223726
# (pass the RUN_TIMESTAMP folder name shown under runs/run_<this>/)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <RUN_TIMESTAMP>"
    echo "  e.g.: $0 20260831_223726"
    echo "  (the folder name under runs/run_<RUN_TIMESTAMP>/)"
    exit 1
fi

RUN_TIMESTAMP="$1"
RUN_ROOT="runs/run_${RUN_TIMESTAMP}"

if [ ! -d "$RUN_ROOT" ]; then
    echo "⚠️  ${RUN_ROOT} not found."
    exit 1
fi

EPISODES=50
SEED=100
INSTANCE=1

SCENARIOS=(baseline)
BACKBONES=(
    "gnn:configs/training_config.yaml"
    "mlp:configs/training_config_mlp.yaml"
)

TMP_CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_CONFIG_DIR"' EXIT

echo "Re-evaluating sweep: ${RUN_ROOT}/"
echo "NOTE: this rebuilds each scenario/backbone's temp config fresh from"
echo "the CURRENT contents of configs/training_config.yaml and"
echo "configs/training_config_mlp.yaml. If either has been edited since"
echo "this sweep was originally trained, evaluation will run under a"
echo "config that doesn't exactly match what these models were trained"
echo "with — worth a quick diff check if anything looks off afterward."
echo

for scenario in "${SCENARIOS[@]}"; do
    scenario_data_dir="data/scenario_${scenario}/instance_${INSTANCE}"

    if [ ! -f "${scenario_data_dir}/agents.npy" ]; then
        echo "⚠️  ${scenario_data_dir}/agents.npy not found — skipping ${scenario}."
        continue
    fi

    echo "################################################################"
    echo "# SCENARIO: $scenario   (data_dir: $scenario_data_dir)"
    echo "################################################################"
    echo

    COMPARISON_ARGS=()

    for entry in "${BACKBONES[@]}"; do
        name="${entry%%:*}"
        base_config="${entry#*:}"
        run_id="${RUN_TIMESTAMP}/${scenario}/${name}"
        seed_dir="${RUN_ROOT}/${scenario}/${name}/seed_${SEED}"

        if [ ! -f "${seed_dir}/models/best_model.zip" ] && [ ! -f "${seed_dir}/ppo_final.zip" ]; then
            echo "⚠️  No trained model found at ${seed_dir} — skipping ${scenario}/${name}."
            continue
        fi

        tmp_config="${TMP_CONFIG_DIR}/${scenario}_${name}.yaml"
        python3 -c "
import yaml
with open('${base_config}') as f:
    cfg = yaml.safe_load(f)
cfg['data_dir'] = '${scenario_data_dir}'
with open('${tmp_config}', 'w') as f:
    yaml.safe_dump(cfg, f)
"

        echo "================================================================"
        echo " [$scenario/$name] config: $base_config -> $tmp_config"
        echo " [$scenario/$name] run_id: $run_id"
        echo "================================================================"

        echo "[$scenario/$name] Re-evaluating baseline (with --data-dir fix)..."
        python3 eval_baseline.py \
            --config "$tmp_config" \
            --data-dir "$scenario_data_dir" \
            --episodes $EPISODES \
            --all-seeds \
            --run-id "$run_id"

        echo "[$scenario/$name] Evaluating PPO (with --data-dir fix — never ran before)..."
        python3 eval_ppo.py \
            --config "$tmp_config" \
            --data-dir "$scenario_data_dir" \
            --all-seeds \
            --episodes $EPISODES \
            --run-id "$run_id"

        echo "[$scenario/$name] Plotting single-seed training diagnostics..."
        python3 plot_training.py --seed "$SEED" --run-id "$run_id"

        echo "[$scenario/$name] Plotting evaluation results..."
        python3 plot_eval.py \
            --all-seeds \
            --run-id "$run_id" \
            --baseline-dir baseline_results \
            --ma-window 5

        COMPARISON_ARGS+=(--resolver "${name}:${run_id}")

        echo "[$scenario/$name] Done."
        echo
    done

    if [ ${#COMPARISON_ARGS[@]} -eq 0 ]; then
        echo "⚠️  No backbones evaluated for ${scenario} — skipping comparison chart."
        continue
    fi

    echo "================================================================"
    echo " [$scenario] Comparing GNN vs MLP (+ baselines) for this scenario"
    echo "================================================================"

    python3 plot_policy_resolver_grid.py \
        "${COMPARISON_ARGS[@]}" \
        --include-ppo \
        --output "${RUN_ROOT}/${scenario}_gnn_vs_mlp_comparison_FIXED.png" \
        --title "Reward (scenario: ${scenario}, data-dir fixed): GNN vs MLP backbone (+ baselines)"

    echo
done

echo "================================================================"
echo " Re-evaluation complete."
echo "================================================================"
echo "Old (buggy, wrong-data) charts are still at:"
for scenario in "${SCENARIOS[@]}"; do
    echo "  ${RUN_ROOT}/${scenario}_gnn_vs_mlp_comparison.png"
done
echo
echo "New (corrected) charts:"
for scenario in "${SCENARIOS[@]}"; do
    echo "  ${RUN_ROOT}/${scenario}_gnn_vs_mlp_comparison_FIXED.png"
done
echo
echo "Compare old vs FIXED per scenario — if baselines now differ"
echo "meaningfully across scenarios (rather than being identical like"
echo "before), that confirms the data-dir fix actually took effect."