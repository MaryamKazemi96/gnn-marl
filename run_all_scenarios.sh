#!/bin/bash
# run_all_scenarios.sh
#
# Runs the full train -> baseline-eval -> plot-training -> ppo-eval ->
# plot-eval pipeline for GNN and MLP (+ every baseline), once per scenario
# (baseline / corridor / wave). Requires generate_scenarios.sh to have
# already been run (needs data/scenario_{baseline,corridor,wave}/instance_1/
# to exist).
#
# Uses INSTANCE 1 of each scenario only, not all 3 instances — 3 scenarios
# x 2 backbones x 3 seeds is already 18 training runs; going to 3 instances
# per scenario would triple that to 54. Confirm the headline result holds
# per scenario first; extend to more instances as a later, separate pass
# if you need the extra statistical robustness (just change INSTANCE
# below and re-run under a new RUN_ROOT).
#
# Does NOT modify your real training_config.yaml / training_config_mlp.yaml
# — generates a temporary per-scenario copy of each with only data_dir
# overridden (via a real YAML load/dump, not text editing, so this is
# robust whether or not your config already has a data_dir line).

set -e

EPISODES=50
SEEDS=(100) 
INSTANCE=1             # which data/scenario_*/instance_N to use

SCENARIOS=(corridor wave baseline)
BACKBONES=(
    "gnn:configs/training_config.yaml"
    "mlp:configs/training_config_mlp.yaml"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}"
TMP_CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_CONFIG_DIR"' EXIT

echo "Sweep folder: ${RUN_ROOT}/"
echo "Scenarios: ${SCENARIOS[*]} (instance ${INSTANCE})"
echo "Backbones: ${BACKBONES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

for scenario in "${SCENARIOS[@]}"; do
    scenario_data_dir="data/scenario_${scenario}/instance_${INSTANCE}"

    if [ ! -f "${scenario_data_dir}/agents.npy" ]; then
        echo "⚠️  ${scenario_data_dir}/agents.npy not found — did you run ./generate_scenarios.sh first?"
        exit 1
    fi

    echo "################################################################"
    echo "# SCENARIO: $scenario   (data_dir: $scenario_data_dir)"
    echo "################################################################"
    echo

    COMPARISON_ARGS=()

    for entry in "${BACKBONES[@]}"; do
        name="${entry%%:*}"
        base_config="${entry#*:}"
        run_id="${TIMESTAMP}/${scenario}/${name}"

        # Build a temp config: real config's contents, data_dir overridden
        # to this scenario's dataset. Real YAML load/dump (not sed), so
        # this works correctly whether data_dir was already set or not.
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

        echo "[$scenario/$name] Starting PPO training..."
        python3 train_ppo.py --config "$tmp_config" --run-id "$run_id"

        echo "[$scenario/$name] Evaluating baseline..."
        python3 eval_baseline.py --config "$tmp_config" --episodes $EPISODES --all-seeds --run-id "$run_id"

        echo "[$scenario/$name] Plotting per-seed training diagnostics..."
        # for s in "${SEEDS[@]}"; do
            # python3 plot_training.py --seed "$s" --run-id "$run_id"
        # done

        echo "[$scenario/$name] Plotting cross-seed training reward (mean +/- std band)..."
        # python3 plot_training.py --run-id "$run_id" --multi-seed

        echo "[$scenario/$name] Evaluating PPO..."
        # python3 eval_ppo.py --all-seeds --episodes $EPISODES --run-id "$run_id"

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

    echo "================================================================"
    echo " [$scenario] Comparing GNN vs MLP (+ baselines) for this scenario"
    echo "================================================================"

    python3 plot_policy_resolver_grid.py \
        "${COMPARISON_ARGS[@]}" \
        --include-ppo \
        --output "${RUN_ROOT}/${scenario}_gnn_vs_mlp_comparison.png" \
        --title "Reward (scenario: ${scenario}): GNN vs MLP backbone (+ baselines)"

    echo
done

echo "================================================================"
echo " All scenarios completed."
echo "================================================================"
echo "Sweep folder: ${RUN_ROOT}/"
for scenario in "${SCENARIOS[@]}"; do
    echo "  ${RUN_ROOT}/${scenario}_gnn_vs_mlp_comparison.png"
done
echo
echo "Reminder: these use instance ${INSTANCE} of each scenario only."
echo "If a result looks surprising, check whether it holds on instance 2/3"
echo "before treating it as a real scenario-driven effect rather than noise."