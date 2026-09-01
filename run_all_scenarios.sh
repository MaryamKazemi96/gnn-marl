#!/bin/bash
# run_all_scenarios.sh


CONFIG="configs/training_config.yaml"
EPISODES=50
SEEDS=(100)
INSTANCE=1             # which data/scenario_*/instance_N to use

SCENARIOS=(wave baseline)
BACKBONES=(gnn:sage mlp:dummy)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="runs/run_${TIMESTAMP}"

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
        backbone="${entry#*:}"
        run_id="${TIMESTAMP}/${scenario}/${name}"

        echo "================================================================"
        echo " [$scenario/$name] --backbone $backbone --data-dir $scenario_data_dir"
        echo " [$scenario/$name] run_id: $run_id"
        echo "================================================================"

        echo "[$scenario/$name] Starting PPO training..."
        python3 train_ppo.py \
            --config "$CONFIG" \
            --backbone "$backbone" \
            --data-dir "$scenario_data_dir" \
            --run-id "$run_id"

        echo "[$scenario/$name] Evaluating baseline..."
        python3 eval_baseline.py \
            --config "$CONFIG" \
            --data-dir "$scenario_data_dir" \
            --episodes $EPISODES \
            --all-seeds \
            --run-id "$run_id"

        echo "[$scenario/$name] Plotting per-seed training diagnostics..."
        for s in "${SEEDS[@]}"; do
            python3 plot_training.py --seed "$s" --run-id "$run_id"
        done

        echo "[$scenario/$name] Plotting cross-seed training reward (mean +/- std band)..."
        # only meaningful with >1 seed — SEEDS currently has just one
        # python3 plot_training.py --run-id "$run_id" --multi-seed

        echo "[$scenario/$name] Evaluating PPO..."
        python3 eval_ppo.py \
            --config "$CONFIG" \
            --data-dir "$scenario_data_dir" \
            --all-seeds \
            --episodes $EPISODES \
            --run-id "$run_id"

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