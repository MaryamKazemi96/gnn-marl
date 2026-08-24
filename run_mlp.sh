
set -e
 
CONFIG="configs/training_config_mlp.yaml"
EPISODES=50
SEEDS=(42 123 456)
 
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${TIMESTAMP}/mlp"
RUN_ROOT="runs/run_${RUN_ID}"
 
echo "================================================================"
echo " MLP run — run_id: ${RUN_ID}"
echo "================================================================"
 
echo "[mlp] Starting PPO training..."
python3 train_ppo.py --config "$CONFIG" --run-id "$RUN_ID"
 
echo "[mlp] Evaluating baseline..."
python3 eval_baseline.py --config "$CONFIG" --episodes $EPISODES --all-seeds --run-id "$RUN_ID"
 
echo "[mlp] Plotting training..."
for s in "${SEEDS[@]}"; do
    python3 plot_training.py --seed "$s" --run-id "$RUN_ID"
done
 
echo "[mlp] Evaluating PPO..."
python3 eval_ppo.py --all-seeds --episodes $EPISODES --run-id "$RUN_ID"
 
echo "[mlp] Plotting evaluation results..."
python3 plot_eval.py --all-seeds --run-id "$RUN_ID" --baseline-dir baseline_results --ma-window 5
 
echo
echo "All tasks completed successfully!"
echo "Sweep folder: ${RUN_ROOT}/"
echo "run_id for later comparison against your GNN run: ${RUN_ID}"