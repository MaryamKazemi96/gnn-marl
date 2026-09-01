#!/bin/bash
# generate_scenarios.sh
#
# Generates the three scenarios matching the reference repo's structure:
#   baseline  - uniform demand over the full map (your current default)
#   corridor  - same demand pattern, but spatially restricted to a narrow
#               band, forcing route overlap (matches her "hard" scenario)
#   wave      - same total task count, but delivered in a few large bursts
#               instead of a steady drip (matches her "wave" scenario)
#
# Each scenario is regenerated N_INSTANCES times with different RNG seeds
# (matching her "-w3-1/2/3" pattern), landing in its own numbered
# subdirectory — data/scenario_{name}/instance_{i}/.
#
# Total task count is held constant across baseline/wave (80 = 10*8 =
# 4*20) so they're comparable — only the ARRIVAL PATTERN differs, not the
# total demand volume.

set -e

# generate_data.py's own sys.path.append only adds the src/ directory,
# letting it import "environment.environment" directly — but that module
# in turn does `from src.utils.ego_graph_builder import ...`, which needs
# the REPO ROOT (parent of src/) on the path, not src/ itself. Without
# this, generate_data.py crashes with "ModuleNotFoundError: No module
# named 'src'" the moment it tries to import Planner. Run this script
# from the repo root (where you'd normally run train_ppo.py etc.) for
# $(pwd) to resolve correctly.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

N_ROBOTS=6
N_INSTANCES=3   # matches reference's 3-instances-per-scenario pattern

echo "================================================================"
echo " Generating scenario: baseline (full map, steady arrivals)"
echo "================================================================"
for i in $(seq 1 $N_INSTANCES); do
    seed=$((1000 + i))
    echo "  instance $i (seed=$seed)..."
    python3 src/data_generation/generate_data.py \
        --n-batches 10 --n-tasks 8 --n-robots $N_ROBOTS \
        --release-interval 15 --region full --seed $seed \
        --output-dir "data/scenario_baseline/instance_${i}"
done

echo
echo "================================================================"
echo " Generating scenario: corridor (spatially restricted, hard contention)"
echo "================================================================"
for i in $(seq 1 $N_INSTANCES); do
    seed=$((2000 + i))
    echo "  instance $i (seed=$seed)..."
    python3 src/data_generation/generate_data.py \
        --n-batches 10 --n-tasks 8 --n-robots $N_ROBOTS \
        --release-interval 15 --region corridor --corridor-width-frac 0.15 --seed $seed \
        --output-dir "data/scenario_corridor/instance_${i}"
done

echo
echo "================================================================"
echo " Generating scenario: wave (same 80 tasks, delivered as 4 big bursts)"
echo "================================================================"
for i in $(seq 1 $N_INSTANCES); do
    seed=$((3000 + i))
    echo "  instance $i (seed=$seed)..."
    python3 src/data_generation/generate_data.py \
        --n-batches 4 --n-tasks 20 --n-robots $N_ROBOTS \
        --release-interval 40 --region full --seed $seed \
        --output-dir "data/scenario_wave/instance_${i}"
done

echo
echo "================================================================"
echo " All scenarios generated."
echo "================================================================"
echo "data/scenario_baseline/instance_{1,2,3}/"
echo "data/scenario_corridor/instance_{1,2,3}/"
echo "data/scenario_wave/instance_{1,2,3}/"
echo
echo "To train against a specific scenario, point a config's data_dir at"
echo "one of the instance directories above, e.g.:"
echo "  data_dir: data/scenario_corridor/instance_1"