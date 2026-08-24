#!/usr/bin/env bash
# Clean V4.2 demonstration recording
# Run from any directory.

set -e

PROJECT="$HOME/Documents/IIT_Jammu/project_folder/DSSE-PMU-Neural"
CTRL="$PROJECT/neural_controller"

clear

echo
echo "======================================================================"
echo "        NEURAL ACTIVE FAULT MANAGEMENT CONTROLLER DESIGN"
echo "                 FINAL V4.2 DEMONSTRATION"
echo "======================================================================"
echo
sleep 3

cd "$CTRL"

echo ">>> STEP 1: FINAL V4.2 TRAINING"
echo
sleep 3

python3 train_multitask_controller_v4_2_final.py ../scenario_data/*.csv

echo
echo "======================================================================"
echo "                 TRAINING COMPLETE"
echo "======================================================================"
echo
echo "The V4.2 model has been trained and saved."
echo
sleep 5

echo
echo ">>> STEP 2: FINAL NEURAL -> WLS DEMONSTRATION"
echo
sleep 3

cd "$PROJECT"

python3 run_neural_wls_demo_final.py

echo
echo "======================================================================"
echo "                    DEMONSTRATION COMPLETE"
echo "======================================================================"
echo
echo "Neural classification -> PMU localization -> WLS weighting"
echo "-> state estimation has been demonstrated for all four cases."
echo
sleep 8
