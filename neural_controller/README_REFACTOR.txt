NEURAL ACTIVE CONTROLLER REFACTOR
===================================

Primary path
------------
train_multitask_controller.py
        |
        +--> fault type model
        |
        +--> affected PMU model
        |
        v
multitask_active_controller.py
        |
        +--> confidence filtering
        +--> 2-window activation persistence
        +--> 3-window recovery persistence
        +--> PMU-level weights
        +--> 12-element measurement weights
        |
        v
wls_neural.py

Files replaced
--------------
1. neural_controller/multitask_active_controller.py
2. neural_controller/train_multitask_controller.py
3. neural_controller/wls_neural.py

Files deliberately NOT replaced
--------------------------------
action_controller.py
train_action_controller.py

Those remain the action-only baseline for comparison.

Important change
----------------
SYNC no longer claims to down-weight "phase data" while actually applying a
single weight to all four PMU measurements. The new controller produces
12 measurement weights:

[Vmag1,Vang1,Imag1,Iang1, Vmag2,Vang2,Imag2,Iang2,
 Vmag3,Vang3,Imag3,Iang3]

For SYNC on PMU2, for example:
[1.0,0.1,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]

For BAD_DATA on PMU2:
[1.0,1.0,1.0,1.0, 0.1,0.1,0.1,0.1, 1.0,1.0,1.0,1.0]

For CLOCK_DRIFT on PMU2:
[1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0]

The WLS solver accepts either 3 PMU weights or all 12 measurement weights.

Install
-------
Copy the three files into:
Neural_Active_Fault_Controller_Code/neural_controller/

Then retrain the primary model:

cd neural_controller
python3 train_multitask_controller.py ../scenario_data/*.csv

This creates:
neural_active_controller.joblib
neural_active_controller.json

Test a scenario:

python3 multitask_active_controller.py \
    ../scenario_data/PMU2_sync_r03.csv \
    neural_active_controller.joblib \
    --output PMU2_sync_neural_active_results.csv

The output contains:
raw neural prediction, confidence, temporal management state,
active fault type/PMU, management action, PMU weights, and 12 measurement
weights.

Do not use simulator truth during inference.
