Set-Location $PSScriptRoot
cls

# KERAS_BACKEND=torch;
# py --3.12 solution.py --env cartpole


$env:KERAS_BACKEND = "torch"
# python solution.py --env cartpole
# python solution.py --env cartpole --resume cartpole_checkpoint_ep1100.keras
# python solution.py --env cartpole --resume cartpole_separete_networks_2\cartpole_checkpoint_ep1600.keras


# Critic-Check #
python solution.py --env cartpole --check-critic cartpole_single_network\cartpole_final.keras
python solution.py --env cartpole --check-critic cartpole_separate_networks_1\cartpole_final.keras
python solution.py --env cartpole --check-critic cartpole_separate_networks_2\cartpole_final.keras
python solution.py --env cartpole --check-critic cartpole_separate_networks_3\cartpole_final.keras

# py -3.13
