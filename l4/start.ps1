Set-Location $PSScriptRoot
cls

# KERAS_BACKEND=torch;
# py --3.12 solution.py --env cartpole


$env:KERAS_BACKEND = "torch"
python solution.py --env cartpole --resume cartpole_checkpoint_ep1100.keras

# py -3.13