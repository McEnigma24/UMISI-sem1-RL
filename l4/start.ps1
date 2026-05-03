Set-Location $PSScriptRoot
cls

# KERAS_BACKEND=torch;
# py --3.12 solution.py --env cartpole


$env:KERAS_BACKEND = "torch"
python solution.py --env cartpole

# py -3.13