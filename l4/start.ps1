Set-Location $PSScriptRoot
cls

# KERAS_BACKEND=torch;
# py --3.12 solution.py --env cartpole

$baseName = "lunar_0_entropy"

$env:KERAS_BACKEND = "torch"
python solution.py --env lunar

# python solution.py --env cartpole
# python solution.py --env cartpole --resume cartpole_checkpoint_ep1100.keras
# python solution.py --env cartpole --resume cartpole_single_network_small_bigger_learning_rate_1\cartpole_checkpoint_ep1600.keras


# Critic-Check #
# python solution.py --env cartpole --check-critic cartpole_single_network\cartpole_final.keras
# python solution.py --env cartpole --check-critic cartpole_separate_networks_1\cartpole_final.keras
# python solution.py --env cartpole --check-critic cartpole_separate_networks_2\cartpole_final.keras
# python solution.py --env cartpole --check-critic cartpole_separate_networks_3\cartpole_final.keras

# py -3.13



# Kod do tworzenia katalogu i przenoszenia outputów
$existingDirs = Get-ChildItem -Directory | Where-Object { $_.Name -match "^$baseName(_\d+)?$" }
$numbers = @()
foreach ($dir in $existingDirs) {
    if ($dir.Name -eq $baseName) {
        $numbers += 0
    } elseif ($dir.Name -match "^$baseName_(\d+)$") {
        $numbers += [int]$matches[1]
    }
}
if ($numbers.Count -eq 0) {
    $dirName = $baseName
} else {
    $maxNum = ($numbers | Measure-Object -Maximum).Maximum
    $dirName = "$baseName_$($maxNum + 1)"
}
New-Item -ItemType Directory -Path $dirName -Force
# Przenieś pliki obrazów z katalogów plots_*
Move-Item -Path "plots_*" -Destination $dirName
# Przenieś pliki wag .keras
Move-Item "*.keras" $dirName -ErrorAction SilentlyContinue