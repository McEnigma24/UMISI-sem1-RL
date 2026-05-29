# Rejestruje .venv z l5 jako nazwany kernel Jupyter (widoczny w „Select Kernel” w Cursorze / VS Code).
# Uruchom z katalogu l5/:  .\register_jupyter_kernel.ps1

Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Brak $py — najpierw: python -m venv .venv  oraz  .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}
& $py -m pip install -q ipykernel
& $py -m ipykernel install --user --name=umisi-l5 --display-name="Python (UMISI l5 .venv)"
Write-Host "OK. W notebooku: Select Kernel -> Python (UMISI l5 .venv)  albo  Enter interpreter path -> $py"
