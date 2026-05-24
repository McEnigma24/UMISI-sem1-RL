Set-Location $PSScriptRoot

Remove-Item -Recurse -Force plots
New-Item -ItemType Directory -Path plots