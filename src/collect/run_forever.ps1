# Runs a collector module until it exits cleanly; restarts it after a crash (it resumes from checkpoints).
# Usage: powershell -File src/collect/run_forever.ps1 -Module src.collect.reviews
param([Parameter(Mandatory = $true)][string]$Module)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null
$name = $Module.Split('.')[-1]
for ($i = 1; $i -le 50; $i++) {
    "$(Get-Date -Format s) start attempt $i" | Out-File -Append -Encoding utf8 "logs\$name.supervisor.log"
    python -m $Module 2>> "logs\$name.stderr.log" | Out-Null
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) exit code $code" | Out-File -Append -Encoding utf8 "logs\$name.supervisor.log"
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 60
}
