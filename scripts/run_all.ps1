$ErrorActionPreference = "Stop"

Write-Host "`n[1/2] Building features and text SVD..." -ForegroundColor Cyan
python -m src.features --no-validate
if ($LASTEXITCODE -ne 0) {
    throw "Feature build or validation failed."
}

Write-Host "`n[2/2] Running final validation..." -ForegroundColor Cyan
python -m src.features.validate
if ($LASTEXITCODE -ne 0) {
    throw "Final validation failed."
}

Write-Host "`nAll pipeline steps completed successfully." -ForegroundColor Green
