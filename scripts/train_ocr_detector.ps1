[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Definition
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_ocr_training_trial.py"
$python = "D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr-train\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Definition -PathType Leaf)) {
    throw "Detector trial definition does not exist: $Definition"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "OCR training Python does not exist: $python"
}

& $python $runner --definition $Definition --kind detector
if ($LASTEXITCODE -ne 0) {
    throw "Detector training trial failed with exit code $LASTEXITCODE"
}
