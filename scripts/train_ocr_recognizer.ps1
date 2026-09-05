[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Definition,

    [Parameter(Mandatory = $true)]
    [ValidateSet("general", "thai")]
    [string]$Track
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_ocr_training_trial.py"
$python = "D:\OCR_Model_Assets\environments\ie-ocr-train\Scripts\python.exe"
$kind = if ($Track -eq "general") { "general_recognizer" } else { "thai_recognizer" }

if (-not (Test-Path -LiteralPath $Definition -PathType Leaf)) {
    throw "Recognizer trial definition does not exist: $Definition"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "OCR training Python does not exist: $python"
}

& $python $runner --definition $Definition --kind $kind
if ($LASTEXITCODE -ne 0) {
    throw "$Track recognizer training trial failed with exit code $LASTEXITCODE"
}
