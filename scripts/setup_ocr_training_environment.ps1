[CmdletBinding()]
param(
    [string]$AssetRoot = 'D:\OCR_Model_Assets',
    [string]$Python310 = '',
    [switch]$SkipVerification
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentRoot = Join-Path $AssetRoot 'environments\ie-ocr-train'
$VendorRoot = Join-Path $AssetRoot 'vendor\PaddleOCR'
$CacheRoot = Join-Path $AssetRoot 'cache'
$TrainingPython = Join-Path $EnvironmentRoot 'Scripts\python.exe'
$ExpectedVendorCommit = 'b03f46425e8ff4442b268ce449e3eef758146cd4'

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if ((Split-Path -Leaf $EnvironmentRoot) -ne 'ie-ocr-train') {
    throw "Refusing unexpected training environment target: $EnvironmentRoot"
}
if ((Split-Path -Leaf $VendorRoot) -ne 'PaddleOCR') {
    throw "Refusing unexpected PaddleOCR vendor target: $VendorRoot"
}

$CFreeGiB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
$DFreeGiB = [math]::Round((Get-PSDrive D).Free / 1GB, 2)
if ($CFreeGiB -lt 15) {
    throw "C: free space is $CFreeGiB GiB; at least 15 GiB is required."
}
if ($DFreeGiB -lt 20) {
    throw "D: free space is $DFreeGiB GiB; at least 20 GiB is required."
}

if ([string]::IsNullOrWhiteSpace($Python310)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $Launcher) {
        $Python310 = (
            & $Launcher.Source -3.10 -c "import sys; print(sys.executable)" 2>$null |
                Select-Object -Last 1
        )
    }
}
if ([string]::IsNullOrWhiteSpace($Python310) -or -not (Test-Path -LiteralPath $Python310)) {
    throw 'Python 3.10 was not found. Pass -Python310 with an explicit executable path.'
}
$ResolvedPythonVersion = (& $Python310 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($ResolvedPythonVersion -ne '3.10') {
    throw "The training environment requires Python 3.10; found $ResolvedPythonVersion."
}

New-Item -ItemType Directory -Force -Path $AssetRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VendorRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot 'pip') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot 'temp') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AssetRoot 'models\paddleocr\training') | Out-Null

if (-not (Test-Path -LiteralPath $VendorRoot)) {
    & git clone --branch v3.7.0 --depth 1 `
        'https://github.com/PaddlePaddle/PaddleOCR.git' $VendorRoot
    Assert-NativeSuccess 'PaddleOCR clone'
}
$ActualVendorCommit = (& git -C $VendorRoot rev-parse HEAD).Trim()
Assert-NativeSuccess 'PaddleOCR revision check'
if ($ActualVendorCommit -ne $ExpectedVendorCommit) {
    throw "PaddleOCR vendor commit mismatch: $ActualVendorCommit"
}
$VendorChanges = & git -C $VendorRoot status --porcelain
Assert-NativeSuccess 'PaddleOCR clean-tree check'
if (-not [string]::IsNullOrWhiteSpace(($VendorChanges -join "`n"))) {
    throw "PaddleOCR vendor checkout is not clean; refusing to train against local edits."
}

if (-not (Test-Path -LiteralPath $TrainingPython)) {
    & $Python310 -m venv $EnvironmentRoot
    Assert-NativeSuccess 'Python virtual-environment creation'
}

$env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip'
$env:TEMP = Join-Path $CacheRoot 'temp'
$env:TMP = $env:TEMP
$CudaBin = Join-Path $EnvironmentRoot 'Lib\site-packages\nvidia\cu13\bin\x86_64'
$CudnnBin = Join-Path $EnvironmentRoot 'Lib\site-packages\nvidia\cudnn\bin'
$env:PATH = "$CudaBin;$CudnnBin;$env:PATH"

& $TrainingPython -m pip install --upgrade pip
Assert-NativeSuccess 'pip upgrade'
& $TrainingPython -m pip install 'paddlepaddle-gpu==3.3.0' `
    -i 'https://www.paddlepaddle.org.cn/packages/stable/cu130/'
Assert-NativeSuccess 'PaddlePaddle GPU installation'
& $TrainingPython -m pip install 'nvidia-cuda-nvrtc==13.0.48'
Assert-NativeSuccess 'NVIDIA CUDA 13.0 NVRTC installation'
& $TrainingPython -m pip install -r (Join-Path $VendorRoot 'requirements.txt')
Assert-NativeSuccess 'PaddleOCR training requirements installation'
& $TrainingPython -m pip install 'paddleocr==3.7.0' 'paddlex==3.7.2'
Assert-NativeSuccess 'PaddleOCR package installation'
& $TrainingPython -m pip check
Assert-NativeSuccess 'pip dependency check'

if (-not $SkipVerification) {
    & $TrainingPython (Join-Path $PSScriptRoot 'verify_ocr_training_environment.py') `
        --environment-root $EnvironmentRoot `
        --vendor-root $VendorRoot `
        --device 'gpu:0' `
        --write-report
    Assert-NativeSuccess 'OCR training environment verification'
}

Write-Host "OCR training environment ready at $EnvironmentRoot"
Write-Host "PaddleOCR vendor commit: $ActualVendorCommit"
Write-Host "C: free before setup: $CFreeGiB GiB; D: free before setup: $DFreeGiB GiB"
