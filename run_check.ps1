param(
    [ValidateSet("random", "uniform")]
    [string]$Mode = "random",
    [string]$Gpu = "0"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python"
}

$env:CUDA_VISIBLE_DEVICES = $Gpu
$Config = "configs\pl_aff_${Mode}10_fullnoise_200.yaml"

Write-Host "Checking PL AFF $Mode 10% full-noise experiment on physical GPU $Gpu"
& $PythonExe scripts\check_experiment.py --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
