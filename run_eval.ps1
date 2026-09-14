param(
    [ValidateSet("random", "uniform")]
    [string]$Mode = "random",
    [string]$Gpu = "0",
    [int]$MaxSamples = 0,
    [int]$Seed = 42,
    [switch]$NoConsistency
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python"
}

$env:CUDA_VISIBLE_DEVICES = $Gpu
$Config = "configs\pl_aff_${Mode}10_fullnoise_200.yaml"
$Arguments = @(
    "scripts\evaluate_full_random_noise.py",
    "--config", $Config,
    "--max-samples", $MaxSamples,
    "--seed", $Seed
)
if ($NoConsistency) { $Arguments += "--no-consistency" }

Write-Host "Evaluating PL AFF $Mode 10% from Gaussian noise with 200 reverse steps on physical GPU $Gpu"
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
