param(
    [ValidateSet('smoke','core','full','server')]
    [string]$Profile = 'core',
    [string]$Model = '',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"

$modelDir = Get-NInferModelDir
if (-not $Model) {
    $Model = if ($env:NINFER_DFLASH2_MODEL) { $env:NINFER_DFLASH2_MODEL } else { "$modelDir\qwen3_8_27b_dflash2.ninfer" }
}
if (-not (Test-Path $Model)) { throw "Missing DFlash2 artifact: $Model" }
Assert-NInferHostMemory -Artifacts @($Model)

$exe = if ($env:NINFER_EXE) { $env:NINFER_EXE } else { '.\build-ninja\apps\ninfer.exe' }
$serve = if ($env:NINFER_SERVE_EXE) { $env:NINFER_SERVE_EXE } else { '.\build-ninja\apps\ninfer-serve.exe' }
$outRoot = if ($env:NINFER_AB_OUT) { $env:NINFER_AB_OUT } else { 'profiles\ab-suite' }
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

function Run-CliSuite([string]$Name,[string]$Kv,[string]$Arms,[string]$Workloads,[int]$Pairs,[int]$Discard,[double]$Cooldown) {
    $out = Join-Path $outRoot $Name
    $argv = @('-m','tools.dflash2_training.ab_suite','--exe',$exe,'--model',$Model,'--out',$out,'--kv-dtype',$Kv,'--arms',$Arms,'--workloads',$Workloads,'--pairs',[string]$Pairs,'--discard',[string]$Discard,'--cooldown',[string]$Cooldown)
    & $Python @argv
    if ($LASTEXITCODE -ne 0) { throw "A/B CLI suite failed: $Name" }
}

function Run-ServerSuite([string]$Name,[string]$Kv,[string]$Arms,[string]$Workloads) {
    $out = Join-Path $outRoot $Name
    $argv = @('-m','tools.dflash2_training.server_ab','--serve',$serve,'--model',$Model,'--out',$out,'--kv-dtype',$Kv,'--arms',$Arms,'--workloads',$Workloads,'--concurrency','1,2,4,8','--repeats','2','--cooldown','8')
    & $Python @argv
    if ($LASTEXITCODE -ne 0) { throw "A/B server suite failed: $Name" }
}

switch ($Profile) {
    'smoke' {
        Run-CliSuite 'smoke-int8' 'int8' 'baseline,dflash2-k15,tree15,tree15-stair,lookup-skip' 'prose,code,lookup-repeat' 2 0 1
    }
    'core' {
        Run-CliSuite 'core-int8' 'int8' 'all' 'all' 4 1 5
        foreach ($kv in @('fp8','rk8v4')) {
            Run-CliSuite "kv-$kv" $kv 'baseline,dflash2-k15,tree15,tree15-stair' 'prose,code,long-context' 4 1 5
        }
        Run-ServerSuite 'server-int8' 'int8' 'baseline,dflash2-k15,tree15,tree15-stair,lookup-skip' 'prose,chat,reasoning,code,lookup-repeat'
    }
    'full' {
        Run-CliSuite 'full-int8' 'int8' 'all' 'all' 6 1 10
        foreach ($kv in @('fp8','rk8v4')) {
            Run-CliSuite "full-$kv" $kv 'all' 'prose,chat,reasoning,code,structured,lookup-repeat,long-context' 5 1 8
        }
        foreach ($kv in @('int8','fp8','rk8v4')) {
            Run-ServerSuite "server-$kv" $kv 'baseline,dflash2-k15,tree15,tree15-stair,lookup-skip' 'prose,chat,reasoning,code,lookup-repeat'
        }
    }
    'server' {
        Run-ServerSuite 'server-int8' 'int8' 'baseline,dflash2-k15,tree15,tree15-stair,lookup-skip' 'prose,chat,reasoning,code,lookup-repeat'
    }
}

Write-Host "A/B profile '$Profile' complete. Results: $outRoot"
