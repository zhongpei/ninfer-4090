# Calibrate Stair target-verification rung costs on one RTX 4090 while keeping the DFlash2
# neural proposal physically fixed at K15. Each arm forces one routing rung by assigning it a
# unit cost and all other rungs a prohibitive cost. Warmup/probes/hysteresis are disabled.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$weights = if ($env:NINFER_DFLASH2_MODEL) { $env:NINFER_DFLASH2_MODEL } else {
    "$modelDir\qwen3_8_27b_dflash2.ninfer"
}
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }
Assert-NInferHostMemory -Artifacts @($weights)

$prompt = @'
Explain why speculative decoding throughput depends on both accepted tokens per round and the
non-linear cost of target verification width. Use a concrete CUDA inference example and discuss
small-M kernel staircase effects.
'@

$configs = @(
    @{ label='rung3';  costs='1,100,100,100' },
    @{ label='rung7';  costs='100,1,100,100' },
    @{ label='rung11'; costs='100,100,1,100' },
    @{ label='rung15'; costs='100,100,100,1' }
)

"config,rep,decode_tok_s,generated_tokens,rounds,seconds_per_round,accepted,tok_per_round,sha256"
foreach ($cfg in $configs) {
    for ($rep = 1; $rep -le 7; $rep++) {
        $stem = "$out\staircost_$($cfg.label)_$rep"
        $stdout = "$stem.txt"
        $stderr = "$stem.err.log"
        $argv = @(
            $weights,'--prompt',$prompt,'--max-new','768','--max-context','8192',
            '--kv-dtype','int8','--spec','dflash2','--draft-tokens','15',
            '--spec-router','stair','--spec-stair-widths','3,7,11,15',
            '--spec-stair-costs',$cfg.costs,'--spec-stair-draft-cost','0',
            '--spec-stair-warmup','0','--spec-stair-probe-period','0',
            '--spec-stair-margin','0','--greedy','--no-thinking'
        )
        & .\build-ninja\apps\ninfer.exe @argv > $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) {
            "$($cfg.label),$rep,FAILED,,,,,,"
            Get-Content $stderr -Tail 6 | ForEach-Object { "    $_" }
            continue
        }
        $log = Get-Content $stderr -Raw
        function M([string]$pattern) {
            if ($log -match $pattern) { return $Matches[1] }
            return ''
        }
        $decode = M 'decode speed\s+([\d.]+) tok/s'
        $generated = M 'generated tokens\s+(\d+)'
        $rounds = M '\S+ rounds\s+(\d+)'
        $accepted = M '\S+ accepted tokens\s+(\d+)'
        $tpr = M '\S+ acceptance length\s+([\d.]+) tok/round'
        $spr = ''
        if ($decode -and $generated -and $rounds -and [double]$decode -gt 0 -and [double]$rounds -gt 0) {
            $spr = ([double]$generated / [double]$decode) / [double]$rounds
        }
        $hash = (Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
        "$($cfg.label),$rep,$decode,$generated,$rounds,$spr,$accepted,$tpr,$hash"
    }
}
