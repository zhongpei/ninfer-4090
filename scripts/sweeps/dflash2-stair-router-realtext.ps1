# Compare fixed-K DFlash2 with the adaptive Stair verification policy on real text.
#
# The existing dflash2-draft-tokens-realtext.ps1 explains why the tiled benchmark corpus is invalid
# for drafting acceptance. This sweep uses the same production-shaped prose prompt and hashes every
# output. Set NINFER_STAIR_COSTS to a comma-separated table measured on the target 4090.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$weights = "$modelDir\qwen3_8_27b_dflash2.ninfer"
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }
Assert-NInferHostMemory -Artifacts @($weights)

$costs = if ($env:NINFER_STAIR_COSTS) { $env:NINFER_STAIR_COSTS } else { '1,1.02,1.05,1.10' }
$widths = if ($env:NINFER_STAIR_WIDTHS) { $env:NINFER_STAIR_WIDTHS } else { '3,7,11,15' }
$prompt = 'Explain how a paged key-value cache lets a transformer serve many concurrent requests ' +
          'without reserving each one its maximum context up front. Cover fragmentation, the ' +
          'block table indirection, and what happens when a request outgrows its allocation.'

$configs = @(
  @{ label='fixed15'; args=@('--spec','dflash2','--draft-tokens','15') },
  @{ label='stair15'; args=@('--spec','dflash2','--draft-tokens','15','--spec-router','stair',
                             '--spec-stair-widths',$widths,'--spec-stair-costs',$costs) }
)

"config,rep,decode_tok_s,generated_tokens,rounds,drafted,accepted,acceptance_rate_pct,tok_per_round,content_sha256"
foreach ($c in $configs) {
  for ($rep = 1; $rep -le 5; $rep++) {
    $stem = "$out\stair_$($c.label)_$rep"
    $log = "$stem.err.log"
    $text = "$stem.txt"
    $argv = @($weights,'--prompt',$prompt,'--max-new','512','--max-context','8192',
              '--kv-dtype','int8','--greedy','--no-thinking') + $c.args
    & .\build-ninja\apps\ninfer.exe @argv > $text 2> $log
    if ($LASTEXITCODE -ne 0) {
      "$($c.label),$rep,FAILED,,,,,,,"
      Get-Content $log -Tail 4 | ForEach-Object { "    $_" }
      continue
    }
    $txt = Get-Content $log -Raw
    $dec = if ($txt -match 'decode speed\s+([\d.]+) tok/s') { $Matches[1] } else { '' }
    $gen = if ($txt -match 'generated tokens\s+(\d+)') { $Matches[1] } else { '' }
    $rounds = if ($txt -match '\S+ rounds\s+(\d+)') { $Matches[1] } else { '' }
    $draft = if ($txt -match '\S+ drafted tokens\s+(\d+)') { $Matches[1] } else { '' }
    $acc = if ($txt -match '\S+ accepted tokens\s+(\d+)') { $Matches[1] } else { '' }
    $rate = if ($txt -match '\S+ acceptance rate\s+([\d.]+)%') { $Matches[1] } else { '' }
    $tpr = if ($txt -match '\S+ acceptance length\s+([\d.]+) tok/round') { $Matches[1] } else { '' }
    $hash = if (Test-Path -LiteralPath $text) {
      try { (Get-FileHash -LiteralPath $text -Algorithm SHA256 -ErrorAction Stop).Hash }
      catch { "ERR:$($_.Exception.GetType().Name)" }
    } else { 'ERR:no-stdout-file' }
    "$($c.label),$rep,$dec,$gen,$rounds,$draft,$acc,$rate,$tpr,$hash"
  }
}
