# DFlash2 lookup A/B on production-shaped generated text.
#
# Compares neural baseline, lookup proposal replacement, lookup head-skip, and head-skip with
# deep-copy promotion. All modes use the same prompt and greedy target output. A static corpus can
# be injected with NINFER_LOOKUP_CORPUS; omit it to isolate local/process lookup.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$weights = if ($env:NINFER_DFLASH2_MODEL) {
    $env:NINFER_DFLASH2_MODEL
} else {
    "$modelDir\qwen3_8_27b_dflash2.ninfer"
}
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }
Assert-NInferHostMemory -Artifacts @($weights)

$k = if ($env:NINFER_DFLASH2_K) { [int]$env:NINFER_DFLASH2_K } else { 15 }
$ngram = if ($env:NINFER_LOOKUP_NGRAM) { [int]$env:NINFER_LOOKUP_NGRAM } else { 5 }
$confidence = if ($env:NINFER_LOOKUP_CONFIDENCE) { $env:NINFER_LOOKUP_CONFIDENCE } else { '0.70' }
$support = if ($env:NINFER_LOOKUP_SUPPORT) { $env:NINFER_LOOKUP_SUPPORT } else { '2' }
$persistent = if ($env:NINFER_LOOKUP_PERSISTENT) { $env:NINFER_LOOKUP_PERSISTENT } else { '262144' }
$corpusArgs = @()
if ($env:NINFER_LOOKUP_CORPUS) {
    $corpusArgs = @('--lookup-corpus-prefix', $env:NINFER_LOOKUP_CORPUS)
}

$prompt = @'
You are editing a C++ inference runtime. Rewrite the following design as a concise implementation
plan while preserving exact identifiers: ProgramImpl, decode_dflash_batch, StateImage, ReplaySSM,
PagedKVCache, candidate_ids, proposal_q, proposal_extents, target_valid_columns. Explain how a
verified prefix is committed and how rejected speculative state is discarded. Then give a small
pseudo-diff showing the same identifiers more than once.
'@

$commonLookup = @('--lookup-ngram',"$ngram",'--lookup-strategy','vote',
                  '--lookup-min-support',"$support",
                  '--lookup-min-confidence',"$confidence",
                  '--lookup-persistent-tokens',"$persistent") + $corpusArgs

$configs = @(
    @{ label='baseline'; args=@() },
    @{ label='replace'; args=$commonLookup + @('--lookup-dflash','replace',
                                               '--lookup-base-drafts','7',
                                               '--lookup-deep-after','0',
                                               '--lookup-deep-drafts','15') },
    @{ label='skip'; args=$commonLookup + @('--lookup-dflash','skip',
                                            '--lookup-base-drafts','7',
                                            '--lookup-deep-after','0',
                                            '--lookup-deep-drafts','15') },
    @{ label='skip_deep'; args=$commonLookup + @('--lookup-dflash','skip',
                                                 '--lookup-base-drafts','7',
                                                 '--lookup-deep-after','2',
                                                 '--lookup-deep-drafts','15') }
)

"config,rep,decode_tok_s,generated,rounds,drafted,accepted,lookup_queries,lookup_hits,lookup_rounds,replace_rounds,head_skip_rounds,lookup_drafted,lookup_accepted,sha256"
foreach ($cfg in $configs) {
    for ($rep = 1; $rep -le 5; $rep++) {
        $stem = "$out\lookup_$($cfg.label)_$rep"
        $stdout = "$stem.txt"
        $stderr = "$stem.err.log"
        $argv = @($weights,'--prompt',$prompt,'--max-new','768','--max-context','8192',
                  '--kv-dtype','int8','--spec','dflash2','--draft-tokens',"$k",
                  '--greedy','--no-thinking') + $cfg.args
        & .\build-ninja\apps\ninfer.exe @argv > $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) {
            "$($cfg.label),$rep,FAILED,,,,,,,,,,,,"
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
        $drafted = M '\S+ drafted tokens\s+(\d+)'
        $accepted = M '\S+ accepted tokens\s+(\d+)'
        $lq = M 'lookup queries\s+(\d+)'
        $lh = M 'lookup hits\s+(\d+)'
        $lr = M 'lookup rounds\s+(\d+)'
        $lrep = M 'lookup replace rounds\s+(\d+)'
        $lskip = M 'lookup head-skip rounds\s+(\d+)'
        $ld = M 'lookup drafted tokens\s+(\d+)'
        $la = M 'lookup accepted tokens\s+(\d+)'
        $hash = (Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
        "$($cfg.label),$rep,$decode,$generated,$rounds,$drafted,$accepted,$lq,$lh,$lr,$lrep,$lskip,$ld,$la,$hash"
    }
}
