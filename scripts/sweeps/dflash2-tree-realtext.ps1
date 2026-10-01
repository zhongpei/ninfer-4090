# Fixed-chain vs runtime-tree DFlash2 A/B on real text.
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

$prompt = @'
Explain how a hybrid transformer with full attention and recurrent Gated DeltaNet layers can
verify speculative branches without copying one complete model state per branch. Discuss parent
state, accepted-path commit, paged KV, and why rejected branches must not become persistent state.
Then provide concise C++-style pseudocode for the transaction.
'@

$configs = @(
    @{ label='chain15'; k=15; args=@('--spec-tree','off') },

    # Same resident b16 drafter, different target tree budgets. These are the rows used to
    # calibrate the 24GB Tree-Stair verify staircase; changing K would also change draft cost.
    @{ label='b16_tree3';  k=15; args=@('--spec-tree','lattice','--spec-tree-nodes','3',
                                        '--spec-tree-spine','3') },
    @{ label='b16_tree7';  k=15; args=@('--spec-tree','lattice','--spec-tree-nodes','7',
                                        '--spec-tree-spine','5') },
    @{ label='b16_tree11'; k=15; args=@('--spec-tree','lattice','--spec-tree-nodes','11',
                                        '--spec-tree-spine','7') },
    @{ label='b16_tree15'; k=15; args=@('--spec-tree','lattice','--spec-tree-nodes','15',
                                        '--spec-tree-spine','7') },

    @{ label='b16_tree_stair'; k=15; args=@('--spec-tree','lattice','--spec-tree-nodes','15',
                                            '--spec-tree-spine','7','--spec-router','stair',
                                            '--spec-stair-widths','3,7,11,15',
                                            '--spec-stair-costs','1.00,1.02,1.05,1.10') }

)

"config,rep,decode_tok_s,generated,rounds,drafted,accepted,tree_rounds,tree_fallback,tree_nodes,tree_accepted,sha256"
foreach ($cfg in $configs) {
    for ($rep = 1; $rep -le 5; $rep++) {
        $stem = "$out\tree_$($cfg.label)_$rep"
        $stdout = "$stem.txt"
        $stderr = "$stem.err.log"
        $argv = @($weights,'--prompt',$prompt,'--max-new','768','--max-context','8192',
                  '--kv-dtype','int8','--spec','dflash2','--draft-tokens',"$($cfg.k)",
                  '--greedy','--no-thinking') + $cfg.args
        & .\build-ninja\apps\ninfer.exe @argv > $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) {
            "$($cfg.label),$rep,FAILED,,,,,,,,,"
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
        $tr = M 'tree rounds\s+(\d+)'
        $tf = M 'tree fallback rounds\s+(\d+)'
        $tn = M 'tree nodes\s+(\d+)'
        $ta = M 'tree accepted drafts\s+(\d+)'
        $hash = (Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
        "$($cfg.label),$rep,$decode,$generated,$rounds,$drafted,$accepted,$tr,$tf,$tn,$ta,$hash"
    }
}
