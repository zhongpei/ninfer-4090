#!/usr/bin/env python3
"""Build the bounded suffix corpus consumed by --lookup-corpus-prefix.

Adapted from TandemLLM tools/build_corpus.py: the suffix array is intentionally ordered only by the
first max_order tokens because the runtime never compares deeper than that. Output is raw little
endian arrays PREFIX.tokens.i32 and PREFIX.suffix.u32 plus PREFIX.meta.json.
"""
from __future__ import annotations
import argparse, fnmatch, hashlib, json, os, time
import numpy as np

DOC_SEP = -1
GLOBS = ("*.py","*.go","*.ts","*.tsx","*.js","*.rs","*.c","*.h","*.cpp","*.cu",
         "*.md","*.txt","*.rst","*.toml","*.yaml","*.yml","*.json","*.sh")
SKIP = {".git","__pycache__","node_modules",".venv","venv","dist","build",".mypy_cache",
        ".pytest_cache"}

def build_suffix_array(tokens: np.ndarray, max_order: int) -> np.ndarray:
    n = int(tokens.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    _, rank = np.unique(tokens, return_inverse=True)
    rank = rank.astype(np.int64, copy=False)
    k = 1
    while k < max_order:
        second = np.full(n, -1, dtype=np.int64)
        if n > k:
            second[:n-k] = rank[k:]
        sa = np.lexsort((second, rank))
        r0, r1 = rank[sa], second[sa]
        changed = np.empty(n, dtype=bool); changed[0] = True
        if n > 1:
            changed[1:] = (r0[1:] != r0[:-1]) | (r1[1:] != r1[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(changed) - 1
        rank = new_rank
        k *= 2
    return np.lexsort((np.arange(n), rank)).astype(np.uint32)

def tokenizer(path: str):
    json_path = path if path.endswith("tokenizer.json") else os.path.join(path, "tokenizer.json")
    if os.path.exists(json_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(json_path)
        return lambda s: tok.encode(s, add_special_tokens=False).ids, json_path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    return lambda s: tok(s, add_special_tokens=False).input_ids, path

def files(root: str, globs: tuple[str,...], max_bytes: int):
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
        for name in sorted(names):
            if not any(fnmatch.fnmatch(name, g) for g in globs):
                continue
            p = os.path.join(base, name)
            try: size = os.path.getsize(p)
            except OSError: continue
            if 0 < size <= max_bytes:
                yield p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF model/tokenizer directory or tokenizer.json")
    ap.add_argument("--src", action="append", required=True, help="source directory; repeatable")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=40_000_000)
    ap.add_argument("--max-file-bytes", type=int, default=400_000)
    ap.add_argument("--globs", default=",".join(GLOBS))
    a=ap.parse_args()
    if not 1 <= a.max_order <= 16: raise SystemExit("--max-order must be 1..16")
    encode, tok_source = tokenizer(a.tokenizer)
    globs=tuple(x.strip() for x in a.globs.split(",") if x.strip())
    chunks=[]; n=0; used=[]
    t0=time.perf_counter()
    for root in a.src:
        count=0
        for p in files(os.path.expanduser(root), globs, a.max_file_bytes):
            try:
                text=open(p,"r",encoding="utf-8").read()
            except (OSError,UnicodeDecodeError):
                continue
            ids=encode(text)
            if not ids: continue
            remaining=a.max_tokens-n
            if remaining <= 1: break
            ids=ids[:remaining-1]
            chunks.append(np.asarray(ids+[DOC_SEP], dtype=np.int32))
            n += len(ids)+1; count += len(ids)
        used.append({"path":root,"tokens":count})
        if n >= a.max_tokens: break
    if not chunks: raise SystemExit("no input text tokenised")
    tokens=np.concatenate(chunks).astype(np.int32, copy=False)
    print(f"tokenised {len(tokens):,} ids in {time.perf_counter()-t0:.1f}s")
    t1=time.perf_counter(); sa=build_suffix_array(tokens,a.max_order)
    print(f"suffix array in {time.perf_counter()-t1:.1f}s")
    tokens.tofile(a.out_prefix+".tokens.i32"); sa.tofile(a.out_prefix+".suffix.u32")
    try:
        digest=hashlib.sha256(open(tok_source,"rb").read()).hexdigest()
    except (OSError,IsADirectoryError):
        digest=""
    with open(a.out_prefix+".meta.json","w") as f:
        json.dump({"max_order":a.max_order,"tokens":len(tokens),"separator":DOC_SEP,
                   "sources":used,"tokenizer_sha256":digest},f,indent=2)
    print(f"wrote {a.out_prefix}.tokens.i32 and .suffix.u32")

if __name__=="__main__":
    main()
