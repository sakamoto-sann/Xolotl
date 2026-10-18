# DSpark with the 1017 llama.cpp Q2_K trunk

This is opt-in. With neither `STRATA_DSPARK` nor `STRATA_SERVE_DSPARK` set, the trunk follows its existing path.
The draft stack uses the trunk's quantized token embedding rows and shares its output head. Every proposed token
is verified by the trunk; the amount of draft precision and expert pruning affect acceptance and cost.
Actual output parity and speed still need to be checked on the chosen hardware and workload.

## Matching weights

Use `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Its MTP tensors occupy shards 44, 45 and 46 of 48.
Obtain `config.json`, `tokenizer.json`, `model.safetensors.index.json` and those three shards at that revision.
Verify metadata and each selected shard using `v41_checkpoint.Checkpoint` and the repository's pinned profile
`tools/v41/checkpoints/deepseek-v41-flash-1017.json` before exporting. No other trunk shards are needed for this export.

```sh
python tools/v41/v41_sidecar.py --checkpoint SOURCE \
  --out SOURCE/1017-MTP-MXFP4.gguf --mtp 0,1,2 --no-vision
python tools/v41/v41_sidecar.py --checkpoint SOURCE \
  --verify SOURCE/1017-MTP-MXFP4.gguf --sample 4
build/v41_sidecar_requant --in SOURCE/1017-MTP-MXFP4.gguf \
  --out SOURCE/1017-MTP-Q2_K.gguf.partial --type Q2_K --threads 6
```

The requantizer's default is one thread. `--threads 1..32` gives workers disjoint row chunks and leaves nonexpert
weights unchanged. Check the output before renaming `.partial` to `.gguf`. The 1017 export has 81 tensors;
9 routed-expert tensors are requantized and 72 other tensors retain their bytes. The C++ GGUF writer may omit
trailing singleton dimensions, which preserve the tensor's geometry.

The engine checks the sidecar's pinned source tags, required tensor names and 1017 shapes before loading GPU
weights. It rejects a partial export or another checkpoint. Source tags establish compatibility; the source shard
SHA-256 checks and a separately recorded sidecar SHA-256 establish the local file provenance.

## Run and check

```sh
build/test_ds41_dspark --validate-only --llama-q2-k \
  -m MODEL-00001-of-00010.gguf --sidecar SOURCE/1017-MTP-Q2_K.gguf
python tools/v41/test_dspark_q2k.py --build build \
  --model MODEL-00001-of-00010.gguf --sidecar SOURCE/1017-MTP-Q2_K.gguf \
  --output NEW_TEST_OUTPUT
```

These checks allocate no GPU memory. They check invalid revision, missing tensors, invalid shape, early CLI
rejection, and byte parity between serial and parallel requantization over multiple row chunks.

For one-shot generation, set `STRATA_DSPARK=SOURCE/1017-MTP-Q2_K.gguf`; for the persistent server, set
`STRATA_SERVE_DSPARK` instead. Pass the existing `--llama-q2-k` option to the trunk in either case.
Use the GPU layer-major prompt path (`--prompt-batch` or `--prefill-chunk` greater than one, at least 17 pass columns).
`STRATA_DSPARK_DEPTH=4` and `STRATA_DSPARK_CONF=0.6` are candidates to measure, not a changed default.
`STRATA_DSPARK_COUNT=COUNTS.txt` records routing counts from this checkpoint. A later run can set
`STRATA_DSPARK_KEEP=COUNTS.txt:64` to load 64 experts per draft block, reducing device weight memory.
Measure acceptance and latency again; counts from one coding prompt may not represent another task.
Leave `STRATA_DSPARK_KEEP` unset until collecting counts from this checkpoint; another model's expert
counts are not a qualification result for 1017.

Measure committed tokens per second, total completion latency, acceptance, token parity, task checks, memory,
and SSD reads against a same-day run of the same binary without DSpark. A draft stack takes VRAM away from the
trunk expert cache and can slow an SSD-bound model down. The one-shot path also re-feeds a prompt tail to seed
the draft window; account for that in first-token latency. The persistent server has a separate prompt seed path.

## Local qualification limits (2026-10-18)

On the measured RTX 3090 / Ryzen 9 3900X / 32 GiB host, full draft weights and a 64-expert draft were slower than
normal one-shot generation on the 4,060-token coding fixture. A 40-expert, confidence-0.6 persistent-server trial
improved one cached regeneration but did not provide a consistent overall speed improvement.

Fresh and first repeated coding outputs matched the normal run. After a one-token response budget, a subsequent
DSpark regeneration produced a longer docstring and hit the same 96-token output limit with incomplete code;
the normal run completed in 85 tokens. The cause (cache lifecycle versus quantized verifier numerics) is not
isolated. Both modes also failed the JSON arithmetic fixture; Japanese sentence-count instruction adherence differed.
These are experimental results, not a lossless-runtime qualification. Keep the normal inference default until the
runtime parity and workload results are resolved. Sidecar compatibility checks alone do not establish output parity.
