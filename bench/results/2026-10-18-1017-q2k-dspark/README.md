# 1017 Q2_K DSpark: local exploratory qualification

The sidecar now loads beside the llama.cpp 1017 Q2_K trunk. **This is experimental; the normal serving default is retained.**
One cached coding regeneration became faster, but fresh latency did not improve and a fixed-budget regeneration
failed the normal-output parity check. Source/weight compatibility passing is not a lossless-runtime qualification.

RTX 3090 24 GiB, Ryzen 9 3900X, 32 GiB RAM, Samsung 970 EVO 1TB NVMe (PCIe 3 x4), Btrfs zstd:3.
No ComfyUI inference ran. Browser/display GPU use remained present. Both modes use the same pinned 1017 Q2_K model.

## Matching auxiliary weights

Official checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`.
Only shards 44/45/46 of 48 were downloaded and SHA-256 verified. The no-vision export has 81 tensors.
MXFP4 sidecar: 7,903,825,792 bytes; Q2_K sidecar: 5,143,449,440 bytes. All 72 nonexpert tensors retain their bytes;
9 routed-expert tensors are requantized. Thirty-six sampled MXFP4 expert matrices match official FP4 exactly;
30 Q8 matrices have worst relative RMS error 0.00608. Serial/six-worker conversion matches byte for byte on a
synthetic multi-chunk fixture. Hashes and provenance are in `weights-manifest.json`.

## One-shot coding

Same executable SHA-256 `5155ec91d2e2cd1f6bcc2b1d4dd8fd9645939a4aa07e8647379564863096ffdb`, 4,060 prompt tokens,
85 output tokens including EOS, temperature 0, seed 42. Context configured 131,072; arena 16 GiB; margin 1.5 GiB;
prompt batch 8,192; chunk 32; readers 8; enc-skip and KV-FP4 on; file pipeline and freq/recency cache on.
Each trial advises only model files DONTNEED; it does not drop global caches. MemorySwapMax=0, 4 GiB availability
floor, one GPU lock. One observation per final profile: these are exploratory results, not a three-repeat speed claim.

| Mode | First token s (load included) | Committed decode tok/s | Process wall s | Estimated decode physical reads GiB |
| --- | ---: | ---: | ---: | ---: |
| Normal | 72.36 | 2.006 | 117.51 | 71.03 |
| DSpark Q2_K, all 128 experts | 132.74 | 1.809 | 182.41 | 86.08 |
| DSpark Q2_K, 64 experts | 128.49 | 1.908 | 175.72 | 82.39 |

All three coding streams have identical token IDs and pass 2,000 comparisons with `bisect_left`.
All own-cgroup swap and OOM counters stayed zero; minimum available memory was about 7.4 GiB or more.
Physical reads use one-second `/proc/PID/io` sample differences around first/last token, with boundary error.
Do not interpret completion-pipeline-only byte counters as all SSD traffic: the verifier can also read CPU fallback experts.
The unpruned draft accepted 3.722 drafts/pass (4.722 tokens/pass, depth 4) but generated more file traffic than normal.

The one-shot DSpark path re-feeds a 128-token prompt tail. The persistent server instead captures the batch's
main-hidden rows; its 128-row draft seed took about 0.05 seconds on the cold coding request.

## Persistent API sequence

Same executable and base settings in both processes. Prompt batch 12,288 (elastic), `STRATA_PRESTAGE_GIB=1.5`,
serve step max 96, pass bias 0. DSpark trial: Q2_K sidecar, 40 experts per block, depth 4, confidence 0.6,
large-prompt eviction threshold 4,096. GPU draft weights measured 2.08 GB. Keep counts come from the unpruned
1017 coding run (three 128-entry rows); this is workload-specific calibration.

Same ordered HTTP sequence, fresh process per mode: cold code, repeated code, code with max_tokens=1,
code after that budget, CSV/JSON filter+sum, Japanese summary. Code limit 96, JSON/Japanese 192.
All client timing is wall time; startup before the listener becomes ready is excluded.

| Request | Normal seconds | DSpark seconds | Quality observation |
| --- | ---: | ---: | --- |
| code-cold | 100.260 | 103.670 | Same complete coding text |
| code-repeat | 26.360 | 23.284 | Same complete coding text |
| code-budget-one | 0.018 | 0.032 | Both honor one token |
| code-after-budget | 24.943 | 28.930 | Normal complete; DSpark length stop / incomplete code |
| json | 70.385 | 63.974 | Both wrong; different totals |
| japanese | 57.992 | 57.707 | Facts retained; normal 3 sentences, DSpark 2 |

Fresh code TTFT: normal 58.973 s, DSpark 61.184 s. First cached regeneration: 26.360 -> 23.284 s (-11.7% time).
That gain did not hold across all requests. After the one-token budget, normal returned the same complete 85-token
function; DSpark wrote a longer docstring and hit 96 tokens with incomplete Python (`finish_reason=length`).
The cause, including cache lifecycle versus quantized verifier numerics, has not been isolated. No cache-corruption
or universal lossless-decoding claim is made. Both fresh and first repeated code outputs pass 2,000 functional cases.

JSON truth: count 10, total 514, IDs R03/R06/R09/R15/R18/R21/R27/R30/R33/R39. Normal returned count 20 / total 1080;
DSpark count 20 / total 1160. Both are wrong. Japanese required facts were preserved by both; the requested two
sentences were met only by the DSpark trial. See `api-*-fixtures.json` for the full synthetic prompts and outputs.

## Checks and limits

CUDA build, CPU MoE build, pinned-sidecar preflight, malformed revision/missing tensor/wrong shape refusal,
quantizer parity and actual coding generation were checked. A proposed extra uniform-batch dispatch flag was
removed: the real file-blob geometry already uses the class-aware GPU path, so that change did not improve speed.
Rebuilding after removal restored the exact initial qualified executable hash above.

No HIP/SYCL toolchain or hardware was available. Independent review was unavailable and subagents were disabled;
a separate read-only self-review was used. No full-128K-input soak or stochastic-sampling parity was checked.
The runtime parity gate is unresolved, so this report does not recommend changing the normal default or merging
as a production-ready lossless decoder. The sidecars and opt-in compatibility code remain available for further work.

A separate Mac DSH 0.2.1-alpha.5 integration check passed with the 40-expert DSpark trial: the model called the read tool and returned a newly generated local fixture exactly (9,509-token input, 160.708 seconds end to end). Private DSH request/event history is not published. This functional check does not close the regenerate-parity gap above.
