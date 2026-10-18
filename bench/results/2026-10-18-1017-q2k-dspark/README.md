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

## Regeneration follow-up: opt-in file-union split

A longer-output repeat probe rules out the one-token request as a necessary trigger. With the original D3 profile, the third coding response had already changed from 85 to 113 tokens before the one-token request. All five complete responses passed 2,000 lower_bound cases when allowed 192 tokens. The earlier 96-token failure was truncation of the longer answer, not proof of corrupted code/cache state.

D3 serves at most eight distinct file experts on the GPU in a call; additional file experts use the CPU activation path (Q8_K rather than GPU Q8_1). With `STRATA_TIER_BATCH=0`, all five complete responses matched the normal 85-token answer and CPU expert count was zero, but warm latency was 26.90-28.31 s.

`STRATA_TIER_BATCH_FILE_SPLIT=1` now optionally bisects only a batched call whose distinct uncached, non-arena, non-prefetched file union exceeds the existing staging slots. It splits before routing observation or any reader/stream work. Each child completes before the next starts; buffer capacities and events are unchanged. This limits file CPU fallback when the configured single-token top-k fits the staging slots. It does not override a deliberately reduced arena PCIe share. Speculative read-ahead (`STRATA_D3_SPEC`) was left off. With the option disabled, the existing batched implementation is used.

| Same coding fixture, API | Cold end-to-end | Warm median (4 sequential requests) | Full-answer equality |
|---|---:|---:|---|
| Normal | 102.24 s | 23.51 s | 5/5 same, 85 tokens |
| Original DSpark D3 | 102.71 s | variable output lengths | 3/5 same, 2/5 longer (113) |
| DSpark, per-token MoE | 101.37 s | 27.02 s | 5/5 same, 85 tokens |
| DSpark, file-union split | 93.70 s | 20.93 s | 5/5 same, 85 tokens |

The exploratory split/normal warm medians differed by 10.96%. The first normal profile reused a previously loaded API (a 12-token smoke preceded this probe), while the other profiles started fresh; the controlled same-executable comparison below supersedes this timing estimate. All five split outputs matched the normal answer and each passed 2,000 cases. Native logs reported zero CPU experts. Observation began during the cold request; all 131 samples had zero inference-cgroup swap/OOM, and minimum observed MemAvailable was 7.15 GiB. These are sequential warm samples of one prompt, not an independent multi-task benchmark or proof of universal greedy parity. See `regeneration-results.json` and `regeneration-fixtures.json` for exact outputs and settings.

The unmodified normal API remains the DSH default. The opt-in candidate is retained for further qualification. A separate default-off replay produced the identical 85-token answer on cold and repeated requests. Synthetic GPU regression passed all six modes: 96 calls/mode with the option disabled and 144 calls/mode enabled, including a 16-expert union, both output paths, zero weights and cache/arena replacement. Enabled output was byte-identical to per-token GPU calls, and CPU experts remained zero. HIP/SYCL coverage remains unavailable; the PR remains draft.

### Final same-executable comparison

Both profiles were started fresh using executable SHA-256 `4eb9c5720541512ce3e9c204f8a5691e47e3e651b7f32941fcd9bd4e312547d2`. The same six-request sequence, 4,060-token prompt, temperature 0, seed 42, 192-token output cap (except the deliberate one-token request), 131,072 context, 16 GiB arena and 1.5 GiB VRAM margin were used. Normal ran first; caches were not globally evicted. This is a sequential warm experiment, not independent randomized trials.

| Profile | Cold end-to-end | Warm median (4 requests) | Normal answer match | Quality |
|---|---:|---:|---|---|
| Normal | 100.859 s | 24.953 s | 5/5, 85 tokens | 2,000 cases each |
| DSpark + file split | 93.293 s | 20.884 s | 5/5, 85 tokens | 2,000 cases each |

Matched warm latency decreased by 16.31%. The original exploratory estimate above is superseded by this pair. Complete per-request timings, configurations and texts are in the JSON files. This establishes parity and a latency improvement for this fixture; longer-context input, other tasks, sampling and long-running stability remain unqualified. The default-off replay also returned the normal answer. Source/test self-review was performed because an independent reviewer was unavailable; HIP/SYCL builds remain missing, so this is still a draft.

Synthetic regression: `test_ds41_pipeline` passed 6 modes x 96 calls with the option disabled; `STRATA_TIER_BATCH_FILE_SPLIT=1 test_ds41_pipeline` passed 6 modes x 144 calls with the option enabled, with byte-identical per-token GPU reference and zero CPU experts. See `file-split-regression.txt`. The initial 256 MiB test-controller cap was insufficient; the successful rerun peaked at 1.2 GiB under a 4 GiB cap, separate from the inference cgroup.
