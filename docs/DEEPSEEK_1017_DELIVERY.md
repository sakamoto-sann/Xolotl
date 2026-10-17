# Experimental SSD expert delivery and cache placement

Source implementation only, 2026-10-17. This work builds on the 1017 loader and Q2_K direct-input changes in
[fork PR #1](https://github.com/sakamoto-sann/Xolotl/pull/1). No build, inference run, or speed measurement has
been performed for this change. Keep the PR draft until the pending checks below are authorized and pass.

## Mechanisms selected from the papers

| Source | Mechanism used here | Scope of this implementation |
| --- | --- | --- |
| [SSD-LLaMA](https://arxiv.org/html/2609.18110v1) | Transfer a completed expert without waiting for every expert read in the layer; overlap its H2D copy with another expert's compute | A bounded completion queue, the existing split readers, and a separate CUDA copy stream |
| [FlashMoE](https://arxiv.org/html/2601.17063v2) | Combine frequency and recency when choosing what to retain | An independently written, deterministic decayed-frequency heuristic for existing RAM and VRAM victim selection |

FlashMoE's learned policy is not implemented. There is no training trace or learned Belady approximation here.
SSD-LLaMA's expert repacking, io_uring submission, lossless rANS transport, and GPU decompression are not implemented.
The existing blob source still assembles an expert from its gate/up/down ranges, including split-file Q2_K input.
The cited papers' hardware and performance results do not establish a speedup on a 3090 or for this 1017 artifact.

## Opt-in interface

The additional `ds41_generate` arguments are:

```text
--file-pipeline
--cache-policy lru|freq-recency
--cache-half-life 64
```

All new behavior is off by default. `freq-recency` enables the existing adaptive RAM arena as well as the new
victim-selection history. VRAM replacement still follows the existing `--vram-lru` / `--no-vram-lru` setting.
The half-life must be finite and positive; 64 routed tokens per layer is an experimental starting value, not a
measured optimum. Selecting `lru` does not undo an explicit `--arena-adapt` argument.

The completion pipeline covers single-token decode and the small multi-token per-expert verification path.
Large prompt chunks retain their existing streaming implementation. History is collected for those chunks on
their first pass, but their bulk placement/rebalancing algorithm is unchanged. The target is the V4.1
per-expert file blob source; the older uniform-format multi-token path is not a new optimized path.
New modes reject CPU-only expert execution; `--file-pipeline` also rejects `STRATA_FILE_GPU=0`.
The CLI does not change the model's routing or weight representation. Engram's SSD row cache remains separate.

## Cache policy

For an expert last updated at token `t0`, the score at layer-local token `t` is:

```text
score(t) = stored_mass * 2^(-(t - t0) / half_life)
stored_mass = score(t) + 1    on a routed access
```

One token advances the layer's clock once, irrespective of the number of routed experts. Observing the same
route stream token by token or in one batch produces the same history. The clocks track evaluated tokens,
including speculative candidates that may later be rejected; this affects placement only and has no rollback.

The lowest score is the preferred victim, with existing last-use ordering breaking ties. A candidate must have
a strictly higher score than the resident before replacing it. Slot capacity, format class, currently routed
expert protection, and existing VRAM admission/demotion filters still apply. A one-off miss can be served without
displacing a hotter resident. Arena capacity and VRAM slot counts retain their existing configuration limits.

## Ownership and resource bounds

- Reader callbacks publish completion only after every part of that expert has finished. The queue mutex makes
  their writes visible to the engine thread. Workers do not call CUDA.
- The transfer stream waits until earlier kernels release the existing staging bank. Each completed expert
  uses a distinct slot in that bank; the compute stream waits on its copy event before launching its kernel.
  Completion order may vary, while output rows and the weighted sum remain indexed by the original routes.
- RAM victims are selected before reads and protected from reuse. Earlier arena DMA users are synchronized
  before readers overwrite a victim. Slot mappings are published after the output stream synchronizes.
- On an early return, a local drain guard waits for callbacks before their borrowed stack arrays are destroyed.
  The existing read pool now waits for every worker to acknowledge the submitted generation, including late
  wakeups; counting only currently busy workers could let a callback outlive its captured arrays.
- Closing the GPU tier drains readers and both affected CUDA streams before freeing their buffers and events.
- The opt-in pipeline uses one additional nonblocking stream and nine events, with no extra GPU expert bank.
  Existing fallback pinned staging can still allocate up to eight maximum-sized expert blobs when no RAM victim
  is available. Direct-to-arena reads and stashed experts avoid that allocation when no fallback is needed.
- History stores one 16-byte entry per layer/expert plus layer clocks: approximately 240 KiB for 40 x 384 experts.
  The completion queue has eight entries and uses no model-sized allocation. Existing RAM/VRAM budget flags
  are not increased. No downloader or Comfy service setting is changed.
- Single-token tier-graph capture is bypassed when the pipeline is enabled, because repeated per-expert launches
  need disjoint metadata regions. The existing default capture path remains in place.

The worker-generation wait correction also applies to the existing small verification path with new flags off.
It is a lifetime correction, not a claimed throughput improvement. Numerical equivalence of the default and
opt-in paths is still a pending runtime check.

## Counters and validation status

Pipeline mode reports completed expert transfers, demand-read logical payload bytes, H2D logical payload bytes,
and host time blocked waiting for a ready expert. H2D includes speculative-stash hits; demand payload excludes
those hits. These counters cover the changed demand paths, not all prefill, CPU, speculative, or Engram traffic.
They do not measure aligned physical SSD reads, read amplification, device bandwidth, or GPU overlap duration.

`test_ds41_delivery.cpp` prepares checks for history aging, workload shifts, admission, batching invariance,
layer independence, bounded completion order, partial reads, duplicate notification rejection, and visibility
of concurrent reader writes. Its CMake target is present but **has not been built or run**.

Performed: source inspection and `git diff --check`. A separate read-only self-review covered buffer lifetimes,
slot bounds, stream/event dependencies, combinations of the two flags, and default branches. No independent
reviewer was available with sub-agents disabled. Neither compilation nor runtime correctness is established.

Pending after build/inference permission is restored:

1. Build the touched backends and run the prepared CPU-only helper tests; exercise the reader pool with delayed
   worker wakeups, repeated generations, and shutdown. Use a race detector where supported.
2. Compare default vs pipeline vs policy vs both flags on identical short prompts and verification batches,
   including all-hit, all-miss, partial/stashed reads, mixed blob sizes, fallback staging and error cleanup.
   Compare routed expert outputs, logits and generated tokens; inspect arena/VRAM byte-check diagnostics.
3. Check CUDA copy/kernel timelines, event reuse across layers and calls, teardown, and memory use. Measure
   physical SSD bytes separately from logical counters, then compare same-day baseline latency and throughput.
4. Include Engram SSD misses, cold/warm caches, prefill and concurrent Comfy work in later resource measurements.
   Choose the RAM/VRAM reserve and download limits from those observations before operational use.

For a later article, distinguish paper results, this source implementation, and eventual hardware measurements.
At this stage only the source implementation and its limitations can be reported.
