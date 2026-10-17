# DeepSeek 1017 Q2_K direct input

The opt-in `ds41_generate --llama-q2-k -m <any-shard.gguf>` source path reads the
existing llama.cpp files directly. It does not merge files, convert model weights
or materialize the Engram tables in RAM. The default native GSQ path remains the
default. CUDA compilation and helper tests have passed; short full-model generation and measurements have passed.
See [runtime qualification](DEEPSEEK_1017_RUNTIME.md) for current evidence and measurement limits.

## Inspected artifact

- Repository: [mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF/tree/2c97dba3934f27a99767a90830e7519cc7b8b2d9/Q2_K)
- Revision: `2c97dba3934f27a99767a90830e7519cc7b8b2d9`.
- Ten Q2_K files, 264,515,416,800 bytes total (about 246.35 GiB).
- 1,046 tensors. Routed gate/up use Q2_K; routed down uses Q3_K. The filename's
  Q2_K label does not mean every tensor has that type.
- The two embedded Engram matrices use Q2_K with shape `[256, 384006168]` and
  `[256, 384016682]`. One row is one 84-byte block, instead of 256 float values.
- The dense weights use Q2_K, the output head Q6_K, and routers/Engram norms BF16.
  The latter are expanded to F32 on the selected backend; tables and expert weights
  stay packed. Token embeddings are dequantized one requested row at a time.

The checked-in `tools/v41/checkpoints/deepseek-v41-flash-1017-q2-k.json` records
the ten published file sizes and SHA-256 values. Header compatibility alone does
not establish the identity of the weight bytes. The runtime checks the input
contract; it does not hash all 265 GB during startup.

## Metadata and n-gram contract

llama.cpp uses different metadata names and omits some official config fields.
The opt-in adapter checks the represented geometry against the pinned official
1017 config (`1bae0e1088993b9b82de08f59de5be5a34e42170`), and supplies only the
missing fields from that explicit profile: KV sources `[2,8,14,20]`, index sources
`[2,8,14,20,24,28,32,36]`, candidate layer 20, 2,048 candidate blocks of size 8,
and the compressed vocabulary/table dimensions.

The 43-entry compression schedule contains 40 main and three draft entries;
only the 40 main entries are used. The token map, 48 bucket primes, 48 offsets,
eight hash multipliers and Engram layer IDs come from the file. Lengths, token
map bounds, bucket offsets and summed row counts are checked. The n-gram width
stays four, with eight heads and 256 elements per row. Three Engram tensor names
are aliased into the existing graph's names; the matrices are not transposed.

Each expert role retains its own shard and absolute offset. Buffered and direct
reads use that shard's descriptor, including partitioned/prefetched reads. All
ten files must be complete before the existing split reader can open the model.
Embedded Engram rows use the existing sorted, deduplicated, page-coalesced reader
and bounded raw row cache. The default two-million-row cache holds about 168 MB
of Q2_K payload, plus index overhead; `STRATA_ENGRAM_CACHE_ROWS` still controls it.

External `--engram`, DSpark sidecars and blob mirrors are refused in this CLI
mode until their matching/provenance and split behavior are qualified. Existing
native export and external FP8 Engram support are described separately.

## Checks that can run without a backend

These read headers and tiny fixtures, without model inference or a C++ build:

```sh
python3 tools/v41/tests/test_q2_input.py
python3 tools/v41/tests/test_checkpoint.py
# After every download is complete:
python3 tools/v41/v41_q2_input.py /models/Q2_K/DeepSeek-V4.1-Flash-1017-Q2_K-00001-of-00010.gguf
```

`v41_q2_input.py --verify` additionally hashes the entire artifact. That is
substantial I/O and must wait until ComfyUI no longer needs priority. The
preflight checks split numbers/counts, duplicate names, overlap, bounds, all
tensor shapes/types, and the pinned sizes. Without `--verify`, it explicitly
reports that weight hashes were not checked.

## Evidence and remaining qualification (2026-10-17)

All ten pinned model shards passed size/SHA-256 verification. Real-file Python preflight and C++ geometry checks
passed for all 1,046 tensors. CUDA build, Q2_K row parity, actual-dimension synthetic expert checks, pipeline
integration and short real generation passed. Arithmetic answered 42; eight 32-token code trials matched token IDs.
See [runtime qualification](DEEPSEEK_1017_RUNTIME.md) for exact versions, speeds, RAM and test limits.

The row decoder follows ggml's Q2_K layout: 16 scale/min bytes, 64 packed code bytes and two half scales.
The llama.cpp converter at `2076412598e99127ea2554467b2b7530856895d3` maps V4/V4.1 tensors without a RoPE
permutation. Matching-reference logits/routing, long context, broad quality and non-CUDA backends remain unqualified.
A separate read-only self-review was performed; no independent reviewer was available with delegation disabled.

Use an explicit RAM/VRAM budget on a 32 GB machine. The 60 GiB arena defaults are unsuitable here.
Initial comparisons use `--route-bias 0 --route-bias-file 0`; existing biased routing/ranks are not a 1017 quality baseline.
