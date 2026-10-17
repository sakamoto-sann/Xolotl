# DeepSeek 1017 Q2_K direct input

The opt-in `ds41_generate --llama-q2-k -m <any-shard.gguf>` source path reads the
existing llama.cpp files directly. It does not merge files, convert model weights
or materialize the Engram tables in RAM. The default native GSQ path remains the
default. CUDA compilation and helper tests have passed; full-model generation is still pending.
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

## Evidence and pending qualification (2026-10-17)

Completed: inspected only the ten remote headers (about 6 MB total), matched all
1,046 tensor names/shapes/types to the direct-input contract, checked declared
payload bounds/overlaps against published sizes, and passed 15 tiny Python
header/offset tests plus the existing 16 source-export tests. These are format
checks, not evidence that C++ compiles or inference is numerically correct.

Source inspection of llama.cpp converter commit
`2076412598e99127ea2554467b2b7530856895d3` found that its V4/V4.1 mapping returns
the named tensors without a RoPE permutation. The row decoder follows ggml's
Q2_K layout: 16 scale/min bytes, 64 packed code bytes and two half scales.
`test_ds41_q2_row` has now passed the 1,024-block comparison with ggml's decoder.
`test_ds41_geometry <file> --llama-q2-k` is built and awaiting all ten complete files.

Pending: run the real-file gates and native regressions, finish weight hash verification, compare logits/routing with a
matching llama.cpp reference, then assess context length, RAM/VRAM and speed on
the target GPU. No throughput, precision or successful 1017 generation is claimed.
Review so far is a separate read-only self-review; no independent reviewer was
available and delegation was disabled.

The initial source-only phase kept builds and inference stopped. The user has since lifted that hold and
prioritized Xolotl; build/helper results and the prepared inference procedure are recorded in the runtime notes.
The existing 60 GiB arena and large prompt defaults are not suitable as-is for
a 32 GB machine. Before any later run, choose an explicit RAM/VRAM budget and
small prompt/context, and use `--route-bias 0 --route-bias-file 0` for initial
reference comparisons. Existing routing biases/ranks are not a 1017 quality baseline.

This record separates source implementation, header evidence and pending runtime
results so it can later support an article without turning unmeasured behavior
into a performance claim.
