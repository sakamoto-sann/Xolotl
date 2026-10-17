# DeepSeek V4.1 Flash 1017 source export

This document covers the opt-in official 1017 checkpoint export into Xolotl's native format.
The separate [Q2_K direct-input path](DEEPSEEK_1017_Q2_K.md) reads llama.cpp split files. Runtime
compatibility and performance remain unqualified until the exported files are
built and tested on the target machine. No 1017 inference result is claimed.

The pinned source is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Its metadata Git blob IDs and all 48
shard sizes/SHA-256 values are recorded in
`tools/v41/checkpoints/deepseek-v41-flash-1017.json`.

## Source selection

Put the original safetensors, config.json, tokenizer.json and
model.safetensors.index.json from that revision in a dedicated directory. Extra
shard directories can be passed with repeated `--shard-dir` flags. Duplicate
copies of a shard are rejected. This tool does not download or delete anything.

Before conversion, explicitly hash the source files:

```sh
python3 tools/v41/v41_checkpoint.py --checkpoint /models/ds41-1017 --verify
```

This reads the full checkpoint and writes source-bound verification markers.
It is substantial disk I/O; run it only when the machine is available. Legacy
empty `.ok` files cannot qualify a 1017 source. Changing or replacing a verified
shard invalidates its marker. Keep source files immutable during conversion.
Source-aware export retains existing sidecar, Engram and unqualified main-file
partial outputs rather than overwriting them. Use dedicated output paths.

The command without `--verify` only checks metadata, file stamps and existing
markers. `--engram-only` selects the shards containing the Engram tables.

## Native export commands

These are commands to run later, not an automatic install or benchmark:

```sh
python3 tools/v41/v41_gguf.py --checkpoint /models/ds41-1017 \
    --out /models/xolotl-1017/V41-1017-Q2_0.gguf --rtn q2_0 --no-engram --threads 2
python3 tools/v41/extract_engram.py --checkpoint /models/ds41-1017 \
    /models/xolotl-1017/engram
python3 tools/v41/v41_sidecar.py --checkpoint /models/ds41-1017 \
    --out /models/xolotl-1017/V41-1017-MTP.gguf --no-vision
python3 tools/v41/serve/export_tokenizer.py /models/ds41-1017 \
    /models/xolotl-1017/tokenizer
```

The main file uses the existing streaming RTN exporter and the native Q2_0
expert format, not Q2_K. This is not the calibrated GSQ model and has no measured
1017 quality baseline. Old calibration maps, prequantized expert directories
and in-place expert holes are rejected with `--checkpoint`; their source
provenance is not yet qualified. Do not substitute the published older GSQ
model, rank.bin or DSpark sidecar for the new 1017 weights.

The native main GGUF and sidecar record their source revision. Source-aware
verification rejects a GGUF/sidecar without the matching revision tag. RTN
worker processes receive the same source selection explicitly, including when
Python uses spawn rather than fork. Without `--checkpoint`, the existing source
selection and metadata remain unchanged.

Engram extraction verifies the pinned source shards, preserves their FP8 rows
and scales, and records the revision in the output index. Point the existing
engine's `--engram` option at this extracted directory so it can read rows from
SSD instead of loading the full tables. Do not mix extracted directories or
manually replace their files. Runtime cross-checking of the main file against
the external Engram index is not added in this change.

Budget disk space before exporting: the official checkpoint is about 510 GB,
the native main export about 163 GB and external Engram about 203 GB, plus a
sidecar and temporary files. Sizes depend on the selected expert format.
The 60 GiB default RAM arena also needs reducing on a 32 GB machine; no memory
or routing defaults are changed here. Build, inference and resource tuning are
separate follow-up steps.

## Small checks

```sh
python3 tools/v41/tests/test_checkpoint.py
```

These tests use tiny temporary files, without network, model weights, torch,
CUDA, a C++ build or inference. Actual export and runtime tests remain pending.
