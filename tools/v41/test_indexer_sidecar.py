"""Generate bounded sidecar validation fixtures and run the CPU validator."""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/llama.cpp/gguf-py"))
import gguf

def write(path, mode, combined=False):
    w = gguf.GGUFWriter(str(path), "wrong-indexer" if mode == "architecture" else ("deepseek41-retrieval" if combined else "deepseek41-indexer"))
    w.add_string("deepseek41.indexer.source_repo", "wrong" if mode == "source" else "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    w.add_string("deepseek41.indexer.source_revision", "wrong" if mode == "revision" else "1bae0e1088993b9b82de08f59de5be5a34e42170")
    count = 35 if combined else 24
    for i in range(count - (mode == "missing")):
        name = f"blk.{i}.indexer.attn_k.weight"
        if mode == "unknown" and i == 0: name = "unrecognized.weight"
        data = np.ones((2, 2), dtype=np.float32)
        if i == 0:
            if mode == "shape": data = np.ones((3, 2), dtype=np.float32)
            if mode == "type": data = data.astype(np.float16)
            if mode == "nonfinite": data[0, 0] = np.nan
        w.add_tensor(name, data)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    if mode == "truncated": path.write_bytes(path.read_bytes()[:-64])
    if mode == "duplicate":
        payload = path.read_bytes()
        old = b"blk.1.indexer.attn_k.weight"
        assert payload.count(old) == 1
        path.write_bytes(payload.replace(old, b"blk.0.indexer.attn_k.weight"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="indexer-sidecar-") as tmp:
        root = Path(tmp)
        modes = ["valid", "shape", "type", "missing", "unknown", "nonfinite", "source", "revision", "architecture", "truncated", "duplicate"]
        for combined in (False, True):
            paths = [root / (str(combined) + mode + ".gguf") for mode in modes]
            for mode, path in zip(modes, paths): write(path, mode, combined)
            subprocess.run([args.validator, *map(str, paths)], check=True)

if __name__ == "__main__": main()
