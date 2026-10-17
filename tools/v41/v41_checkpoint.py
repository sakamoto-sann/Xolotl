"""Pinned V4.1 source selection and offline shard verification; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile


PROFILE = Path(__file__).with_name("checkpoints") / "deepseek-v41-flash-1017.json"


def stamp(path):
    s = path.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "device": s.st_dev, "inode": s.st_ino}


class Checkpoint:
    def __init__(self, root, shard_dirs=(), profile=PROFILE):
        self.root = Path(root).resolve()
        self.dirs = [str(self.root)] + [str(Path(p).resolve()) for p in shard_dirs]
        self.profile = json.loads(Path(profile).read_text())
        self.repo = self.profile["repo"]
        self.revision = self.profile["revision"]
        self.name = self.profile["name"]
        self.url = f"https://huggingface.co/{self.repo}/tree/{self.revision}"
        self.check_metadata()

    def check_metadata(self):
        for name, oid in self.profile["metadata_blobs"].items():
            data = (self.root / name).read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if actual != oid:
                raise ValueError(f"{name}: not from pinned {self.repo}@{self.revision}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        if set(index["weight_map"].values()) != set(self.profile["shards"]):
            raise ValueError("checkpoint index and pinned shard set differ")

    def path(self, name):
        if name not in self.profile["shards"] or Path(name).name != name:
            raise ValueError(f"shard is not in pinned source: {name}")
        found = [Path(d) / name for d in self.dirs if (Path(d) / name).is_file()]
        if not found:
            raise FileNotFoundError(f"missing source shard: {name}")
        if len({str(p.resolve()) for p in found}) != 1:
            raise ValueError(f"ambiguous source shard: {name}")
        return found[0]

    def identity(self, name, path):
        return {"repo": self.repo, "revision": self.revision,
                "sha256": self.profile["shards"][name]["sha256"], **stamp(path)}

    def verified_path(self, name):
        path = self.path(name)
        expected = self.identity(name, path)
        if expected["size"] != self.profile["shards"][name]["size"]:
            raise ValueError(f"source shard size mismatch: {name}")
        marker = Path(str(path) + ".xolotl-verified.json")
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise ValueError(f"{name}: run v41_checkpoint.py --verify for this pinned source")
        return str(path)

    def verify_shard(self, name):
        path = self.path(name)
        before = self.identity(name, path)
        if before["size"] != self.profile["shards"][name]["size"]:
            raise ValueError(f"source shard size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != before["sha256"]:
            raise ValueError(f"source shard SHA-256 mismatch: {name}")
        if self.identity(name, path) != before:
            raise ValueError(f"source shard changed during verification: {name}")
        marker = Path(str(path) + ".xolotl-verified.json")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as fp:
            temporary = Path(fp.name)
            json.dump(before, fp, sort_keys=True)
        try:
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)


def add_arguments(parser):
    parser.add_argument("--checkpoint", help="official 1017 HF source directory; pinned profile, opt-in")
    parser.add_argument("--shard-dir", action="append", default=[], help="additional source shard directory")


def from_arguments(args):
    if args.shard_dir and not args.checkpoint:
        raise ValueError("--shard-dir requires --checkpoint")
    return Checkpoint(args.checkpoint, args.shard_dir) if args.checkpoint else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--verify", action="store_true", help="hash shards and write source-bound markers")
    parser.add_argument("--engram-only", action="store_true", help="check only shards containing Engram tables")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    source = from_arguments(args)
    names = sorted(source.profile["shards"])
    if args.engram_only:
        index = json.loads((source.root / "model.safetensors.index.json").read_text())["weight_map"]
        names = sorted({v for k, v in index.items() if ".engram.embed." in k})
        if not names:
            raise ValueError("checkpoint contains no Engram tables")
    for name in names:
        path = source.verify_shard(name) if args.verify else source.verified_path(name)
        print(path, flush=True)
    print(f"verified {len(names)} shards: {source.repo}@{source.revision}")


if __name__ == "__main__":
    main()
