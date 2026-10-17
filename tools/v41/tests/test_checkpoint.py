"""Source identity checks on tiny files; no model, network, torch or GPU."""
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import struct
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v41_checkpoint import Checkpoint


class HeaderProvider:
    def get(self, layer, expert, role, source):
        return source.info("layers.1.engram.embed.weight")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.name = "model-00001-of-00001.safetensors"
        self.data = b"fixture checkpoint weights"
        (self.root / self.name).write_bytes(self.data)
        metadata = {"config.json": {}, "tokenizer.json": {},
                    "model.safetensors.index.json": {"weight_map": {"layers.1.engram.embed.weight": self.name}}}
        blobs = {}
        for name, value in metadata.items():
            data = json.dumps(value).encode()
            (self.root / name).write_bytes(data)
            blobs[name] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps({"repo": "fixture/1017", "revision": "1017",
            "name": "fixture", "metadata_blobs": blobs,
            "shards": {self.name: {"size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}}}))

    def source(self, shard_dirs=()):
        return Checkpoint(self.root, shard_dirs, profile=self.profile)

    def test_legacy_ok_marker_does_not_verify_1017(self):
        (self.root / (self.name + ".ok")).touch()
        with self.assertRaisesRegex(ValueError, "--verify"):
            self.source().verified_path(self.name)

    def test_verified_source_survives_worker_serialization(self):
        source = self.source()
        source.verify_shard(self.name)
        worker = pickle.loads(pickle.dumps(source))
        self.assertEqual(worker.verified_path(self.name), str(self.root / self.name))

    def test_same_size_wrong_weights_are_rejected(self):
        (self.root / self.name).write_bytes(b"X" * len(self.data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.source().verify_shard(self.name)
        self.assertFalse((self.root / (self.name + ".xolotl-verified.json")).exists())

    def test_changed_weights_invalidate_marker(self):
        source = self.source()
        source.verify_shard(self.name)
        (self.root / self.name).write_bytes(b"Y" * len(self.data))
        with self.assertRaisesRegex(ValueError, "--verify"):
            source.verified_path(self.name)

    def test_previous_revision_marker_is_rejected(self):
        source = self.source()
        source.verify_shard(self.name)
        marker = self.root / (self.name + ".xolotl-verified.json")
        data = json.loads(marker.read_text())
        data["revision"] = "previous"
        marker.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "--verify"):
            source.verified_path(self.name)

    def test_changed_metadata_is_rejected(self):
        (self.root / "config.json").write_text('{"different_model": true}')
        with self.assertRaisesRegex(ValueError, "not from pinned"):
            self.source()

    def test_shards_on_another_drive(self):
        other = self.root / "ssd"
        other.mkdir()
        (self.root / self.name).rename(other / self.name)
        source = self.source([other])
        source.verify_shard(self.name)
        self.assertEqual(source.verified_path(self.name), str(other / self.name))

    def test_ambiguous_copies_are_rejected(self):
        other = self.root / "ssd"
        other.mkdir()
        (other / self.name).write_bytes(self.data)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.source([other]).verify_shard(self.name)

    def test_path_outside_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in pinned"):
            self.source().verified_path("../previous.safetensors")

    def test_engram_extraction_retains_existing_partial(self):
        import extract_engram
        source = self.tensor_fixture()
        output = self.root / "engram"
        output.mkdir()
        partial = output / "engram.safetensors.partial"
        partial.write_bytes(b"retain this Engram output")
        with mock.patch.object(extract_engram.CP, "Checkpoint", return_value=source), \
                mock.patch.object(sys, "argv", ["extract_engram.py", "--checkpoint", str(self.root), str(output)]):
            with self.assertRaisesRegex(ValueError, "existing output retained"):
                extract_engram.main()
        self.assertEqual(partial.read_bytes(), b"retain this Engram output")
        self.assertFalse((output / "model.safetensors.index.json").exists())

    def tensor_fixture(self):
        header = json.dumps({"layers.1.engram.embed.weight":
                             {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        data = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        (self.root / self.name).write_bytes(data)
        profile = json.loads(self.profile.read_text())
        profile["shards"][self.name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.profile.write_text(json.dumps(profile))
        source = self.source()
        source.verify_shard(self.name)
        return source

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_spawn_export_worker_keeps_source_selection(self):
        import concurrent.futures
        import v41_gguf as export
        source = self.tensor_fixture()
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                initializer=export._worker_init, initargs=(HeaderProvider(), ["/missing/legacy"], source)) as pool:
            result = pool.submit(export._worker_get, (0, 0, "gate")).result(timeout=20)
        self.assertEqual(result[:2], ("F32", [1]))
        self.assertEqual(result[3], 4)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_explicit_source_retains_unqualified_partial(self):
        import v41_gguf as export
        source = self.tensor_fixture()
        target = str(self.root / "output.gguf")
        partial = Path(target + ".partial")
        partial.write_bytes(b"retain this interrupted output")
        with mock.patch.object(export.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            with self.assertRaisesRegex(ValueError, "partial file retained"):
                export.write_shard(target, [], [], export.Source(checkpoint=source), HeaderProvider(), 1, "fixture")
        self.assertEqual(partial.read_bytes(), b"retain this interrupted output")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_explicit_source_resumes_qualified_zero_tensor_partial(self):
        import v41_gguf as export
        source = self.tensor_fixture()
        target = str(self.root / "output.gguf")
        hdr = export.header([], [])
        Path(target + ".partial").write_bytes(hdr + b"interrupted tensor")
        Path(target + ".partial.json").write_text(json.dumps({
            "signature": hashlib.sha256(hdr + b"fixture|fixture/1017@1017").hexdigest(), "completed": 0}))
        with mock.patch.object(export.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            export.write_shard(target, [], [], export.Source(checkpoint=source), HeaderProvider(), 1, "fixture")
        self.assertEqual(Path(target).read_bytes(), hdr)
        self.assertFalse(Path(target + ".partial.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_explicit_source_rejects_legacy_journal_without_revision(self):
        import v41_gguf as export
        source = self.tensor_fixture()
        target = str(self.root / "output.gguf")
        hdr = export.header([], [])
        Path(target + ".partial").write_bytes(b"retain previous output")
        Path(target + ".partial.json").write_text(json.dumps({
            "signature": hashlib.sha256(hdr + b"fixture").hexdigest(), "completed": 0}))
        with self.assertRaisesRegex(ValueError, "different plan/source"):
            export.write_shard(target, [], [], export.Source(checkpoint=source), HeaderProvider(), 1, "fixture")
        self.assertEqual(Path(target + ".partial").read_bytes(), b"retain previous output")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_legacy_source_still_accepts_legacy_marker(self):
        import v41_gguf as export
        self.tensor_fixture()
        (self.root / (self.name + ".ok")).touch()
        source = export.Source(dirs=[str(self.root)])
        try:
            self.assertIsNone(source.checkpoint)
            self.assertEqual(source.info("layers.1.engram.embed.weight")[:2], ("F32", [1]))
        finally:
            for fd in source.fds.values():
                os.close(fd)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "export integration requires numpy")
    def test_older_gguf_cannot_be_verified_as_1017(self):
        import v41_gguf as export
        source = self.tensor_fixture()
        path = self.root / "old.gguf"
        path.write_bytes(export.header([], [export.MD.kv_string("strata.v41.source_revision", "previous")]))
        with self.assertRaisesRegex(ValueError, "selected checkpoint revision"):
            export.verify(str(path), export.Source(checkpoint=source), HeaderProvider(), {}, [])


if __name__ == "__main__":
    unittest.main()
