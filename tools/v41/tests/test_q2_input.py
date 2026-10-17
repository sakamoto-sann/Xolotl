"""Tiny header/offset fixtures only. No build, weight conversion or inference."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v41_q2_input as Q
import v41_gguf_meta as M
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_reader import TensorInfo
import struct


def tiny_header(path, no, tensors):
    records = [M.kv_string('general.architecture', 'deepseek41'), M.kv_u32('split.count', 2),
               M.kv_u32('split.no', no), M.kv_u32('split.tensors.count', 2)]
    def string(s):
        data = s.encode()
        return struct.pack('<Q', len(data)) + data
    raw = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(records)) + b''.join(records)
    for t in tensors:
        raw += string(t.name) + struct.pack('<I', len(t.shape)) + struct.pack('<' + 'Q'*len(t.shape), *t.shape)
        raw += struct.pack('<IQ', t.type_id, t.offset)
    raw += bytes((-len(raw)) % 32)
    raw += bytes(max((t.offset + t.expected_bytes() for t in tensors), default=0))
    path.write_bytes(raw)
    return Q.GGUFFile(path)


class InputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = [self.root / f'model-{i:05d}-of-00002.gguf' for i in (1, 2)]
        self.names = ['blk.0.ffn_gate_exps.weight', 'blk.0.ffn_down_exps.weight']
        self.headers = [tiny_header(p, i, [TensorInfo(self.names[i], [256, 1, 2], 10, 'Q2_K', 0)])
                        for i, p in enumerate(self.paths)]

    def validate(self):
        return Q.validate_headers(self.headers, [p.stat().st_size for p in self.paths])

    def test_any_split_resolves_whole_family(self):
        self.assertEqual(Q.split_paths(self.paths[1]), self.paths)

    def test_roles_keep_distinct_shards_and_absolute_offsets(self):
        spans = self.validate()
        for i, name in enumerate(self.names):
            path, offset, size = Q.expert_span(self.headers, name, 1)
            self.assertEqual(path, str(self.paths[i]))
            self.assertEqual(offset, self.headers[i].data_start + 84)
            self.assertEqual(size, 84)
            self.assertEqual(spans[name][2], 168)

    def test_missing_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'split count'):
            Q.validate_headers(self.headers[:1], [self.paths[0].stat().st_size])

    def test_wrong_split_number_is_rejected(self):
        self.headers[1].metadata['split.no'] = 0
        with self.assertRaisesRegex(ValueError, 'split count/number'):
            self.validate()

    def test_mixed_split_totals_are_rejected(self):
        self.headers[1].metadata['split.tensors.count'] = 3
        with self.assertRaisesRegex(ValueError, 'split tensor count'):
            self.validate()

    def test_duplicate_tensor_is_rejected(self):
        self.headers[1].tensors[0].name = self.names[0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_truncated_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            Q.validate_headers(self.headers, [p.stat().st_size - 1 for p in self.paths])

    def test_out_of_range_expert_is_rejected(self):
        for expert in (-1, 2):
            with self.assertRaisesRegex(ValueError, 'out of range'):
                Q.expert_span(self.headers, self.names[0], expert)

    def test_misaligned_payload_is_rejected(self):
        self.headers[1].tensors[0].offset = 1
        with self.assertRaisesRegex(ValueError, 'invalid tensor layout'):
            self.validate()

    def test_overlapping_payloads_are_rejected(self):
        self.headers[0].tensors.append(TensorInfo('overlap', [256], 10, 'Q2_K', 0))
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.validate()

    def test_header_checks_do_not_hash_or_read_weights(self):
        with mock.patch.object(Q.hashlib, 'sha256', side_effect=AssertionError('must not hash')):
            self.assertEqual(len(self.validate()), 2)

    def q2_contract(self):
        return [SimpleNamespace(tensors=[SimpleNamespace(name=name, shape=shape, type_id=kind)
                 for name, (shape, kind) in Q.expected_tensors().items()])]

    def test_full_q2_contract_includes_embedded_engram_and_mixed_down_types(self):
        headers = self.q2_contract()
        Q.validate_q2_tensors(headers)
        self.assertEqual(len(headers[0].tensors), 1046)
        tables = [t for t in headers[0].tensors if 'engram_embd' in t.name]
        self.assertEqual([t.shape[0] for t in tables], [256, 256])
        self.assertEqual([t.type_id for t in tables], [10, 10])

    def test_wrong_dense_quantization_is_rejected(self):
        headers = self.q2_contract()
        headers[0].tensors[0].type_id = 1
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            Q.validate_q2_tensors(headers)

    def test_wrong_ngram_row_width_is_rejected(self):
        headers = self.q2_contract()
        next(t for t in headers[0].tensors if 'engram_embd' in t.name).shape = [512, 384006168]
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            Q.validate_q2_tensors(headers)

    def test_unexpected_draft_stack_is_rejected(self):
        headers = self.q2_contract()
        headers[0].tensors.append(SimpleNamespace(name='blk.40.unqualified_draft', shape=[1], type_id=0))
        with self.assertRaisesRegex(ValueError, 'tensor names'):
            Q.validate_q2_tensors(headers)


if __name__ == '__main__':
    unittest.main()
