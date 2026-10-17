"""Read-only preflight for split GGUF input. Headers by default; full SHA only with --verify."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile

PROFILE = Path(__file__).with_name('checkpoints') / 'deepseek-v41-flash-1017-q2-k.json'


def split_paths(path):
    path = Path(path)
    match = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', path.name)
    if not match:
        return [path]
    stem, no, count = match.groups()
    no, count = int(no), int(count)
    if not 1 <= no <= count <= 1024:
        raise ValueError('invalid split file number/count')
    return [path.with_name(f'{stem}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count + 1)]


def validate_headers(headers, file_sizes):
    """Validate tensor locations without reading any payload. Returns name -> (path, offset, bytes)."""
    if not headers or len(headers) != len(file_sizes):
        raise ValueError('missing shard headers/sizes')
    first = headers[0].metadata
    if first.get('general.architecture') != 'deepseek41':
        raise ValueError('not a deepseek41 input')
    result, total = {}, 0
    declared = first.get('split.tensors.count')
    for i, (g, size) in enumerate(zip(headers, file_sizes)):
        meta = g.metadata
        if meta.get('split.count', 1) != len(headers) or meta.get('split.no', 0) != i:
            raise ValueError('split count/number mismatch')
        if len(headers) > 1 and (declared is None or meta.get('split.tensors.count') != declared):
            raise ValueError('split tensor count mismatch')
        spans = []
        for t in g.tensors:
            if t.name in result:
                raise ValueError('duplicate tensor: ' + t.name)
            n = t.expected_bytes()
            if not t.shape or any(d <= 0 for d in t.shape) or not n or t.offset % g.alignment:
                raise ValueError('invalid tensor layout: ' + t.name)
            start = g.data_start + t.offset
            if start < g.data_start or start + n > size:
                raise ValueError('truncated tensor: ' + t.name)
            spans.append((start, start + n))
            result[t.name] = (str(g.path), start, n)
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('overlapping tensor payloads')
        total += len(g.tensors)
    if declared is not None and declared != total:
        raise ValueError('total tensor count mismatch')
    return result


def expert_span(headers, name, expert):
    for g in headers:
        for t in g.tensors:
            if t.name != name:
                continue
            if len(t.shape) != 3 or not 0 <= expert < t.shape[2]:
                raise ValueError('expert id/shape out of range')
            size = t.expected_bytes()
            if not size or size % t.shape[2]:
                raise ValueError('invalid expert byte size')
            stride = size // t.shape[2]
            return str(g.path), g.data_start + t.offset + expert * stride, stride
    raise ValueError('missing expert tensor: ' + name)


def check_profile(headers, paths, profile):
    """Strictly bind the preflight to the published 10-file artifact (not a filename assertion alone)."""
    expected = profile['shards']
    if {p.name for p in paths} != set(expected):
        raise ValueError('not the pinned Q2_K shard family')
    for p in paths:
        if p.stat().st_size != expected[p.name]['size']:
            raise ValueError('incomplete/wrong-size shard: ' + p.name)
    m = headers[0].metadata
    for key, value in profile['metadata'].items():
        if m.get(key) != value:
            raise ValueError('unsupported input metadata: ' + key)
    validate_q2_tensors(headers)
    for layer in range(40):
        for role in ('gate', 'up', 'down'):
            expert_span(headers, f'blk.{layer}.ffn_{role}_exps.weight', 383)


def expected_tensors():
    """The inspected 1017 Q2_K contract, matching the opt-in C++ geometry's shapes and storage types."""
    result = {'token_embd.weight': ([5120, 129280], 10), 'output.weight': ([5120, 129280], 14),
              'output_norm.weight': ([5120], 0)}
    for layer in range(40):
        prefix = f'blk.{layer}.'
        def add(name, shape, type_id):
            result[prefix + name] = (shape, type_id)
        for site in ('attn', 'ffn'):
            add(f'hc_{site}_fn.weight', [20480, 24], 10)
            add(f'hc_{site}_base.weight', [24], 0)
            add(f'hc_{site}_scale.weight', [3], 0)
            add(f'{site}_norm.weight', [5120], 0)
        for name, shape, type_id in (
            ('attn_sinks.weight', [64], 0), ('attn_q_a.weight', [5120, 1280], 10),
            ('attn_q_b.weight', [1280, 32768], 10), ('attn_q_a_norm.weight', [1280], 0),
            ('attn_kv.weight', [5120, 512], 10), ('attn_kv_a_norm.weight', [512], 0),
            ('attn_output_a.weight', [4096, 8192], 10), ('attn_output_b.weight', [8192, 5120], 10),
            ('ffn_gate_inp.weight', [5120, 384], 30), ('exp_probs_b.bias', [384], 0),
            ('exp_probs_b_vl.bias', [384], 0)):
            add(name, shape, type_id)
        for role in ('gate', 'up', 'down'):
            shape, type_id = ([2304, 5120], 11) if role == 'down' else ([5120, 2304], 10)
            add(f'ffn_{role}_exps.weight', shape + [384], type_id)
            add(f'ffn_{role}_shexp.weight', shape, type_id)
        if layer in (2, 8, 14, 20):
            for name, shape, type_id in (
                ('attn_compressor_kv.weight', [5120, 512], 10), ('attn_compressor_norm.weight', [512], 0),
                ('indexer.attn_k.weight', [512, 128], 10), ('indexer.k_norm.weight', [128], 0)):
                add(name, shape, type_id)
            if layer != 20:
                add('attn_compressor_gate.weight', [5120, 512], 10)
        if layer in (2, 8, 14, 20, 24, 28, 32, 36):
            add('indexer.attn_q_b.weight', [1280, 4096], 10)
            add('indexer.proj.weight', [5120, 32], 10)
        if layer in (1, 14):
            add('engram_q.weight', [5120, 4], 30)
            add('engram_k.weight', [5120, 4], 30)
            add('engram_wkv.weight', [6144, 25600], 10)
            add('engram_embd.weight', [256, 384006168 if layer == 1 else 384016682], 10)
    return result


def validate_q2_tensors(headers):
    actual = {t.name: (t.shape, t.type_id) for g in headers for t in g.tensors}
    wanted = expected_tensors()
    if set(actual) != set(wanted):
        raise ValueError('unexpected/missing 1017 tensor names')
    for name, info in wanted.items():
        if actual[name] != info:
            raise ValueError('unsupported tensor shape/type: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='any completed member of the Q2_K family')
    parser.add_argument('--verify', action='store_true', help='read all 265 GB to verify SHA-256; substantial I/O')
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text())
    paths = split_paths(args.model)
    headers = [GGUFFile(p) for p in paths]
    spans = validate_headers(headers, [p.stat().st_size for p in paths])
    check_profile(headers, paths, profile)
    if args.verify:
        for p in paths:
            digest = hashlib.sha256()
            with p.open('rb') as fp:
                for block in iter(lambda: fp.read(8 << 20), b''):
                    digest.update(block)
            if digest.hexdigest() != profile['shards'][p.name]['sha256']:
                raise ValueError('SHA-256 mismatch: ' + p.name)
            print('SHA-256 verified:', p.name, flush=True)
    print(f'{len(paths)} headers, {len(spans)} tensors validated; full hashes {"verified" if args.verify else "NOT checked"}')
    print('No model conversion, backend initialization or inference was performed.')


if __name__ == '__main__':
    main()
