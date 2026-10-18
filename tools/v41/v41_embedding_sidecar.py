"""Write an opt-in host-mapped BF16 embedding GGUF from the fully verified pinned1017 shard."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'../../third_party/llama.cpp/gguf-py'))
import gguf
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
src=Path(a.source);dest=Path(a.out);tmp=Path(str(dest)+'.tmp')
manifest=Path(str(dest)+'.manifest.json')
if dest.exists() or tmp.exists() or manifest.exists():raise ValueError('output or manifest exists; retained')
def stamp(path):
 s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
source_stamp=stamp(src)
expected='43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7'
def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(16<<20):h.update(block)
 return h.hexdigest()
if src.stat().st_size!=1323858272 or digest_file(src)!=expected:raise ValueError('not the pinned1017 embedding shard')
with src.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0]
 if not 0<n<16<<20:raise ValueError('header size')
 header=json.loads(f.read(n))
t=header['embed.weight'];shape=[129280,5120]
if t['dtype']!='BF16' or t['shape']!=shape or t['data_offsets']!=[0,1323827200]:raise ValueError('embedding schema mismatch')
arr=np.memmap(src,dtype='<u2',mode='r',offset=8+n,shape=tuple(shape))
for start in range(0,shape[0],128):
 if np.any((arr[start:start+128]&0x7f80)==0x7f80):raise ValueError('nonfinite BF16 embedding')
w=gguf.GGUFWriter(str(tmp),'deepseek41-embedding')
for key,value in {'source_repo':'deepseek-ai/DeepSeek-V4.1-Flash-1017','source_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','source_tensor':'embed.weight','source_shard_sha256':expected}.items():w.add_string('deepseek41.embedding.'+key,value)
w.add_tensor('token_embd.weight',arr,raw_dtype=gguf.GGMLQuantizationType.BF16)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
reader=gguf.GGUFReader(str(tmp));tensor=reader.tensors[0]
if len(reader.tensors)!=1 or list(tensor.shape)!=[5120,129280] or int(tensor.tensor_type)!=30:raise ValueError('written schema mismatch; staged file retained')
# Compare all payload bits in bounded chunks, independent of float conversion.
actual=np.asarray(tensor.data).view(np.uint16).reshape(shape)
for start in range(0,shape[0],128):
 if not np.array_equal(actual[start:start+128],arr[start:start+128]):raise ValueError('written payload differs; staged file retained')
# Close output mmap before rename so this also works on Windows.
del actual,tensor,reader
if stamp(src)!=source_stamp:raise ValueError('source changed during conversion; staged output retained')
with manifest.open('x') as mf:mf.write(json.dumps({'source':str(src),'source_shard_sha256':expected,'full_shard_verified':True,'tensor':'embed.weight','output_tensor':'token_embd.weight','dtype':'BF16','shape':shape,'all_payload_bits_verified':True,'output_sha256':digest_file(tmp)},indent=2))
tmp.rename(dest);print('verified original BF16 embedding GGUF:',dest,flush=True)
