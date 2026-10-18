"""CPU-only tiny fixture generator/validator driver; no GPU initialization."""
import argparse,sys,subprocess,json,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--test',required=True);p.add_argument('--out',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.repo)/'third_party/llama.cpp/gguf-py'));import gguf
root=Path(a.out);root.mkdir(exist_ok=False)
meta={'source_repo':'deepseek-ai/DeepSeek-V4.1-Flash-1017','source_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','source_tensor':'embed.weight','source_shard_sha256':'43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7'}
def make(name,change=None,arch='deepseek41-embedding',tensor='token_embd.weight',data=None,dtype=gguf.GGMLQuantizationType.BF16,extra=False,missing=False):
 dest=root/(name+'.gguf');w=gguf.GGUFWriter(str(dest),arch);m={**meta,**(change or {})}
 for k,v in m.items():w.add_string('deepseek41.embedding.'+k,v)
 if data is None:data=np.full((4,256),0x3f80,np.uint16)
 if not missing:w.add_tensor(tensor,data,raw_dtype=dtype)
 if extra:w.add_tensor('other',np.zeros((1,256),np.float32))
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close();return dest
valid=make('valid');bad=[]
for key in meta:bad.append(make('bad-'+key,{key:'wrong'}))
bad+=[make('bad-arch',arch='other'),make('bad-name',tensor='other'),make('bad-shape',data=np.ones((4,128),np.uint16)),make('bad-type',data=np.ones((4,256),np.float32),dtype=gguf.GGMLQuantizationType.F32),make('bad-extra',extra=True),make('bad-missing',missing=True)]
for name,bits in [('nan',0x7fc1),('positive-inf',0x7f80),('negative-inf',0xff80)]:
 arr=np.full((4,256),0x3f80,np.uint16);arr[-1,-1]=bits;bad.append(make('bad-'+name,data=arr))
trunc=root/'bad-truncated.gguf';trunc.write_bytes(valid.read_bytes()[:-1]);bad.append(trunc)
# Valid finite edge values: signed zero and smallest positive/negative subnormal.
edge=np.full((4,256),0x3f80,np.uint16);edge[0,:4]=[0,0x8000,1,0x8001];edges=make('valid-edge',data=edge)
subprocess.run([a.test,str(valid),*[str(x) for x in bad]],check=True)
subprocess.run([a.test,str(edges),*[str(x) for x in bad]],check=True)
(root/'results.json').write_text(json.dumps({'valid':2,'invalid':len(bad),'passed':True},indent=2))
