#!/usr/bin/env python3
"""Requires the pinned 1017 trunk and matching sidecar, NumPy and gguf-py; allocates no GPU memory.
Run with LD_LIBRARY_PATH set as for ds41_generate. Synthetic expert rows span multiple quantizer chunks.
"""
from pathlib import Path
import sys, subprocess, os, hashlib, argparse
import numpy as np
parser=argparse.ArgumentParser(description='CPU-only DSpark 1017 sidecar rejection and parallel requantization parity checks')
parser.add_argument('--build',type=Path,required=True)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--sidecar',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
repo=Path(__file__).resolve().parents[2];out=a.output;out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(repo/'third_party/llama.cpp/gguf-py'))
import gguf
source=a.sidecar
r=gguf.GGUFReader(str(source))
subprocess.run([str(a.build/'test_ds41_dspark'),'--validate-only','--llama-q2-k','--sidecar',str(source),'-m',str(a.model)],check=True)
rev='1bae0e1088993b9b82de08f59de5be5a34e42170'
for name,revision,complete in [('wrong-revision','wrong',False),('partial',rev,False),('wrong-shape',rev,True)]:
 p=out/(name+'.gguf');w=gguf.GGUFWriter(str(p),'deepseek41-sidecar')
 w.add_string('deepseek41.sidecar.source_revision',revision)
 w.add_string('deepseek41.sidecar.source','https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/'+rev)
 if complete:
  for t in r.tensors:w.add_tensor(t.name,np.zeros(1,np.float32))
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
 model=a.model
 cmd=[str(a.build/'test_ds41_dspark'),'--validate-only','--llama-q2-k','--sidecar',str(p),'-m',str(model)]
 result=subprocess.run(cmd,capture_output=True,text=True)
 assert result.returncode==1,(name,result)
 expected={'wrong-revision':'pinned official','partial':'sidecar missing','wrong-shape':'shape mismatch'}[name]
 assert expected in result.stderr,(name,result.stderr)
 print(name,'correctly refused',result.stderr.strip(),flush=True)
 env=os.environ.copy();env['STRATA_DSPARK']=str(p)
 result=subprocess.run([str(a.build/'ds41_generate'),'-m',str(model),'--llama-q2-k','--ids','3','-n','1'],env=env,capture_output=True,text=True)
 assert result.returncode==2 and 'DSpark preflight' in result.stderr and 'CUDA0' not in result.stderr,result.stderr
# Tiny nine-stack sidecar proves that parallel chunk ownership preserves every output byte, including nonexperts.
p=out/'synthetic-mx.gguf';w=gguf.GGUFWriter(str(p),'deepseek41-sidecar');rng=np.random.default_rng(42)
for b in range(3):
 for role in ('w1','w2','w3'):
  data=rng.integers(0,256,size=(2,8195,16,17),dtype=np.uint8);data[...,0]=120
  data=data.reshape(2,8195,16*17)
  w.add_tensor(f'mtp.{b}.ffn.experts.{role}.weight',data,raw_dtype=gguf.GGMLQuantizationType.MXFP4)
w.add_tensor('norm.weight',np.arange(64,dtype=np.float32))
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
for threads in (1,6):
 subprocess.run([str(a.build/'v41_sidecar_requant'),'--in',str(p),'--out',str(out/f'q2-{threads}.gguf'),'--type','Q2_K','--threads',str(threads)],check=True)
a=(out/'q2-1.gguf').read_bytes();b=(out/'q2-6.gguf').read_bytes();assert a==b
print('serial and 6-thread Q2_K sidecar output byte-identical',hashlib.sha256(a).hexdigest(),flush=True)
