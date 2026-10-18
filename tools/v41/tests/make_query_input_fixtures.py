"""Tiny CPU artifacts exercise optional retrieval profile validation without a model/GPU."""
import argparse,struct,subprocess,copy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--test',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
def string(s):b=s.encode();return struct.pack('<Q',len(b))+b
layers=[2,8,14,20,24,28,32,36]
indexer=[f'blk.{l}.indexer.{role}.weight' for l in layers for role in (['attn_q_b','proj','attn_k','k_norm'] if l<24 else ['attn_q_b','proj'])]
compressor=[f'blk.{l}.attn_compressor_{role}.weight' for l in [2,8,14,20] for role in (['kv','gate','norm'] if l!=20 else ['kv','norm'])]
query=[f'blk.{l}.attn_q_a.weight' for l in layers]
assert (len(indexer),len(compressor),len(query))==(24,11,8)
def artifact(path,names,arch,mode=None):
 names=list(names);dims=[[4,2] for _ in names];types=[0]*len(names)
 meta={'general.architecture':arch,'deepseek41.indexer.source_repo':'deepseek-ai/DeepSeek-V4.1-Flash-1017','deepseek41.indexer.source_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170'}
 if mode=='unknown':names[-1]='blk.3.unexpected.weight'
 if mode=='duplicate':names[-1]=names[0]
 if mode=='shape':dims[-1]=[2,4]
 if mode=='type':types[-1]=1
 if mode=='source':meta['deepseek41.indexer.source_repo']='other/model'
 if mode=='revision':meta['deepseek41.indexer.source_revision']='0'*40
 if mode=='missing-meta':del meta['deepseek41.indexer.source_revision']
 if mode=='arch':meta['general.architecture']='deepseek41-retrieval-qinput-bad'
 if mode=='count':names.pop();dims.pop();types.pop()
 if mode=='legacy-arch':meta['general.architecture']='deepseek41-indexer'
 header=struct.pack('<4sIQQ',b'GGUF',3,len(names),len(meta))
 for k,v in meta.items():header+=string(k)+struct.pack('<I',8)+string(v)
 for i,n in enumerate(names):header+=string(n)+struct.pack('<I',2)+struct.pack('<QQ',*dims[i])+struct.pack('<IQ',types[i],i*32)
 data=bytearray(struct.pack('<8I',0x80000000,1,0x3f800000,0,0,0,0,0)*len(names))
 if mode in ['nan','inf']:struct.pack_into('<I',data,len(data)-4,0x7fc00000 if mode=='nan' else 0x7f800000)
 result=header+b'\0'*((-len(header))%32)+data
 if mode=='truncated':result=result[:-1]
 path.write_bytes(result)
for count,names,arch in [(24,indexer,'deepseek41-indexer'),(35,indexer+compressor,'deepseek41-retrieval'),(43,indexer+compressor+query,'deepseek41-retrieval-qinput')]:
 valid=out/f'valid{count}.gguf';artifact(valid,names,arch)
 invalid=[]
 for mode in ['unknown','duplicate','shape','type','source','revision','missing-meta','arch','count','nan','inf','truncated']+(['legacy-arch'] if count!=24 else []):
  path=out/f'{count}-{mode}.gguf';artifact(path,names,arch,mode);invalid.append(str(path))
 subprocess.run([a.test,str(valid),*invalid],check=True)
print('VALIDATOR_PROFILES_24_35_43_PASS')
