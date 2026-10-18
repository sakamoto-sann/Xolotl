"""Optional 1017 indexer precision restoration. Tensor ranges only; full-shard hashes are not verified."""
import sys,json,struct,urllib.request,hashlib,time,os,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description="Fetch only pinned 1017 indexer tensor ranges and write an F32 GGUF sidecar; no GPU initialization or full-shard download.")
parser.add_argument("--checkpoint-root",required=True,help="Directory with pinned config and safetensors index metadata")
parser.add_argument("--cache-dir",required=True,help="Dedicated directory for downloaded indexer ranges")
parser.add_argument("--out",required=True,help="New GGUF output; existing files are retained")
parser.add_argument("--include-compressor",action="store_true",help="Experimental: also restore the 11 compressed-KV projection/norm tensors")
args=parser.parse_args()
srcroot=Path(args.checkpoint_root);out=Path(args.cache_dir);dest=Path(args.out)
if dest.exists() or Path(str(dest)+".tmp").exists():raise ValueError("output or staged output exists; retained")
out.mkdir(parents=True,exist_ok=True)
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/"../../third_party/llama.cpp/gguf-py"))
from v41_checkpoint import Checkpoint
from v41_gguf import Source
import gguf
cp=Checkpoint(srcroot);wm=json.loads((srcroot/"model.safetensors.index.json").read_text())["weight_map"]
names=sorted(k for k in wm if ".attn.indexer." in k or (args.include_compressor and ".attn.compressor." in k))
if len(names)!=(43 if args.include_compressor else 32) or any("/" in n or "\\" in n for n in names):raise ValueError("unsupported indexer tensor set")
manifest={"repo":cp.repo,"revision":cp.revision,"verification":"HTTPS byte ranges from pinned revision; individual tensor hashes, not full-shard verification","tensors":{}}
old_manifest={}
if (out/"range-manifest.json").exists():
 old_manifest=json.loads((out/"range-manifest.json").read_text())
 if old_manifest.get("repo")!=cp.repo or old_manifest.get("revision")!=cp.revision:raise ValueError("cached manifest source mismatch")
def fetch(shard,start,n):
 expected_total=cp.profile["shards"][shard]["size"]
 url=f"https://huggingface.co/{cp.repo}/resolve/{cp.revision}/{shard}?download=true&xolotl_range={start}-{start+n-1}"
 req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{start+n-1}","Accept-Encoding":"identity"})
 with urllib.request.urlopen(req,timeout=90) as response:
  if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{start+n-1}/{expected_total}":raise RuntimeError("range response mismatch")
  data=response.read(n+1)
  if len(data)!=n:raise RuntimeError("range size mismatch")
 return data
headers={}
for shard in sorted({wm[n] for n in names}):
 path=out/(shard+".header.json")
 if path.exists():
  obj=json.loads(path.read_text())
  if obj.get("repo")!=cp.repo or obj.get("revision")!=cp.revision:raise ValueError("cached header belongs to another source")
  h=obj["header"];base=obj["base"]
 else:
  length=struct.unpack("<Q",fetch(shard,0,8))[0];assert 0<length<16*1024**2
  h=json.loads(fetch(shard,8,length));base=8+length
  path.write_text(json.dumps({"repo":cp.repo,"revision":cp.revision,"header":h,"base":base}))
 headers[shard]=(h,base)
 print("header",shard,flush=True)
for name in names:
 shard=wm[name];h,base=headers[shard];item=h[name];a,b=item["data_offsets"];assert 0<=a<b<=cp.profile["shards"][shard]["size"]-base
 path=out/(name+".bin")
 known=old_manifest.get("tensors",{}).get(name)
 if path.exists() and known:
  data=path.read_bytes()
  if hashlib.sha256(data).hexdigest()!=known["sha256"]:raise ValueError("cached tensor hash mismatch")
 else:
  data=fetch(shard,base+a,b-a)
  if path.exists() and path.read_bytes()!=data:raise ValueError("untracked cached tensor differs; retained")
  if not path.exists():
   tmp=path.with_suffix(".partial");tmp.write_bytes(data);tmp.replace(path)
 if len(data)!=b-a:raise ValueError("tensor byte count mismatch")
 manifest["tensors"][name]={"shard":shard,"dtype":item["dtype"],"shape":item["shape"],"offset":base+a,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
 (out/"range-manifest.json").write_text(json.dumps(manifest,indent=2))
 print("tensor",name,item["dtype"],len(data),flush=True)
class LocalRange(Source):
 def __init__(self):super().__init__(dirs=[str(srcroot)])
 def fp8_lut(self):
  bits=np.arange(256,dtype=np.uint8);e=((bits>>3)&15).astype(np.int32);m=(bits&7).astype(np.float32)
  values=np.where(e==0,m*np.float32(2**-9),(1+m/8)*np.exp2(e-7)).astype(np.float32)
  values=np.copysign(values,np.where(bits&128,-1.0,1.0)).astype(np.float32)
  values[(bits&127)==127]=np.nan
  assert values[0x7e]==448 and values[1]==2**-9 and np.signbit(values[128])
  return values
 def _shard(self,shard):return headers[shard]
 def read(self,name,start=0,n=None):
  data=(out/(name+".bin")).read_bytes()
  n=len(data)-start if n is None else n
  assert 0<=start<=len(data) and start+n<=len(data)
  return data[start:start+n]
source=LocalRange()

writer=gguf.GGUFWriter(str(dest)+".tmp","deepseek41-retrieval" if args.include_compressor else "deepseek41-indexer")
writer.add_string("deepseek41.indexer.source_repo",cp.repo);writer.add_string("deepseek41.indexer.source_revision",cp.revision)
writer.add_string("deepseek41.indexer.source_manifest_sha256",hashlib.sha256((out/"range-manifest.json").read_bytes()).hexdigest())
converted=[]
for name in names:
 if not name.endswith(".weight"):continue
 layer=int(name.split(".")[1]);role=name.split(".")[-2]
 target={"wq_b":"attn_q_b","weights_proj":"proj","wk":"attn_k","k_norm":"k_norm"}.get(role,role)
 array=np.ascontiguousarray(source.f32(name),np.float32);assert np.isfinite(array).all()
 output=f"blk.{layer}.indexer.{target}.weight"
 if ".attn.compressor." in name:
  target={"wkv":"kv","wgate":"gate","norm":"norm"}[role]
  output=f"blk.{layer}.attn_compressor_{target}.weight"
 writer.add_tensor(output,array)
 converted.append({"source":name,"name":output,"shape":list(array.shape),"bytes":array.nbytes,"sha256":hashlib.sha256(array.tobytes()).hexdigest()})
if len(converted)!=(35 if args.include_compressor else 24):raise ValueError("unexpected output tensor count")
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close();Path(str(dest)+".tmp").rename(dest)
manifest["sidecar"]={"path":str(dest),"bytes":dest.stat().st_size,"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"tensors":converted}
(out/"sidecar-manifest.json").write_text(json.dumps(manifest,indent=2))
print("SIDECAR",manifest["sidecar"]["bytes"],manifest["sidecar"]["sha256"],flush=True)
