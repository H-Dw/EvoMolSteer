"""Bounded tensor-only diagnostic: no FLOWR, candidates, LLM or experiment mutation."""
import concurrent.futures,gc,json,os,sys,threading,time
from pathlib import Path
import torch
from agenticgen.uniguide.chunking import concurrent_map
mode=sys.argv[1];destination=Path(sys.argv[2]);torch.set_num_threads(1)
torch.cuda.init();parent=torch.cuda.current_stream();device=torch.cuda.current_device()
base=torch.full((512,512),0.001,device='cuda');torch.cuda.synchronize()
seen=set();lock=threading.Lock();rows=[];checksum=None
local=threading.local();pool=None
if mode=='persistent':pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
def body(i):
 with lock:seen.add(int(torch.cuda.current_stream().cuda_stream))
 x=base.detach().clone().requires_grad_(True);y=x@base
 g=torch.autograd.grad(y.sum(),x)[0]
 return float(g.sum().item())
def persistent(i):
 with torch.cuda.device(device):
  if not hasattr(local,'stream'):local.stream=torch.cuda.Stream(device=device)
  local.stream.wait_stream(parent)
  with torch.cuda.stream(local.stream):
   try:return body(i)
   finally:local.stream.synchronize()
def sample(label):
 torch.cuda.synchronize()
 free,total=torch.cuda.mem_get_info()
 return dict(label=label,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=free,total=total,unique_streams=len(seen),host_threads=len(list(Path('/proc/self/task').iterdir())))
rows.append(sample('baseline'));started=time.time();interrupted=None
try:
 for i in range(20):
  values=list(pool.map(persistent,range(2))) if pool else concurrent_map(body,range(2),workers=2 if mode=='current' else 1)
  if checksum is None:checksum=values[0]
  if any(v!=checksum for v in values):raise RuntimeError('Matrix arithmetic changed')
  del values;gc.collect();rows.append(sample('iteration_'+str(i+1)))
  if rows[-1]['allocated']>4*1024**3:interrupted='engineering probe memory ceiling';break
finally:
 if pool:pool.shutdown(wait=True)
rows.append(sample('after_pool_shutdown'));torch.cuda.empty_cache();rows.append(sample('after_empty_cache'))
cleared=hasattr(torch._C,'_cuda_clearCublasWorkspaces')
if cleared:torch._C._cuda_clearCublasWorkspaces();gc.collect();torch.cuda.empty_cache();rows.append(sample('after_clear_blas_workspaces'))
result=dict(mode=mode,torch=str(torch.__version__),hip=str(torch.version.hip),gpu=torch.cuda.get_device_name(),matrix_shape=[512,512],workers=1 if mode=='sequential' else 2,iterations_completed=sum(x['label'].startswith('iteration_') for x in rows),seconds=time.time()-started,checksum=checksum,workspace_clear_available=cleared,interrupted=interrupted,rows=rows,random_draws=0,molecular_candidates=0,api_calls=0,environment={k:os.environ.get(k) for k in ['HIPBLAS_WORKSPACE_CONFIG','CUBLAS_WORKSPACE_CONFIG','ROCBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF','GPU_MAX_HW_QUEUES','HSA_NO_SCRATCH_RECLAIM']})
destination.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n')
print(json.dumps(dict(mode=mode,path=str(destination),seconds=result['seconds'],iterations=result['iterations_completed'],baseline=rows[0],before_clear=next(x for x in rows if x['label']=='after_empty_cache'),after_clear=rows[-1],checksum=checksum)))
