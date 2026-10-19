"""Compare the native bridge with FlashInfer's Python API and an FP32 reference.
Run alone on the GPU. Includes CUDA graph capture/replay.
"""
import argparse
from pathlib import Path
import torch, flashinfer, ctypes, json
parser = argparse.ArgumentParser()
parser.add_argument('--library', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit.cubin_loader import setup_cubin_loader
print('versions',torch.__version__,flashinfer.__version__,flush=True)
torch.manual_seed(93)
m=gen_trtllm_gen_fused_moe_sm100_module();m.build_and_load();setup_cubin_loader(str(m.get_library_path()))
E,H,I,K=16,2560,640,10
w1=(torch.randn(E,2*I,H,device='cuda')*.03).to(torch.float8_e4m3fn)
w2=(torch.randn(E,H,I,device='cuda')*.03).to(torch.float8_e4m3fn)
s1=torch.ones(E,2*I//128,H//128,device='cuda');s2=torch.ones(E,H//128,I//128,device='cuda')
work=torch.empty(192*1024*1024,dtype=torch.uint8,device='cuda')
lib=ctypes.CDLL(args.library);fn=lib.pie_flashinfer_fp8
fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]+[ctypes.c_void_p]*9+[ctypes.c_int]*5;fn.restype=ctypes.c_int
lib.pie_flashinfer_error.restype=ctypes.c_char_p
results=[]
cases=[]
for M in (1,4,128):
 x=torch.randn(M,H,device='cuda').bfloat16();scale=torch.pow(2,torch.ceil(torch.log2(x.float().view(M,H//128,128).abs().amax(-1).clamp_min(1.e-4)/448)))
 q=(x.float().view(M,H//128,128)/scale[...,None]).reshape(M,H).to(torch.float8_e4m3fn);xs=scale.T.contiguous()
 ids=torch.stack([torch.randperm(E,device='cuda')[:K] for _ in range(M)]).int();p=torch.rand(M,K,device='cuda');p/=p.sum(-1,keepdim=True)
 wf=torch.cat(w1.chunk(2,1)[::-1],dim=1)
 y=trtllm_fp8_block_scale_routed_moe((ids,p),None,q,xs,wf,s1,w2,s2,E,K,None,None,I,0,E,1.,routing_method_type=4,enable_pdl=False)
 if isinstance(y,list):y=y[0]
 torch.cuda.synchronize();print('python done',M,flush=True)
 cases.append((M,x,q,xs,ids,p,wf,y))
for M,x,q,xs,ids,p,wf,y in cases:
 # Loading the native bridge resets the shared cubin callback; all default tactics are now cached.
 out=torch.empty_like(x)
 call_args=[torch.cuda.current_stream().cuda_stream,work.data_ptr(),work.numel()]+[a.data_ptr() for a in (q,xs,ids,p,wf,s1,w2,s2,out)]+[M,H,I,E,K]
 rc=fn(*call_args)
 if rc:raise RuntimeError(lib.pie_flashinfer_error().decode())
 torch.cuda.synchronize();assert torch.equal(out,y),(M,(out-y).abs().max().item())
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  call_args[0]=torch.cuda.current_stream().cuda_stream
  rc=fn(*call_args)
  if rc:raise RuntimeError(lib.pie_flashinfer_error().decode())
 graph.replay();torch.cuda.synchronize();assert torch.equal(out,y)
 # FP32 reference checks gate/up ordering and routing; FP8 intermediate rounding differs.
 ref=torch.zeros(M,H,device='cuda')
 xd=q.float()*xs.T.repeat_interleave(128,-1)
 for j in range(K):
  selected=w1[ids[:,j].long()].float();z=torch.bmm(selected,xd[:,:,None]).squeeze(-1)
  a,b=z.chunk(2,dim=-1);h=torch.nn.functional.silu(a)*b
  v=torch.bmm(w2[ids[:,j].long()].float(),h[:,:,None]).squeeze(-1)
  ref+=p[:,j,None]*v
 err=(out.float()-ref).norm()/ref.norm();cos=torch.nn.functional.cosine_similarity(out.float().flatten(),ref.flatten(),dim=0)
 assert err.item() < 0.06 and cos.item() > 0.995, (M, err.item(), cos.item())
 print('PASS',M,'bridge exact; ref relative L2',err.item(),'cos',cos.item(),flush=True)
 results.append(dict(rows=M,bridge_exact=True,reference_relative_l2=err.item(),cosine=cos.item()))
args.output.write_text(json.dumps(results, indent=2))
