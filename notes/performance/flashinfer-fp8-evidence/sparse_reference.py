import torch,json
from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe
E,H,I,K,M=16,2560,640,10,128
up=torch.zeros(E,2*I,H,device='cuda');down=torch.zeros(E,H,I,device='cuda')
for j in range(I):up[:,j,j]=1;up[:,I+j,(j+3)%H]=2
for j in range(H):down[:,j,j%I]=1
up=torch.cat(up.chunk(2,1)[::-1],dim=1).to(torch.float8_e4m3fn);down=down.to(torch.float8_e4m3fn)
s1=torch.cat([torch.full((E,I//128,H//128),.25,device='cuda'),torch.full((E,I//128,H//128),.5,device='cuda')],dim=1)
s2=(.5+.25*(torch.arange(E,device='cuda')%3)).view(E,1,1).expand(E,H//128,I//128).contiguous()
x=((torch.arange(H,device='cuda')%8+1)/8).repeat(M,1).bfloat16();scale=torch.full((H//128,M),1/256,device='cuda');q=(x.float()*256).to(torch.float8_e4m3fn)
ids=((torch.arange(M*K,device='cuda')+K)%E).view(M,K).int();p=((torch.arange(M*K,device='cuda')%K)+1).view(M,K).float()/55
out=trtllm_fp8_block_scale_routed_moe((ids,p),None,q,scale,up,s1,down,s2,E,K,None,None,I,0,E,1.,routing_method_type=4,enable_pdl=False)
print(out[0,:16].tolist())
a=x.float()[:,:I]*.5;b=x.float()[:,(torch.arange(I,device='cuda')+3)%H]*.5
r=(torch.nn.functional.silu(a)*b).repeat(1,4)*(p*(.5+.25*(ids%3))).sum(-1)[:,None]
print(r[0,:16].tolist());print('L2',((out-r).norm()/r.norm()).item(),'maxrel',((out-r).abs()/r.abs()).max().item())

open("/workspace/qwen-flashinfer-benchmark/sparse-golden.json","w").write(json.dumps(out[:,:8].tolist()))
