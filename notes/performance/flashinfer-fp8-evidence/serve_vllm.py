import ctypes,os
ctypes.CDLL(None).prctl(41,1,0,0,0)
os.environ['PATH']='/tmp/pie-vllm-runtime/bin:/tmp/pie-vllm-runtime/lib/python3.12/site-packages/nvidia/cu13/bin:'+os.environ['PATH']
os.environ['CUDA_HOME']='/tmp/pie-vllm-runtime/lib/python3.12/site-packages/nvidia/cu13'
os.environ['FLASHINFER_WORKSPACE_BASE']='/tmp/vllm-flashinfer-cache'
os.environ['FLASHINFER_CUBIN_DIR']='/root/.cache/flashinfer/cubins'
os.environ['MAX_JOBS']='12'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
args=['/tmp/pie-vllm-runtime/bin/vllm','serve','/workspace/models/Qwen3.8-Flash-Next-FP8','--served-model-name','qwen-fp8','--host','127.0.0.1','--port','8000','--tensor-parallel-size','1','--gpu-memory-utilization','0.90','--max-model-len','2048','--max-num-seqs','1','--max-num-batched-tokens','128','--engram-config','{"cpu_offload":true}','--limit-mm-per-prompt','{"image":0,"video":0}','--no-enable-prefix-caching','--reasoning-parser','qwen3']
os.execv(args[0],args)
