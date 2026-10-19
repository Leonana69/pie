import ctypes,os,sys
ctypes.CDLL(None).prctl(41,1,0,0,0)
os.environ['PIE_HOME']='/workspace/qwen-fused-benchmark/state'
os.environ['LD_LIBRARY_PATH']='/tmp/pie-cuda13/nvidia/cu13/lib:/usr/local/cuda/lib64'
os.execv(sys.argv[1],[sys.argv[1],'-c','/workspace/qwen-fused-benchmark/config.toml','serve'])
