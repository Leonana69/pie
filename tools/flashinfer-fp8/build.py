"""Build an optional native PIE bridge against FlashInfer 0.7.0.post1.

Put the Python environment's bin directory on PATH so ninja is available.
"""
import argparse
import json
from pathlib import Path
import subprocess

import flashinfer
import tvm_ffi
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit import env

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
args = parser.parse_args()
if flashinfer.__version__ != "0.7.0.post1":
    raise RuntimeError("Bridge pins the FlashInfer 0.7.0.post1 native ABI")
module = gen_trtllm_gen_fused_moe_sm100_module()
module.build_and_load()
ffi = Path(tvm_ffi.__file__).parent
args.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    "c++", "-std=c++17", "-O2", "-shared", "-fPIC",
    str(Path(__file__).with_name("bridge.cpp")),
    "-I" + str(ffi / "include"), "-I" + str(args.cuda / "include"),
    "-L" + str(ffi / "lib"), "-ltvm_ffi",
    "-L" + str(args.cuda / "lib64"), "-lcudart", "-ldl",
    "-Wl,-rpath," + str(ffi / "lib"), "-Wl,-rpath," + str(args.cuda / "lib64"),
    "-DPIE_FI_MODULE=" + json.dumps(str(module.get_library_path())),
    "-DPIE_FI_CUBINS=" + json.dumps(str(env.FLASHINFER_CUBIN_DIR)),
    "-o", str(args.output),
], check=True)
print(args.output)
