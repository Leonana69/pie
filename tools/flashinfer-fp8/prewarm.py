"""Download and verify default Qwen-shape cubins before native serving.

Run alone on the GPU after build.py. The 16-expert case covers the optional test;
the 512-expert case covers the full model. Peak live tensors need about 3 GiB.
"""
import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

HIDDEN, INTERMEDIATE, TOP_K = 2560, 640, 10
for experts in (16, 512):
    up = torch.zeros(experts, 2 * INTERMEDIATE, HIDDEN,
                     device="cuda", dtype=torch.float8_e4m3fn)
    down = torch.zeros(experts, HIDDEN, INTERMEDIATE,
                       device="cuda", dtype=torch.float8_e4m3fn)
    up_scales = torch.ones(experts, 2 * INTERMEDIATE // 128, HIDDEN // 128,
                           device="cuda")
    down_scales = torch.ones(experts, HIDDEN // 128, INTERMEDIATE // 128,
                             device="cuda")
    for rows in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.ones(HIDDEN // 128, rows, device="cuda")
        ids = torch.arange(TOP_K, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
        weights = torch.full((rows, TOP_K), 1 / TOP_K, device="cuda")
        trtllm_fp8_block_scale_routed_moe(
            (ids, weights), None, x, scales, up, up_scales, down, down_scales,
            experts, TOP_K, None, None, INTERMEDIATE, 0, experts, 1.,
            routing_method_type=4, enable_pdl=False,
        )
        torch.cuda.synchronize()
        print(f"Warmed experts={experts}, rows={rows}", flush=True)
    del up, down, up_scales, down_scales
