use super::common::{Gpu, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::{
    Tensor,
    linear::{fp8_compute, moe::GroupSeat},
};

fn decode(code: u8) -> f32 {
    let e = (code >> 3) & 15;
    let m = code & 7;
    let v = if e == 0 {
        f32::from(m) / 512.
    } else {
        (1. + f32::from(m) / 8.) * 2f32.powi(i32::from(e) - 7)
    };
    if code & 128 != 0 { -v } else { v }
}
fn rounded(x: f32) -> f32 {
    let mut best = 0u8;
    let mut error = f32::INFINITY;
    for c in 0..127u8 {
        let d = (decode(c) - x.abs()).abs();
        if d < error || (d == error && c & 1 == 0) {
            error = d;
            best = c;
        }
    }
    decode(best) * if x.is_sign_negative() { -1. } else { 1. }
}
fn quantized(x: &[u16], k: usize) -> Vec<f32> {
    let mut out = vec![0.; x.len()];
    for (row, values) in x.chunks(k).enumerate() {
        for (b, block) in values.chunks(128).enumerate() {
            let max = block
                .iter()
                .map(|v| from_bf16(*v).abs())
                .fold(1e-4f32, f32::max);
            let scale = 2f32.powf((max / 448.).log2().ceil());
            for (j, value) in block.iter().enumerate() {
                out[row * k + b * 128 + j] = rounded(from_bf16(*value) / scale) * scale;
            }
        }
    }
    out
}
fn inputs(rows: usize, k: usize) -> Vec<u16> {
    (0..rows * k)
        .map(|i| {
            to_bf16(if rows > 1 && i / k == 0 {
                0.
            } else {
                ((i * 37 % 137) as f32 - 68.) / 39. * 2f32.powi((i % k / 128) as i32 - 2)
            })
        })
        .collect()
}
fn assert_close(got: &[u16], want: &[f32]) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let expected = from_bf16(to_bf16(w));
        assert!(
            (from_bf16(g) - expected).abs() <= 0.012 * expected.abs().max(1.),
            "at {i}: got {} expected {expected} (unrounded {w})",
            from_bf16(g)
        );
    }
}

#[test]
fn fp8_experts_use_each_selected_experts_block_scales_for_both_input_layouts() {
    const E: usize = 3;
    const N: usize = 256;
    const K: usize = 640;
    for tokens in [1usize, 3, 13] {
        const TOP: usize = 2;
        let mut gpu = Gpu::open();
        let ctx = gpu.ctx();

        let routes: Vec<i32> = (0..tokens * TOP)
            .map(|i| {
                if i % 11 == 7 {
                    -1
                } else if i % 11 == 8 {
                    E as i32
                } else {
                    (i % E) as i32
                }
            })
            .collect();
        let codes: Vec<u8> = (0..E * N * K)
            .map(|i| (24 + i * 13 % 64) as u8 | if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let scales: Vec<f32> = (0..E * (N / 128) * (K / 128))
            .map(|i| (0.7 + 0.13 * i as f32) * 2f32.powi((i % 5) as i32 - 3))
            .collect();
        for by_token in [false, true] {
            let rows = if by_token { tokens } else { tokens * TOP };
            let x = inputs(rows, K);
            let q = quantized(&x, K);
            let mut want = vec![0.; tokens * TOP * N];
            for r in 0..tokens * TOP {
                let e = routes[r] as usize;
                if e >= E {
                    continue;
                }
                let xr = if by_token { r / TOP } else { r };
                for n in 0..N {
                    for k in 0..K {
                        want[r * N + n] += q[xr * K + k]
                            * decode(codes[(e * N + n) * K + k])
                            * scales[(e * (N / 128) + n / 128) * (K / 128) + k / 128];
                    }
                }
            }
            let a = Tensor::new(gpu.up(&x), rows as u32, K as u32, Dtype::Bf16);
            let c = Tensor::new(gpu.up(&codes), E as u32, (N * K) as u32, Dtype::E4m3);
            let s = Tensor::new(
                gpu.up(&scales),
                E as u32,
                (N / 128 * K / 128 * 4) as u32,
                Dtype::U8,
            );
            let r = Tensor::new(gpu.up(&routes), tokens as u32, TOP as u32, Dtype::I32);
            let mut y = Tensor::new(
                gpu.zeros(tokens * TOP * N * 2),
                (tokens * TOP) as u32,
                N as u32,
                Dtype::Bf16,
            );
            fp8_compute::experts(
                &ctx,
                "test.fp8",
                a,
                c,
                s,
                None,
                r,
                &mut y,
                GroupSeat::RESIDENT,
            )
            .unwrap();
            gpu.sync();
            assert_close(&gpu.down::<u16>(y.ptr, tokens * TOP * N), &want);
        }
    }
}

#[test]
fn fp8_grouped_stage_preserves_inactive_routes_and_relocates_the_window() {
    for experts in [1usize, 16] {
        let (tokens, top, n, k) = (4usize, 2usize, 128usize, 640usize);
        let mut gpu = Gpu::open();
        let ctx = gpu.ctx();

        for by_token in [false, true] {
            let rows = if by_token { tokens } else { tokens * top };
            let x = inputs(rows, k);
            let q = quantized(&x, k);
            // Every nibble is 1.0, every weight scale is 1.0.
            let a = Tensor::new(gpu.up(&x), rows as u32, k as u32, Dtype::Bf16);
            let c = Tensor::new(
                gpu.up(&vec![0x38u8; experts * n * k]),
                experts as u32,
                (n * k) as u32,
                Dtype::E4m3,
            );
            let s = Tensor::new(
                gpu.up(&vec![1f32; experts * n / 128 * k / 128]),
                experts as u32,
                (n / 128 * k / 128 * 4) as u32,
                Dtype::U8,
            );
            let r = Tensor::new(
                gpu.up(&vec![0i32; tokens * top]),
                tokens as u32,
                top as u32,
                Dtype::I32,
            );
            let initial = vec![to_bf16(-99.); tokens * top * n];
            let mut y = Tensor::new(
                gpu.up(&initial),
                (tokens * top) as u32,
                n as u32,
                Dtype::Bf16,
            );
            let stage = gpu.up(&[1u32, 2u32]);
            ctx.arm_stage(stage);
            fp8_compute::experts(
                &ctx,
                "test.fp8",
                a,
                c,
                s,
                None,
                r,
                &mut y,
                GroupSeat::RESIDENT,
            )
            .unwrap();
            gpu.sync();
            ctx.disarm_stage();
            let got = gpu.down::<u16>(y.ptr, tokens * top * n);
            for route in 0..tokens * top {
                for col in 0..n {
                    if (4..6).contains(&route) {
                        let xr = if by_token { route / top } else { route };
                        assert_close(
                            &got[route * n + col..route * n + col + 1],
                            &[q[xr * k..(xr + 1) * k].iter().sum()],
                        );
                    } else {
                        assert_eq!(got[route * n + col], to_bf16(-99.));
                    }
                }
            }
        }
    }
}

#[test]
fn fp8_f16_and_graph_replay_recompute_scales() {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let a = Tensor::new(gpu.up(&vec![0x3800u16; 640]), 1, 640, Dtype::F16);
    let w = Tensor::new(gpu.up(&vec![0x38u8; 128 * 640]), 1, 128 * 640, Dtype::E4m3);
    let sf = Tensor::new(gpu.up(&[0.3f32; 5]), 1, 20, Dtype::U8);
    let routes = Tensor::new(gpu.up(&[0i32]), 1, 1, Dtype::I32);
    let mut y = Tensor::new(gpu.zeros(128 * 2), 1, 128, Dtype::F16);
    let mut fire = || {
        fp8_compute::experts(
            &ctx,
            "test.fp8",
            a,
            w,
            sf,
            None,
            routes,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .unwrap()
    };
    fire();
    gpu.sync();
    let check = |s| assert_eq!(s, rt::cudaError::cudaSuccess);
    unsafe {
        check(rt::cudaStreamBeginCapture(
            ctx.stream().cast(),
            rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
        ));
        fire();
        let mut graph = core::ptr::null_mut();
        check(rt::cudaStreamEndCapture(
            ctx.stream().cast(),
            &raw mut graph,
        ));
        let mut exec = core::ptr::null_mut();
        check(rt::cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0));
        let updated = vec![0x4000u16; 640]; // f16 2.0; result = 640 * 2 * .3 = 384
        check(rt::cudaMemcpy(
            a.ptr as *mut _,
            updated.as_ptr().cast(),
            1280,
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ));
        check(rt::cudaGraphLaunch(exec, ctx.stream().cast()));
        gpu.sync();
        assert!(gpu.down::<u16>(y.ptr, 128).iter().all(|v| *v == 0x5e00));
        check(rt::cudaGraphExecDestroy(exec));
        check(rt::cudaGraphDestroy(graph));
    }
}
