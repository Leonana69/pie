use super::common::{Gpu, captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype;
use kernels_cuda::{Tensor, linear::silu};

#[test]
fn the_scaled_silu_projection_preserves_rounding_and_staged_rows() {
    for (rows, n, k, scale) in [
        (1usize, 320usize, 10240usize, 0.25f32),
        (4, 320, 10240, -0.5),
        (2, 320, 10240, 0.),
    ] {
        let mut gpu = Gpu::open();
        let ctx = gpu.ctx();
        let x: Vec<_> = (0..(rows + 2) * k)
            .map(|i| {
                to_bf16((i as i32 * 13 % 31 - 15) as f32 / 16. * 2f32.powi(-((i % 13) as i32)))
            })
            .collect();
        let w: Vec<_> = (0..n * k)
            .map(|i| {
                to_bf16((i as i32 * 17 % 37 - 18) as f32 / 32. * 2f32.powi(-((i % 11) as i32)))
            })
            .collect();
        let a = Tensor::new(gpu.up(&x), rows as u32, k as u32, Dtype::Bf16);
        let weight = Tensor::new(gpu.up(&w), n as u32, k as u32, Dtype::Bf16);
        let sentinel = to_bf16(-99.);
        let mut out = Tensor::new(
            gpu.up(&vec![sentinel; (rows + 2) * n]),
            rows as u32,
            n as u32,
            Dtype::Bf16,
        );
        // The active rows start at one; both surrounding rows must survive.
        let window = gpu.up(&[rows as u32, 1u32]);
        ctx.arm_stage(window);
        silu::matmul_silu_scaled(&ctx, a, weight, scale, &mut out).unwrap();
        let graph = captured(&ctx, || {
            silu::matmul_silu_scaled(&ctx, a, weight, scale, &mut out).unwrap()
        });
        replay(&ctx, graph);
        gpu.sync();
        forget(graph);
        let got: Vec<u16> = gpu.down(out.ptr, (rows + 2) * n);
        assert!(
            got[..n]
                .iter()
                .chain(&got[(rows + 1) * n..])
                .all(|v| *v == sentinel)
        );
        for r in 1..=rows {
            for c in 0..n {
                // Wide dynamic range, accumulated in FP64 as a reference;
                // the kernel's FP32 reduction may differ by an output ULP.
                let dot: f64 = (0..k)
                    .map(|j| {
                        f64::from(from_bf16(x[r * k + j])) * f64::from(from_bf16(w[c * k + j]))
                    })
                    .sum();
                let v = from_bf16(to_bf16(dot as f32)) * scale;
                let want = from_bf16(to_bf16(v / (1. + (-v).exp())));
                let value = from_bf16(got[r * n + c]);
                assert!(
                    (value - want).abs() <= 0.008 * want.abs().max(0.01),
                    "row {r}, col {c}: {value} != {want}"
                );
            }
        }
        ctx.arm_stage(gpu.up(&[0u32, 1u32]));
        silu::matmul_silu_scaled(&ctx, a, weight, scale, &mut out).unwrap();
        gpu.sync();
        assert_eq!(gpu.down::<u16>(out.ptr, (rows + 2) * n), got);
        ctx.disarm_stage();
        // Decline prefill and incompatible weights, so the dispatcher falls back.
        assert!(!silu::supported(
            &ctx,
            Tensor { rows: 8, ..a },
            weight,
            Tensor { rows: 8, ..out }
        ));
        assert!(!silu::supported(
            &ctx,
            a,
            Tensor {
                dtype: Dtype::F16,
                ..weight
            },
            out
        ));
    }
}

#[test]
#[ignore = "manual GPU timing: run alone, --nocapture --test-threads=1"]
fn benchmark_qwen_scaled_silu_projection() {
    use kernels_cuda::{elemwise::norm, linear::gemm};
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let tuning = ctx.tuning();
    let ctx = ctx.with_tuning(kernels_cuda::tuning::Tuning {
        gemv_narrow: true,
        gemm_rows_m: 8,
        gemm_narrow_n: 8,
        gemm_skinny_rows: 0,
        pdl: true,
        ..tuning
    });
    let a = Tensor::new(gpu.up(&vec![to_bf16(0.125); 10240]), 1, 10240, Dtype::Bf16);
    // Distinct, nonconstant matrices avoid a cache/compression-friendly
    // constant-weight microbenchmark. Allocation and upload are not timed.
    let weights: Vec<_> = (0..97u64)
        .map(|bank| {
            let mut seed = bank + 1;
            let values: Vec<u16> = (0..320 * 10240)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let bits = (seed >> 32) as u16;
                    (bits & 0x8000) | 0x3a00 | (bits & 0x1ff)
                })
                .collect();
            Tensor::new(gpu.up(&values), 320, 10240, Dtype::Bf16)
        })
        .collect();
    let mut out = Tensor::new(gpu.zeros(320 * 2), 1, 320, Dtype::Bf16);
    let mut body = |fused: bool| {
        for &w in &weights {
            if fused {
                silu::matmul_silu_scaled(&ctx, a, w, 0.25, &mut out).unwrap();
            } else {
                gemm::matmul(&ctx, a, w, &mut out).unwrap();
                norm::silu_scaled(&ctx, 0.25, &mut out).unwrap();
            }
        }
    };
    body(false);
    body(true);
    gpu.sync();
    let baseline = captured(&ctx, || body(false));
    let fused = captured(&ctx, || body(true));
    for iteration in 0..6 {
        for (label, graph) in if iteration % 2 == 0 {
            [("before", baseline), ("after", fused)]
        } else {
            [("after", fused), ("before", baseline)]
        } {
            for _ in 0..10 {
                replay(&ctx, graph);
            }
            gpu.sync();
            let start = std::time::Instant::now();
            for _ in 0..100 {
                replay(&ctx, graph);
            }
            gpu.sync();
            eprintln!(
                "HC_PROJECTION {label} repeat={iteration} us_per_projection={:.4}",
                start.elapsed().as_secs_f64() * 1e6 / 9700.
            );
        }
    }
    forget(baseline);
    forget(fused);
}
