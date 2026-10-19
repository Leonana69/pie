#![cfg(feature = "cuda")]

use engine_cuda::{Boot, Knobs, Lane, Shell};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Value, Weight, ops};
use poem_compiler::Budget;

struct Micro;
impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let x = ops::layout::embed(
            &inputs.tokens(),
            &Weight::sym("embed", [128, 10240], Dtype::Bf16),
            128,
        );
        let y = ops::linear::matmul(&x, &Weight::sym("projection", [320, 10240], Dtype::Bf16));
        ops::elemwise::silu_scaled(0.25, &y)
    }
}

#[test]
fn a_scaled_silu_projection_serves_with_prefill_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.zt");
    let mut writer = ztensor::Writer::create(&path).unwrap();
    let embed: Vec<u8> = vec![0x3f80u16; 128 * 10240]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    writer
        .add("embed", [128, 10240], ztensor::Leaf::BF16, &embed)
        .unwrap();
    // 10240 * 1/4096 = 2.5, and the epilogue is SiLU(0.625).
    let weights: Vec<u8> = vec![0x3980u16; 320 * 10240]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    writer
        .add("projection", [320, 10240], ztensor::Leaf::BF16, &weights)
        .unwrap();
    writer.finish().unwrap();
    let trace = poem::trace_hybrid("scaled-silu-projection", &Micro, Platform::Cuda);
    let src = ztensor::Source::open(&path).unwrap();
    let contract = poem::import::own_contract(&src, &trace.params, 1, Platform::Cuda).unwrap();
    for (graphs, fused) in [
        ("off", false),
        ("off", true),
        ("shaped", false),
        ("shaped", true),
    ] {
        let mut shell = Shell::load(Boot {
            trace: trace.clone(),
            contract: &contract,
            checkpoint: &path,
            budget: Budget::new(1, 16),
            patches: None,
            voxels: None,
            page_size: 16,
            context: 16,
            slots: 1,
            kv_budget: None,
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            ordinal: 0,
            knobs: Knobs {
                chip: Some(engine_cuda::chip::profile(&format!(
                    "[engine]\ngraphs = \"{graphs}\"\n[fuse]\nmatmul_silu_scaled = {fused}\n"
                ))),
                ..Knobs::default()
            },
            cache_dir: None,
            runahead: engine::runahead::Runahead::F1,
            residency: engine_cuda::experts::Plan::default(),
            deferred_tier: false,
            world: engine_cuda::World::default(),
            comm: core::ptr::null_mut(),
        })
        .unwrap();
        shell.open(0).unwrap();
        for tokens in [&[2u32, 7, 11, 13, 17][..], &[19u32][..]] {
            let result = shell
                .fire(&[Lane {
                    slot: 0,
                    word: 0,
                    tokens,
                }])
                .unwrap();
            assert_eq!(result[0].len(), 320);
            let want = 0.625f32 / (1. + (-0.625f32).exp());
            assert!(
                result[0].iter().all(|v| (*v - want).abs() < 0.002),
                "{:?}",
                result[0]
            );
        }
    }
}
