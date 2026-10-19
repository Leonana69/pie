#![cfg(feature = "cuda")]

use engine_cuda::{Boot, Graphs, Knobs, Lane, Shell};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Precision, Value, Weight, ops};
use poem_compiler::Budget;

struct Micro;
impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let x = ops::layout::embed(
            &inputs.tokens(),
            &Weight::sym("embed", [128, 640], Dtype::Bf16),
            128,
        );
        let logits = ops::linear::matmul(&x, &Weight::sym("router", [2, 640], Dtype::Bf16));
        let (routes, weights) = ops::linear::moe_topk_softmax(&logits, 2, 1);
        let precision = Precision::ACTED
            .into_iter()
            .find(|p| p.dtype == Dtype::E4m3tile128)
            .unwrap();
        let bank = Weight::at("experts", [2, 128, 640], precision);
        let routed = ops::linear::moe_matmul_select_quant(&x, &bank, &routes, 1);
        ops::linear::moe_weighted_sum(&routed, &weights)
    }
}

#[test]
fn an_fp8_expert_with_five_scale_columns_serves() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.zt");
    let mut writer = ztensor::Writer::create(&path).unwrap();
    // All input rows are ones; route to expert zero. Its five block scales
    // are deliberately not powers of two: the fallback must keep them.
    let ones: Vec<u8> = vec![0x3f80u16; 128 * 640]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    writer
        .add("embed", [128, 640], ztensor::Leaf::BF16, &ones)
        .unwrap();
    let router: Vec<u8> = (0..1280)
        .flat_map(|i| (if i < 640 { 0x3f80u16 } else { 0xbf80u16 }).to_le_bytes())
        .collect();
    writer
        .add(
            "experts",
            [2, 128, 640],
            ztensor::Leaf::E4M3,
            &vec![0x38; 2 * 128 * 640],
        )
        .unwrap();
    let factors = [0.3f32, 0.7, 0.1, 0.2, 0.4, 1., 1., 1., 1., 1.];
    let scales: Vec<u8> = factors.into_iter().flat_map(f32::to_le_bytes).collect();
    writer
        .add("experts.scales", [2, 1, 5], ztensor::Leaf::F32, &scales)
        .unwrap();
    writer
        .add("router", [2, 640], ztensor::Leaf::BF16, &router)
        .unwrap();
    writer.finish().unwrap();
    let trace = poem::trace_hybrid("fp8-five-scales", &Micro, Platform::Cuda);
    let src = ztensor::Source::open(&path).unwrap();
    let contract = poem::import::own_contract(&src, &trace.params, 1, Platform::Cuda).unwrap();
    for graphs in [Graphs::Off, Graphs::default()] {
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
                chip: Some(engine_cuda::chip::pinned(graphs, None)),
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
            assert_eq!(result[0].len(), 128);
            // BF16(128 * (0.3 + 0.7 + 0.1 + 0.2 + 0.4)) = 218.
            assert!(
                result[0].iter().all(|v| (*v - 218.).abs() < 0.01),
                "{:?}",
                result[0]
            );
        }
    }
}
