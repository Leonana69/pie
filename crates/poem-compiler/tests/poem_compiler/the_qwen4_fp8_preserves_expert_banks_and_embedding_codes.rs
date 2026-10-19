#[test]
fn the_fp8_deployment_preserves_expert_banks_and_embedding_codes() {
    use poem::{Dtype, Platform};
    use poem_ir::{ActQuant, Linear, Operands, Operation};
    for model in ["qwen38-flash-next", "qwen38-flash-next-mini"] {
        let name = format!("{model}-bf16-e4m3tile128a8-kv-bf16");
        let trace = poem_compiler::catalog::deployment(&name)
            .expect("Qwen FP8 deployment")
            .trace(Platform::Cuda);
        let find = |name: &str| trace.params.iter().find(|p| p.name == name).unwrap();
        let experts = if model.ends_with("mini") { 16 } else { 512 };
        assert_eq!(
            find("layer.0.experts_gate_up").shape,
            vec![experts, 1280, 2560]
        );
        assert_eq!(find("layer.0.experts_gate_up").dtype, Dtype::E4m3);
        assert_eq!(
            find("layer.0.experts_gate_up.scales").shape,
            vec![experts, 10, 20]
        );
        assert_eq!(
            find("layer.0.experts_down.scales").shape,
            vec![experts, 20, 5]
        );
        assert_eq!(find("ple.table").dtype, Dtype::E4m3tensor);
        assert_eq!(find("ple.table.scales").dtype, Dtype::F32);
        assert_eq!(find("embed").dtype, Dtype::Bf16);
        let lowp = trace
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.op,
                    Operation::Linear(Linear::MoeMatmulSelectActQuant {
                        quant: ActQuant::E4m3 { gran: 128 },
                        ..
                    })
                )
            })
            .count();
        assert!(lowp >= 2, "experts must request A8 in the traced plan");
        let folded = poem_compiler::fuse::fuse(trace, &["linear.matmul_silu_scaled"]);
        let count = folded
            .nodes
            .iter()
            .filter(|n| n.op.name() == "linear.matmul_silu_scaled")
            .count();
        assert_eq!(count, if model.ends_with("mini") { 9 } else { 97 });
    }
}

#[test]
#[ignore = "requires the official checkpoint at PIE_QWEN_FP8_CHECKPOINT; reads metadata only"]
fn the_official_fp8_checkpoint_builds_a_contract() {
    let path = std::env::var("PIE_QWEN_FP8_CHECKPOINT").expect("checkpoint path");
    let mut shards: Vec<_> = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();
    let source = ztensor_compat::open_all(&shards).unwrap();
    let row =
        poem_compiler::catalog::deployment("qwen38-flash-next-bf16-e4m3tile128a8-kv-bf16").unwrap();
    let contract = row.contract(&source, poem::Platform::Cuda).unwrap();
    // Constructing the symbolic contract alone does not validate nested
    // scale reshapes. Compile the real metadata exactly as serving does.
    let metadata = checkpoint::file::read::parse_metadata(std::path::Path::new(&path)).unwrap();
    checkpoint::plan::compile(
        &metadata,
        &contract,
        checkpoint::plan::StorageTarget::for_backend(checkpoint::types::BackendKind::Cuda, 0, 1),
    )
    .unwrap();
    let find = |name: &str| contract.tensors.iter().find(|t| t.name == name).unwrap();
    assert_eq!(find("layer.0.experts_gate_up").expr.sources().len(), 1024);
    assert_eq!(find("layer.0.experts_down").expr.sources().len(), 512);
    assert_eq!(find("ple.table").expr.sources().len(), 128);
    assert!(
        find("layer.0.experts_down.scales")
            .expr
            .sources()
            .iter()
            .all(|name| name.ends_with(".weight_scale_inv"))
    );
    assert!(
        find("ple.table.scales")
            .expr
            .sources()
            .iter()
            .all(|name| name.ends_with("ngram_embedding.weight_scale"))
    );
}
