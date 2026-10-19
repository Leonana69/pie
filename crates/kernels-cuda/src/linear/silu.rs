//! Decode-sized BF16 projection with a scaled SiLU epilogue. Qwen's FP8
//! experts retain their own kernels; its HC down projections are BF16.
use crate::{
    Tensor,
    error::Error,
    jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse},
};
use dtype::Dtype;

/// The deliberately narrow initial fast path; all other shapes retain GEMM
/// autotuning followed by the ordinary staged SiLU.
pub fn supported(ctx: &Ctx, act: Tensor, weight: Tensor, out: Tensor) -> bool {
    act.dtype == Dtype::Bf16
        && weight.dtype == Dtype::Bf16
        && out.dtype == Dtype::Bf16
        && (1..=4).contains(&act.rows)
        && (1..=4).contains(&ctx.opaque_rows(act.rows as i32))
        && weight.rows == 320
        && weight.width == 10240
        && act.width == weight.width
        && out.rows == act.rows
        && out.width == weight.rows
        && act.ptr != 0
        && weight.ptr != 0
        && out.ptr != 0
        && aligned16(act.ptr)
        && aligned16(weight.ptr)
}

pub fn matmul_silu_scaled(
    ctx: &Ctx,
    act: Tensor,
    weight: Tensor,
    scale: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_silu_scaled";
    if !supported(ctx, act, weight, *out) {
        return Err(refuse(OP, "needs aligned BF16 [1..4,10240] x [320,10240]"));
    }
    let m = ctx.opaque_rows(act.rows as i32);
    // Qwen HC: 320 columns x 10240 K. Sixteen warps distribute the
    // long reduction enough to beat a separate tuned GEMM and SiLU on B200.
    let fire = Fire::at(
        "linear/silu.cuh",
        "::pie::linear::matmul_silu_scaled_bf16_kernel<16, 2>",
    )
    .apply(Launch::grid([weight.rows, m as u32, 1], [32, 16, 1]).programmatic(ctx.tuning().pdl));
    ctx.fire(
        OP,
        fire,
        &[
            ArgValue::Ptr(weight.ptr),
            ArgValue::Ptr(act.ptr),
            ArgValue::Ptr(out.ptr),
            ArgValue::I32(weight.rows as i32),
            ArgValue::I32(weight.width as i32),
            ArgValue::F32(scale),
            ctx.stage(),
        ],
    )
}
