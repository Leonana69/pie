//! W8A8 expert fallback for released FP8 codes with original f32 block factors.
//! Used when the tcgen05 scale layout cannot represent the bank (e.g. K=640).
//! Inputs are quantized in 128-element blocks; decode streams weights, prefill
//! groups routes for mma.sync. Scale tensor widths are measured in bytes.

use super::moe::GroupSeat;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/fp8_compute.cuh";

fn supported() -> bool {
    crate::jit::device::compute_capability_major().is_some_and(|major| major >= 9)
}

fn quantize(ctx: &Ctx, op: &'static str, x: Tensor, stage_fan: u32) -> Result<(u64, u64), Error> {
    if !supported() {
        return Err(refuse(
            op,
            "FP8 activation computation requires SM90 or newer",
        ));
    }
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = stated(op, x.rows)?;
    let k = stated(op, x.width)?;
    let bytes = u64::from(x.rows) * u64::from(x.width);
    let scale_at = bytes.next_multiple_of(256);
    let total = scale_at + u64::from(x.rows) * u64::from(x.width.div_ceil(128)) * 4;
    let total =
        usize::try_from(total).map_err(|_| refuse(op, "FP8 activation workspace exceeds usize"))?;
    let codes = ctx.scratch_stream(op, "linear.fp8_activations", total)? as usize as u64;
    let scales = codes + scale_at;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::fp8_quantize_128<{t}>")),
        )
        .apply(Launch::grid(
            [x.rows, x.width.div_ceil(128), 1],
            [128, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(scales),
            rows.arg(),
            k.arg(),
            stated(op, stage_fan)?.arg(),
            ctx.stage(),
        ],
    )?;
    Ok((codes, scales))
}

#[allow(clippy::too_many_arguments)]
pub fn experts(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    if y.rows == 0 {
        return Ok(());
    }
    if x.width == 0
        || y.width == 0
        || codes.rows == 0
        || routes.width == 0
        || routes.dtype != dtype::Dtype::I32
        || scales.dtype != dtype::Dtype::U8
    {
        return Err(refuse(
            op,
            "FP8 experts require nonempty axes, i32 routes and byte-addressed f32 scales",
        ));
    }
    let route_count = routes
        .rows
        .checked_mul(routes.width)
        .ok_or_else(|| refuse(op, "expert route count overflows"))?;
    if y.rows != route_count
        || x.dtype != y.dtype
        || (x.rows != routes.rows && x.rows != route_count)
        || bias.is_some()
    {
        return Err(refuse(
            op,
            "FP8 expert input, output, routes and bias shapes do not match",
        ));
    }

    if codes.dtype != dtype::Dtype::E4m3
        || (!x.width.is_multiple_of(128) || !y.width.is_multiple_of(128))
    {
        return Err(refuse(
            op,
            "FP8 expert axes require whole 128-element tiles",
        ));
    }
    let code_k = x.width;
    let scale_k = x.width / 128 * 4;
    let scale_n = y.width / 128;
    if codes.width
        != y.width
            .checked_mul(code_k)
            .ok_or_else(|| refuse(op, "FP8 expert width overflows"))?
        || scales.width
            != scale_n
                .checked_mul(scale_k)
                .ok_or_else(|| refuse(op, "FP8 scale width overflows"))?
        || codes.rows != scales.rows
    {
        return Err(refuse(
            op,
            "FP8 expert code and scale planes do not match N, K and expert count",
        ));
    }
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let by_token = x.rows == routes.rows;
    let (a, s) = quantize(ctx, op, x, if by_token { 1 } else { routes.width })?;
    if codes.rows <= 4096 && route_count > 2 * codes.rows {
        return experts_grouped(ctx, op, a, s, codes, scales, routes, y, seat, by_token);
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::fp8_expert_stream<{t}>")),
        )
        .apply(Launch::grid([y.rows, y.width.div_ceil(16), 1], [128, 1, 1])),
        &[
            ArgValue::Ptr(a),
            ArgValue::Ptr(s),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |b| b.arg()),
            y.arg(),
            stated(op, routes.width)?.arg(),
            stated(op, if by_token { routes.width } else { 1 })?.arg(),
            stated(op, y.width)?.arg(),
            stated(op, x.width)?.arg(),
            stated(op, codes.rows)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn experts_grouped(
    ctx: &Ctx,
    op: &'static str,
    a: u64,
    s: u64,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    by_token: bool,
) -> Result<(), Error> {
    // Same counting sort and bounded work list as PIE's MXFP4 path.
    const GROUP: u32 = 8;
    let experts = codes.rows;
    let count = y.rows;
    let offsets_at = (count as usize).next_multiple_of(64);
    let work_cap = count.div_ceil(GROUP) + experts;
    let work_at = (offsets_at + experts as usize + 2).next_multiple_of(64);
    let order = ctx.scratch_stream(op, "moe_route_order", (work_at + work_cap as usize) * 4)?
        as usize as u64;
    let offsets = order + offsets_at as u64 * 4;
    let work = order + work_at as u64 * 4;
    ctx.fire(
        op,
        Fire::at("linear/quant.cuh", symbol("::pie::linear::moe_route_order"))
            .apply(Launch::grid([1, 1, 1], [1024, 1, 1]).smem(2 * (experts + 1) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            stated(op, work_cap)?.arg(),
            GROUP.arg(),
            stated(op, routes.width)?.arg(),
            stated(op, count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
        ],
    )?;
    let t = dtype_dispatch!(op, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::fp8_expert_grouped<{t}>")),
        )
        .apply(Launch::grid(
            [work_cap + 1, y.width.div_ceil(64), 1],
            [128, 1, 1],
        )),
        &[
            ArgValue::Ptr(a),
            ArgValue::Ptr(s),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            codes.arg(),
            scales.arg(),
            y.arg(),
            stated(op, if by_token { routes.width } else { 1 })?.arg(),
            stated(op, y.width)?.arg(),
            stated(op, codes.width / y.width)?.arg(),
            stated(op, experts)?.arg(),
            stated(op, work_cap)?.arg(),
            stated(op, count)?.arg(),
            stated(op, routes.width)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}
