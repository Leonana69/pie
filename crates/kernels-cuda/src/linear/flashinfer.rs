//! Opt-in native FlashInfer FP8 MoE bridge. The bridge owns no device memory:
//! scratch belongs to PIE's stream/region arena and survives graph replay.
use crate::{
    Tensor,
    error::Error,
    jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse},
};
use dtype::Dtype;
use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

const OP: &str = "linear.flashinfer_fp8_moe";
const FILE: &str = "linear/flashinfer_fp8.cuh";
type Run = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    usize,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i32,
    i32,
    i32,
    i32,
    i32,
) -> i32;
struct Bridge {
    _library: libloading::Library,
    run: Run,
    error: unsafe extern "C" fn() -> *const c_char,
}
static PATH: OnceLock<Option<String>> = OnceLock::new();
pub fn enabled() -> bool {
    path().is_some()
}
fn path() -> Option<&'static str> {
    PATH.get_or_init(|| {
        std::env::var("PIE_FLASHINFER_MOE_LIBRARY")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}
fn bridge() -> Result<&'static Bridge, Error> {
    static BRIDGE: OnceLock<Result<Bridge, String>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| unsafe {
            // The explicitly configured native plugin uses the versioned ABI in tools/flashinfer-fp8.
            let lib =
                libloading::Library::new(path().ok_or("FlashInfer bridge is not configured")?)
                    .map_err(|e| e.to_string())?;
            let abi = lib
                .get::<unsafe extern "C" fn() -> i32>(b"pie_flashinfer_abi\0")
                .map_err(|e| e.to_string())?;
            if abi() != 1 {
                return Err("FlashInfer bridge ABI mismatch".into());
            }
            let run = *lib
                .get::<Run>(b"pie_flashinfer_fp8\0")
                .map_err(|e| e.to_string())?;
            let error = *lib
                .get::<unsafe extern "C" fn() -> *const c_char>(b"pie_flashinfer_error\0")
                .map_err(|e| e.to_string())?;
            tracing::info!("FlashInfer TRT-LLM FP8 MoE bridge enabled");
            Ok(Bridge {
                _library: lib,
                run,
                error,
            })
        })
        .as_ref()
        .map_err(|e| refuse(OP, e.clone()))
}

#[allow(clippy::too_many_arguments)]
pub fn experts(
    ctx: &Ctx,
    x: Tensor,
    up: Tensor,
    up_scales: Tensor,
    down: Tensor,
    down_scales: Tensor,
    routes: Tensor,
    weights: Tensor,
    inter: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    let (m, h, e, top) = (y.rows, x.width, up.rows, routes.width);
    if m == 0 {
        return Ok(());
    }
    if x.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || weights.dtype != Dtype::F32
        || routes.dtype != Dtype::I32
        || up.dtype != Dtype::E4m3
        || down.dtype != Dtype::E4m3
        || up_scales.dtype != Dtype::U8
        || down_scales.dtype != Dtype::U8
        || h == 0
        || inter == 0
        || !h.is_multiple_of(128)
        || !inter.is_multiple_of(128)
        || top == 0
        || top > 128
        || top > e
        || x.rows != m
        || routes.rows != m
        || weights.rows != m
        || weights.width != top
        || y.width != h
        || down.rows != e
        || up.width as u64 != 2 * u64::from(inter) * u64::from(h)
        || down.width as u64 != u64::from(h) * u64::from(inter)
        || up_scales.rows != e
        || down_scales.rows != e
        || u64::from(up_scales.width) != 2 * u64::from(inter / 128) * u64::from(h / 128) * 4
        || u64::from(down_scales.width) != u64::from(inter / 128) * u64::from(h / 128) * 4
    {
        return Err(refuse(
            OP,
            "expected resident BF16 rows, F32 routing weights and 128-block FP8 expert banks",
        ));
    }
    let ptr = |name, bytes| ctx.scratch_stream(OP, name, bytes).map(|p| p as u64);
    let q = ptr("flashinfer.q", m as usize * h as usize)?;
    let s = ptr("flashinfer.scales", m as usize * (h / 128) as usize * 4)?;
    let ids = ptr("flashinfer.ids", m as usize * top as usize * 4)?;
    let probs = ptr("flashinfer.probs", m as usize * top as usize * 4)?;
    let out = ptr("flashinfer.out", m as usize * h as usize * 2)?;
    // Bounded experimental deployment: full Qwen <=128-token fires. Fail closed
    // on exhaustion in the bridge, never allocate during graph capture.
    let work_bytes = 192usize * 1024 * 1024;
    let work = ctx.scratch_stream(OP, "flashinfer.workspace", work_bytes)?;
    let b = bridge()?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::flashinfer_fp8_prepare")
            .apply(Launch::grid([m, h / 128, 1], [128, 1, 1])),
        &[
            x.arg(),
            routes.arg(),
            weights.arg(),
            ArgValue::Ptr(q),
            ArgValue::Ptr(s),
            ArgValue::Ptr(ids),
            ArgValue::Ptr(probs),
            m.arg(),
            h.arg(),
            top.arg(),
            ctx.stage(),
        ],
    )?;
    // SAFETY: shape/precision checked above; all pointers are live device allocations.
    // The C ABI catches exceptions and enqueues only onto ctx.stream().
    let code = unsafe {
        (b.run)(
            ctx.stream(),
            work,
            work_bytes,
            q,
            s,
            ids,
            probs,
            up.ptr,
            up_scales.ptr,
            down.ptr,
            down_scales.ptr,
            out,
            m as i32,
            h as i32,
            inter as i32,
            e as i32,
            top as i32,
        )
    };
    if code != 0 {
        let detail = unsafe { CStr::from_ptr((b.error)()) }
            .to_string_lossy()
            .into_owned();
        return Err(refuse(OP, detail));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::flashinfer_fp8_finish")
            .apply(Launch::grid([m, 1, 1], [256, 1, 1])),
        &[ArgValue::Ptr(out), y.arg(), m.arg(), h.arg(), ctx.stage()],
    )
}

/// Reorder both FP8 code halves and their factors once, in place at load.
pub fn swap_gate_up(ctx: &Ctx, codes: Tensor, scales: Tensor) -> Result<(), Error> {
    for tensor in [codes, scales] {
        if !tensor.width.is_multiple_of(8) {
            return Err(refuse(OP, "gate/up planes must contain whole 32-bit words"));
        }
        let words = tensor.width / 8;
        ctx.fire(
            OP,
            Fire::at(FILE, "::pie::linear::flashinfer_swap_legs")
                .apply(Launch::flat(words * tensor.rows, 256)),
            &[tensor.arg(), words.arg(), tensor.rows.arg()],
        )?;
    }
    Ok(())
}
