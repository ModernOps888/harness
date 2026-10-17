pub const GEMM_SHADER: &str = include_str!("gemm.wgsl");
pub const GEMV_SHADER: &str = include_str!("gemv.wgsl");
pub const GEMV_Q4_SHADER: &str = include_str!("gemv_q4.wgsl");
pub const SPARSE_GEMV_Q4_SHADER: &str = include_str!("sparse_gemv_q4.wgsl");
pub const GEMM_Q4_BATCHED_SHADER: &str = include_str!("gemm_q4_batched.wgsl");
pub const RMSNORM_SHADER: &str = include_str!("rmsnorm.wgsl");
pub const FUSED_ADD_RMSNORM_SHADER: &str = include_str!("fused_add_rmsnorm.wgsl");
pub const SWIGLU_SHADER: &str = include_str!("swiglu.wgsl");
pub const ROPE_SHADER: &str = include_str!("rope.wgsl");
pub const FLASH_ATTN_GQA_SHADER: &str = include_str!("flash_attn_gqa.wgsl");
pub const MOE_ROUTER_TOPK_SHADER: &str = include_str!("moe_router_topk.wgsl");

