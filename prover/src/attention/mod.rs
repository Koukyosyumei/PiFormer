pub mod layernorm;
pub mod linear;
pub mod projection;
pub mod ternary_check;

pub use linear::{
    prove_linear_attention, verify_linear_attention, LinearAttentionInstance, LinearAttentionProof,
};
pub use ternary_check::{
    prove_ternary_weights, verify_ternary_weights, TernaryWeightInstance, TernaryWeightProof,
};
