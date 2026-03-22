//! End-to-end integration demo for PiFormer.
//!
//! Constructs a tiny linear-attention instance (2 tokens, 2-dim heads),
//! runs the prover, and verifies the proof.
//!
//! Usage:  cargo run --bin example

use ark_ff::PrimeField;
use piformer_prover::{
    attention::{LinearAttentionInstance},
    lookup::LassoInstance,
    pcs::HyraxParams,
    prover::{PiFormerProver, PiFormerWitness},
    verifier::PiFormerVerifier,
    F,
};

fn main() {
    let seq_len = 2usize;
    let d_head = 2usize;
    let bits_per_chunk = 4usize; // 2^4 = 16 entries per sub-table
    let _c = 1usize; // single sub-table (c=1 here; Lasso handles decomposition)

    // ── Define lookup table: T[i] = i + 1 ──
    let table_size = 1usize << bits_per_chunk;
    let table: Vec<F> = (0..table_size).map(|i| F::from((i + 1) as u64)).collect();

    // ── Build small Q, K, V matrices (seq_len × d_head) ──
    let q = vec![
        vec![F::from(1u64), F::from(2u64)],
        vec![F::from(3u64), F::from(4u64)],
    ];
    let k = vec![
        vec![F::from(0u64), F::from(1u64)],
        vec![F::from(2u64), F::from(3u64)],
    ];
    let v = vec![
        vec![F::from(5u64), F::from(6u64)],
        vec![F::from(7u64), F::from(8u64)],
    ];

    // ── φ(x) = table[x] ──
    let phi = |x: F| -> F {
        let xi = x.into_bigint().as_ref()[0] as usize;
        table[xi % table_size]
    };
    let apply_phi = |m: &Vec<Vec<F>>| -> Vec<Vec<F>> {
        m.iter().map(|row| row.iter().map(|&x| phi(x)).collect()).collect()
    };

    let phi_q = apply_phi(&q);
    let phi_k = apply_phi(&k);

    // ── context[i][j] = Σ_t φ(K)[t][i] · V[t][j] ──
    let context: Vec<Vec<F>> = (0..d_head)
        .map(|i| {
            (0..d_head)
                .map(|j| (0..seq_len).map(|t| phi_k[t][i] * v[t][j]).sum())
                .collect()
        })
        .collect();

    // ── out[t][j] = Σ_i φ(Q)[t][i] · context[i][j] ──
    let out: Vec<Vec<F>> = (0..seq_len)
        .map(|t| {
            (0..d_head)
                .map(|j| (0..d_head).map(|i| phi_q[t][i] * context[i][j]).sum())
                .collect()
        })
        .collect();

    // ── Build Lasso instances ──
    let build_lasso = |mat: &Vec<Vec<F>>| -> LassoInstance {
        let indices: Vec<usize> = mat
            .iter()
            .flatten()
            .map(|x| x.into_bigint().as_ref()[0] as usize % table_size)
            .collect();
        let outputs: Vec<F> = indices.iter().map(|&i| table[i]).collect();
        LassoInstance {
            tables: vec![table.clone()],
            query_indices: indices,
            outputs,
            bits_per_chunk,
        }
    };

    let q_lasso = build_lasso(&q);
    let k_lasso = build_lasso(&k);

    // ── Hyrax setup: sigma = bits_per_chunk / 2 = 2 ──
    let sigma = bits_per_chunk / 2;
    println!("Generating Hyrax parameters (sigma={sigma})…");
    let params = HyraxParams::new(sigma);

    // ── Assemble witness ──
    let inst = LinearAttentionInstance {
        seq_len,
        d_head,
        q,
        k,
        v,
        phi_q,
        phi_k,
        context,
        out,
        q_lasso,
        k_lasso,
    };
    let witness = PiFormerWitness {
        block_witnesses: vec![vec![inst]],
    };

    // ── Prove ──
    println!("Proving linear attention…");
    let proof = PiFormerProver::prove(&witness, &params);

    // ── Verify ──
    println!("Verifying…");
    match PiFormerVerifier::verify(&proof, &witness, &params) {
        Ok(()) => println!("✓ Proof verified successfully!"),
        Err(e) => {
            eprintln!("✗ Verification FAILED: {e}");
            std::process::exit(1);
        }
    }
}
