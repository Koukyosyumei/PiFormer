//! End-to-end integration test / demo for PiFormer.
//!
//! Builds a tiny linear-attention instance (4 tokens, 4-dim heads),
//! runs the prover, and verifies the proof.
//!
//! Usage:  cargo run --bin example

use ark_ff::PrimeField;
use piformer_prover::{
    F,
    attention::LinearAttentionInstance,
    lookup::LassoInstance,
    pcs::HyraxParams,
    prover::{PiFormerProver, PiFormerWitness},
    verifier::PiFormerVerifier,
};

fn main() {
    let seq_len        = 4usize;
    let d_head         = 4usize;
    let bits_per_chunk = 4usize;   // 2^4 = 16 entries per sub-table
    let c              = 2usize;   // two sub-tables per activation

    // --- Define sub-tables: T_k[i] = i*(k+1) ---
    let table_size = 1usize << bits_per_chunk;
    let tables: Vec<Vec<F>> = (0..c)
        .map(|k| (0..table_size).map(|i| F::from((i * (k + 1)) as u64)).collect())
        .collect();

    // --- Build small Q, K, V matrices (deterministic) ---
    let make_mat = |seed: u64| -> Vec<Vec<F>> {
        (0..seq_len)
            .map(|t| {
                (0..d_head)
                    .map(|d| F::from((t as u64 * 7 + d as u64 * 3 + seed) % 16))
                    .collect()
            })
            .collect()
    };
    let q = make_mat(1);
    let k = make_mat(2);
    let v = make_mat(3);

    // --- Apply φ: φ(x) = T_0[x & mask] + T_1[(x >> m) & mask] ---
    let mask = (1usize << bits_per_chunk) - 1;
    let phi = |x: F| -> F {
        let xi  = PrimeField::into_bigint(x).as_ref()[0] as usize;
        let ch0 = xi & mask;
        let ch1 = (xi >> bits_per_chunk) & mask;
        tables[0][ch0] + tables[1][ch1]
    };
    let apply_phi = |m: &Vec<Vec<F>>| -> Vec<Vec<F>> {
        m.iter().map(|row| row.iter().map(|&x| phi(x)).collect()).collect()
    };
    let phi_q = apply_phi(&q);
    let phi_k = apply_phi(&k);

    // --- context[i][j] = Σ_t φ(K)[t][i] · V[t][j] ---
    let context: Vec<Vec<F>> = (0..d_head)
        .map(|i| {
            (0..d_head)
                .map(|j| (0..seq_len).map(|t| phi_k[t][i] * v[t][j]).sum())
                .collect()
        })
        .collect();

    // --- out[t][j] = Σ_i φ(Q)[t][i] · context[i][j] ---
    let out: Vec<Vec<F>> = (0..seq_len)
        .map(|t| {
            (0..d_head)
                .map(|j| (0..d_head).map(|i| phi_q[t][i] * context[i][j]).sum())
                .collect()
        })
        .collect();

    // --- Build Lasso instances ---
    let build_lasso = |mat: &Vec<Vec<F>>| -> LassoInstance {
        let mut indices = Vec::new();
        let mut outputs = Vec::new();
        for row in mat {
            for &x in row {
                let xi = PrimeField::into_bigint(x).as_ref()[0] as usize;
                indices.push(xi);
                outputs.push(phi(x));
            }
        }
        LassoInstance { tables: tables.clone(), query_indices: indices, outputs, bits_per_chunk }
    };

    let q_lasso = build_lasso(&q);
    let k_lasso = build_lasso(&k);

    // --- Hyrax setup (nu = bits_per_chunk/2, sigma = bits_per_chunk - nu) ---
    let nu    = bits_per_chunk / 2;
    let sigma = bits_per_chunk - nu;
    println!("Generating Hyrax parameters (sigma={sigma})...");
    let params = HyraxParams::new(sigma);

    // --- Assemble witness ---
    let inst = LinearAttentionInstance {
        seq_len, d_head, q, k, v, phi_q, phi_k, context, out, q_lasso, k_lasso,
    };
    let witness = PiFormerWitness { block_witnesses: vec![vec![inst]] };

    // --- Prove ---
    println!("Proving linear attention...");
    let proof = PiFormerProver::prove(&witness, &params);

    // --- Verify ---
    println!("Verifying...");
    match PiFormerVerifier::verify(&proof, &witness, &params) {
        Ok(())  => println!("✓ Proof verified successfully!"),
        Err(e)  => eprintln!("✗ Verification FAILED: {e}"),
    }
}

