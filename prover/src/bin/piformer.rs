//! PiFormer CLI — ZK Proof of Inference for transformer models.
//!
//! # Workflow
//!
//! ```text
//! # 1. One-time offline setup
//! piformer setup --weights model.json --seq-len 32 --pk model.pk --vk model.vk
//!
//! # 2. Prove an inference run
//! piformer prove --pk model.pk --witness witness.json --proof proof.bin
//!
//! # 3. Verify the proof
//! piformer verify --vk model.vk --proof proof.bin
//!
//! # 4. Inspect a key or proof file
//! piformer inspect model.vk
//!
//! # 5. End-to-end smoke test with a tiny synthetic model
//! piformer sample --output-dir sample/
//! ```
//!
//! # File formats
//!
//! | Extension | Contents                                         |
//! |-----------|--------------------------------------------------|
//! | `*.pk`    | Proving key (G1 weight commitments + raw weights)|
//! | `*.vk`    | Verifying key (G1 weight commitments only)       |
//! | `*.bin`   | Proof bundle (proof + public instances)          |
//! | `*.json`  | Human-readable weights or witness (F as hex)     |

mod piformer {
    pub mod codec;
    pub mod json_io;
    pub mod sample;
}

use std::{
    fs,
    io::{self},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, Subcommand};
use piformer_prover::{
    pcs::HyraxParams,
    prover::prove,
    setup::preprocess_transformer_model,
    transcript::Transcript,
    verifier::verify,
};

use piformer::{codec, json_io, sample};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// PiFormer — succinct ZK proofs of transformer inference.
#[derive(Parser)]
#[command(
    name = "piformer",
    version,
    propagate_version = true,
    about = "Succinct ZK proofs of transformer inference (Hyrax PCS + Sumcheck + Lasso)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Preprocess model weights → proving key + verifying key (run once).
    Setup {
        /// Path to the model weights JSON file.
        #[arg(short, long, value_name = "FILE")]
        weights: PathBuf,

        /// Sequence length used during inference.
        #[arg(short, long, value_name = "N")]
        seq_len: usize,

        /// Output path for the proving key.
        #[arg(long, value_name = "FILE", default_value = "model.pk")]
        pk: PathBuf,

        /// Output path for the verifying key.
        #[arg(long, value_name = "FILE", default_value = "model.vk")]
        vk: PathBuf,
    },

    /// Generate a ZK proof from a proving key and witness.
    Prove {
        /// Path to the proving key (*.pk).
        #[arg(short, long, value_name = "FILE")]
        pk: PathBuf,

        /// Path to the witness JSON file.
        #[arg(short, long, value_name = "FILE")]
        witness: PathBuf,

        /// Output path for the proof bundle.
        #[arg(long, value_name = "FILE", default_value = "proof.bin")]
        proof: PathBuf,

        /// Fiat-Shamir transcript domain separator.
        #[arg(long, value_name = "LABEL", default_value = "piformer")]
        transcript_label: String,
    },

    /// Verify a proof against a verifying key.
    Verify {
        /// Path to the verifying key (*.vk).
        #[arg(short, long, value_name = "FILE")]
        vk: PathBuf,

        /// Path to the proof bundle (*.bin).
        #[arg(short, long, value_name = "FILE")]
        proof: PathBuf,

        /// Fiat-Shamir transcript domain separator (must match prove).
        #[arg(long, value_name = "LABEL", default_value = "piformer")]
        transcript_label: String,
    },

    /// Print human-readable information about a key or proof file.
    Inspect {
        /// File to inspect (*.pk, *.vk, or *.bin proof bundle).
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Run a full setup→prove→verify cycle on a small synthetic model (demo/test).
    Sample {
        /// Directory to write the generated sample files.
        #[arg(long, value_name = "DIR", default_value = "sample")]
        output_dir: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Setup { weights, seq_len, pk, vk } => run_setup(&weights, seq_len, &pk, &vk),
        Command::Prove { pk, witness, proof, transcript_label } => {
            run_prove(&pk, &witness, &proof, &transcript_label)
        }
        Command::Verify { vk, proof, transcript_label } => {
            run_verify(&vk, &proof, &transcript_label)
        }
        Command::Inspect { file } => run_inspect(&file),
        Command::Sample { output_dir } => run_sample(&output_dir),
    };
    if let Err(e) = result {
        eprintln!("\nerror: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

fn run_setup(weights_path: &Path, seq_len: usize, pk_path: &Path, vk_path: &Path) -> io::Result<()> {
    println!("=== PiFormer Setup ===");

    // Load weights
    eprint!("  Loading weights from {}  ... ", weights_path.display());
    let json_str = fs::read_to_string(weights_path)?;
    let jw: json_io::JsonWeights =
        serde_json::from_str(&json_str).map_err(|e| io_err(format!("weights JSON: {e}")))?;
    let weights = json_io::weights_from_json(jw).map_err(io_err)?;
    eprintln!(
        "ok  ({} block(s), d_model={}, d_ff={}, vocab={})",
        weights.num_blocks, weights.d_model, weights.d_ff, weights.vocab_size
    );

    // Preprocess (offline Hyrax commitment phase)
    eprint!("  Preprocessing (offline commitment phase) ... ");
    let t0 = Instant::now();
    let lasso_params = HyraxParams::new(2); // default sigma=2 (4-bit tables)
    let pk = preprocess_transformer_model(weights, seq_len, &lasso_params);
    eprintln!("done ({:.2}s)", t0.elapsed().as_secs_f64());

    // Write verifying key (slim — no raw weights)
    eprint!("  Writing verifying key → {}  ... ", vk_path.display());
    let vk_bytes = codec::encode_vk(&pk.vk).map_err(io_err)?;
    fs::write(vk_path, &vk_bytes)?;
    eprintln!("ok  ({} bytes)", vk_bytes.len());

    // Write proving key (full — includes weights)
    eprint!("  Writing proving key   → {}  ... ", pk_path.display());
    let pk_bytes = codec::encode_pk(&pk).map_err(io_err)?;
    fs::write(pk_path, &pk_bytes)?;
    eprintln!("ok  ({} bytes)", pk_bytes.len());

    println!();
    println!("  Proving key  : {}", pk_path.display());
    println!("  Verifying key: {}", vk_path.display());
    Ok(())
}

fn run_prove(
    pk_path: &Path,
    witness_path: &Path,
    proof_path: &Path,
    label: &str,
) -> io::Result<()> {
    println!("=== PiFormer Prove ===");

    // Load proving key
    eprint!("  Loading proving key from {}  ... ", pk_path.display());
    let pk_bytes = fs::read(pk_path)?;
    let pk = codec::decode_pk(&pk_bytes).map_err(io_err)?;
    eprintln!(
        "ok  ({} block(s), d_model={}, seq_len={})",
        pk.vk.num_blocks, pk.vk.d_model, pk.vk.seq_len
    );

    // Load witness
    eprint!("  Loading witness from {}  ... ", witness_path.display());
    let witness_str = fs::read_to_string(witness_path)?;
    let jw: json_io::JsonWitness =
        serde_json::from_str(&witness_str).map_err(|e| io_err(format!("witness JSON: {e}")))?;
    let (witness, inst_attn, inst_ffn, lasso_sigma) =
        json_io::witness_from_json(jw).map_err(io_err)?;
    let lasso_params = HyraxParams::new(lasso_sigma);
    eprintln!("ok");

    // Prove
    eprint!("  Generating proof  ... ");
    let t0 = Instant::now();
    let mut transcript = Transcript::new(label.as_bytes());
    let proof = prove(&pk, &witness, &inst_attn, &inst_ffn, &mut transcript, &lasso_params)
        .map_err(io_err)?;
    eprintln!("done ({:.2}s)", t0.elapsed().as_secs_f64());

    // Write proof bundle
    eprint!("  Writing proof → {}  ... ", proof_path.display());
    let proof_bytes =
        codec::encode_proof_bundle(&proof, &inst_attn, &inst_ffn, lasso_sigma).map_err(io_err)?;
    fs::write(proof_path, &proof_bytes)?;
    eprintln!("ok  ({} bytes)", proof_bytes.len());

    println!();
    println!("  Proof file: {}", proof_path.display());
    Ok(())
}

fn run_verify(vk_path: &Path, proof_path: &Path, label: &str) -> io::Result<()> {
    println!("=== PiFormer Verify ===");

    // Load verifying key
    eprint!("  Loading verifying key from {}  ... ", vk_path.display());
    let vk_bytes = fs::read(vk_path)?;
    let vk = codec::decode_vk(&vk_bytes).map_err(io_err)?;
    eprintln!(
        "ok  ({} block(s), d_model={}, seq_len={})",
        vk.num_blocks, vk.d_model, vk.seq_len
    );

    // Load proof bundle
    eprint!("  Loading proof from {}  ... ", proof_path.display());
    let proof_bytes = fs::read(proof_path)?;
    let (proof, inst_attn, inst_ffn, lasso_sigma) =
        codec::decode_proof_bundle(&proof_bytes).map_err(io_err)?;
    let lasso_params = HyraxParams::new(lasso_sigma);
    eprintln!("ok");

    // Verify
    eprint!("  Verifying  ... ");
    let t0 = Instant::now();
    let mut transcript = Transcript::new(label.as_bytes());
    match verify(&proof, &vk, &inst_attn, &inst_ffn, &mut transcript, &lasso_params) {
        Ok(()) => {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!("VALID ({elapsed:.3}s)");
            println!();
            println!("  ✓  Proof is VALID.");
            Ok(())
        }
        Err(e) => {
            eprintln!("INVALID");
            Err(io_err(format!("proof verification failed: {e}")))
        }
    }
}

fn run_inspect(path: &Path) -> io::Result<()> {
    let bytes = fs::read(path)?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "vk" => {
            let vk = codec::decode_vk(&bytes).map_err(io_err)?;
            println!("File type    : Verifying Key");
            println!("Path         : {}", path.display());
            println!("Size         : {} bytes", bytes.len());
            println!("Blocks       : {}", vk.num_blocks);
            println!("Sequence len : {}", vk.seq_len);
            println!("d_model      : {}", vk.d_model);
            println!("Vocab size   : {}", vk.vocab_size);
            for (i, bvk) in vk.block_vks.iter().enumerate() {
                println!("  Block[{}]: seq_len={}, d_model={}", i, bvk.seq_len, bvk.d_model);
            }
        }
        "pk" => {
            let pk = codec::decode_pk(&bytes).map_err(io_err)?;
            println!("File type    : Proving Key");
            println!("Path         : {}", path.display());
            println!("Size         : {} bytes", bytes.len());
            println!("Blocks       : {}", pk.vk.num_blocks);
            println!("Sequence len : {}", pk.vk.seq_len);
            println!("d_model      : {}", pk.vk.d_model);
            println!("Vocab size   : {}", pk.vk.vocab_size);
            for (i, bpk) in pk.block_pks.iter().enumerate() {
                println!(
                    "  Block[{}]: q_w={}×{}, ffn_w1={}×{}",
                    i,
                    bpk.q_pk.w.len(),
                    bpk.q_pk.w.first().map_or(0, |r: &Vec<_>| r.len()),
                    bpk.ffn_pk.w1.len(),
                    bpk.ffn_pk.w1.first().map_or(0, |r: &Vec<_>| r.len()),
                );
            }
        }
        "bin" => {
            let (proof, inst_attn, inst_ffn, lasso_sigma) =
                codec::decode_proof_bundle(&bytes).map_err(io_err)?;
            println!("File type    : Proof Bundle");
            println!("Path         : {}", path.display());
            println!("Size         : {} bytes", bytes.len());
            println!("Blocks       : {}", proof.block_proofs.len());
            println!("Attn seq_len : {}", inst_attn.seq_len);
            println!("Attn d_head  : {}", inst_attn.d_head);
            println!("Lasso sigma  : {}", lasso_sigma);
            println!(
                "Q lasso queries : {}",
                inst_attn.q_lasso.outputs.len()
            );
            println!(
                "FFN lasso queries: {}",
                inst_ffn.activation_lasso.outputs.len()
            );
        }
        _ => {
            return Err(io_err(format!(
                "unrecognised extension '.{ext}' — expected .pk, .vk, or .bin"
            )));
        }
    }
    Ok(())
}

fn run_sample(out_dir: &Path) -> io::Result<()> {
    const T: usize = 2;       // seq_len
    const D: usize = 2;       // d_model
    const D_FF: usize = 4;    // d_ff
    const V: usize = 2;       // vocab_size
    const M_BITS: usize = 4;  // activation bit-width for lasso

    println!("=== PiFormer Sample (T={T}, D={D}, D_FF={D_FF}, V={V}) ===");
    fs::create_dir_all(out_dir)?;

    // --- Weights ---
    let weights = sample::build_zero_weights(1, D, D_FF, V);
    let weights_path = out_dir.join("weights.json");
    let jw = json_io::weights_to_json(&weights);
    fs::write(&weights_path, serde_json::to_string_pretty(&jw).unwrap())?;
    println!("  Wrote {}", weights_path.display());

    // --- Setup ---
    let pk_path = out_dir.join("model.pk");
    let vk_path = out_dir.join("model.vk");
    run_setup(&weights_path, T, &pk_path, &vk_path)?;

    // --- Witness ---
    let (witness, inst_attn, inst_ffn) =
        sample::build_zero_witness(T, D, D_FF, V, M_BITS);
    let lasso_sigma = M_BITS / 2;
    let jw = json_io::witness_to_json(&witness, &inst_attn, &inst_ffn, lasso_sigma);
    let witness_path = out_dir.join("witness.json");
    fs::write(&witness_path, serde_json::to_string_pretty(&jw).unwrap())?;
    println!("  Wrote {}", witness_path.display());

    // --- Prove ---
    let proof_path = out_dir.join("proof.bin");
    run_prove(&pk_path, &witness_path, &proof_path, "piformer-sample")?;

    // --- Verify ---
    run_verify(&vk_path, &proof_path, "piformer-sample")?;

    println!();
    println!("All sample files written to '{}'.", out_dir.display());
    println!("Use 'piformer inspect <file>' to examine any output.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn io_err(msg: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::Other, msg.to_string())
}
