//! Fiat-Shamir transcript built on SHA3-256.
//!
//! Usage:
//!   let mut tr = Transcript::new(b"MyProtocol");
//!   tr.append_field(b"claim", &val);
//!   let r: F = tr.challenge_field(b"round_0");

use ark_ff::PrimeField;
use sha3::{Digest, Sha3_256};

pub struct Transcript {
    hasher: Sha3_256,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(label);
        Self { hasher }
    }

    /// Absorb raw bytes with a domain-separation label.
    pub fn append_bytes(&mut self, label: &[u8], data: &[u8]) {
        self.hasher.update(label);
        self.hasher.update((data.len() as u64).to_le_bytes());
        self.hasher.update(data);
    }

    /// Absorb a single field element.
    pub fn append_field<F: PrimeField>(&mut self, label: &[u8], f: &F) {
        let mut buf = Vec::new();
        f.serialize_compressed(&mut buf).unwrap();
        self.append_bytes(label, &buf);
    }

    /// Absorb a slice of field elements.
    pub fn append_field_vec<F: PrimeField>(&mut self, label: &[u8], v: &[F]) {
        self.hasher.update(label);
        self.hasher.update((v.len() as u64).to_le_bytes());
        for f in v {
            let mut buf = Vec::new();
            f.serialize_compressed(&mut buf).unwrap();
            self.hasher.update(&buf);
        }
    }

    /// Squeeze a field element challenge (non-interactive via Fiat-Shamir).
    pub fn challenge_field<F: PrimeField>(&mut self, label: &[u8]) -> F {
        self.hasher.update(label);
        let hash = self.hasher.clone().finalize();
        // Feed hash back into state so future challenges are independent.
        self.hasher.update(&hash);
        F::from_le_bytes_mod_order(&hash)
    }
}
