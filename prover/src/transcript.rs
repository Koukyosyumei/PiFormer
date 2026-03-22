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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F;

    #[test]
    fn test_challenge_is_deterministic() {
        // Identical transcripts must produce identical challenges.
        let mut t1 = Transcript::new(b"test");
        t1.append_field(b"x", &F::from(42u64));
        let c1 = t1.challenge_field::<F>(b"chall");

        let mut t2 = Transcript::new(b"test");
        t2.append_field(b"x", &F::from(42u64));
        let c2 = t2.challenge_field::<F>(b"chall");

        assert_eq!(c1, c2, "same transcript state must yield the same challenge");
    }

    #[test]
    fn test_different_init_labels_give_different_challenges() {
        let mut t1 = Transcript::new(b"proto_a");
        let c1 = t1.challenge_field::<F>(b"chall");

        let mut t2 = Transcript::new(b"proto_b");
        let c2 = t2.challenge_field::<F>(b"chall");

        assert_ne!(c1, c2, "different init labels must produce different challenges");
    }

    #[test]
    fn test_different_challenge_labels_give_different_results() {
        let mut t1 = Transcript::new(b"test");
        let c1 = t1.challenge_field::<F>(b"label_a");

        let mut t2 = Transcript::new(b"test");
        let c2 = t2.challenge_field::<F>(b"label_b");

        assert_ne!(c1, c2, "different challenge labels must produce different results");
    }

    #[test]
    fn test_different_appended_data_changes_challenge() {
        let mut t1 = Transcript::new(b"test");
        t1.append_field(b"x", &F::from(1u64));
        let c1 = t1.challenge_field::<F>(b"chall");

        let mut t2 = Transcript::new(b"test");
        t2.append_field(b"x", &F::from(2u64));
        let c2 = t2.challenge_field::<F>(b"chall");

        assert_ne!(c1, c2, "different appended values must change the challenge");
    }

    #[test]
    fn test_sequential_challenges_are_distinct() {
        let mut t = Transcript::new(b"test");
        let c1 = t.challenge_field::<F>(b"r1");
        let c2 = t.challenge_field::<F>(b"r2");
        let c3 = t.challenge_field::<F>(b"r3");

        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_append_bytes_affects_challenge() {
        let mut t1 = Transcript::new(b"test");
        t1.append_bytes(b"data", b"hello");
        let c1 = t1.challenge_field::<F>(b"chall");

        let mut t2 = Transcript::new(b"test");
        t2.append_bytes(b"data", b"world");
        let c2 = t2.challenge_field::<F>(b"chall");

        assert_ne!(c1, c2, "different byte payloads must yield different challenges");
    }

    #[test]
    fn test_append_field_vec_affects_challenge() {
        let v1 = vec![F::from(1u64), F::from(2u64), F::from(3u64)];
        let v2 = vec![F::from(1u64), F::from(2u64), F::from(99u64)];

        let mut t1 = Transcript::new(b"test");
        t1.append_field_vec(b"vec", &v1);
        let c1 = t1.challenge_field::<F>(b"chall");

        let mut t2 = Transcript::new(b"test");
        t2.append_field_vec(b"vec", &v2);
        let c2 = t2.challenge_field::<F>(b"chall");

        assert_ne!(c1, c2, "different field vec contents must change the challenge");
    }

    #[test]
    fn test_prover_verifier_transcript_sync() {
        // Simulates a 3-round interactive protocol where prover and verifier
        // must derive the same sequence of challenges from the same transcript.
        let values = [F::from(10u64), F::from(20u64), F::from(30u64)];

        let mut prover_t = Transcript::new(b"protocol");
        for v in &values {
            prover_t.append_field(b"round", v);
        }
        let prover_challenge = prover_t.challenge_field::<F>(b"final");

        let mut verifier_t = Transcript::new(b"protocol");
        for v in &values {
            verifier_t.append_field(b"round", v);
        }
        let verifier_challenge = verifier_t.challenge_field::<F>(b"final");

        assert_eq!(prover_challenge, verifier_challenge);
    }
}
