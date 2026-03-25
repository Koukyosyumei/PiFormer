use crate::field::F;
use crate::poly::DenseMLPoly; // DenseMLPolyのパスはご自身の構成に合わせてください
use ark_ff::Field;
use ark_ff::Zero;

/// 2次元の行列（VecのVec）を DenseMLPoly (多線形延長) に変換します。
///
/// ゼロ知識証明のハイパーキューブに適合させるため、
/// 行数(rows)と列数(cols)をそれぞれ「2のべき乗」にパディング（ゼロ埋め）します。
/// データは行優先（Row-Major）でフラットな1次元ベクトルに展開されます。
pub fn mat_to_mle(mat: &[Vec<F>], rows: usize, cols: usize) -> DenseMLPoly {
    // 1. 行と列のサイズを直近の2のべき乗に切り上げる
    // (例: 3行5列 -> 4行8列 に拡張)
    // .max(1) は、サイズが0の場合に 2^0 = 1 に最低限保つための安全策です。
    let r_p2 = rows.next_power_of_two().max(1);
    let c_p2 = cols.next_power_of_two().max(1);

    let total_size = r_p2 * c_p2;

    // 2. パディング領域を含めたゼロ配列を確保
    let mut evals = vec![F::zero(); total_size];

    // 3. 元の行列のデータを行優先(Row-Major)でコピー
    for i in 0..rows {
        // 安全策: もし引数 `mat` の内部Vecの長さが `cols` より短い/長い場合のパニックを防ぐ
        let current_cols = mat.get(i).map_or(0, |row| row.len()).min(cols);

        for j in 0..current_cols {
            // パディング後の列数(c_p2)を使ってインデックスを計算する！
            // ここを `cols` で計算すると、行と行の間にパディングが正しく挿入されません。
            evals[i * c_p2 + j] = mat[i][j];
        }
    }

    // 4. DenseMLPolyとして返す
    DenseMLPoly::new(evals)
}

/// 1次元のベクトルを DenseMLPoly に変換します。
///
/// ベクトルの長さを「2のべき乗」にパディングします。
/// 少なくとも変数が1つ（長さ2）になるように `.max(2)` としています。
pub fn vec_to_mle(vec: &[F], len: usize) -> DenseMLPoly {
    // 1. 直近の2のべき乗に切り上げ（最低でも長さ2にする）
    let padded_len = len.next_power_of_two().max(2);

    // 2. ゼロで初期化
    let mut evals = vec![F::zero(); padded_len];

    // 3. データのコピー
    let copy_len = vec.len().min(len);
    for i in 0..copy_len {
        evals[i] = vec[i];
    }

    // 4. DenseMLPolyとして返す
    DenseMLPoly::new(evals)
}

pub fn eval_rows(poly: &DenseMLPoly, _n_row_vars: usize, r_row: &[F]) -> Vec<F> {
    let mut p = poly.clone();
    for &r in r_row {
        p = p.fix_first_variable(r);
    }
    p.evaluations
}

pub fn eval_cols(poly: &DenseMLPoly, n_row_vars: usize, r_col: &[F]) -> Vec<F> {
    let n_p2_rows = 1 << n_row_vars;
    let n_p2_cols = poly.evaluations.len() / n_p2_rows;
    (0..n_p2_rows)
        .map(|i| {
            DenseMLPoly::new(poly.evaluations[i * n_p2_cols..(i + 1) * n_p2_cols].to_vec())
                .evaluate(r_col)
        })
        .collect()
}

pub fn eval_cols_ternary(w_raw: &[Vec<F>], r_out: &[F], d_in: usize, d_out: usize) -> Vec<F> {
    // 1. eq(j, r_out) を事前計算 (O(d_out))
    // 後のループで使い回すことで、計算量を劇的に減らす
    let eq_evals = compute_eq_evals(r_out, d_out);

    let mut res = vec![F::ZERO; d_in.next_power_of_two()];

    // 2. コアループ: 有限体乗算 (Mul) はゼロ！
    for k in 0..d_in {
        let mut sum = F::ZERO;
        for j in 0..d_out {
            // 重みが 0 の場合はメモリアクセスと計算を完全にスキップ (Sparsityの活用)
            if w_raw[k][j] == F::ONE {
                sum += eq_evals[j];
            } else if w_raw[k][j] == F::ZERO - F::ONE {
                sum -= eq_evals[j]
            } else if w_raw[k][j] != F::ZERO {
                unreachable!("Weight must be ternary [-1, 0, 1]")
            }
        }
        res[k] = sum;
    }
    res
}

/// eq(j, r) 多項式のすべての評価点を O(D) で計算するヘルパー
pub fn compute_eq_evals(r: &[F], n: usize) -> Vec<F> {
    let mut evals = vec![F::ONE; n.next_power_of_two()];
    for (i, &ri) in r.iter().enumerate() {
        let bit_step = 1 << i;
        for j in 0..bit_step {
            evals[j + bit_step] = evals[j] * ri;
            evals[j] = evals[j] * (F::ONE - ri);
        }
    }
    evals
}

pub fn combine(a: &[F], b: &[F]) -> Vec<F> {
    let mut res = a.to_vec();
    res.extend_from_slice(b);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;

    // -----------------------------------------------------------------------
    // mat_to_mle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mat_to_mle_2x2_exact() {
        let mat = vec![
            vec![F::from(1u64), F::from(2u64)],
            vec![F::from(3u64), F::from(4u64)],
        ];
        let mle = mat_to_mle(&mat, 2, 2);
        assert_eq!(mle.evaluations.len(), 4);
        assert_eq!(mle.evaluations[0], F::from(1u64)); // mat[0][0]
        assert_eq!(mle.evaluations[1], F::from(2u64)); // mat[0][1]
        assert_eq!(mle.evaluations[2], F::from(3u64)); // mat[1][0]
        assert_eq!(mle.evaluations[3], F::from(4u64)); // mat[1][1]
    }

    #[test]
    fn test_mat_to_mle_non_power_of_two_pads_correctly() {
        // 3x3 → r_p2=4, c_p2=4, total=16
        let mat = vec![
            vec![F::from(1u64), F::from(2u64), F::from(3u64)],
            vec![F::from(4u64), F::from(5u64), F::from(6u64)],
            vec![F::from(7u64), F::from(8u64), F::from(9u64)],
        ];
        let mle = mat_to_mle(&mat, 3, 3);
        assert_eq!(mle.evaluations.len(), 16);
        assert_eq!(mle.evaluations[0], F::from(1u64));
        assert_eq!(mle.evaluations[2], F::from(3u64));
        assert_eq!(mle.evaluations[3], F::zero()); // col padding
        assert_eq!(mle.evaluations[4], F::from(4u64));
        assert_eq!(mle.evaluations[7], F::zero()); // col padding
                                                   // Row 3 (all zero, row padding)
        assert_eq!(mle.evaluations[12], F::zero());
        assert_eq!(mle.evaluations[15], F::zero());
    }

    #[test]
    fn test_mat_to_mle_single_row() {
        let mat = vec![vec![F::from(10u64), F::from(20u64)]];
        let mle = mat_to_mle(&mat, 1, 2);
        assert_eq!(mle.evaluations.len(), 2);
        assert_eq!(mle.evaluations[0], F::from(10u64));
        assert_eq!(mle.evaluations[1], F::from(20u64));
    }

    #[test]
    fn test_mat_to_mle_single_element() {
        let mat = vec![vec![F::from(42u64)]];
        let mle = mat_to_mle(&mat, 1, 1);
        assert_eq!(mle.evaluations.len(), 1);
        assert_eq!(mle.evaluations[0], F::from(42u64));
    }

    #[test]
    fn test_mat_to_mle_all_zeros() {
        let mat = vec![vec![F::zero(); 4]; 4];
        let mle = mat_to_mle(&mat, 4, 4);
        assert!(mle.evaluations.iter().all(|&x| x.is_zero()));
    }

    #[test]
    fn test_mat_to_mle_reported_dims_larger_than_actual() {
        // Declare 4x4 but only provide 2x2 data — extra entries should be zero
        let mat = vec![
            vec![F::from(1u64), F::from(2u64)],
            vec![F::from(3u64), F::from(4u64)],
        ];
        let mle = mat_to_mle(&mat, 4, 4);
        assert_eq!(mle.evaluations.len(), 16);
        assert_eq!(mle.evaluations[0], F::from(1u64));
        assert_eq!(mle.evaluations[1], F::from(2u64));
        // Row 1 starts at index 4 (c_p2=4)
        assert_eq!(mle.evaluations[4], F::from(3u64));
        assert_eq!(mle.evaluations[5], F::from(4u64));
        // Remaining entries must be zero
        for idx in [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert_eq!(
                mle.evaluations[idx],
                F::zero(),
                "index {idx} should be zero"
            );
        }
    }

    // -----------------------------------------------------------------------
    // vec_to_mle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_to_mle_exact_power_of_two() {
        let v = vec![
            F::from(10u64),
            F::from(20u64),
            F::from(30u64),
            F::from(40u64),
        ];
        let mle = vec_to_mle(&v, 4);
        assert_eq!(mle.evaluations.len(), 4);
        assert_eq!(mle.evaluations[0], F::from(10u64));
        assert_eq!(mle.evaluations[3], F::from(40u64));
    }

    #[test]
    fn test_vec_to_mle_pads_to_next_power_of_two() {
        let v = vec![F::from(1u64), F::from(2u64), F::from(3u64)];
        let mle = vec_to_mle(&v, 3);
        assert_eq!(mle.evaluations.len(), 4);
        assert_eq!(mle.evaluations[0], F::from(1u64));
        assert_eq!(mle.evaluations[1], F::from(2u64));
        assert_eq!(mle.evaluations[2], F::from(3u64));
        assert_eq!(mle.evaluations[3], F::zero()); // padding
    }

    #[test]
    fn test_vec_to_mle_minimum_size() {
        // len=1 → padded to 2 (enforced minimum)
        let v = vec![F::from(99u64)];
        let mle = vec_to_mle(&v, 1);
        assert_eq!(mle.evaluations.len(), 2);
        assert_eq!(mle.evaluations[0], F::from(99u64));
        assert_eq!(mle.evaluations[1], F::zero());
    }

    #[test]
    fn test_vec_to_mle_reported_len_larger() {
        // len=8 but only 3 elements provided — remaining should be zero
        let v = vec![F::from(5u64), F::from(6u64), F::from(7u64)];
        let mle = vec_to_mle(&v, 8);
        assert_eq!(mle.evaluations.len(), 8);
        assert_eq!(mle.evaluations[0], F::from(5u64));
        assert_eq!(mle.evaluations[2], F::from(7u64));
        for i in 3..8 {
            assert_eq!(mle.evaluations[i], F::zero());
        }
    }

    #[test]
    fn test_vec_to_mle_all_zeros() {
        let v = vec![F::zero(); 4];
        let mle = vec_to_mle(&v, 4);
        assert!(mle.evaluations.iter().all(|&x| x.is_zero()));
    }
}
