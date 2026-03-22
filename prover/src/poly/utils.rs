use crate::field::F;
use crate::poly::DenseMLPoly; // DenseMLPolyのパスはご自身の構成に合わせてください
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
