"""
Quantization utilities for PiFormer → Rust ZK prover alignment.

All arithmetic is performed in Python's arbitrary-precision integers.
Conversion to BN254 field elements (mod p) happens only at JSON-serialization
time via ``int_to_field_hex``.

Protocol invariants matched to layernorm.rs / lasso.rs / range.rs:
  - LayerNorm: sum_x, var_x, sigma, y are small non-negative Python ints.
  - Lasso: query_indices are in [0, 2^num_bits - 1]; table entries are
    non-negative Python ints; outputs = sum of sub-table lookups.
  - Range-proof residuals (sigma lo/hi, y lo/hi) must fit in 32 bits.
  - Projections / context matrix: arbitrary signed Python ints → field elements.
"""

from __future__ import annotations

import math
from typing import List, Tuple

# ---------------------------------------------------------------------------
# BN254 scalar field modulus
# ---------------------------------------------------------------------------

BN254_P: int = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)

# ---------------------------------------------------------------------------
# Field-element helpers
# ---------------------------------------------------------------------------


def int_to_field_hex(n: int) -> str:
    """Encode an arbitrary Python integer as a 0x-prefixed 64-nibble big-endian
    hex string representing the canonical representative in BN254 Fr."""
    return f"0x{(n % BN254_P):064x}"


def field_hex_to_int(s: str) -> int:
    """Decode a 0x-prefixed hex string to a Python int (canonical representative)."""
    return int(s, 16) % BN254_P


# ---------------------------------------------------------------------------
# Integer matrix helpers
# ---------------------------------------------------------------------------


def mat_mul_int(
    A: List[List[int]], B: List[List[int]]
) -> List[List[int]]:
    """Integer matrix multiplication (no mod).  A: (m, k), B: (k, n) → (m, n)."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    assert len(B) == k
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
    ]


def mat_add_int(
    A: List[List[int]], B: List[List[int]]
) -> List[List[int]]:
    """Element-wise integer matrix addition."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_transpose(M: List[List[int]]) -> List[List[int]]:
    """Transpose a 2-D list."""
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def mat_to_json(M: List[List[int]]) -> List[List[str]]:
    return [[int_to_field_hex(v) for v in row] for row in M]


def vec_to_json(v: List[int]) -> List[str]:
    return [int_to_field_hex(x) for x in v]


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def quantize_to_int(x: float, scale: float, num_bits: int) -> int:
    """Clamp-round quantize: returns integer in [0, 2^num_bits - 1]."""
    max_val = (1 << num_bits) - 1
    return max(0, min(max_val, round(x / scale)))


def apply_phi_int(
    x_flat: List[int],
    tables_int: List[List[int]],
    bits_per_chunk: int,
) -> Tuple[List[int], List[int]]:
    """
    Apply the additively decomposed lookup activation φ in integer arithmetic.

    Args:
        x_flat: flat list of query indices, each already in
                [0, 2^(c*bits_per_chunk) - 1].
        tables_int: list of c sub-tables, each of size 2^bits_per_chunk.
        bits_per_chunk: bits consumed by each sub-table.

    Returns:
        (outputs, query_indices)  — both are flat lists of ints.
        outputs[j] = Σ_k tables_int[k][ chunk_k(x_flat[j]) ]
    """
    c = len(tables_int)
    chunk_size = 1 << bits_per_chunk
    mask = chunk_size - 1

    outputs: List[int] = []
    query_indices: List[int] = x_flat  # already the full indices

    for idx in x_flat:
        out = 0
        for k in range(c):
            chunk = (idx >> (k * bits_per_chunk)) & mask
            out += tables_int[k][chunk]
        outputs.append(out)

    return outputs, query_indices


# ---------------------------------------------------------------------------
# LayerNorm witness computation
# ---------------------------------------------------------------------------


def compute_ln_witness(
    x_rows: List[List[int]],
    gamma: List[int],
    beta: List[int],
    d: int,
) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
    """
    Compute LayerNorm witness in exact integer arithmetic matching the Rust
    protocol in ``attention/layernorm.rs``.

    Protocol:
      sum_x[i]   = Σ_j x[i][j]
      sq_sum_x[i]= Σ_j x[i][j]²  (sum of squares; matches Rust sq_sum_x field)
      actual_var[i] = Σ_j (d·x[i][j] − sum_x[i])²  = d·(d·sq_sum_x − sum_x²)
      sigma[i]  = floor(√actual_var[i] / d)
                  i.e. largest s such that (d·s)² ≤ actual_var[i]
      sig_d     = d · sigma[i]
      expr[i,j] = gamma[j]·(d·x[i][j] − sum_x[i]) + beta[j]·sig_d
      y[i][j]   = floor((2·expr + sig_d) / (2·sig_d))

    All arguments and return values are Python ints (no field reduction).

    Returns:
        (y, sum_x, sq_sum_x, sigma)

    Raises:
        ValueError if any y value is negative (beta_floor is too small).
    """
    t = len(x_rows)

    sum_x = [sum(row) for row in x_rows]

    # sq_sum_x[i] = Σ_j x[i][j]² — what the Rust prover reads via the "var_x" JSON key
    sq_sum_x: List[int] = []
    # actual_var[i] = Σ_j (d·x[i][j] − sum_x[i])² — used only for sigma computation
    actual_var: List[int] = []
    for i in range(t):
        sq_sum_x.append(sum(x_rows[i][j] ** 2 for j in range(d)))
        actual_var.append(sum((d * x_rows[i][j] - sum_x[i]) ** 2 for j in range(d)))

    # sigma[i] = largest int s.t. (d·s)² ≤ actual_var[i]
    sigma: List[int] = []
    for v in actual_var:
        s = math.isqrt(v) // d
        # Correct for potential off-by-one from integer square root
        while (d * (s + 1)) ** 2 <= v:
            s += 1
        while s > 0 and (d * s) ** 2 > v:
            s -= 1
        sigma.append(s)

    y: List[List[int]] = []
    for i in range(t):
        sig_d = d * sigma[i]
        row_y: List[int] = []
        for j in range(d):
            if sig_d == 0:
                # Degenerate case: all x[i][j] are equal → y = 0 (by convention)
                row_y.append(0)
                continue
            expr = gamma[j] * (d * x_rows[i][j] - sum_x[i]) + beta[j] * sig_d
            yij = (2 * expr + sig_d) // (2 * sig_d)
            if yij < 0:
                raise ValueError(
                    f"y[{i}][{j}] = {yij} < 0. Increase beta_floor. "
                    f"expr={expr}, sig_d={sig_d}, gamma={gamma[j]}, beta={beta[j]}"
                )
            row_y.append(yij)
        y.append(row_y)

    return y, sum_x, sq_sum_x, sigma


def min_beta_floor(
    x_rows: List[List[int]],
    gamma: List[int],
    d: int,
) -> int:
    """
    Compute the minimum integer BETA_FLOOR to add to every beta[j] so that
    all LayerNorm y values are non-negative.

    This is conservative: returns the smallest floor that guarantees
    ``compute_ln_witness`` succeeds with ``beta_exported[j] = beta[j] + floor``.
    """
    if not x_rows:
        return 0

    t, _d = len(x_rows), len(x_rows[0])
    sum_x = [sum(row) for row in x_rows]
    var_x = [
        sum((d * x_rows[i][j] - sum_x[i]) ** 2 for j in range(d))
        for i in range(t)
    ]
    sigma_list = []
    for v in var_x:
        s = math.isqrt(v) // d
        while (d * (s + 1)) ** 2 <= v:
            s += 1
        while s > 0 and (d * s) ** 2 > v:
            s -= 1
        sigma_list.append(s)

    floor_needed = 0
    for i in range(t):
        sig_d = d * sigma_list[i]
        if sig_d == 0:
            continue
        for j in range(d):
            # Worst case: beta = 0
            expr_no_beta = gamma[j] * (d * x_rows[i][j] - sum_x[i])
            # We need: expr_no_beta + beta_floor * sig_d >= -sig_d/2 + 1
            # (so that (2*expr + sig_d) // (2*sig_d) >= 0)
            # Rearranging: beta_floor >= (-sig_d/2 + 1 - expr_no_beta) / sig_d
            #                          = 1 - expr_no_beta/sig_d - 1/2
            min_floor_ij = math.ceil((-expr_no_beta - sig_d // 2) / sig_d) + 1
            if min_floor_ij > floor_needed:
                floor_needed = min_floor_ij

    return max(0, floor_needed)


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------


def extract_ternary_weight_matrix(
    weight_float: List[List[float]],
    alpha: float,
    quant_scale: int = 1,
) -> List[List[int]]:
    """
    Extract an integer effective weight matrix from a ternary layer.

    Python implementation of TernaryLinear._quantize + integer scaling:
      delta   = 0.7 * mean(|w|)
      ternary = sign threshold on delta
      w_eff   = ternary * round(alpha * quant_scale)

    ``quant_scale`` scales alpha to an integer; set to 1 if alpha itself
    should stay as-is (e.g. alpha ≈ integer after training).

    Args:
        weight_float: out_features × in_features float matrix.
        alpha: learnable scalar scale from TernaryLinear.alpha.
        quant_scale: integer scale applied to alpha.

    Returns:
        Integer matrix of same shape with entries in
        { -round(alpha*quant_scale), 0, round(alpha*quant_scale) }.
    """
    import statistics

    flat = [abs(w) for row in weight_float for w in row]
    delta = 0.7 * statistics.mean(flat) if flat else 0.0
    alpha_int = max(1, round(alpha * quant_scale))

    result = []
    for row in weight_float:
        int_row = []
        for w in row:
            if w > delta:
                int_row.append(alpha_int)
            elif w < -delta:
                int_row.append(-alpha_int)
            else:
                int_row.append(0)
        result.append(int_row)
    return result


def extract_phi_tables_int(
    tables_float: List[List[float]],
    output_scale: int = 1,
) -> List[List[int]]:
    """
    Convert floating-point sub-tables to non-negative integer sub-tables.

    table_int[k][i] = max(0, round(tables_float[k][i] * output_scale))
    """
    return [
        [max(0, round(v * output_scale)) for v in tbl]
        for tbl in tables_float
    ]
