"""
Quantization utilities for PiFormer → Rust ZK prover alignment.

All arithmetic is performed in Python's arbitrary-precision integers.
Conversion to BN254 field elements (mod p) happens only at JSON-serialization
time via ``int_to_field_hex``.

Protocol invariants matched to layernorm.rs / lasso.rs / range.rs:
  - LayerNorm: sum_x, sq_sum_x, inv_sigma, norm_x, y are small non-negative
    Python ints computed via the Lasso-based inv_sqrt lookup.
  - Lasso: query_indices are in [0, 2^num_bits - 1]; table entries are
    non-negative Python ints; outputs = sum of sub-table lookups.
  - Range-proof residuals (y lo/hi) must fit in 32 bits.
  - Projections / context matrix: arbitrary signed Python ints → field elements.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# BN254 scalar field modulus
# ---------------------------------------------------------------------------

BN254_P: int = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)

# ---------------------------------------------------------------------------
# Lasso inv-sqrt lookup constants (must match inv_sqrt.rs)
# ---------------------------------------------------------------------------

INV_SQRT_SCALE: int = 1 << 16   # 65536
CHUNK_BITS: int = 16
CHUNK_SIZE: int = 1 << CHUNK_BITS  # 65536

_INV_SQRT_T1: Optional[List[int]] = None


def _get_inv_sqrt_t1() -> List[int]:
    """Build (once) and return T_1: T_1[h] = round(65536 / sqrt((h+0.5)*65536))."""
    global _INV_SQRT_T1
    if _INV_SQRT_T1 is not None:
        return _INV_SQRT_T1
    scale = float(INV_SQRT_SCALE)
    chunk_f = float(CHUNK_SIZE)
    t1 = []
    for h in range(CHUNK_SIZE):
        x_center = (h + 0.5) * chunk_f
        if x_center > 0.0:
            val = round(scale / math.sqrt(x_center))
        else:
            val = round(scale / math.sqrt(0.5 * chunk_f))
        t1.append(val)
    _INV_SQRT_T1 = t1
    return t1


def lookup_inv_sqrt(var_x: int) -> int:
    """
    Look up approximate 1/sqrt(var_x) using the two-table Lasso decomposition.

    Matches Rust ``lookup_inv_sqrt`` in inv_sqrt.rs:
      T_0[lo] = 0  (k=0, low bits)
      T_1[hi] = round(65536 / sqrt((hi+0.5)*65536))  (k=1, high bits)
      result = T_0[lo] + T_1[hi] = T_1[hi]
    """
    t1 = _get_inv_sqrt_t1()
    hi = (var_x >> CHUNK_BITS) & (CHUNK_SIZE - 1)
    return t1[min(hi, CHUNK_SIZE - 1)]

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
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Compute LayerNorm witness using the Lasso inv-sqrt lookup, matching the
    Rust protocol in ``attention/layernorm.rs``.

    Protocol:
      sum_x[i]    = Σ_j x[i][j]
      sq_sum_x[i] = Σ_j x[i][j]²
      var_x[i]    = d·(d·sq_sum_x[i] − sum_x[i]²)  (= Σ_j (d·x−sum)²)
      inv_sigma[i]= lookup_inv_sqrt(var_x[i])         (= T_1[var_x >> 16])
      norm_x[i,j] = (d·x[i][j] − sum_x[i]) · inv_sigma[i]
      expr[i,j]   = gamma[j]·norm_x[i,j] + beta[j]·INV_SQRT_SCALE
      y[i][j]     = round(expr / INV_SQRT_SCALE)
                  = (expr + INV_SQRT_SCALE//2) // INV_SQRT_SCALE

    All arguments and return values are Python ints (no field reduction).

    Returns:
        (y, sum_x, sq_sum_x)   — 3 values

    Raises:
        ValueError if any y value is negative (beta_floor is too small).
    """
    t = len(x_rows)
    S = INV_SQRT_SCALE

    sum_x = [sum(row) for row in x_rows]
    sq_sum_x: List[int] = [
        sum(x_rows[i][j] ** 2 for j in range(d)) for i in range(t)
    ]

    y: List[List[int]] = []
    for i in range(t):
        var_x = d * (d * sq_sum_x[i] - sum_x[i] ** 2)
        var_x = max(0, var_x)
        inv_sigma = lookup_inv_sqrt(var_x)

        row_y: List[int] = []
        for j in range(d):
            norm_x = (d * x_rows[i][j] - sum_x[i]) * inv_sigma
            expr = gamma[j] * norm_x + beta[j] * S
            yij = (expr + S // 2) // S
            if yij < 0:
                raise ValueError(
                    f"y[{i}][{j}] = {yij} < 0. Increase beta_floor. "
                    f"expr={expr}, inv_sigma={inv_sigma}, gamma={gamma[j]}, beta={beta[j]}"
                )
            row_y.append(yij)
        y.append(row_y)

    return y, sum_x, sq_sum_x


def min_beta_floor(
    x_rows: List[List[int]],
    gamma: List[int],
    d: int,
) -> int:
    """
    Compute the minimum integer BETA_FLOOR to add to every beta[j] so that
    all LayerNorm y values are non-negative.

    Uses the Lasso-based formula: y = (gamma*norm_x + beta*S + S//2) // S >= 0
    ⟹ beta >= ceil((-gamma*norm_x - S//2) / S)
    """
    if not x_rows:
        return 0

    S = INV_SQRT_SCALE
    t = len(x_rows)
    sum_x = [sum(row) for row in x_rows]
    sq_sum_x = [sum(x_rows[i][j] ** 2 for j in range(d)) for i in range(t)]

    floor_needed = 0
    for i in range(t):
        var_x = max(0, d * (d * sq_sum_x[i] - sum_x[i] ** 2))
        inv_sigma = lookup_inv_sqrt(var_x)
        if inv_sigma == 0:
            continue
        for j in range(d):
            norm_x = (d * x_rows[i][j] - sum_x[i]) * inv_sigma
            # expr_no_beta = gamma[j] * norm_x
            # Need: gamma[j]*norm_x + beta_floor*S + S//2 >= 0
            # ⟹ beta_floor >= ceil((-gamma[j]*norm_x - S//2) / S)
            expr_no_beta = gamma[j] * norm_x
            min_floor_ij = math.ceil((-expr_no_beta - S // 2) / S)
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
