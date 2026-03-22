from .activation import StructuredLookupActivation
from .projection import PowerOfTwoLinear
from .attention import LinearAttentionLayer
from .model import PiFormerBlock, PiFormerFFN, PiFormerModel
from .export import export_model, export_weights_rust, export_witness_rust, export_all
from .witness import WitnessGenerator
from .quant import (
    BN254_P,
    int_to_field_hex,
    field_hex_to_int,
    compute_ln_witness,
    min_beta_floor,
    extract_ternary_weight_matrix,
    extract_phi_tables_int,
    apply_phi_int,
    quantize_to_int,
)

__all__ = [
    "StructuredLookupActivation",
    "PowerOfTwoLinear",
    "LinearAttentionLayer",
    "PiFormerBlock",
    "PiFormerFFN",
    "PiFormerModel",
    # Export
    "export_model",
    "export_weights_rust",
    "export_witness_rust",
    "export_all",
    # Witness generation
    "WitnessGenerator",
    # Quantization utilities
    "BN254_P",
    "int_to_field_hex",
    "field_hex_to_int",
    "compute_ln_witness",
    "min_beta_floor",
    "extract_ternary_weight_matrix",
    "extract_phi_tables_int",
    "apply_phi_int",
    "quantize_to_int",
]
