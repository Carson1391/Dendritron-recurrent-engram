"""Core data structures for the Dendritron build pipeline."""

from .capacity import (
    FIXED_COMPUTE_FRACTION,
    FIXED_MEMORY_FRACTION,
    SparseCapacityLedger,
    required_compute_for_memory,
)
from .config import DendritronConfig, tiny_smoke_config
from .definition_bank import (
    DEFINITION_READOUT_MARKER,
    DefinitionSense,
    DictionaryWord,
    canonical_definition_text,
    normalize_word,
)
from .definition_store import FrozenDefinitionPayload, FrozenDefinitionStore
from .expert_graph import BranchSpec, ExpertGraph, ExpertRecord
from .hash_engram import HashEngramAddresses, HashEngramAddressor
from .jtd import JTDIndex, JTDSourceRecord, SurfaceMemoryIndex
from .memory_pipeline import SurfaceMemoryAddressor, SurfaceMemoryPlan
from .retrieval import LongestEngramRouter, MemoryCandidate, ResolvedMemory
from .shared_skill_subspace import (
    AdapterCoefficients,
    LoRAFactors,
    SharedSkillBasis,
    fit_shared_skill_basis,
)
from .tokenizer import (
    CanonicalTokenProjection,
    LOCKED_QWEN_TOKENIZER_ID,
    TokenizerContract,
    align_qwen_input,
    boundary_token_ids,
    build_canonical_token_projection,
)

__all__ = [
    "BranchSpec",
    "AdapterCoefficients",
    "DendritronConfig",
    "DEFINITION_READOUT_MARKER",
    "DefinitionSense",
    "DictionaryWord",
    "ExpertGraph",
    "ExpertRecord",
    "FrozenDefinitionPayload",
    "FrozenDefinitionStore",
    "HashEngramAddresses",
    "HashEngramAddressor",
    "FIXED_COMPUTE_FRACTION",
    "FIXED_MEMORY_FRACTION",
    "JTDIndex",
    "JTDSourceRecord",
    "CanonicalTokenProjection",
    "LOCKED_QWEN_TOKENIZER_ID",
    "LongestEngramRouter",
    "LoRAFactors",
    "MemoryCandidate",
    "ResolvedMemory",
    "SurfaceMemoryAddressor",
    "SurfaceMemoryIndex",
    "SurfaceMemoryPlan",
    "SparseCapacityLedger",
    "SharedSkillBasis",
    "TokenizerContract",
    "align_qwen_input",
    "boundary_token_ids",
    "build_canonical_token_projection",
    "canonical_definition_text",
    "fit_shared_skill_basis",
    "normalize_word",
    "required_compute_for_memory",
    "tiny_smoke_config",
]
