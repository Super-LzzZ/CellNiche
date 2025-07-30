from importlib import import_module
from types import ModuleType
from typing import Dict, Any

__version__ = "0.1.5"
__author__  = "ZMLiang <lzzzmmgpt@gmail.com>"

_EXPORTS = {
    # ── CLI (functions) ───────────────────────────────────────
    "cli"          : (".main", "main"),
    "parse_args"    : (".main", "parse_args"),

    # ── Core model (classes) ─────────────────────────────────
    "Model"         : (".model", "Model"),
    "Encoder"       : (".model", "Encoder"),

    # ── Training (function + class) ──────────────────────────
    "run"           : (".trainer", "run"),
    "Trainer"       : (".trainer", "Trainer"),

    # ── Sampler (class) ──────────────────────────────────────
    "Sampler"       : (".sampler", "NeighborSampler"),

    # ── Utils (function) ─────────────────────────────────────
    "clustering_st" : (".utils", "clustering_st"),
    "match_labels" : (".utils", "match_labels"),
    "refine_spatial_domains" : (".utils", "refine_spatial_domains"),
    "setup_seed" : (".utils", "setup_seed"),
    
    "compute_entropy_batch_mixing" : (".utils", "compute_entropy_batch_mixing"),
    "compute_ilisi" : (".utils", "compute_ilisi"),
    "compute_seurat_alignment_score" : (".utils", "compute_seurat_alignment_score"),
    "compute_avg_silhouette_width_batch" : (".utils", "compute_avg_silhouette_width_batch"),
}

_cache: Dict[str, Any] = {}

def __getattr__(name: str) -> Any:
    if name in _cache:
        return _cache[name]
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _EXPORTS[name]
    module: ModuleType = import_module(module_path, __package__)
    obj = getattr(module, attr_name)
    _cache[name] = obj
    return obj

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))

__all__ = list(_EXPORTS.keys())