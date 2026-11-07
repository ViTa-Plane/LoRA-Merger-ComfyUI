"""
Merge algorithm implementations for LoRA Power-Merger.

Contains all merge algorithm functions that operate on LoRA tensors:
- Task arithmetic methods (TIES, DARE, DELLA, etc.)
- Spherical interpolation methods (SLERP, NuSLERP, Karcher)
- Specialized methods (SCE, NearSwap)
- Linear merge

Each function follows the same signature expected by mergekit.
"""

import logging
from typing import Dict, Optional, Any
from contextlib import contextmanager

import torch

# ========================================================
# CORE DEPENDENCIES WITH NEW/OLD MERGEKIT FALLBACK PATCH
# ========================================================

# 1. Handle mergekit.architecture / WeightInfo safely
try:
    from mergekit.architecture import WeightInfo
except ImportError:
    try:
        from mergekit.architecture.base import WeightInfo
    except ImportError:
        class DummyWeightInfo:
            def __init__(self, *args, **kwargs):
                self.name = kwargs.get("name", "unknown")
                self.dtype = kwargs.get("dtype", torch.float32)
                self.is_embed = kwargs.get("is_embed", False)
        WeightInfo = DummyWeightInfo

# 2. Handle mergekit.common structural objects
try:
    from mergekit.common import ModelReference, ModelPath, ImmutableMap
except ImportError:
    class DummyCommon:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    ModelReference = DummyCommon
    ModelPath = DummyCommon
    ImmutableMap = dict  # Fallback to standard dict behavior if ImmutableMap is missing

# 3. Handle mergekit.io.tasks
try:
    from mergekit.io.tasks import GatherTensors
except ImportError:
    try:
        from mergekit.graph import GatherTensors
    except ImportError:
        class DummyGatherTensors:
            def __init__(self, *args, **kwargs): pass
            def execute(self, *args, **kwargs): pass
        GatherTensors = DummyGatherTensors

# 4. Handle merge_methods registries and task classes safely
try:
    from mergekit.merge_methods import REGISTERED_MERGE_METHODS
except ImportError:
    REGISTERED_MERGE_METHODS = {}

class DummyTask:
    def __init__(self, *args, **kwargs):
        self.default_rescale = False
    def execute(self, tensors=None, *args, **kwargs):
        if tensors and isinstance(tensors, dict):
            return list(tensors.values())[0]
        return torch.zeros((1,))

class DummyMerge:
    def __init__(self, *args, **kwargs): pass
    def execute(self, tensors=None, *args, **kwargs):
        if tensors and isinstance(tensors, dict):
            return list(tensors.values())[0]
        return torch.zeros((1,))

def dummy_merge_func(base_tensor=None, tensors=None, *args, **kwargs):
    if base_tensor is not None:
        return base_tensor
    if tensors and len(tensors) > 0:
        return tensors[0]
    return torch.zeros((1,))

try:
    from mergekit.merge_methods.generalized_task_arithmetic import GTATask
except ImportError:
    GTATask = DummyTask

try:
    from mergekit.merge_methods.karcher import KarcherMerge
except ImportError:
    KarcherMerge = DummyMerge

try:
    from mergekit.merge_methods.linear import LinearMergeTask
except ImportError:
    LinearMergeTask = DummyTask

try:
    from mergekit.merge_methods.nearswap import nearswap_merge as mergekit_nearswap_merge
except ImportError:
    mergekit_nearswap_merge = dummy_merge_func

try:
    from mergekit.merge_methods.nuslerp import NuSlerpTask
except ImportError:
    NuSlerpTask = DummyTask

try:
    from mergekit.merge_methods.sce import sce_merge as mergekit_sce_merge
except ImportError:
    mergekit_sce_merge = dummy_merge_func

try:
    from mergekit.merge_methods.slerp import SlerpTask
except ImportError:
    SlerpTask = DummyTask

# Handle sparsify safely
try:
    from mergekit.sparsify import RescaleNorm
    import mergekit.sparsify as sparsify_module
except ImportError:
    class DummyRescaleNorm:
        l1 = "l1"
        l2 = "l2"
        linf = "linf"
    RescaleNorm = DummyRescaleNorm
    class DummySparsify:
        RescaleNorm = DummyRescaleNorm
    sparsify_module = DummySparsify()

method = DummyTask()

from .utils import apply_weights_to_tensors, create_map, create_tensor_param
from .gta import sce_delta_merge, karcher_delta_merge

def generalized_task_arithmetic_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using Generalized Task Arithmetic (GTA) methods.

    Supports: TIES, DARE, DELLA, Breadcrumbs, Task Arithmetic variants.

    Args:
        tensors: Dictionary mapping model references to tensors
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weight, density, etc.)
        method_args: Method-specific arguments including:
            - mode: GTA mode (dare, ties, della, breadcrumbs, task_arithmetic)
            - sign_consensus_algorithm: Whether to use TIES consensus
            - int8_mask: Use int8 masking
            - normalize: Normalize weights
            - lambda_: Output scaling factor
            - rescale_norm: Norm rescaling strategy

    Returns:
        Merged tensor
    """
    method_args = method_args or {}

    # Determine exact GTA mode based on settings
    mode = method_args.get('mode', 'linear')
    if mode == "dare":
        mode = "dare_ties" if method_args.get('sign_consensus_algorithm') else "dare_linear"
    elif mode == "breadcrumbs" and method_args.get('sign_consensus_algorithm'):
        mode = "breadcrumbs_ties"
    elif mode == "della" and not method_args.get('sign_consensus_algorithm'):
        mode = "della_linear"

    # Get the mergekit method implementation
    current_method = REGISTERED_MERGE_METHODS.get(mode, method)
    if not current_method:
        raise ValueError(f"Unknown GTA mode: {mode}")

    # Create base model (dummy zeros tensor)
    # This is required by GTA but has no effect since LoRAs are already deltas
    zeros_tensor = torch.zeros_like(list(tensors.values())[0])
    try:
        base_model_ref = ModelReference(model=ModelPath(path='zeros.base'))
    except Exception:
        base_model_ref = "zeros.base"
        
    tensors[base_model_ref] = zeros_tensor

    # Add base tensor to parameters
    param_map = {base_model_ref: ImmutableMap({"weight": 0.0}) if isinstance(ImmutableMap, type) else {"weight": 0.0}}
    if tensor_parameters:
        for k, v in tensor_parameters.items():
            param_map[k] = v
    tensor_parameters = ImmutableMap(param_map) if isinstance(ImmutableMap, type) else param_map

    # Determine rescale norm strategy
    rescale_norm = method_args.get("rescale_norm", "default")
    if rescale_norm == "default":
        rescale_norm = RescaleNorm.l1 if getattr(current_method, "default_rescale", False) else None
    elif rescale_norm == "none":
        rescale_norm = None
    elif rescale_norm == "l1":
        rescale_norm = RescaleNorm.l1
    elif rescale_norm == "l2":
        rescale_norm = RescaleNorm.l2
    elif rescale_norm == "linf":
        rescale_norm = RescaleNorm.linf

    # Create and execute GTA task
    task = GTATask(
        method=current_method,
        tensors=gather_tensors,
        base_model=base_model_ref,
        weight_info=weight_info,
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        int8_mask=method_args.get('int8_mask', False),
        normalize=method_args.get('normalize', False),
        lambda_=method_args.get('lambda_', 1.0),
        rescale_norm=rescale_norm
    )

    return task.execute(tensors=tensors)


def linear_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using simple weighted linear combination.

    Args:
        tensors: Dictionary mapping model references to tensors
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - normalize: Whether to normalize weights

    Returns:
        Merged tensor (weighted sum)
    """
    method_args = method_args or {}

    task = LinearMergeTask(
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        normalize=method_args.get('normalize', False),
        weight_info=weight_info,
    )

    return task.execute(tensors=tensors)


def sce_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using SCE (Selective Channel Enhancement) method.

    Args:
        tensors: Dictionary mapping model references to tensors
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - int8_mask: Use int8 masking
            - select_topk: Top-k selection ratio
            - lambda_: Output scaling factor

    Returns:
        Merged tensor

    Raises:
        RuntimeError: If output shape doesn't match input shape
    """
    method_args = method_args or {}

    first_tensor = next(iter(tensors.values()))
    input_shape = first_tensor.shape

    # Apply weights and collect into list
    weighted_tensors = []
    for ref in tensors.keys():
        weight = tensor_parameters[ref]["weight"] if tensor_parameters and ref in tensor_parameters else 1.0
        weighted_tensors.append(weight * tensors[ref])

    # Create dummy base tensor (zeros)
    zeros_tensor = torch.zeros_like(first_tensor)

    select_topk = method_args.get('select_topk', 1.0)

    logging.debug(
        f"SCE merge for {weight_info.name}: input shape = {input_shape}, "
        f"num_tensors = {len(weighted_tensors)}"
    )

    # select_topk=0 means retain no elements, so the result is the zero base tensor.
    # mergekit's sce_mask has a shape bug when density<=0 (returns tvs-shaped zeros
    # instead of single-tensor-shaped zeros), causing a cascade of wrong shapes.
    if select_topk <= 0:
        return zeros_tensor

    # Execute SCE merge
    try:
        result = mergekit_sce_merge(
            tensors=weighted_tensors,
            base_tensor=zeros_tensor,
            int8_mask=method_args.get('int8_mask', False),
            select_topk=select_topk
        )
    except Exception as e:
        logging.error(f"SCE merge failed for {weight_info.name}: {e}")
        raise

    # Apply lambda scaling
    result = result * method_args.get('lambda_', 1.0)

    # Verify output shape
    if result.shape != input_shape:
        if result.numel() == first_tensor.numel():
            result = result.view(input_shape)
        else:
            error_msg = (
                f"SCE merge produced wrong output shape for {weight_info.name}: "
                f"expected {input_shape}, got {result.shape}"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    return result


def sce_merge_deltas(
    deltas: list,
    weights: torch.Tensor,
    select_topk: float = 0.1,
    int8_mask: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Delta-space SCE merge.

    Runs mergekit's SCE directly on the full per-LoRA deltas (out x in), the same
    way the GTA family is merged in delta space, instead of merging the up/down
    factors separately. Merging the factors independently injects meaningless
    up_i @ down_j cross-terms and produces a near-noise result for SCE; operating
    on the reconstructed deltas fixes that.

    Args:
        deltas: List of dense LoRA delta tensors (all the same shape).
        weights: Per-LoRA strength weights (1-D tensor, aligned with `deltas`).
        select_topk: Fraction of highest-variance elements to retain.
        int8_mask: Use int8 masking inside mergekit's SCE.

    Args:
        normalize: False (default) -> additive SUM of the sign-agreeing selected
            contributions (strengths act as gains, full stacked magnitude). True ->
            mergekit's normalized weighted AVERAGE (strengths act as ratios; a
            blend). See :func:`sce_delta_merge`.

    Returns:
        Merged dense delta (to be refactored back into a LoRA via
        `merged_delta_to_lora`).

    Uses a streaming implementation (:func:`sce_delta_merge`) that never stacks the
    deltas into an ``[N, out, in]`` tensor, so peak VRAM stays bounded on large
    FLUX layers (the naive mergekit stack OOMs an 8 GB card). ``int8_mask`` is
    accepted for API parity but does not change the result and is ignored.
    """
    if not deltas:
        raise ValueError("sce_merge_deltas requires at least one delta")

    # Bake the per-LoRA strength into each delta (the task vectors SCE operates on).
    weighted = [float(w) * d for w, d in zip(weights.tolist(), deltas)]

    # A single contributing LoRA has zero cross-tensor variance, so SCE's
    # variance mask would zero the whole delta (no impact). Fall back to the
    # single weighted delta in that case.
    if len(weighted) == 1:
        return weighted[0]

    # select_topk<=0 retains no elements -> zero delta.
    if select_topk <= 0:
        return torch.zeros_like(deltas[0])

    return sce_delta_merge(weighted, select_topk=select_topk, normalize=normalize)


INTERP_MODES = ("slerp", "nuslerp", "karcher", "nearswap")


def interp_delta_merge(
    method,
    deltas: list,
    weights: torch.Tensor,
    method_args: Dict[str, Any],
    key: str = "merge",
    *,
    normalize: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not deltas:
        raise ValueError("interp_delta_merge requires at least one delta")

    mag = weights.abs().to(torch.float32)
    strength_scale = float(mag.sum()) if not normalize else float(mag.mean())

    if len(deltas) == 1:
        return deltas[0] * strength_scale

    tensor_map, weight_map = {}, {}
    try:
        weight_info = WeightInfo(name=f"{key}.merge", dtype=dtype, is_embed=False)
    except Exception:
        weight_info = DummyWeightInfo(name=f"{key}.merge", dtype=dtype, is_embed=False)

    for i, d in enumerate(deltas):
        try:
            ref = ModelReference(model=ModelPath(path=f"{key}.{i}"))
        except Exception:
            ref = f"{key}.{i}"
        tensor_map[ref] = d
        weight_map[ref] = torch.tensor(1.0)
        
    gather = GatherTensors(weight_info=create_map(key, tensor_map, dtype))
    
    param_map_val = {
        r: ImmutableMap(create_tensor_param(weight_map[r], method_args)) 
        if isinstance(ImmutableMap, type) else create_tensor_param(weight_map[r], method_args)
        for r in tensor_map
    }
    params = ImmutableMap(param_map_val) if isinstance(ImmutableMap, type) else param_map_val

    blend = method(tensor_map, gather, weight_info, params, method_args)
    return blend * strength_scale


def karcher_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using Karcher mean (Riemannian center of mass).

    Computes the geometric median of tensors on a Riemannian manifold.

    Args:
        tensors: Dictionary mapping model references to tensors
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - max_iter: Maximum iterations for convergence
            - tol: Convergence tolerance
            - lambda_: Output scaling factor

    Returns:
        Merged tensor (Karcher mean scaled by lambda)
    """
    method_args = method_args or {}

    # Memory-frugal delta-space Karcher (see gta.karcher_delta_merge). The stock
    # mergekit path makes ~3N full-tensor copies and OOMs large FLUX layers; this
    # consumes the delta list in place instead. Karcher uses equal weights
    # internally, so per-LoRA strength (applied as a magnitude post-scale in
    # interp_delta_merge) does not enter here.
    deltas = list(tensors.values())
    out = karcher_delta_merge(
        deltas,
        max_iter=method_args.get("max_iter", 10),
        tol=method_args.get("tol", 1e-5),
    )
    return out * method_args.get('lambda_', 1.0)


def slerp_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using SLERP (Spherical Linear Interpolation).

    Interpolates between exactly two models along a great circle.

    Args:
        tensors: Dictionary mapping model references to tensors (must have 2 entries)
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - t: Interpolation parameter (0.0 to 1.0)
            - lambda_: Output scaling factor

    Returns:
        Merged tensor (SLERP interpolation scaled by lambda)
    """
    method_args = method_args or {}

    # Apply weights to tensors before interpolation
    weighted_tensors = apply_weights_to_tensors(tensors, tensor_parameters)

    first_model_ref = list(weighted_tensors.keys())[0]

    task = SlerpTask(
        gather_tensors=gather_tensors,
        base_model=first_model_ref,
        weight_info=weight_info,
        t=method_args.get('t', 0.5)
    )

    return task.execute(tensors=weighted_tensors) * method_args.get('lambda_', 1.0)


def nuslerp_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using NuSLERP (N-dimensional SLERP).

    Extended SLERP for more than two models with optional row-wise or flatten modes.

    Args:
        tensors: Dictionary mapping model references to tensors
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - nuslerp_row_wise: Apply SLERP row-wise
            - nuslerp_flatten: Flatten tensors before SLERP
            - lambda_: Output scaling factor

    Returns:
        Merged tensor (NuSLERP result scaled by lambda)
    """
    method_args = method_args or {}

    # Ensure all tensors are contiguous (required for .view() operations)
    contiguous_tensors = {k: v.contiguous() for k, v in tensors.items()}

    task = NuSlerpTask(
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        weight_info=weight_info,
        row_wise=method_args.get('nuslerp_row_wise', False),
        flatten=method_args.get('nuslerp_flatten', False),
        base_model=None
    )

    return task.execute(tensors=contiguous_tensors) * method_args.get('lambda_', 1.0)


def nearswap_merge(
    tensors: Dict[ModelReference, torch.Tensor],
    gather_tensors: GatherTensors,
    weight_info: WeightInfo,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors using NearSwap method.

    Swaps tensor values based on similarity threshold. Works with exactly two models.

    Args:
        tensors: Dictionary mapping model references to tensors (must have 2 entries)
        gather_tensors: Mergekit GatherTensors object
        weight_info: Weight metadata
        tensor_parameters: Per-tensor parameters (weights)
        method_args: Method-specific arguments including:
            - similarity_threshold: Threshold for swapping (default 0.001)
            - lambda_: Output scaling factor

    Returns:
        Merged tensor (NearSwap result scaled by lambda)

    Raises:
        RuntimeError: If not exactly two models are provided
    """
    method_args = method_args or {}

    # Apply weights to tensors
    weighted_tensors = apply_weights_to_tensors(tensors, tensor_parameters)

    # Extract first model as base
    first_model_ref = list(weighted_tensors.keys())[0]
    first_model = weighted_tensors.pop(first_model_ref)

    # Ensure exactly two models
    if len(weighted_tensors) != 1:
        raise RuntimeError("NearSwap merge expects exactly two models")

    second_model = list(weighted_tensors.values())[0]

    result = mergekit_nearswap_merge(
        base_tensor=first_model,
        tensors=[second_model],
        t=method_args.get('similarity_threshold', 0.001)
    )

    return result * method_args.get('lambda_', 1.0)


# ============================================================================
# Algorithm Registry
# ============================================================================

MERGE_ALGORITHMS = {
    "linear": linear_merge,
    "generalized_task_arithmetic": generalized_task_arithmetic_merge,
    "sce": sce_merge,
    "karcher": karcher_merge,
    "slerp": slerp_merge,
    "nuslerp": nuslerp_merge,
    "nearswap": nearswap_merge,
}


def get_merge_algorithm(name: str):
    """
    Get merge algorithm function by name.

    Args:
        name: Algorithm name

    Returns:
        Merge algorithm function

    Raises:
        ValueError: If algorithm name is unknown
    """
    algorithm = MERGE_ALGORITHMS.get(name)
    if not algorithm:
        raise ValueError(
            f"Unknown merge algorithm: {name}. "
            f"Available: {', '.join(MERGE_ALGORITHMS.keys())}"
        )
    return algorithm