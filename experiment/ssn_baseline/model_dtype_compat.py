"""Explicit future runtime boundary. No ML imports or construction at import time.

Stage-2 commands parse this file only. build_model will load local generic V1
parameters and run the AUTHOR's constructor shape forward; it must only be
called by a separately authorized manual pilot/runtime entry, not preparation.
"""
from pathlib import Path
from .contracts import MODEL_CONFIG, INPUT_OUTPUT


def build_model(*, checkpoint_path, config):
    # No runtime CLI is supplied in stage 2; pilot budget/gate is not frozen.
    expected_keys = set(MODEL_CONFIG) | {"epochs"}
    if set(config) != expected_keys or any(config[k] != v for k, v in MODEL_CONFIG.items()):
        raise RuntimeError("Use the independently frozen candidate and explicit future epoch budget")
    if type(config["epochs"]) is not int or config["epochs"] <= 0:
        raise RuntimeError("Epoch budget must be explicitly frozen before construction")
    from .core_v1_dtype_compat.supersimplenet import SuperSimpleNet
    runtime = dict(config)
    runtime["local_backbone_path"] = str(Path(checkpoint_path).resolve())
    return SuperSimpleNet(tuple(INPUT_OUTPUT["input_hw"]), runtime)


def feature_targets(input_mask, feature_hw=(160, 58)):
    """Future Tensor-only GT mapping, matching author train.py lines70-78."""
    import torch
    import torch.nn.functional as F
    if tuple(input_mask.shape[-2:]) != (640, 232):
        raise RuntimeError("Expected already prepared binary input-space GT")
    mask = F.interpolate(input_mask.float().unsqueeze(1), size=feature_hw, mode="bilinear", align_corners=True)
    return torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))


def native_scores(smoothed_input_logits, classification_logits, native_hw):
    """Future postprocessing only; accepts eval outputs, never calls a model.

The author's eval already upsamples and blurs LOGITS to the input canvas.
Native inverse mapping is the explicit adaptation before sigmoid, not an
interpolation of probabilities and not test-pool normalization.
"""
    import torch
    import torch.nn.functional as F
    if tuple(smoothed_input_logits.shape) != (1, 1, 640, 232) or classification_logits.numel() != 1:
        raise RuntimeError("Single input-space eval map and one classification logit required")
    if len(native_hw) != 2 or any(type(x) is not int or x <= 0 for x in native_hw):
        raise RuntimeError("Native dimensions must come from the same image's verified metadata")
    native_logits = F.interpolate(smoothed_input_logits.float(), size=tuple(native_hw), mode="bilinear", align_corners=False)
    probability = torch.sigmoid(native_logits).to(dtype=torch.float32)
    classification_score = torch.sigmoid(classification_logits.float()).reshape(())
    return {"native_probability": probability, "classification_score": classification_score,
            "localization_max": probability.max(), "classification_logit": classification_logits.float().reshape(())}
