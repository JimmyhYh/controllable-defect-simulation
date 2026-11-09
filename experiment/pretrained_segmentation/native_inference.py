"""Native geometry and label-free FP32 inference; runtime Torch import is lazy."""
from __future__ import annotations
import numpy as np
from . import common as c
from .data import starts, normalized

SCORE_RULE = {"window": 256, "stride": 128, "tail": "include H-256; sorted unique", "pad": "right reflect; discard before stitching",
              "normalization": "RGB/255 then ImageNet mean/std; no resize or contrast stretch",
              "window_probability": "raw float32 anomaly softmax, no augmentation or smoothing",
              "stitching": "float32 sum in ascending y order, uint16 coverage, float32 equal-weight division; each native pixel once in final map",
              "image_score": "maximum of saved native float32 probability map", "precision": "FP32", "batch": 1, "workers": 0}
THRESHOLD_RULE = {"q": .95, "method": "linear", "comparison": "strictly greater", "score_dtype_for_quantile": "float64 conversion of native float32 maxima", "normal_images": 200}

def geometry(height, width):
    c.require(type(height) is int and height >= 256 and type(width) is int and 2 <= width < 256, "Unsupported native calibration geometry")
    ys = starts(height)
    coverage = [0] * height
    for y in ys:
        for row in range(y, y + 256): coverage[row] += 1
    c.require(ys == sorted(set(ys)) and ys[0] == 0 and ys[-1] == height - 256 and min(coverage) > 0 and max(coverage) < 65536, "Native window coverage/tail invalid")
    return {"ys": ys, "window_count": len(ys), "native_shape": [height, width], "native_pixels": height * width,
            "coverage_min": min(coverage), "coverage_max": max(coverage), "padded_width": 256}

def scalar_verification():
    """Fictitious scalar protocol checks, never actual normal scores/threshold."""
    value = float(np.quantile(np.arange(200, dtype=np.float64), .95, method="linear"))
    c.require(abs(value - 189.05) < 1e-10, "Linear quantile scalar convention differs")
    strict = [x > .5 for x in [0., .5, .5, 1.]]
    c.require(strict == [False, False, False, True], "Strict threshold tie convention differs")
    average = np.float32(np.float32(.2) + np.float32(.8)) / np.float32(2)
    c.require(float(average) == .5, "FP32 overlap scalar convention differs")
    return {"scope": "fictitious scalars only, no image/probability payload", "toy_linear_quantile_0_to_199": value,
            "toy_strict_gt_half": strict, "toy_FP32_overlap_average": float(average), "actual_threshold_computed": False}

def native_tile(rgb, y):
    c.require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "Invalid RGB input")
    h, w, _ = rgb.shape
    c.require(y in starts(h) and 2 <= w < 256, "Invalid label-free window coordinate")
    tile = np.pad(rgb[y:y+256], ((0, 0), (0, 256-w), (0, 0)), mode="reflect")
    return normalized(tile)

def predict_native(model, rgb):
    """Accept RGB only; never read/pass GT, labels, thresholds or role scores."""
    import torch
    h, w, _ = rgb.shape
    spec = geometry(h, w)
    total = np.zeros((h, w), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.uint16)
    with torch.inference_mode():
        for y in spec["ys"]:
            x = torch.from_numpy(native_tile(rgb, y)).unsqueeze(0).to("cuda")
            logits = model(x)
            c.require(logits.dtype == torch.float32 and logits.shape == (1, 2, 256, 256), "Native logits dtype/shape differs")
            tile = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
            c.require(tile.dtype == np.float32 and np.isfinite(tile).all() and tile.min() >= 0 and tile.max() <= 1, "Invalid window probability; preserve uncertain forward")
            total[y:y+256] += tile[:, :w]
            coverage[y:y+256] += 1
            del x, logits, tile
    c.require((coverage > 0).all(), "Uncovered native pixels")
    np.divide(total, coverage.astype(np.float32), out=total)
    c.require(total.dtype == np.float32 and np.isfinite(total).all() and total.min() >= 0 and total.max() <= 1, "Invalid native stitched probability")
    return total, len(spec["ys"])
