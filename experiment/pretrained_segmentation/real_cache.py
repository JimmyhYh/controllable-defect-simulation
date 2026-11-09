"""Read-only complete E cache parser. No GT decode, threshold search or metrics."""
from __future__ import annotations
import numpy as np
from . import common as c
from . import formal_common as f
from . import real_scoring_common as r

def probability_path(request):
    return f"probabilities/{request['index']:04d}_{request['id']}.npy"

def checked_probability(request, row):
    name = probability_path(request)
    c.require(row["index"] == request["index"] and row["id"] == request["id"] and row["probability"] == name and row["source_image_sha256"] == request["image_sha256"] and row["window_forwards"] == request["window_count"], "E saved row/order/source differs")
    c.require(c.sha(r.RUN / name) == row["probability_sha256"], "E probability payload identity differs")
    prob = np.load(r.RUN / name, allow_pickle=False)
    c.require(prob.dtype == np.float32 and list(prob.shape) == request["native_shape"] and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, "E probability shape/dtype/range differs")
    c.require(type(row["score"]) is float and row["score"] == float(prob.max()), "E saved native maximum differs")
    return prob

def verify_results():
    from .real_scoring import verify_preparation, binding
    p, req = verify_preparation(); bound = binding(p)
    c.resources(2, 1)
    c.require(f.strict_seal(r.RUN)["identity"] == bound, "E complete cache binding differs")
    expected = {"transaction.json", "scores.json", "E1004评分结果.md", "complete.json"} | {probability_path(x) for x in req}
    c.require(c.paths(r.RUN) == expected, "Unknown/missing E cache member")
    scores = c.load(r.RUN / "scores.json"); transaction = c.load(r.RUN / "transaction.json")
    c.require(scores["binding"] == transaction["binding"] == bound and transaction["intended_images"] == 1004 and transaction["intended_windows"] == p["planned_window_forwards"], "E transaction binding differs")
    c.require(scores["images_scored"] == scores["raw_E_image_decodes"] == len(scores["items"]) == 1004 and scores["window_forwards"] == p["planned_window_forwards"] and scores["constructor_forwards"] == 0 and scores["native_pixels"] == p["planned_native_pixels"], "E complete denominators differ")
    c.require(scores["state_before"] == scores["state_after"] == p["expected_state_hash"] and scores["buffers_before"] == scores["buffers_after"] == p["expected_buffers_hash"], "E parameter/BN signatures differ")
    c.require(scores["GT_read_by_predictor"] is False and scores["metrics_computed"] is False and scores["training_or_parameter_updates"] == 0 and scores["score_rule"] == p["score_rule"], "E scoring-only scope differs")
    for request, row in zip(req, scores["items"]):
        prob = checked_probability(request, row); del prob
    return {"E_scores_verified": True, "images": 1004, "native_pixels": p["planned_native_pixels"], "window_forwards_recorded": p["planned_window_forwards"],
            "original_threshold": p["threshold"], "additional_forwards": 0, "GT_decodes_or_metrics": 0, "meaning": p["meaning"]}

def iter_metric_inputs():
    """Future CPU caller: full verify first, then ordered probability + GT metadata.

    This function does NOT decode GT or compute a metric. Caller must independently
    freeze/authorize GT reading and actual metric implementation before using it.
    """
    verify_results()
    requests = c.load(r.PREP / "requests.json"); gt = c.load(r.PREP / "metric_sources.json")
    rows = c.load(r.RUN / "scores.json")["items"]
    for request, source, row in zip(requests, gt, rows):
        c.require(source["index"] == request["index"] and source["id"] == request["id"] and source["native_shape"] == request["native_shape"], "CPU GT source order/shape metadata differs")
        yield source, checked_probability(request, row), row["score"]
