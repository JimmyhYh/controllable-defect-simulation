"""Read-only verification of one complete Q probability/threshold cache; no quantile."""
from __future__ import annotations

import json
import math

from . import calibration as a


def verify_results(group, seed, p, requests, models):
    import numpy as np
    root, runtime = a.paths(group, seed); entry = a.find_model(models, group, seed); bound = a.binding(p, entry)
    a.t.resources()
    scores, threshold = a.verify_saved_metadata(group, seed, p, requests, models)
    a.c.require(scores["binding"] == threshold["binding"] == bound and threshold["score_source_sha256"] == a.t.sha(root / "scores.json"), "Saved score/threshold identity differs")
    a.c.require(scores["images_scored"] == scores["raw_Q_image_decodes"] == len(scores["items"]) == 200 and scores["window_forwards"] == 844 and scores["constructor_forwards"] == 0 and scores["native_pixels"] == 29202876, "Saved count differs")
    a.c.require(scores["state_before"] == scores["state_after"] == entry["expected_state_hash"] and scores["buffers_before"] == scores["buffers_after"] == entry["expected_buffers_hash"], "Saved model state/BN differs")
    a.c.require(scores["GT_read_by_predictor"] is False and scores["training_or_parameter_updates"] == 0 and scores["score_rule"] == a.SCORE_RULE, "Unexpected scoring scope")
    expected_journal = []
    for request in requests:
        expected_journal.append({"event": "decode_start", "index": request["index"], "id": request["id"]})
        expected_journal.extend({"event": "window_start", "index": request["index"], "id": request["id"], "window_index": i, "y": y} for i, y in enumerate(request["ys"]))
    journal = [json.loads(line) for line in (root / "window_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    a.c.require(journal == expected_journal, "Decode/window intentions not once-only original order")
    saved_values = []
    for row, request in zip(scores["items"], requests):
        name = f"probabilities/{request['index']:03d}_{request['id']}.npy"
        a.c.require(row["index"] == request["index"] and row["id"] == request["id"] and row["source_image_sha256"] == request["image_sha256"] and row["probability"] == name and row["window_forwards"] == request["window_count"], "Saved input order/hash differs")
        prob = np.load(root / name, allow_pickle=False)
        a.c.require(prob.dtype == np.float32 and list(prob.shape) == request["native_shape"] and np.isfinite(prob).all() and 0 <= prob.min() <= prob.max() <= 1, "Invalid saved native float32 probability")
        a.c.require(row["score"] == float(prob.max()), "Saved image max differs")
        saved_values.append(row["score"])
    tau = threshold["threshold"]
    a.c.require(type(tau) is float and math.isfinite(tau) and min(saved_values) <= tau <= max(saved_values), "Invalid saved threshold")
    a.c.require(threshold["rule"] == a.THRESHOLD_RULE and threshold["normal_denominator"] == 200 and threshold["independent_validation"] is False and threshold["quantile_evaluations"] == 1, "Saved threshold rule differs")
    fp = sum(value > tau for value in saved_values)
    a.c.require(threshold["FP"] == fp and threshold["TN"] == 200-fp, "Saved FP/TN differs at original threshold")
    return {"calibration_complete": True, "cache_verified": True, "group": group, "seed": seed, "images": 200, "windows": 844,
            "threshold": tau, "FP": fp, "TN": 200-fp, "normal_denominator": 200, "independent_validation": False,
            "additional_forwards": 0, "additional_quantile_evaluations": 0, "training_updates": 0}
