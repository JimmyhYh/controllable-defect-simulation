"""Explicit final fixed-window scoring. Predictor does not accept/read GT."""
from __future__ import annotations
import numpy as np
from . import common as c

def score_binding():
    from .training import verify_run
    verify_run()
    return {"training_complete_sha256": c.sha(c.RUN / "complete.json"), "model_complete_sha256": c.sha(c.MODEL / "complete.json"),
            "final_weight_sha256": c.sha(c.MODEL / "final_weights.safetensors"), "pilot_complete_sha256": c.sha(c.PREP / "complete.json")}

def verify_scores():
    from .preparation import strict_directories
    expected = score_binding()
    seal = strict_directories(c.SCORES)
    result = c.load(c.SCORES / "result.json")
    c.require(seal["identity"] == expected and result["binding"] == expected and result["input_forwards"] == 21, "Score identity/input count differs")
    items = c.load(c.INPUT / "protocol.json")["items"]
    expected_files = {"transaction.json", "result.json", "complete.json"} | {f"probabilities/{i:02d}_{item['id']}.npy" for i, item in enumerate(items)}
    c.require(c.paths(c.SCORES) == expected_files, "Unknown score artifact/lock")
    c.require([r["id"] for r in result["items"]] == c.R5 + c.N16 and result["state_before"] == result["state_after"] and result["buffers_before"] == result["buffers_after"], "Scoring order/state mutation")
    for i, row in enumerate(result["items"]):
        p = np.load(c.SCORES / row["probability"], allow_pickle=False)
        width = items[i]["width"]
        c.require(p.dtype == np.float32 and p.shape == (256, 256) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, "Invalid raw probability")
        c.require(row["score"] == float(p[:, :width].max()), "Native valid-window score changed")
    return result

def predict_probability(model, image):
    """Inputs contain normalized RGB only; no target/label/GT argument."""
    import torch
    with torch.inference_mode():
        out = torch.softmax(model(torch.from_numpy(image.copy()).unsqueeze(0).to("cuda")), dim=1)[0, 1]
    return out.detach().cpu().numpy().astype(np.float32, copy=False)

def score_probe():
    if c.SCORES.exists():
        result = verify_scores()
        return {"scores_complete": True, "reused": True, "additional_forwards": 0, "images": len(result["items"])}
    from .preparation import verify_pilot
    p, ip, _ = verify_pilot(decode=False, source_bytes=False)
    expected = score_binding()
    c.runtime_environment(); c.resources(4.5, 20, gpu=True)
    c.SCORES.mkdir(parents=True, exist_ok=False)
    c.save(c.SCORES / "transaction.json", {"binding": expected, "meaning": "once-only final scoring; partial/uncertain forward must not auto resume"})
    try:
        from .model import build_model, state_hash, buffers_hash
        from safetensors.torch import load_file
        model, _ = build_model(c.SEED, p)
        model.load_state_dict(load_file(str(c.MODEL / "final_weights.safetensors"), device="cpu"), strict=True)
        model.to("cuda").eval()
        before, buffers = state_hash(model.state_dict()), buffers_hash(model)
        train = c.load(c.RUN / "result.json")
        c.require(before == train["final_state_hash"] and buffers == train["initial_buffers_hash"], "Final scoring state/buffer mismatch")
        rows = []
        for i, item in enumerate(ip["items"]):
            with np.load(c.INPUT / item["cache"], allow_pickle=False) as z:
                image = z["image"]  # Deliberately never access target/valid members.
            prob = predict_probability(model, image)
            c.require(prob.shape == (256, 256) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, "Invalid probability after forward; preserve")
            name = f"probabilities/{i:02d}_{item['id']}.npy"
            (c.SCORES / "probabilities").mkdir(exist_ok=True)
            np.save(c.SCORES / name, prob, allow_pickle=False)
            rows.append({"id": item["id"], "probability": name, "score": float(prob[:, :item["width"]].max())})
        result = {"binding": expected, "items": rows, "input_forwards": 21, "constructor_forwards": 0,
                  "state_before": before, "state_after": state_hash(model.state_dict()), "buffers_before": buffers, "buffers_after": buffers_hash(model),
                  "raw_float32_softmax": True, "smoothing_or_pool_normalization": False, "GT_read_by_predictor": False}
        c.require(result["state_before"] == result["state_after"] and result["buffers_before"] == result["buffers_after"], "State mutated during scoring")
        c.save(c.SCORES / "result.json", result); c.seal(c.SCORES, expected)
    except Exception as exc:
        c.save(c.SCORES / "failure.json", {"error": repr(exc), "rule": "uncertain/partial forward preserved; no auto retry or completion"})
        raise
    verify_scores()
    return {"scores_complete": True, "reused": False, "input_forwards": 21, "AP_computed": False}
