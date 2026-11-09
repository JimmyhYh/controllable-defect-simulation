"""Explicit CPU-only final training-window AP; never invokes a model."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from . import common as c

def metric_binding():
    from .scoring import verify_scores
    verify_scores()
    return {"score_complete_sha256": c.sha(c.SCORES / "complete.json"), "pilot_complete_sha256": c.sha(c.PREP / "complete.json"),
            "exact_AP_source_sha256": c.sha(c.ROOT / "fewshot_synthetic/metrics.py"), "implementation": c.code_identity()}

def verify_metrics():
    from .preparation import strict_directories
    identity = metric_binding()
    c.require(strict_directories(c.METRICS)["identity"] == identity, "Metric cache identity differs")
    result = c.load(c.METRICS / "metrics.json")
    p = c.load(c.PREP / "protocol.json")
    c.require(result["binding"] == identity and result["valid_pixels"] == p["valid_pixels"] and result["positive_pixels"] == p["positive_pixels"], "Metric actual denominator differs")
    c.require(result["step"] == 2000 and result["unique_images_and_windows"] == 21 and result["classification_threshold_gate"] is False, "Metric budget/meaning differs")
    return result

def compute_probe():
    if c.METRICS.exists():
        return verify_metrics()
    from .preparation import verify_pilot
    from .data import read_cache
    from .scoring import verify_scores
    p, ip, _ = verify_pilot(decode=True, source_bytes=False)
    scores = verify_scores(); identity = metric_binding()
    c.resources(3, 80)
    c.METRICS.mkdir(parents=True, exist_ok=False)
    c.save(c.METRICS / "transaction.json", {"binding": identity, "meaning": "exact final fixed-window fitting metrics only"})
    try:
        pairs = []
        for item, row in zip(ip["items"], scores["items"]):
            pack = read_cache(item)
            prob = np.load(c.SCORES / row["probability"], allow_pickle=False)
            valid = pack["valid"]
            pairs.append((prob[valid], pack["target"][valid]))
        def blocks():
            for score, label in pairs:
                yield score, label
        # Frozen implementation, explicit independent destination; no old globals changed.
        from fewshot_synthetic.metrics import exact_ap
        exact = exact_ap(blocks, c.METRICS / "exact", identity, p["valid_pixels"], p["positive_pixels"])
        from sklearn.metrics import average_precision_score
        all_scores = np.concatenate([x[0] for x in pairs]); all_y = np.concatenate([x[1] for x in pairs])
        reference = float(average_precision_score(all_y, all_scores))
        c.require(abs(reference - exact["AP"]) <= 1e-10, "Global exact AP/sklearn mismatch")
        normalized = all_scores.copy(); normalized[normalized == 0] = np.float32(0)
        _, counts = np.unique(normalized.view(np.uint32), return_counts=True)
        c.require(exact["distinct_scores"] == len(counts) and exact["tied_score_values"] == int((counts > 1).sum()), "Actual score tie counts mismatch")
        rows = []
        (c.METRICS / "previews").mkdir()
        for i, (item, pair) in enumerate(zip(ip["items"], pairs)):
            score, label = pair
            pack = read_cache(item)
            prob = np.load(c.SCORES / scores["items"][i]["probability"], allow_pickle=False)
            ap = float(average_precision_score(label, score)) if label.any() else None
            rows.append({"id": item["id"], "kind": item["kind"], "valid_pixels": len(label), "positive_pixels": int(label.sum()),
                         "AP": ap, "mean_probability": float(score.mean()), "max_probability": float(score.max()),
                         "fraction_pixels_strict_gt_0_5": float((score > .5).mean()), "nonconstant": bool(score.min() < score.max())})
            width = item["width"]
            heat = np.zeros((256, width, 3), np.uint8)
            heat[:, :, 0] = np.rint(prob[:, :width] * 255).astype(np.uint8)
            heat[:, :, 2] = np.rint((1 - prob[:, :width]) * 255).astype(np.uint8)
            canvas = Image.new("RGB", (3 * width + 20, 286), "#333333")
            canvas.paste(Image.fromarray(pack["rgb_uint8"][:, :width]), (0, 30))
            canvas.paste(Image.fromarray(pack["target"][:, :width] * 255).convert("RGB"), (width + 10, 30))
            canvas.paste(Image.fromarray(heat), (2 * width + 20, 30))
            ImageDraw.Draw(canvas).text((0, 5), f"{item['id']} RGB | GT | p(anomaly), fixed [0,1] blue->red", fill="white")
            canvas.save(c.METRICS / f"previews/{i:02d}_{item['id']}.png")
        real_rows = rows[:5]
        checks = {"global_AP_at_least_0_95": exact["AP"] >= .95, "each_R_window_AP_at_least_0_85": all(x["AP"] >= .85 for x in real_rows),
                  "each_R_probability_nonconstant": all(x["nonconstant"] for x in real_rows),
                  "finite_training": c.load(c.RUN / "result.json")["all_finite"] is True,
                  "scoring_state_buffers_unchanged": scores["state_before"] == scores["state_after"] and scores["buffers_before"] == scores["buffers_after"]}
        result = {"binding": identity, "step": 2000, "global_AP": exact["AP"], "sklearn_global_AP": reference,
                  "absolute_difference": abs(exact["AP"] - reference), "valid_pixels": p["valid_pixels"], "positive_pixels": p["positive_pixels"],
                  "unique_images_and_windows": 21, "R_windows": 5, "normal_windows": 16, "all_items": rows,
                  "normal_AP": None, "normal_AP_reason": "no positive GT; NA", "classification_threshold_gate": False,
                  "gate_checks": checks, "gate_passed": all(checks.values()), "meaning": "same21 training windows fitting only; no Q/E/U, independent validation or screen effect"}
        c.save(c.METRICS / "metrics.json", result)
        c.save(c.METRICS / "小试最终CPU验收.md", f"# 最终2000步同池技术验收\n\n全池精确像素AP={result['global_AP']:.8%}，gate={result['gate_passed']}。实际21训练窗、有效{p['valid_pixels']}／阳性{p['positive_pixels']}像素。所有5异常与16正常逐项结果及固定[0,1]概率预览已保留；正常像素0.5占比只描述，不构成分类门槛。结论限训练拟合，不是未见效果；未自动进入正式训练。\n")
        c.seal(c.METRICS, identity)
    except Exception as exc:
        c.save(c.METRICS / "failure.json", {"error": repr(exc), "rule": "preserve partial exact AP; do not retry or approximate"})
        raise
    return verify_metrics()
