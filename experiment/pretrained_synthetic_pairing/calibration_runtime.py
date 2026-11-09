"""Explicit once-only single-model Q runtime. Import is stdlib only."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import calibration as a


def preflight():
    a.c.require(Path(sys.executable).resolve() == (a.c.ENV / "Scripts/python.exe").resolve(), "Use .venv-pretrained-seg for explicit calibration")
    command = [str(a.c.CPU / "Scripts/python.exe"), "-B", "-X", "utf8", "-m", "pretrained_synthetic_pairing.calibration", "verify"]
    checked = subprocess.run(command, cwd=a.c.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=360,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    a.c.require(checked.returncode == 0, "Independent CPU admission failed: " + checked.stderr[-1800:])
    a.c.require(json.loads(checked.stdout)["preparation_verified"], "CPU admission did not verify")
    from .resources import environment_metadata
    a.c.require(environment_metadata() == a.t.load(a.PREP / "environment.json"), "Runtime interpreter/dependencies differ")
    return a.t.load(a.PREP / "protocol.json"), a.t.load(a.PREP / "requests.json"), a.t.load(a.PREP / "models.json")


def append_intention(root, row):
    with (root / "window_journal.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def predict_native(model, rgb, ys, before_window):
    """Label-free RGB predictor; no GT/labels/thresholds passed or read."""
    import numpy as np
    import torch
    height, width, channels = rgb.shape
    a.c.require(rgb.dtype == np.uint8 and channels == 3 and ys == a.geometry(height, width)["ys"], "Native RGB/coordinates differ")
    total = np.zeros((height, width), np.float32); coverage = np.zeros((height, width), np.uint16)
    mean = np.asarray([.485, .456, .406], np.float32); std = np.asarray([.229, .224, .225], np.float32)
    with torch.inference_mode():
        for index, y in enumerate(ys):
            tile = np.pad(rgb[y:y+256], ((0, 0), (0, 256-width), (0, 0)), mode="reflect")
            normalized = np.ascontiguousarray(((tile.astype(np.float32) / np.float32(255) - mean) / std).transpose(2, 0, 1))
            tensor = torch.from_numpy(normalized).unsqueeze(0).to("cuda")
            before_window(index, y)  # Durable intention before an uncertain CUDA call.
            logits = model(tensor)
            a.c.require(logits.dtype == torch.float32 and logits.shape == (1, 2, 256, 256) and bool(torch.isfinite(logits).all()), "Nonfinite/invalid native logits")
            prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
            a.c.require(prob.dtype == np.float32 and np.isfinite(prob).all() and 0 <= prob.min() <= prob.max() <= 1, "Invalid window probability")
            total[y:y+256] += prob[:, :width]; coverage[y:y+256] += 1
            del tensor, logits, tile, normalized, prob
    a.c.require(bool((coverage > 0).all()), "Uncovered native pixels")
    np.divide(total, coverage.astype(np.float32), out=total)
    a.c.require(np.isfinite(total).all() and 0 <= total.min() <= total.max() <= 1, "Invalid native stitched probability")
    return total, len(ys)


def run(group, seed):
    p, requests, models = preflight()
    entry = a.find_model(models, group, seed); root, runtime = a.paths(group, seed); bound = a.binding(p, entry)
    if root.exists():
        # No Torch, model construction or threshold computation on cache reuse.
        a.guard(cpu=True, cache=root)
        from .calibration_cache import verify_results
        return verify_results(group, seed, p, requests, models)
    a.c.require(a.new_run_allowed(root.exists(), runtime.exists()), "Unknown/partial output/runtime exists; no automatic restart")
    snapshot = a.t.resources(model=True)
    a.guard(cpu=False, destinations=(root, runtime), raw_q=True, cache=root, temporary=runtime / "tmp")
    a.predecessor_ready(group, seed, p, requests, models)
    root.mkdir(parents=True, exist_ok=False); runtime.mkdir(parents=True, exist_ok=False)
    a.t.save(root / "transaction.json", {"binding": bound, "resources": snapshot, "mode": "once-only Q200; partial/uncertain preserve and stop"})
    a.t.save(runtime / "owner.json", bound)
    rows = []; samples = []; current_index = None
    try:
        temp = runtime / "tmp"; temp.mkdir(exist_ok=False)
        os.environ.update({"TEMP": str(temp), "TMP": str(temp), "TORCH_HOME": str(temp / "torch"), "HF_HOME": str(temp / "hf"),
                           "XDG_CACHE_HOME": str(temp / "xdg"), "TORCHINDUCTOR_CACHE_DIR": str(temp / "inductor"),
                           "TRITON_CACHE_DIR": str(temp / "triton"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        import tempfile
        tempfile.tempdir = str(temp)
        started = time.monotonic()
        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import load_file
        from pretrained_segmentation.model import build_model, state_hash, buffers_hash
        model, initial = build_model(seed, {"compatibility": p["model_compatibility"]})
        a.c.require(initial == entry["initialization"], "Generic constructor/initial BN identity differs")
        model.load_state_dict(load_file(entry["final_weight_path"], device="cpu"), strict=True)
        model.to("cuda").eval()
        before = state_hash(model.state_dict()); buffers = buffers_hash(model)
        a.c.require(before == entry["expected_state_hash"] and buffers == entry["expected_buffers_hash"], "Final model state/BN differs")
        torch.cuda.reset_peak_memory_stats()
        (root / "probabilities").mkdir(exist_ok=False)
        for request in requests:
            current_index = request["index"]
            snap = a.c.resource_snapshot(); snap["image_index"] = current_index
            a.c.require(snap["available_RAM_GiB"] >= 2 and snap["E_free_GiB"] >= 1, "Per-image resource boundary stop")
            samples.append(snap)
            append_intention(root, {"event": "decode_start", "index": current_index, "id": request["id"]})
            raw = (a.SPLIT.parent / request["image"]).read_bytes()
            a.c.require(hashlib.sha256(raw).hexdigest() == request["image_sha256"], "Actual Q image source hash differs")
            with Image.open(io.BytesIO(raw)) as im:
                a.c.require(im.mode in ("RGB", "L") and im.size == (request["width"], request["height"]), "Actual Q image mode/size differs")
                rgb = np.asarray(im.convert("RGB")).copy()
            def before_window(index, y):
                append_intention(root, {"event": "window_start", "index": current_index, "id": request["id"], "window_index": index, "y": y})
            prob, count = predict_native(model, rgb, request["ys"], before_window)
            a.c.require(count == request["window_count"] and list(prob.shape) == request["native_shape"], "Actual geometry differs")
            name = f"probabilities/{current_index:03d}_{request['id']}.npy"
            with (root / name).open("xb") as stream:
                np.save(stream, prob, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            rows.append({"index": current_index, "id": request["id"], "source_image_sha256": request["image_sha256"],
                         "probability": name, "score": float(prob.max()), "window_forwards": count})
            print(json.dumps({"model": f"{group}_{seed}", "image": current_index+1, "of": 200, "RAM_GiB": snap["available_RAM_GiB"]}), flush=True)
            del raw, rgb, prob
        torch.cuda.synchronize()
        after = state_hash(model.state_dict()); after_buffers = buffers_hash(model)
        a.c.require(after == before and after_buffers == buffers, "Scoring mutated parameters/BN")
        result = {"binding": bound, "items": rows, "images_scored": 200, "window_forwards": sum(r["window_forwards"] for r in rows), "constructor_forwards": 0,
                  "state_before": before, "state_after": after, "buffers_before": buffers, "buffers_after": after_buffers,
                  "GT_read_by_predictor": False, "raw_Q_image_decodes": 200, "native_pixels": 29202876, "score_rule": a.SCORE_RULE,
                  "elapsed_seconds": time.monotonic()-started, "resource_samples": samples, "sampled_min_RAM_GiB": min(s["available_RAM_GiB"] for s in samples),
                  "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated()/2**20, "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved()/2**20,
                  "training_or_parameter_updates": 0}
        a.t.save(root / "scores.json", result)
        scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
        tau = float(np.quantile(scores, a.THRESHOLD_RULE["q"], method=a.THRESHOLD_RULE["method"]))
        fp = int((scores > tau).sum())
        threshold = {"binding": bound, "score_source_sha256": a.t.sha(root / "scores.json"), "threshold": tau, "rule": a.THRESHOLD_RULE,
                     "FP": fp, "TN": 200-fp, "normal_denominator": 200, "independent_validation": False, "quantile_evaluations": 1}
        a.t.save(root / "threshold.json", threshold)
        a.t.save(root / "正常校准结果.md", f"# {group}_{seed}正常校准\n\n最终10000步；Q200／844窗，原生max，q.95／linear／严格大于，tau={tau:.17g}；FP{fp}/200、TN{200-fp}/200。只是同池校准，非独立验证；无异常召回／AP或屏幕结论。完整缓存仅核验，不重求分位数，未知事务保留停止。\n")
        a.t.save(root / "runtime_manifest.json", {"binding": bound, "manifest": a.runtime_manifest(runtime)})
        a.t.seal(root, a.run_files(requests), bound)
    except BaseException as exc:
        if not (root / "failure.json").exists():
            a.t.save(root / "failure.json", {"error": repr(exc), "current_request_index": current_index, "completed_images": len(rows), "rule": "uncertain/partial forward preserved; no auto retry, threshold completion or intersection"})
        raise
    from .calibration_cache import verify_results
    return verify_results(group, seed, p, requests, models)
