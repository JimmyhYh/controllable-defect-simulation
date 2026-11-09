"""Manual SSN pilot runtime. No eager ML imports; CPU --help/status are safe.

This module is only parsed during stage3. train/score are NEVER called by a
preparation command. Complete caches verify read-only; partial runs stop.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from . import resources as r
from . import pilot_dtype_preparation as p
from . import pilot_config as c


def paths(name):
    r.require(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name), "Unsafe run name")
    return p.RUNS / name, p.MODELS / name


def admission():
    # Preserve original/new environment identities without cross-importing sites.
    old_python = r.load(r.OUT / "protocol.json")["environment"]["identity"]["executable"]
    result = subprocess.run([old_python, "-B", "-X", "utf8", "-m", "ssn_baseline.pilot_dtype_preparation", "verify"], cwd=r.ROOT, capture_output=True, text=True, encoding="utf-8", check=True, timeout=180)
    r.require(json.loads(result.stdout)["pilot_preparation_ready"], "CPU admission failed")
    return r.load(p.OUT / "protocol.json")


def checkpoint_binding():
    return {"protocol_sha256": r.digest(p.OUT / "protocol.json"), "preparation_complete": r.digest(p.OUT / "complete.json")}


def verify_checkpoint(path):
    path = Path(path).resolve()
    r.require(path.is_relative_to(p.MODELS.resolve()) and path.suffix == ".pt", "Only own pilot checkpoints may resume")
    info = r.load(path.with_suffix(".json"))
    r.require(info["binding"] == checkpoint_binding() and r.digest(path) == info["sha256"], "Checkpoint provenance/hash changed")
    r.require(info["epoch"] in c.CHECKPOINT_EPOCHS and info["step"] == info["epoch"] * c.UPDATES_PER_EPOCH, "Checkpoint cursor invalid")
    return info


def verify_train(name):
    run, models = paths(name)
    r.verify_seal(run / "train")
    summary = r.load(run / "train/summary.json")
    r.require(summary["binding"] == checkpoint_binding() and summary["completed_steps"] == c.STEPS and summary["completed_epochs"] == c.EPOCHS, "Not the final fixed-budget result")
    for filename, sha in summary["model_files"].items():
        r.require(r.digest(models / filename) == sha, "Saved model/checkpoint file changed")
    r.require(r.files_under(models) == set(summary["model_files"]), "Unknown/missing pilot model file")
    for epoch in summary["checkpoint_epochs_saved_in_this_run"]:
        verify_checkpoint(models / f"checkpoint_e{epoch:04}.pt")
    return summary


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def resource_admission():
    r.require(Path(sys.executable).resolve() == (r.ROOT / ".venv-ssn/Scripts/python.exe").resolve(), "Model runtime must use isolated .venv-ssn")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= c.RESOURCES["start_RAM_GiB"] and snap["free_disk_GiB"] >= c.RESOURCES["start_disk_GiB"], "Release resources before manual model runtime; no run started")
    r.require(len(snap["GPUs"]) == 1 and snap["GPUs"][0]["free_MiB"] / 1024 >= c.RESOURCES["start_free_VRAM_GiB"], "GPU resource gate")
    return snap


def model_environment():
    # Called only after a destination and durable transaction have been reserved.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    def no_network(event, args):
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("No network or arbitrary pickle execution in pilot")
    sys.addaudithook(no_network)
    import torch
    r.require(torch.cuda.is_available(), "CUDA required; no silent CPU or precision fallback")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    return torch


def cached_inputs(training):
    import numpy as np
    manifest = r.load(p.data.OUT / "manifest.json")
    result = {}
    for row in manifest["samples"]:
        with np.load(p.data.OUT / row["cache"], allow_pickle=False) as z:
            item = {"row": row, "image": z["image"].copy()}
            if training:
                item.update({key: z[key].copy() for key in ["mask_input", "mask_feature"]})
            result[row["id"]] = item
    return result


def runtime_GT_check(torch, cached):
    import numpy as np
    from .model_dtype_compat import feature_targets
    for item in cached.values():
        source = torch.from_numpy(item["mask_input"].copy()).unsqueeze(0).to("cuda")
        with torch.no_grad():
            actual = feature_targets(source).cpu().numpy()[0, 0]
        r.require(np.array_equal(actual, item["mask_feature"]), "Actual Torch interpolation differs from CPU cache; stop before model init")


def memory_RAM():
    # Query RAM without launching nvidia-smi at each update.
    import ctypes
    class M(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ["total", "free", "tp", "ap", "tv", "av", "ex"]]
    m = M(); m.length = ctypes.sizeof(m)
    r.require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)), "RAM sampling failed")
    return m.free / 2**30


def save_tensor_file(torch, payload, path):
    r.require(not path.exists() and not path.with_suffix(path.suffix + ".partial").exists(), "Unknown model save transaction")
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temp)
    os.replace(temp, path)


def train(name, resume_from=None):
    protocol = admission()
    run, models = paths(name)
    if run.exists():
        if (run / "train/complete.json").exists():
            return {"reused": True, "training_complete": True, "summary": verify_train(name)}
        raise RuntimeError("Partial/unknown pilot run retained; use a separately named manual own-checkpoint continuation")
    r.require(not models.exists(), "Unknown model directory")
    if resume_from:
        r.require(name != "initial", "Resume requires NEW run name, preserving original run")
        resume_info = verify_checkpoint(resume_from)
        r.require(resume_info["epoch"] < c.EPOCHS, "Final checkpoint is not a new training start")
    else:
        r.require(name == "initial", "Fresh initialization only uses fixed initial run name")
    start_resource = resource_admission()
    (run / "train").mkdir(parents=True, exist_ok=False)
    r.write(run / "train/transaction.json", {"binding": checkpoint_binding(), "run": name, "manual_only": True,
            "resume_from": str(Path(resume_from).resolve()) if resume_from else None, "start_resource": start_resource,
            "runtime_import_or_model_operations_started_at_reservation": False})
    epoch_start, history = 0, []
    saved_epochs = []
    started = time.perf_counter(); min_RAM = start_resource["free_RAM_GiB"]
    try:
        models.mkdir(parents=True, exist_ok=False)
        torch = model_environment()
        import numpy as np
        cached = cached_inputs(training=True)
        runtime_GT_check(torch, cached)
        random.seed(c.INIT_SEED); np.random.seed(c.INIT_SEED)
        torch.manual_seed(c.INIT_SEED); torch.cuda.manual_seed_all(c.INIT_SEED)
        from .model_dtype_compat import build_model
        from .objective import loss_terms
        model = build_model(checkpoint_path=protocol["checkpoint_path"], config=protocol["model_config"]).to("cuda")
        optimizer, scheduler = model.get_optimizers()
        exposure = {"R5": 0, "N16": 0, "internal_feature_views": 0, "returned_positive_view_labels": 0}
        initial_hash = state_hash(model)
        if resume_from:
            saved = torch.load(Path(resume_from), map_location="cuda", weights_only=True)
            r.require(saved["binding"] == checkpoint_binding() and saved["completed_step"] == resume_info["step"], "Resume payload binding mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler_state = dict(saved["scheduler"])
            scheduler_state["milestones"] = Counter(scheduler_state["milestones"])
            scheduler.load_state_dict(scheduler_state)
            epoch_start, history, exposure = saved["completed_epoch"], saved["history"], saved["exposure"]
            random.setstate(saved["python_rng"])
            nr = saved["numpy_rng"]
            np.random.set_state((nr["algorithm"], np.array(nr["keys"], dtype=np.uint32), nr["position"], nr["has_gauss"], nr["cached_gaussian"]))
            torch.set_rng_state(saved["torch_cpu_rng"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["torch_cuda_rng"]])
            r.require(state_hash(model) == resume_info["state_sha256"], "Restored model/BN state differs")
            initial_hash = saved["initial_state_sha256"]
        rows = r.load(p.OUT / "sequence.json")
        min_RAM = min(min_RAM, memory_RAM())
        model.train()
        for epoch in range(epoch_start + 1, c.EPOCHS + 1):
            for row in rows[(epoch - 1) * 32:epoch * 32]:
                free = memory_RAM(); min_RAM = min(min_RAM, free)
                r.require(free >= c.RESOURCES["running_RAM_GiB"], "Runtime RAM stop; preserve last complete checkpoint")
                lrs = [x["lr"] for x in optimizer.param_groups]
                r.require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(lrs, p.expected_lrs(row["step"]))), "Actual scheduler cursor/LR mismatch")
                item = cached[row["id"]]
                image = torch.from_numpy(item["image"]).unsqueeze(0).to("cuda")
                mask = torch.from_numpy(item["mask_feature"]).float().reshape(1, 1, 160, 58).to("cuda")
                label = torch.tensor([int(row["role"] == "R5")], dtype=torch.float32, device="cuda")
                optimizer.zero_grad()
                maps, scores, mask_aug, labels_aug = model(image, mask, label)
                losses = loss_terms(maps, scores, mask_aug, labels_aug, torch.ones(1, dtype=torch.bool, device="cuda"))
                r.require(all(bool(torch.isfinite(v).all()) for v in losses.values() if isinstance(v, torch.Tensor)), "Nonfinite loss")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                r.require(bool(torch.isfinite(norm)), "Nonfinite gradient norm")
                optimizer.step()
                exposure[row["role"]] += 1; exposure["internal_feature_views"] += 2
                exposure["returned_positive_view_labels"] += int(labels_aug.detach().sum().item())
                values = {k: float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v) for k, v in losses.items()}
                history.append({"step": row["step"], "epoch": epoch, "id": row["id"], "role": row["role"], "lrs": lrs, "losses": values})
                if row["step"] % 100 == 0:
                    print(json.dumps({"step": row["step"], "of": c.STEPS, "epoch": epoch, "loss": values["total"], "elapsed_seconds_this_run": time.perf_counter() - started, "free_RAM_GiB": free}), flush=True)
            scheduler.step()
            if epoch in c.CHECKPOINT_EPOCHS:
                nr = np.random.get_state(); ss = scheduler.state_dict(); ss["milestones"] = dict(ss["milestones"])
                payload = {"binding": checkpoint_binding(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": ss, "completed_epoch": epoch, "completed_step": epoch * 32, "history": history.copy(),
                    "exposure": exposure.copy(), "initial_state_sha256": initial_hash, "python_rng": random.getstate(),
                    "numpy_rng": {"algorithm": nr[0], "keys": nr[1].tolist(), "position": nr[2], "has_gauss": nr[3], "cached_gaussian": nr[4]},
                    "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all()}
                target = models / f"checkpoint_e{epoch:04}.pt"
                save_tensor_file(torch, payload, target)
                r.write(target.with_suffix(".json"), {"binding": checkpoint_binding(), "epoch": epoch, "step": epoch * 32,
                    "sha256": r.digest(target), "state_sha256": state_hash(model), "lrs_after_epoch": [x["lr"] for x in optimizer.param_groups]})
                saved_epochs.append(epoch)
        r.require(len(history) == c.STEPS and exposure["R5"] == exposure["N16"] == 800, "Actual update/exposure mismatch")
        save_tensor_file(torch, model.state_dict(), models / "final_weights.pt")
        r.write(run / "train/history.json", history)
        summary = {"binding": checkpoint_binding(), "completed_steps": c.STEPS, "completed_epochs": c.EPOCHS,
            "exposure": exposure, "checkpoint_epochs_saved_in_this_run": saved_epochs, "initial_state_sha256": initial_hash,
            "final_state_sha256": state_hash(model), "elapsed_seconds_this_run": time.perf_counter() - started,
            "minimum_sampled_free_RAM_GiB_this_run": min_RAM, "CUDA_peak_allocated_MiB_this_run": torch.cuda.max_memory_allocated() / 2**20,
            "CUDA_peak_reserved_MiB_this_run": torch.cuda.max_memory_reserved() / 2**20,
            "model_files": {n: r.digest(models / n) for n in sorted(r.files_under(models))}, "training_pool_scored": False,
            "gate": "await separate final-weight training-pool score and CPU metrics; no Q/E/U", "full_recovery_empirically_verified": False}
        r.write(run / "train/summary.json", summary); r.seal(run / "train", "ssn-manual-pilot-training-v1")
    except BaseException as exc:
        r.write(run / "train/failure.json", {"error": repr(exc), "last_completed_update": history[-1]["step"] if history else epoch_start * 32,
            "own_complete_checkpoint_epochs_this_run": saved_epochs, "rule": "retain; no auto retry, recovery only to NEW named run from own complete checkpoint"})
        raise
    return {"training_complete": True, "summary": verify_train(name)}


def score(name):
    protocol = admission(); summary = verify_train(name)
    run, models = paths(name); out = run / "scores"
    if out.exists():
        r.verify_seal(out)
        identity = r.load(out / "identity.json")
        r.require(identity["binding"] == checkpoint_binding() and identity["final_weight_sha256"] == r.digest(models / "final_weights.pt"), "Score identity changed")
        return {"scores_complete": True, "reused": True, "new_forwards": 0}
    resource = resource_admission()
    out.mkdir(parents=True, exist_ok=False)
    r.write(out / "transaction.json", {"binding": checkpoint_binding(), "every_selected_input_once": True, "resource": resource,
            "runtime_import_or_model_operations_started_at_reservation": False})
    records = []; hook = None; min_RAM = resource["free_RAM_GiB"]
    try:
        torch = model_environment()
        import numpy as np
        from .model_dtype_compat import build_model, native_scores
        model = build_model(checkpoint_path=protocol["checkpoint_path"], config=protocol["model_config"]).to("cuda")
        model.load_state_dict(torch.load(models / "final_weights.pt", map_location="cuda", weights_only=True), strict=True)
        model.eval(); before = state_hash(model)
        r.require(before == summary["final_state_sha256"], "Final parameter/BN identity differs")
        cached = cached_inputs(training=False)
        capture = {}
        hook = model.discriminator.seg.register_forward_hook(lambda module, args, output: capture.update(logits=output.detach()))
        with torch.inference_mode():
            for item in cached.values():
                free = memory_RAM(); min_RAM = min(min_RAM, free)
                r.require(free >= c.RESOURCES["running_RAM_GiB"], "Scoring RAM stop; preserve uncertain/partial transaction without automatic repeats")
                row = item["row"]
                logits, cls = model(torch.from_numpy(item["image"]).unsqueeze(0).to("cuda"))
                outputs = native_scores(logits, cls, row["coordinates"]["native_hw"])
                feature = torch.sigmoid(capture["logits"].float()).cpu().numpy()[0, 0]
                native = outputs["native_probability"].cpu().numpy()[0, 0]
                r.require(np.isfinite(feature).all() and np.isfinite(native).all() and bool(torch.isfinite(cls).all()), "Nonfinite final score")
                np.savez_compressed(out / f"{row['id']}.npz", feature_probability=feature, native_probability=native)
                records.append({"id": row["id"], "role": row["role"], "classification_logit": float(cls.item()),
                    "classification_score": float(outputs["classification_score"].item()), "native_localization_max": float(outputs["localization_max"].item())})
        hook.remove(); hook = None
        r.require(state_hash(model) == before, "Scoring mutated parameters/BN buffers")
        r.write(out / "scores.json", records)
        r.write(out / "identity.json", {"binding": checkpoint_binding(), "final_weight_sha256": r.digest(models / "final_weights.pt"),
            "state_before_and_after": before, "inputs": len(records), "new_model_forwards": len(records), "score_function_GT_payload_reads": 0,
            "minimum_sampled_free_RAM_GiB": min_RAM,
            "constructor_shape_probe_separately_disclosed": True, "note": "final-weight training-pool diagnostic only; no threshold calibration"})
        r.seal(out, "ssn-pilot-training-pool-scores-v1")
    except BaseException as exc:
        if hook is not None:
            hook.remove()
        r.write(out / "failure.json", {"error": repr(exc), "minimum_sampled_free_RAM_GiB": min_RAM, "completed_ids": [x["id"] for x in records], "rule": "retain uncertain forward state; do not automatically complete or repeat"})
        raise
    return {"scores_complete": True, "reused": False, "input_forwards": 21}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "score", "status", "verify-train"])
    parser.add_argument("--run-name", default="initial")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    if args.command == "status":
        run, _ = paths(args.run_name)
        result = {"run": args.run_name, "exists": run.exists(), "training_complete_marker": (run / "train/complete.json").is_file(), "model_operations": 0}
    elif args.command == "verify-train":
        admission(); result = verify_train(args.run_name)
    elif args.command == "train":
        result = train(args.run_name, args.resume_from)
    else:
        r.require(args.resume_from is None, "Scoring does not resume training")
        result = score(args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
