"""Explicit USER-STARTED formal training; metadata verification is CPU-only."""
from __future__ import annotations
import math
import os
from pathlib import Path
import time
from collections import Counter
from . import common as c
from . import formal_common as f

class ResourceStop(Exception):
    pass

def binding(protocol):
    return {"formal_complete_sha256": c.sha(f.PREP / "complete.json"), "protocol_sha256": c.sha(f.PREP / "protocol.json"),
            "seed": f.SEED, "steps": f.STEPS, "implementation": protocol["implementation"], "pilot_implementation": protocol["pilot_implementation"]}

def verify_checkpoint(folder, bound, seq):
    folder = Path(folder)
    seal = f.strict_seal(folder)
    c.require(c.paths(folder) == {"model.safetensors", "optimizer.pt", "rng.json", "history.json", "checkpoint.json", "complete.json"} and seal["identity"] == bound, "Unknown/foreign formal checkpoint member")
    row, history = c.load(folder / "checkpoint.json"), c.load(folder / "history.json")
    step = row["step"]
    c.require(row["binding"] == bound and type(step) is int and 1 <= step <= f.STEPS and step == len(history), "Formal checkpoint cursor differs")
    c.require(row["buffers_hash"] == row["initial_buffers_hash"] and row["next_lr"] == (c.lr_at(step + 1, f.STEPS) if step < f.STEPS else None), "Formal saved BN/recovery LR differs")
    for entry, req in zip(history, seq):
        c.require(entry["request"] == req and entry["lr"] == c.lr_at(req["update"], f.STEPS) and entry["finite"] is True and all(math.isfinite(entry[k]) for k in ["loss", "CE", "Dice"]), "Formal saved history differs")
    c.require([x["request"]["update"] for x in history] == list(range(1, step + 1)), "Formal actual update prefix differs")
    c.require([g["lr"] for g in row["optimizer_groups"]] == c.lr_at(step, f.STEPS) and [g["name"] for g in row["optimizer_groups"]] == ["encoder", "decoder_head"], "Saved optimizer group LR/name differs")
    return row, history

def validate_model_tree(bound, seq, final=False, complete=False):
    c.paths(f.MODEL)
    expected = {"initialization.json"} | ({"final_weights.safetensors"} if final else set()) | ({"complete.json"} if complete else set())
    files = {p.name for p in f.MODEL.iterdir() if p.is_file()}
    dirs = {p.name for p in f.MODEL.iterdir() if p.is_dir()}
    c.require(files == expected and dirs <= {"checkpoints", "resource_stops"}, "Unknown formal model root file/directory/lock")
    initial = c.load(f.MODEL / "initialization.json")
    rows = []
    for name in dirs:
        members = list((f.MODEL / name).iterdir())
        c.require(members and all(p.is_dir() for p in members), "Unknown/empty formal checkpoint container")
        for folder in members:
            row, history = verify_checkpoint(folder, bound, seq)
            multiple = 1000 if name == "checkpoints" else 100
            c.require(folder.name == f"step_{row['step']:05d}" and row["step"] % multiple == 0 and (name != "resource_stops" or row["step"] % 1000 != 0), "Checkpoint name/type/cursor differs")
            c.require(row["initial_buffers_hash"] == initial["initial_buffers_hash"], "Formal checkpoint initial BN differs")
            rows.append((row["step"], folder.resolve()))
    return rows

def verify_training():
    from .formal_preparation import verify
    p, items, seq = verify(decode=False)
    bound = binding(p)
    c.require(f.strict_seal(f.RUN)["identity"] == bound and f.strict_seal(f.MODEL)["identity"] == bound, "Formal final identity differs")
    c.require(c.paths(f.RUN) == {"binding.json", "transaction.json", "result.json", "complete.json"}, "Unknown formal final result member")
    validate_model_tree(bound, seq, final=True, complete=True)
    result = c.load(f.RUN / "result.json"); initial = c.load(f.MODEL / "initialization.json")
    c.require(result["binding"] == bound and result["step"] == f.STEPS and result["all_finite"] is True, "Formal final budget/finite record differs")
    for step in range(1000, 10001, 1000):
        row, history = verify_checkpoint(f.MODEL / "checkpoints" / f"step_{step:05d}", bound, seq)
        c.require(row["step"] == step, "Formal scheduled checkpoint missing")
    row, history = verify_checkpoint(f.MODEL / "checkpoints/step_10000", bound, seq)
    c.require(row["state_hash"] == result["final_state_hash"] and row["buffers_hash"] == initial["initial_buffers_hash"] == result["initial_buffers_hash"] == result["final_buffers_hash"], "Formal final state/BN signatures differ")
    c.require(c.sha(f.MODEL / "final_weights.safetensors") == result["final_weight_sha256"] == c.sha(f.MODEL / "checkpoints/step_10000/model.safetensors"), "Formal final serialized identity differs")
    actual = dict(Counter(x["request"]["id"] for x in history))
    c.require(actual == p["exposure_by_id"] == result["exposure_by_id"] and result["real_exposures"] == result["normal_exposures"] == 5000, "Formal actual exposure differs")
    return {"formal_training_complete": True, "step": f.STEPS, "seed": f.SEED, "scheduled_checkpoints": list(range(1000, 10001, 1000)), "final_weight_sha256": result["final_weight_sha256"], "scoring_or_calibration": False, "GPU_complete_resume_verified": False}

def save_checkpoint(folder, model, optimizer, history, bound, initial, resources, elapsed, kind, prior_peaks):
    import torch
    from safetensors.torch import save_file
    from .model import state_hash, buffers_hash
    from .training import rng_state
    c.require(not folder.exists(), "Existing/incomplete formal checkpoint")
    folder.mkdir(parents=True, exist_ok=False)
    state = {n: v.detach().cpu().contiguous().clone() for n, v in model.state_dict().items()}
    save_file(state, str(folder / "model.safetensors"))
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    c.save(folder / "rng.json", rng_state()); c.save(folder / "history.json", history)
    step = len(history)
    row = {"binding": bound, "step": step, "kind": kind, "state_hash": state_hash(state), "buffers_hash": buffers_hash(model),
           "initial_buffers_hash": initial["initial_buffers_hash"], "next_lr": c.lr_at(step + 1, f.STEPS) if step < f.STEPS else None,
           "optimizer_groups": [{"name": g["name"], "lr": g["lr"], "parameter_tensors": len(g["params"])} for g in optimizer.param_groups],
           "peak_cuda_allocated_MiB": max(prior_peaks[0], torch.cuda.max_memory_allocated() / 2**20),
           "peak_cuda_reserved_MiB": max(prior_peaks[1], torch.cuda.max_memory_reserved() / 2**20),
           "resources": resources, "elapsed_seconds": elapsed, "GPU_restore_measured": False}
    c.require(row["buffers_hash"] == initial["initial_buffers_hash"], "Encoder BN buffers changed")
    c.save(folder / "checkpoint.json", row); c.seal(folder, bound)

def train(resume_from=None):
    from .formal_preparation import verify, request_input
    p, items, seq = verify(decode=False)
    bound = binding(p)
    if (f.RUN / "complete.json").is_file() or (f.MODEL / "complete.json").is_file():
        c.require(resume_from is None, "Completed formal run is read-only")
        return verify_training()
    f.runtime_environment()
    start_resources = c.resources(4.5, 20, gpu=True)
    restore = None
    if resume_from is None:
        c.require(not f.RUN.exists() and not f.MODEL.exists(), "Unknown/partial formal run; no automatic restart")
        f.RUN.mkdir(parents=True, exist_ok=False); f.MODEL.mkdir(parents=True, exist_ok=False)
        c.save(f.RUN / "binding.json", bound)
        c.save(f.RUN / "transaction.json", {"manual_entry": True, "budget": f.STEPS, "resources_before_model": start_resources})
    else:
        c.require(c.load(f.RUN / "binding.json") == bound and c.paths(f.RUN) == {"binding.json", "transaction.json"} and not any(p.is_dir() for p in f.RUN.rglob("*")), "Unknown formal run/failure/finalization; no recovery")
        restore = Path(resume_from).resolve()
        c.require(restore.is_relative_to(f.MODEL.resolve()) and restore.parent.name in {"checkpoints", "resource_stops"}, "Restore only this formal run's own checkpoint")
        rows = validate_model_tree(bound, seq)
        c.require(rows and restore in [path for step, path in rows if step == max(x for x, _ in rows)], "Explicit restore must be latest complete own point")
        row, _ = verify_checkpoint(restore, bound, seq)
        c.require(row["step"] < f.STEPS, "Final checkpoint exists but finalization incomplete; preserve and diagnose")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from safetensors.torch import load_file, save_file
    from .model import build_model, make_optimizer, loss_value, set_training_mode, state_hash, buffers_hash
    from .training import restore_rng
    started = time.monotonic()
    try:
        model, initial = build_model(f.SEED, p)
        model.to("cuda"); optimizer = make_optimizer(model)
        history, resource_log, done, prior, prior_peaks = [], [], 0, 0.0, (0.0, 0.0)
        if restore is None:
            c.save(f.MODEL / "initialization.json", initial)
        else:
            row, history = verify_checkpoint(restore, bound, seq)
            c.require(c.load(f.MODEL / "initialization.json") == initial, "Fresh formal initialization differs")
            model.load_state_dict(load_file(str(restore / "model.safetensors"), device="cuda"), strict=True)
            optimizer.load_state_dict(torch.load(restore / "optimizer.pt", map_location="cpu", weights_only=True))
            c.require(state_hash(model.state_dict()) == row["state_hash"] and buffers_hash(model) == initial["initial_buffers_hash"], "Restored formal tensors differ")
            c.require([g["lr"] for g in optimizer.param_groups] == c.lr_at(row["step"], f.STEPS), "Restored optimizer rates differ")
            restore_rng(c.load(restore / "rng.json"))
            done, prior, resource_log = row["step"], row["elapsed_seconds"], row["resources"]
            prior_peaks = (row["peak_cuda_allocated_MiB"], row["peak_cuda_reserved_MiB"])
        torch.cuda.reset_peak_memory_stats()
        for index in range(done, f.STEPS):
            req = seq[index]; pack = request_input(items[req["item_index"]], req)
            x = torch.from_numpy(pack["image"]).unsqueeze(0).to("cuda")
            y = torch.from_numpy(pack["target"]).unsqueeze(0).to("cuda")
            valid = torch.from_numpy(pack["valid"]).unsqueeze(0).to("cuda")
            rates = c.lr_at(index + 1, f.STEPS)
            for group, rate in zip(optimizer.param_groups, rates): group["lr"] = rate
            set_training_mode(model); optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss, ce, dice = loss_value(logits, y, valid)
            c.require(bool(torch.isfinite(loss)), "Formal nonfinite loss")
            loss.backward()
            c.require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.requires_grad), "Formal missing/nonfinite gradient")
            optimizer.step()
            c.require(all(bool(torch.isfinite(v).all()) for v in model.parameters()), "Formal nonfinite updated parameter")
            history.append({"request": req, "lr": rates, "loss": float(loss.detach()), "CE": float(ce.detach()), "Dice": float(dice.detach()), "finite": True})
            del x, y, valid, logits, loss, ce, dice, pack
            low = False
            if (index + 1) % 100 == 0:
                snap = c.resources(0, 0); snap["step"] = index + 1; resource_log.append(snap)
                low = snap["free_RAM_GiB"] < 2
                print({"step": index + 1, "of": f.STEPS, "loss": history[-1]["loss"], "LR": rates, "elapsed_seconds": prior + time.monotonic() - started}, flush=True)
            if (index + 1) % 1000 == 0 or low:
                parent = "checkpoints" if (index + 1) % 1000 == 0 else "resource_stops"
                save_checkpoint(f.MODEL / parent / f"step_{index+1:05d}", model, optimizer, history, bound, initial, resource_log, prior + time.monotonic() - started, "resource_stop" if low else "scheduled", prior_peaks)
            if low:
                raise ResourceStop("RAM below2GiB at known boundary; full checkpoint saved; explicit user recovery only")
        torch.cuda.synchronize()
        final = {n: v.detach().cpu().contiguous().clone() for n, v in model.state_dict().items()}
        save_file(final, str(f.MODEL / "final_weights.safetensors"))
        c.require(c.sha(f.MODEL / "final_weights.safetensors") == c.sha(f.MODEL / "checkpoints/step_10000/model.safetensors"), "Formal final/checkpoint serialization differs")
        result = {"binding": bound, "step": f.STEPS, "all_finite": True, "real_exposures": 5000, "normal_exposures": 5000,
                  "exposure_by_id": dict(Counter(x["request"]["id"] for x in history)), "initial_buffers_hash": initial["initial_buffers_hash"],
                  "final_buffers_hash": buffers_hash(model), "final_state_hash": state_hash(final), "final_weight_sha256": c.sha(f.MODEL / "final_weights.safetensors"),
                  "elapsed_seconds": prior + time.monotonic() - started, "resumed_from": str(restore) if restore else None,
                  "peak_cuda_allocated_MiB": max(prior_peaks[0], torch.cuda.max_memory_allocated() / 2**20), "peak_cuda_reserved_MiB": max(prior_peaks[1], torch.cuda.max_memory_reserved() / 2**20),
                  "cuda_peak_scope": "training segments after peak reset, including saved prior segments on resume; not system VRAM",
                  "sampled_min_free_RAM_GiB": min(x["free_RAM_GiB"] for x in resource_log), "resource_samples": resource_log,
                  "scoring_calibration_or_AP": False, "GPU_complete_resume_verified": False}
        c.require(result["initial_buffers_hash"] == result["final_buffers_hash"], "Formal final BN changed")
        c.save(f.RUN / "result.json", result)
        validate_model_tree(bound, seq, final=True)
        c.seal(f.MODEL, bound); c.seal(f.RUN, bound)
    except (KeyboardInterrupt, ResourceStop):
        print("STOPPED: no implicit retry; latest full own checkpoint requires explicit manual resume", flush=True)
        raise
    except Exception as exc:
        c.save(f.RUN / "failure.json", {"error": repr(exc), "rule": "preserve; no automatic recovery or changed settings"})
        raise
    return verify_training()
