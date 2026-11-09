"""Explicit USER-STARTED pilot training; import/verification is model-free."""
from __future__ import annotations
import os
from pathlib import Path
import time
from . import common as c

class ResourceStop(Exception):
    pass

def binding(protocol):
    return {"pilot_complete_sha256": c.sha(c.PREP / "complete.json"), "protocol_sha256": c.sha(c.PREP / "protocol.json"),
            "seed": c.SEED, "steps": 2000, "implementation": protocol["implementation"]}

def verify_checkpoint(folder, bound):
    from .preparation import strict_directories
    seal = strict_directories(folder)
    c.require(c.paths(folder) == {"model.safetensors", "optimizer.pt", "rng.json", "history.json", "checkpoint.json", "complete.json"}, "Unknown checkpoint member/lock")
    c.require(seal["identity"] == bound, "Checkpoint belongs to another protocol/run")
    row = c.load(Path(folder) / "checkpoint.json")
    history = c.load(Path(folder) / "history.json")
    c.require(row["binding"] == bound and row["step"] == len(history) and 1 <= row["step"] <= 2000, "Checkpoint cursor differs")
    c.require([x["update"] for x in history] == list(range(1, row["step"] + 1)) and all(x["lr"] == c.lr_at(x["update"]) and x["finite"] is True for x in history), "Checkpoint actual LR/finite prefix differs")
    seq = c.load(c.PREP / "requests.json")
    c.require([x["item_index"] for x in history] == [x["item_index"] for x in seq[:row["step"]]], "Actual sequence prefix changed")
    c.require(row["buffers_hash"] == row["initial_buffers_hash"], "Encoder BN buffers changed")
    c.require(row["next_lr"] == (c.lr_at(row["step"] + 1) if row["step"] < 2000 else None), "Saved recovery LR differs")
    return row, history

def validate_model_tree(bound, final=False, complete=False):
    """Never absorb unknown root artifacts/locks into a new complete seal."""
    c.paths(c.MODEL)
    expected_files = {"initialization.json"} | ({"final_weights.safetensors"} if final else set()) | ({"complete.json"} if complete else set())
    files = {p.name for p in c.MODEL.iterdir() if p.is_file()}
    directories = {p.name for p in c.MODEL.iterdir() if p.is_dir()}
    c.require(files == expected_files and directories <= {"checkpoints", "resource_stops"}, "Unknown model root member or lock; no update/resume")
    rows = []
    for name in directories:
        members = list((c.MODEL / name).iterdir())
        c.require(members and all(p.is_dir() for p in members), "Unknown/empty checkpoint container")
        for folder in members:
            row, _ = verify_checkpoint(folder, bound)
            c.require(folder.name == f"step_{row['step']:04d}" and row["step"] % (400 if name == "checkpoints" else 100) == 0, "Checkpoint directory/name/cursor differs")
            rows.append((row["step"], folder.resolve()))
    return rows

def verify_run():
    from .preparation import verify_pilot, strict_directories
    p, _, _ = verify_pilot(decode=False, source_bytes=False)
    bound = binding(p)
    c.require(strict_directories(c.RUN)["identity"] == bound and strict_directories(c.MODEL)["identity"] == bound, "Run/model completion identity differs")
    c.require(c.paths(c.RUN) == {"binding.json", "transaction.json", "result.json", "complete.json"}, "Unknown run result member")
    validate_model_tree(bound, final=True, complete=True)
    result = c.load(c.RUN / "result.json")
    c.require(result["binding"] == bound and result["step"] == 2000 and result["all_finite"] is True, "Final budget/finite result differs")
    required = [400, 800, 1200, 1600, 2000]
    for step in required:
        row, _ = verify_checkpoint(c.MODEL / "checkpoints" / f"step_{step:04d}", bound)
        c.require(row["step"] == step, "Checkpoint name/cursor mismatch")
    row, history = verify_checkpoint(c.MODEL / "checkpoints/step_2000", bound)
    initial = c.load(c.MODEL / "initialization.json")
    c.require(row["state_hash"] == result["final_state_hash"] and row["buffers_hash"] == initial["initial_buffers_hash"] == result["initial_buffers_hash"], "Final parameter/buffer metadata inconsistent")
    c.require(c.sha(c.MODEL / "final_weights.safetensors") == result["final_weight_sha256"] == c.sha(c.MODEL / "checkpoints/step_2000/model.safetensors"), "Final weight identity mismatch")
    c.require(result["real_exposures"] == sum(x["item_index"] < 5 for x in history) == 1000 and result["normal_exposures"] == 1000, "Actual exposure composition differs")
    return {"training_complete": True, "steps": 2000, "scheduled_checkpoints": required, "final_weight_sha256": result["final_weight_sha256"],
            "actual_scoring_or_AP": False, "runtime_resumed": result["resumed_from"], "GPU_full_resume_verified": False}

def rng_state():
    import random, numpy as np, torch
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state().tolist(), "torch_cuda_all": [x.tolist() for x in torch.cuda.get_rng_state_all()]}

def restore_rng(p):
    import random, numpy as np, torch
    def tuples(x):
        return tuple(tuples(v) for v in x) if isinstance(x, list) else x
    random.setstate(tuples(p["python"]))
    n = p["numpy"]; np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(torch.tensor(p["torch_cpu"], dtype=torch.uint8, device="cpu"))
    c.require(len(p["torch_cuda_all"]) == torch.cuda.device_count(), "Restore CUDA device count differs")
    torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8, device="cpu") for x in p["torch_cuda_all"]])

def save_checkpoint(folder, model, optimizer, history, bound, initial, resources, elapsed, kind="scheduled"):
    import torch
    from safetensors.torch import save_file
    from .model import state_hash, buffers_hash
    c.require(not folder.exists(), "Checkpoint already exists or partially saved")
    folder.mkdir(parents=True, exist_ok=False)
    state = {n: t.detach().cpu().contiguous().clone() for n, t in model.state_dict().items()}
    save_file(state, str(folder / "model.safetensors"))
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    c.save(folder / "rng.json", rng_state())
    step = len(history)
    c.save(folder / "history.json", history)
    row = {"binding": bound, "step": step, "kind": kind, "state_hash": state_hash(state), "buffers_hash": buffers_hash(model),
           "initial_buffers_hash": initial["initial_buffers_hash"], "next_lr": c.lr_at(step + 1) if step < 2000 else None,
           "resources": resources, "elapsed_seconds": elapsed, "checkpoint_restore_GPU_tested": False}
    c.require(row["buffers_hash"] == row["initial_buffers_hash"], "BN buffers changed during training")
    c.save(folder / "checkpoint.json", row)
    c.seal(folder, bound)
    return row

def train_probe(resume_from=None):
    from .preparation import verify_pilot
    from .data import read_cache
    p, ip, seq = verify_pilot()
    bound = binding(p)
    if (c.RUN / "complete.json").is_file() or (c.MODEL / "complete.json").is_file():
        c.require(resume_from is None, "Completed run is read-only; no resume")
        return verify_run()
    c.runtime_environment()
    start_resources = c.resources(4.5, 20, gpu=True)
    restore = None
    if resume_from is None:
        c.require(not c.RUN.exists() and not c.MODEL.exists(), "Unknown/partial run; explicit own complete checkpoint required")
        c.RUN.mkdir(parents=True, exist_ok=False); c.MODEL.mkdir(parents=True, exist_ok=False)
        c.save(c.RUN / "binding.json", bound)
        c.save(c.RUN / "transaction.json", {"manual_entry": True, "budget": 2000, "resources_before_model": start_resources})
    else:
        c.require(c.load(c.RUN / "binding.json") == bound and c.paths(c.RUN) == {"binding.json", "transaction.json"}, "Unknown partial run/failure/finalization; diagnose, do not resume")
        c.require(not any(p.is_dir() for p in c.RUN.rglob("*")), "Unknown unsealed run directory/lock; no resume")
        restore = Path(resume_from).resolve()
        c.require(restore.is_relative_to(c.MODEL.resolve()) and restore.parent.name in {"checkpoints", "resource_stops"}, "Resume only this run's own full checkpoint")
        rows = validate_model_tree(bound)
        c.require(rows and restore in [path for step, path in rows if step == max(s for s, _ in rows)], "Resume requires latest full own checkpoint")
        rr, history = verify_checkpoint(restore, bound)
        c.require(rr["step"] < 2000, "Final checkpoint exists but finalization incomplete; diagnose separately")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from safetensors.torch import save_file, load_file
    from .model import build_model, make_optimizer, set_training_mode, loss_value, state_hash, buffers_hash
    started = time.monotonic()
    try:
        model, initial = build_model(c.SEED, p)
        model.to("cuda"); optimizer = make_optimizer(model)
        history, resource_log, done, prior_elapsed = [], [], 0, 0.0
        if restore is None:
            c.save(c.MODEL / "initialization.json", initial)
        else:
            rr, history = verify_checkpoint(restore, bound)
            saved_initial = c.load(c.MODEL / "initialization.json")
            c.require(saved_initial == initial, "Fresh initialization differs before own-state restore")
            model.load_state_dict(load_file(str(restore / "model.safetensors"), device="cuda"), strict=True)
            optimizer.load_state_dict(torch.load(restore / "optimizer.pt", map_location="cpu", weights_only=True))
            c.require(state_hash(model.state_dict()) == rr["state_hash"] and buffers_hash(model) == initial["initial_buffers_hash"], "Restored state differs")
            restore_rng(c.load(restore / "rng.json"))
            done, prior_elapsed = rr["step"], rr["elapsed_seconds"]
            resource_log = rr["resources"]
        torch.cuda.reset_peak_memory_stats()
        for index in range(done, 2000):
            req = seq[index]; item = ip["items"][req["item_index"]]
            pack = read_cache(item)
            x = torch.from_numpy(pack["image"].copy()).unsqueeze(0).to("cuda")
            y = torch.from_numpy(pack["target"].copy()).unsqueeze(0).to("cuda")
            valid = torch.from_numpy(pack["valid"].copy()).unsqueeze(0).to("cuda")
            lr = c.lr_at(index + 1)
            for group, rate in zip(optimizer.param_groups, lr): group["lr"] = rate
            set_training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, ce, dice = loss_value(logits, y, valid)
            c.require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            c.require(all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in model.parameters() if t.requires_grad), "Missing/nonfinite gradient")
            optimizer.step()
            c.require(all(bool(torch.isfinite(t).all()) for t in model.parameters()), "Nonfinite updated parameter")
            history.append({"update": index + 1, "item_index": req["item_index"], "id": item["id"], "loss": float(loss.detach()), "CE": float(ce.detach()), "Dice": float(dice.detach()), "lr": lr, "finite": True})
            del x, y, valid, logits, loss, ce, dice
            low = False
            if (index + 1) % 100 == 0:
                snap = c.resources(0, 0)
                snap["step"] = index + 1; resource_log.append(snap)
                low = snap["free_RAM_GiB"] < 2
                print({"step": index + 1, "of": 2000, "loss": history[-1]["loss"], "LR": lr, "elapsed_seconds": prior_elapsed + time.monotonic() - started}, flush=True)
            if (index + 1) % 400 == 0 or low:
                parent = "checkpoints" if (index + 1) % 400 == 0 else "resource_stops"
                folder = c.MODEL / parent / f"step_{index + 1:04d}"
                save_checkpoint(folder, model, optimizer, history, bound, initial, resource_log, prior_elapsed + time.monotonic() - started, "resource_stop" if low else "scheduled")
            if low:
                raise ResourceStop("Sampled RAM below2GiB; own complete stop checkpoint saved; manual review/resume only")
        torch.cuda.synchronize()
        final = {n: t.detach().cpu().contiguous().clone() for n, t in model.state_dict().items()}
        save_file(final, str(c.MODEL / "final_weights.safetensors"))
        c.require(c.sha(c.MODEL / "final_weights.safetensors") == c.sha(c.MODEL / "checkpoints/step_2000/model.safetensors"), "Final serialized weight differs from final checkpoint")
        result = {"binding": bound, "step": 2000, "all_finite": True, "real_exposures": 1000, "normal_exposures": 1000,
                  "initial_buffers_hash": initial["initial_buffers_hash"], "final_buffers_hash": buffers_hash(model),
                  "final_state_hash": state_hash(final), "final_weight_sha256": c.sha(c.MODEL / "final_weights.safetensors"),
                  "elapsed_seconds": prior_elapsed + time.monotonic() - started, "resumed_from": str(restore) if restore else None,
                  "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20, "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
                  "sampled_min_free_RAM_GiB": min(s["free_RAM_GiB"] for s in resource_log), "resource_samples": resource_log,
                  "scoring_or_AP_performed": False, "gate": "not evaluated; explicit final scoring then CPU metrics required"}
        c.require(result["final_buffers_hash"] == result["initial_buffers_hash"], "Final encoder BN buffers changed")
        c.save(c.RUN / "result.json", result)
        validate_model_tree(bound, final=True)
        c.seal(c.MODEL, bound); c.seal(c.RUN, bound)
    except (KeyboardInterrupt, ResourceStop):
        print("STOPPED: no implicit retry; only latest full own checkpoint can be explicitly resumed", flush=True)
        raise
    except Exception as exc:
        c.save(c.RUN / "failure.json", {"error": repr(exc), "rule": "preserve; no automatic recovery or changed settings"})
        raise
    return verify_run()
