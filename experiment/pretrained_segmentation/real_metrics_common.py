"""CPU-only real-metric identities and explicit GT/output boundaries."""
from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import sys
from . import common as c
from . import formal_common as f
from . import real_scoring_common as r
from fewshot_synthetic.metrics import exact_ap, BINS, CHUNK
from fewshot_synthetic.common import resource

PREP = c.BASE / "real_metrics_preparation_v1"
RUN = c.BASE / "real_metrics_v1"
EVIDENCE = c.BASE / "real_scoring_verification_v1"
CODE = ["real_metrics_common.py", "real_metrics.py", "real_metrics_cli.py", "README_real_metrics.md"]
PIXELS = 146411528

def implementation():
    return c.code_identity(CODE)

def require_closed_members(files, directories, manifest_files):
    """Pure file/directory-set admission; accepts invented sets for scalar audit."""
    expected_dirs = set()
    for name in manifest_files:
        parent = Path(name).parent
        while str(parent) != ".":
            expected_dirs.add(parent.as_posix()); parent = parent.parent
    c.require(set(files) == set(manifest_files) | {"complete.json"} and set(directories) == expected_dirs,
              "Unknown/missing upstream metric source file, lock or directory")

def source_trees():
    """Re-check closed E/preparation/evidence directories on every reuse path."""
    for folder in [r.PREP, r.RUN, EVIDENCE, r.Q_VERIFICATION]:
        seal = c.load(folder / "complete.json")
        require_closed_members(c.paths(folder), {x.relative_to(folder).as_posix() for x in folder.rglob("*") if x.is_dir()}, seal["files"])
        f.strict_seal(folder)  # Source hashes too; E probabilities hashed, not parsed.

def cpu_environment():
    return {"interpreter": str(Path(sys.executable).resolve()), "interpreter_sha256": c.sha(sys.executable),
            "python": sys.version, "packages": {n: md.version(n) for n in ["numpy", "Pillow", "scipy", "scikit-learn"]}}

def guard(gt=False, arrays=False, writing=False):
    blocked = lambda n: n.split(".")[0] in c.BLOCKED or n == "pretrained_segmentation.model"
    c.require(not any(blocked(n) for n in sys.modules), "Fresh CPU metric process required")
    split = c.load(c.DATA / "split.json")
    allowed = {(c.DATA / x["mask"]).resolve() for x in split["E_official1004"]} if gt else set()
    def audit(event, args):
        if event == "import" and blocked(args[0]): raise RuntimeError("No model imports in CPU metrics")
        if event == "socket.connect" or event.startswith("pickle."): raise RuntimeError("No network/pickle in CPU metrics")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}: return
            path = Path(name).resolve(); mode, flags = args[1] or "", args[2] or 0
            write = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if write:
                c.require(writing and any(path.is_relative_to(p) for p in [PREP, RUN]), "Write outside independent metric destinations")
            elif path.is_relative_to(c.ROOT / "data"):
                c.require(path == (c.DATA / "split.json").resolve() or path in allowed, "RGB/GT read outside explicit E GT scope")
            elif path.suffix.lower() in {".npy", ".npz", ".uint32"}:
                # Preparation may HASH frozen E NPY files, never np.load them.
                c.require(path.is_relative_to(r.RUN / "probabilities") or arrays and path.is_relative_to(RUN), "Other array/cache payload forbidden")
    sys.addaudithook(audit)

def probe_reference():
    """JSON/hash evidence only: do not re-run AP or read old probe arrays/tables."""
    proof_dir = c.BASE / "probe_acceptance_verification_v1"
    seal = f.strict_seal(proof_dir); proof = c.load(proof_dir / "verification.json")
    c.require(seal["identity"] == proof["identity"], "Probe independent evidence identity differs")
    summary = c.load(c.METRICS / "metrics.json"); exact = c.load(c.METRICS / "exact/metrics.json")
    complete = c.load(c.METRICS / "complete.json"); exact_complete = c.load(c.METRICS / "exact/complete.json")
    c.require(proof["identity"]["metrics_complete_sha256"] == c.sha(c.METRICS / "complete.json") and proof["identity"]["metrics_sha256"] == c.sha(c.METRICS / "metrics.json") == complete["files"]["metrics.json"], "Probe AP source differs")
    c.require(complete["files"]["exact/complete.json"] == c.sha(c.METRICS / "exact/complete.json") and exact_complete["files"]["metrics.json"] == c.sha(c.METRICS / "exact/metrics.json"), "Probe exact result metadata differs")
    c.require(summary["binding"]["exact_AP_source_sha256"] == c.sha(c.ROOT / "fewshot_synthetic/metrics.py"), "Frozen exact AP implementation differs")
    c.require(summary["global_AP"] == exact["AP"] == proof["global_AP"] and summary["absolute_difference"] == abs(summary["global_AP"]-summary["sklearn_global_AP"]) <= 1e-10, "Probe sklearn crosscheck differs")
    c.require(exact["pixels"] == 1225984 and exact["positive_pixels"] == 24482 and exact["distinct_scores"] == proof["distinct_float32_scores"] and exact["tied_score_values"] == proof["tied_score_values"], "Probe actual ties/denominators differ")
    return {"evidence": "sealed same21 training-window AP/sklearn/actual ties; not recomputed", "pixels": exact["pixels"],
            "positive_pixels": exact["positive_pixels"], "absolute_difference": summary["absolute_difference"],
            "distinct_scores": exact["distinct_scores"], "tied_score_values": exact["tied_score_values"]}

def source_context():
    """Only source hashes/JSON/coordinates; no GT bytes or probability parsing."""
    from .real_scoring import verify_preparation, binding
    p, requests = verify_preparation()
    seal = f.strict_seal(r.RUN)  # Hashes E NPY bytes only. Never np.load here.
    c.require(seal["identity"] == binding(p), "E result binding differs")
    proof_seal = f.strict_seal(EVIDENCE); proof = c.load(EVIDENCE / "verification.json")
    c.require(proof_seal["identity"] == proof["identity"] and proof["summary"]["E_scores_verified"] is True and proof["all_1004_native_probabilities_verified"] is True, "Full E evidence missing")
    refs = {"preparation_complete_sha256": r.PREP / "complete.json", "scoring_complete_sha256": r.RUN / "complete.json",
            "scores_sha256": r.RUN / "scores.json", "transaction_sha256": r.RUN / "transaction.json", "formal_model_complete_sha256": f.MODEL / "complete.json",
            "final_weight_sha256": f.MODEL / "final_weights.safetensors", "original_threshold_sha256": c.BASE / p["threshold_source"],
            "normal_calibration_verification_complete_sha256": r.Q_VERIFICATION / "complete.json"}
    for key, path in refs.items(): c.require(proof["identity"][key] == c.sha(path), "E audit source differs: " + key)
    c.require(proof["identity"]["implementation"] == r.implementation() and proof["summary"]["images"] == 1004 and proof["summary"]["native_pixels"] == PIXELS and proof["summary"]["window_forwards_recorded"] == 4270, "E full audit denominators differ")
    scores = c.load(r.RUN / "scores.json"); gt = c.load(r.PREP / "metric_sources.json")
    c.require(scores["binding"] == seal["identity"] and len(scores["items"]) == len(gt) == len(requests) == 1004 and sum(x["native_pixels"] for x in requests) == PIXELS, "E complete metadata differs")
    c.require(scores["state_before"] == scores["state_after"] == p["expected_state_hash"] and scores["buffers_before"] == scores["buffers_after"] == p["expected_buffers_hash"], "E parameter/BN evidence differs")
    inventory = c.load(c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json")["files"]
    for request, source, row in zip(requests, gt, scores["items"]):
        c.require(source["index"] == row["index"] == request["index"] and source["id"] == row["id"] == request["id"] and source["native_shape"] == request["native_shape"], "GT/E metadata order differs")
        c.require(source["mask"] == f"test/{source['id']}_GT.png" and source["mask_sha256"] == inventory[source["mask"]], "GT path/source hash metadata differs")
        c.require(row["probability"] == f"probabilities/{row['index']:04d}_{row['id']}.npy" and row["probability_sha256"] == seal["files"][row["probability"]], "E probability hash metadata differs")
    c.require([x["id"] for x in requests] == sorted(x["id"] for x in requests), "Frozen first-ID example rule requires original ascending IDs")
    reference = probe_reference(); sources = dict(p["upstream"])
    for path in [r.PREP / n for n in ["complete.json", "protocol.json", "requests.json", "metric_sources.json"]] + [EVIDENCE / "complete.json", EVIDENCE / "verification.json", r.RUN / "complete.json"]:
        sources[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    for name, digest in seal["files"].items(): sources[(r.RUN / name).relative_to(c.ROOT).as_posix()] = digest
    for path in [c.METRICS / n for n in ["complete.json", "metrics.json", "exact/complete.json", "exact/metrics.json"]] + [c.BASE / "probe_acceptance_verification_v1/complete.json", c.BASE / "probe_acceptance_verification_v1/verification.json"]:
        sources[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    return p, requests, gt, scores, sources, reference
