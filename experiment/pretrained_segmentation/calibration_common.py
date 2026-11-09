"""Independent normal calibration identities and explicit read/write scope."""
from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import sys
from . import common as c
from . import formal_common as f

PREP = c.BASE / "normal_calibration_preparation_v1"
RUN = c.BASE / "normal_calibration_v1"
RUNTIME = c.BASE / "normal_calibration_runtime_v1"
CODE = ["calibration_common.py", "native_inference.py", "normal_calibration.py", "calibration_cli.py", "README_calibration.md"]

def implementation():
    return {name: c.sha(Path(__file__).parent / name) for name in CODE}

def guard(cpu=True, raw_q=False, probability_cache=False):
    forbidden = lambda n: n.split(".")[0] in c.BLOCKED or n == "pretrained_segmentation.model"
    c.require(not cpu or not any(forbidden(n) for n in sys.modules), "CPU calibration preparation contains model imports")
    q = c.load(c.DATA / "split.json")["Q_normal200"]
    allowed = {(c.DATA / r["image"]).resolve() for r in q} if raw_q else set()
    def audit(event, args):
        if cpu and event == "import" and forbidden(args[0]):
            raise RuntimeError("CPU calibration preparation forbids model imports")
        if event == "socket.connect" or cpu and event.startswith("pickle."):
            raise RuntimeError("No network/pickle execution authorized")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}: return
            path = Path(name).resolve()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in [PREP, RUN, RUNTIME]), "Write outside independent normal calibration outputs")
            elif path.is_relative_to(c.ROOT / "data"):
                c.require(path == (c.DATA / "split.json").resolve() or path in allowed, "Raw Q/E/U or GT access outside explicit scoring scope")
            elif path.suffix.lower() == ".npy" or "probabilities" in path.parts:
                c.require(probability_cache and path.is_relative_to(RUN / "probabilities"), "Probability payload access not authorized by this entry")
    sys.addaudithook(audit)

def require_fresh_runtime(is_present):
    """Pure metadata admission; no filesystem mutation or model operation."""
    c.require(is_present is False, "Existing/unknown calibration runtime directory; preserve and diagnose before any new run")

def runtime_environment():
    c.require(Path(sys.executable).resolve() == (c.ENV / "Scripts/python.exe").resolve(), "Use .venv-pretrained-seg for actual calibration")
    actual = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in md.distributions()}
    expected = {x["name"].lower().replace("_", "-"): x["version"] for x in c.load(c.BASE / "environment_v1/environment.json")["metadata"]["distributions"]}
    c.require(actual == expected, "Calibration runtime dependency identity differs")
    import tempfile
    require_fresh_runtime(RUNTIME.exists())
    c.paths(RUNTIME); RUNTIME.mkdir(parents=True, exist_ok=False)
    os.environ.update({"TEMP": str(RUNTIME), "TMP": str(RUNTIME), "TORCH_HOME": str(RUNTIME / "torch"),
                       "HF_HOME": str(RUNTIME / "hf"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    tempfile.tempdir = str(RUNTIME)

def upstream():
    """Hash/metadata only; do not read raw calibration images, GT or probabilities."""
    from .formal_training import verify_training
    from .formal_preparation import verify
    summary = verify_training(); formal, _, _ = verify(decode=False)
    folder = c.BASE / "baseline_training_verification_v1"
    seal = f.strict_seal(folder); v = c.load(folder / "verification.json")
    c.require(seal["identity"] == v["identity"], "Formal training verification seal differs")
    for key, path in [("preparation_complete_sha256", f.PREP / "complete.json"), ("formal_model_complete_sha256", f.MODEL / "complete.json"),
                      ("formal_run_complete_sha256", f.RUN / "complete.json"), ("final_weight_sha256", f.MODEL / "final_weights.safetensors")]:
        c.require(v["identity"][key] == c.sha(path), "Formal training verification source differs: " + key)
    c.require(summary["step"] == 10000 and summary["seed"] == f.SEED and v["updates"] == 10000 and len(v["checkpoints"]) == 10, "Formal final training not verified")
    mapping = dict(formal["upstream"])
    for name in f.CODE:
        path = Path(__file__).parent / name
        c.require(c.sha(path) == formal["implementation"][name], "Frozen formal implementation differs")
        mapping[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    for path in [f.PREP / "complete.json", f.PREP / "protocol.json", f.RUN / "complete.json", f.RUN / "result.json",
                 f.MODEL / "complete.json", f.MODEL / "initialization.json", f.MODEL / "final_weights.safetensors", folder / "complete.json", folder / "verification.json"]:
        mapping[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    c.verify_upstream(mapping)
    return mapping, formal, summary
