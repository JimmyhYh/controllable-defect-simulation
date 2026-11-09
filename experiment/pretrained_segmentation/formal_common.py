"""Independent formal identities and CPU/runtime boundaries; no pilot mutation."""
from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import sys
from . import common as c

SEED = 2026102202
STEPS = 10000
PREP = c.BASE / "formal_preparation_v1"
RUN = c.BASE / "baseline_v1"
MODEL = c.ROOT / "models/pretrained_segmentation/ksdd2_r5_v1/baseline_2026102202"
RUNTIME = c.BASE / "formal_runtime_v1"
CODE = ["formal_common.py", "formal_preparation.py", "formal_training.py", "formal_cli.py", "README_formal.md"]

def implementation():
    return {name: c.sha(Path(__file__).parent / name) for name in CODE}

def strict_seal(folder):
    folder = Path(folder)
    result = c.verify_seal(folder)
    expected = set()
    for name in result["files"]:
        parent = Path(name).parent
        while str(parent) != ".":
            expected.add(parent.as_posix()); parent = parent.parent
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_dir()}
    c.require(actual == expected, "Unknown/missing formal directory or directory-shaped lock")
    return result

def guard(cpu=True):
    """Own guard: training pool only, independent destinations, no old global edits."""
    forbidden = lambda n: n.split(".")[0] in c.BLOCKED or n == "pretrained_segmentation.model"
    c.require(not cpu or not any(forbidden(n) for n in sys.modules), "CPU formal preparation contains model imports")
    split = c.load(c.DATA / "split.json")
    allowed = {(c.DATA / r[k]).resolve() for r in split["R_real5"] + split["N_train1885"] for k in ["image", "mask"]}
    def audit(event, args):
        if cpu and event == "import" and forbidden(args[0]):
            raise RuntimeError("CPU formal preparation forbids model imports")
        if event == "socket.connect" or cpu and event.startswith("pickle."):
            raise RuntimeError("No network/pickle execution authorized")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return
            path = Path(name).resolve()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in [PREP, RUN, MODEL, RUNTIME]), "Write outside independent formal outputs")
            elif path.is_relative_to(c.ROOT / "data"):
                c.require(path in allowed or path == (c.DATA / "split.json").resolve(), "Q/E/U or nontraining raw input forbidden")
    sys.addaudithook(audit)

def runtime_environment():
    c.require(Path(sys.executable).resolve() == (c.ENV / "Scripts/python.exe").resolve(), "Use independent .venv-pretrained-seg")
    actual = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in md.distributions()}
    expected = {x["name"].lower().replace("_", "-"): x["version"] for x in c.load(c.BASE / "environment_v1/environment.json")["metadata"]["distributions"]}
    c.require(actual == expected, "Formal runtime dependency set differs")
    import tempfile
    c.paths(RUNTIME)  # Reject existing redirects; this is not a scientific completion.
    RUNTIME.mkdir(parents=True, exist_ok=True)
    os.environ.update({"TEMP": str(RUNTIME), "TMP": str(RUNTIME), "TORCH_HOME": str(RUNTIME / "torch"),
                       "HF_HOME": str(RUNTIME / "hf"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    tempfile.tempdir = str(RUNTIME)

def upstream():
    """Read-only accepted gate metadata and protected code/resources, no probabilities."""
    from .preparation import verify_pilot
    from .training import verify_run
    pilot, _, _ = verify_pilot(decode=False, source_bytes=False)
    verify_run()
    folder = c.BASE / "probe_acceptance_verification_v1"
    seal = strict_seal(folder)
    accepted = c.load(folder / "verification.json")
    c.require(seal["identity"] == accepted["identity"], "Accepted gate seal differs")
    for key, path in [("metrics_sha256", c.METRICS / "metrics.json"), ("metrics_complete_sha256", c.METRICS / "complete.json"),
                      ("score_complete_sha256", c.SCORES / "complete.json"), ("input_complete_sha256", c.INPUT / "complete.json"),
                      ("pilot_complete_sha256", c.PREP / "complete.json"), ("training_complete_sha256", c.RUN / "complete.json"),
                      ("model_complete_sha256", c.MODEL / "complete.json"),
                      ("training_verification_complete_sha256", c.BASE / "probe_training_verification_v1/complete.json")]:
        c.require(accepted["identity"][key] == c.sha(path), "Accepted source binding differs: " + key)
    metrics = c.load(c.METRICS / "metrics.json")
    c.require(accepted["gate_passed"] is True and metrics["gate_passed"] is True and all(metrics["gate_checks"].values()), "Final pilot gate not passed")
    c.require(metrics["step"] == 2000 and metrics["valid_pixels"] == 1225984 and metrics["positive_pixels"] == 24482, "Accepted budget/denominators differ")
    result = c.upstream_files()
    for name in c.CODE:
        path = Path(__file__).parent / name
        c.require(c.sha(path) == pilot["implementation"][name], "Frozen pilot implementation differs")
        result[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    for path in [c.INPUT / "complete.json", c.INPUT / "protocol.json", c.PREP / "complete.json", c.PREP / "protocol.json",
                 c.MODEL / "complete.json", c.RUN / "complete.json", c.RUN / "result.json", c.SCORES / "complete.json",
                 c.METRICS / "complete.json", c.METRICS / "metrics.json", folder / "complete.json", folder / "verification.json"]:
        result[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    return result, pilot
