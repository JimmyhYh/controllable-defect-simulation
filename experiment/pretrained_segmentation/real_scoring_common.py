"""Independent E-only identities, CPU admission and explicit runtime scope."""
from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import sys
from . import common as c
from . import formal_common as f
from . import calibration_common as a

PREP = c.BASE / "real_scoring_preparation_v1"
RUN = c.BASE / "real_evaluation_v1"
RUNTIME = c.BASE / "real_scoring_runtime_v1"
Q_VERIFICATION = c.BASE / "normal_calibration_verification_v1"
TAU = 7.264203168233501e-06
CODE = ["real_scoring_common.py", "real_scoring.py", "real_cache.py", "real_scoring_cli.py", "README_real_scoring.md"]

def implementation():
    return c.code_identity(CODE)

def guard(cpu=True, raw_e=False, probabilities=False):
    forbidden = lambda n: n.split(".")[0] in c.BLOCKED or n == "pretrained_segmentation.model"
    c.require(not cpu or not any(forbidden(n) for n in sys.modules), "CPU real preparation contains model imports")
    rows = c.load(c.DATA / "split.json")["E_official1004"]
    allowed = {(c.DATA / r["image"]).resolve() for r in rows} if raw_e else set()
    def audit(event, args):
        if cpu and event == "import" and forbidden(args[0]):
            raise RuntimeError("CPU E preparation/cache verification forbids model imports")
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("No network or pickle execution in E scoring")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}: return
            path = Path(name).resolve(); mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in [PREP, RUN, RUNTIME]), "Write outside independent E destinations")
            elif path.is_relative_to(c.ROOT / "data"):
                c.require(path == (c.DATA / "split.json").resolve() or path in allowed, "Q/E/U raw or GT access outside E RGB scoring scope")
            elif path.suffix.lower() == ".npy" or "probabilities" in path.parts:
                c.require(probabilities and path.is_relative_to(RUN / "probabilities"), "Probability/input array payload not authorized")
    sys.addaudithook(audit)

def fresh_runtime(present):
    c.require(present is False, "Existing/unknown E runtime; preserve before any forward")

def runtime_environment():
    c.require(Path(sys.executable).resolve() == (c.ENV / "Scripts/python.exe").resolve(), "Use .venv-pretrained-seg for E scoring")
    actual = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in md.distributions()}
    expected = {x["name"].lower().replace("_", "-"): x["version"] for x in c.load(c.BASE / "environment_v1/environment.json")["metadata"]["distributions"]}
    c.require(actual == expected, "E runtime dependency identity differs")
    import tempfile
    fresh_runtime(RUNTIME.exists()); c.paths(RUNTIME); RUNTIME.mkdir(parents=True, exist_ok=False)
    os.environ.update({"TEMP": str(RUNTIME), "TMP": str(RUNTIME), "TORCH_HOME": str(RUNTIME / "torch"),
                       "HF_HOME": str(RUNTIME / "hf"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    tempfile.tempdir = str(RUNTIME)

def upstream():
    """Full formal hash checks; Q payloads are NOT read, only prior audit evidence."""
    from .native_inference import SCORE_RULE, THRESHOLD_RULE
    sources, formal, summary = a.upstream()
    f.strict_seal(a.PREP)
    qp = c.load(a.PREP / "protocol.json")
    c.require(qp["implementation"] == a.implementation() and qp["score_rule"] == SCORE_RULE and qp["threshold_rule"] == THRESHOLD_RULE, "Frozen Q implementation/rules differ")
    c.verify_upstream(qp["upstream"])
    qs = c.load(a.RUN / "complete.json")
    c.require(c.sha(a.RUN / "complete.json") == "07440d4222c9843f544cf7b0118da60dd1cd92e2a70ae9efe5dd2d8dc740a171", "Original Q completion changed")
    # Deliberately do not strict_seal Q RUN: that would read all Q NPY bytes.
    c.require(c.paths(a.RUN) == set(qs["files"]) | {"complete.json"}, "Unknown/missing Q cache member")
    c.require({p.relative_to(a.RUN).as_posix() for p in a.RUN.rglob("*") if p.is_dir()} == {"probabilities"}, "Unknown Q cache directory")
    evidence_seal = f.strict_seal(Q_VERIFICATION)
    evidence = c.load(Q_VERIFICATION / "verification.json")
    c.require(evidence_seal["identity"] == evidence["identity"], "Q independent evidence identity differs")
    refs = {"calibration_complete_sha256": a.RUN / "complete.json", "scores_sha256": a.RUN / "scores.json",
            "threshold_sha256": a.RUN / "threshold.json", "preparation_complete_sha256": a.PREP / "complete.json",
            "formal_model_complete_sha256": f.MODEL / "complete.json", "final_weight_sha256": f.MODEL / "final_weights.safetensors"}
    for key, path in refs.items():
        c.require(evidence["identity"][key] == c.sha(path), "Q independent audit source differs: " + key)
    c.require(evidence["identity"]["implementation"] == a.implementation(), "Q audited implementation differs")
    threshold = c.load(a.RUN / "threshold.json"); scores = c.load(a.RUN / "scores.json")
    c.require(threshold["binding"] == qs["identity"] == scores["binding"] and threshold["rule"] == THRESHOLD_RULE, "Q score/threshold binding differs")
    c.require(threshold["threshold"] == TAU == evidence["summary"]["threshold"] and threshold["FP"] == 10 and threshold["TN"] == 190 and threshold["normal_denominator"] == 200, "Original saved threshold or audit counts differ")
    c.require(evidence["summary"]["calibration_verified"] is True and evidence["summary"]["images"] == 200 and evidence["summary"]["window_forwards_recorded"] == 844, "Q full audit not established")
    trained = c.load(f.RUN / "result.json")
    c.require(scores["images_scored"] == scores["raw_Q_image_decodes"] == 200 and scores["window_forwards"] == 844 and scores["constructor_forwards"] == 0, "Q recorded completion counts differ")
    c.require(scores["state_before"] == scores["state_after"] == trained["final_state_hash"] and scores["buffers_before"] == scores["buffers_after"] == trained["initial_buffers_hash"] == trained["final_buffers_hash"], "Q final state/BN binding differs")
    c.require(threshold["score_source_sha256"] == c.sha(a.RUN / "scores.json") and threshold["binding"]["final_weight_sha256"] == summary["final_weight_sha256"], "Q final weight/score source differs")
    for name in ["scores.json", "threshold.json"]:
        c.require(qs["files"][name] == c.sha(a.RUN / name), "Q manifest metadata differs")
    for name in a.CODE:
        path = Path(__file__).parent / name; sources[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    for path in [a.PREP / "complete.json", a.PREP / "protocol.json", a.PREP / "requests.json", a.RUN / "complete.json", a.RUN / "scores.json", a.RUN / "threshold.json", Q_VERIFICATION / "complete.json", Q_VERIFICATION / "verification.json"]:
        sources[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    return sources, formal, summary
