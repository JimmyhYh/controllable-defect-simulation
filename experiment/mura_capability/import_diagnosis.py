"""One-shot, import-only write-stack diagnosis; never constructs a detector.

prepare/static-check/verify use only the standard library. run owns exactly one
diagnostic child. Complete transcripts are verified, never replayed.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent.parent
BASE = ROOT / "results/mura_capability/week1_v1"
DAY1 = BASE / "day1_admission_v1"
OUT = BASE / "import_diagnosis_v1"
OLD = ROOT / "results/mura_response/semu_applicability_v1"
DAY1_SHA = "be272e26ff988e192d2d834bdb40b9ed2a9b6b2bd4c1e99e9f114c7730e2da1e"
SEQUENCE = ("numpy", "torch", "faiss", "torchvision.models",
            "patchcore.common", "patchcore.patchcore", "patchcore.sampler")
HEAVY = ("numpy", "scipy", "PIL", "torch", "torchvision", "faiss",
         "timm", "patchcore", "huggingface_hub", "safetensors", "httpx2")
PREP_FILES = {"sources.json", "protocol.json", "preparation_complete.json"}
CORE_NAMES = ("Pillow", "faiss-cpu", "numpy", "scipy", "timm", "torch", "torchvision", "tqdm")
EXTRA_NAMES = ("huggingface-hub", "safetensors", "httpx", "httpx2", "hf-xet")
PAYLOAD_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif",
               ".webp", ".npy", ".npz", ".pth", ".pt", ".pkl", ".pickle",
               ".faiss", ".safetensors"}
FORBIDDEN_IMPORTS = ("mura_response", "controlled_generation", "detector",
                     "localization_training", "localization_calibration", "pod_probe")
ENV_NAMES = ("TEMP", "TMP", "TMPDIR")
CURRENT_IMPORT = None
FIRST_EVENT = None
PROFILE_LOCK = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message + "; preserve state; no automatic retry or substitution")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + "\n")


def env():
    return {k: os.environ.get(k) for k in ENV_NAMES}


def own_implementation():
    return {str(Path(__file__).resolve()): digest(__file__),
            str(ROOT / "mura_capability/__init__.py"): digest(ROOT / "mura_capability/__init__.py")}


def versions():
    result = {}
    for name in CORE_NAMES + EXTRA_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in (
                "total", "available", "page", "page_available",
                "virtual", "virtual_available", "extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(Memory)]
    query.restype = ctypes.c_int
    require(query(ctypes.byref(memory)), "RAM query failed")
    result = {"available_RAM_GiB": memory.available / 2**30,
              "free_D_GiB": shutil.disk_usage(ROOT).free / 2**30}
    require(result["available_RAM_GiB"] >= 2 and result["free_D_GiB"] >= 40,
            "CPU metadata/import diagnosis resource admission not met")
    return result


def directory_stats(folder):
    rows = []
    for p in sorted(folder.rglob("*")):
        require(not p.is_symlink() and not p.is_junction(), "Redirected protected member")
        stat = p.stat()
        rows.append([p.relative_to(folder).as_posix(), "D" if p.is_dir() else "F",
                     stat.st_size, stat.st_mtime_ns])
    return {"members": len(rows),
            "stat_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}


def day1_binding(full=False):
    require(digest(DAY1 / "complete.json") == DAY1_SHA, "Day-1 complete identity differs")
    complete = read(DAY1 / "complete.json")
    require({p.name for p in DAY1.iterdir()} == set(complete["outputs"]) | {"complete.json"},
            "Day-1 strict inventory differs")
    require(all(digest(DAY1 / n) == h for n, h in complete["outputs"].items()),
            "Day-1 metadata changed")
    sources = read(DAY1 / "sources.json")
    if full:
        for p, h in sources["protected_files_sha256"].items():
            require(digest(p) == h, "Protected source/asset metadata differs: " + p)
        for name, expected in sources["protected_directory_stats"].items():
            require(directory_stats(OLD / name) == expected, "Protected directory differs: " + name)
    expected_versions = sources["runtime"]["packages_metadata_only"]
    current_versions = versions()
    require(all(current_versions[n] == v for n, v in expected_versions.items()),
            "Frozen core package metadata differs")
    require(digest(sys.executable) == sources["runtime"]["executable_sha256"],
            "Interpreter identity differs")
    failure = OLD / "fit_scores_v1/PatchCore"
    original_failure = read(failure / "failure.json")
    require(original_failure["completed_unique_inputs"] == 0
            and not (failure / "complete.json").exists()
            and not list((failure / "records").iterdir())
            and not list((failure / "heatmaps").iterdir()), "Old failure state differs")
    workpoints = {}
    for model in ("PatchCore", "G1"):
        folder = OLD / "normal_scoring_v1" / model
        require(digest(folder / "complete.json") == sources["models"][model]["complete_sha256"],
                "Normal model complete differs")
        q = read(folder / "q_result.json")
        require(q["threshold"] == sources["models"][model]["threshold"], "Frozen tau differs")
        workpoints[model] = {"complete_sha256": digest(folder / "complete.json"),
                            "threshold": q["threshold"]}
    evidence = PROJECT / "报告/Mura能力说明第1天资源与真实材料准入.md"
    text = evidence.read_text(encoding="utf-8")
    match = re.search(chr(96) * 3 + r"json\n(.*?)\n" + chr(96) * 3, text, re.S)
    require(match is not None, "Independent day-1 proof missing")
    proof = json.loads(match.group(1))
    require(proof["status"] == "DAY1_ADMISSION_CACHE_VERIFIED"
            and proof["complete_sha256"] == DAY1_SHA
            and proof["protected_files_sha256_unchanged"] == len(sources["protected_files_sha256"])
            and proof["protected_directory_stat_signatures_unchanged"] == len(sources["protected_directory_stats"])
            and proof["model_import_or_forward_or_calibration_or_fitting"] == 0,
            "Independent day-1 evidence differs")
    return {"day1_complete_sha256": DAY1_SHA,
            "day1_metadata_sha256": {n: digest(DAY1 / n) for n in sorted(complete["outputs"])},
            "day1_independent_report_sha256": digest(evidence),
            "day1_independent_proof": proof,
            "protected_files": len(sources["protected_files_sha256"]),
            "protected_directories": len(sources["protected_directory_stats"]),
            "old_failed_metadata_sha256": {n: digest(failure / n)
                                           for n in ("started.json", "sources.json", "failure.json", "inflight.json")},
            "normal_workpoints": workpoints,
            "implementation": own_implementation(), "package_metadata": current_versions,
            "interpreter": sys.executable, "interpreter_sha256": digest(sys.executable),
            "source_chain_identity": {
                str(p): digest(p) for p in (
                    ROOT / "mura_response/inference_runtime.py",
                    ROOT / "mura_response/fit_inference.py",
                    ROOT / "vendor/patchcore-inspection/src/patchcore/patchcore.py",
                    ROOT / "vendor/patchcore-inspection/src/patchcore/backbones.py",
                    ROOT / ".venv/Lib/site-packages/timm/models/_builder.py",
                    ROOT / ".venv/Lib/site-packages/timm/models/_hub.py",
                    Path(sys.base_prefix) / "Lib/tempfile.py")}}


def classify_open(path, mode, flags):
    path = Path(path).resolve()
    mode = mode if isinstance(mode, str) else ""
    flags = flags if isinstance(flags, int) else 0
    writing = any(c in mode for c in "wax+") or bool(
        flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and not path.is_relative_to(OUT.resolve()):
        return "OUTSIDE_WRITE"
    if path.suffix.lower() in PAYLOAD_EXT:
        return "FORBIDDEN_PAYLOAD_OPEN"
    return None


def forbidden_call(filename, name):
    file = filename.replace("\\", "/").lower()
    if "/mura_response/" in file or "/controlled_generation/" in file:
        return "FORBIDDEN_PROJECT_CALL"
    if "/torch/cuda/" in file and name in {
            "_lazy_init", "init", "is_available", "device_count", "mem_get_info",
            "manual_seed", "manual_seed_all", "set_device", "synchronize",
            "reset_peak_memory_stats", "current_device", "get_device_properties",
            "_exchange_device", "_maybe_exchange_device"}:
        return "FORBIDDEN_CUDA_CALL"
    if file.endswith("/torch/nn/modules/module.py") and name in ("__init__", "_call_impl", "_wrapped_call_impl"):
        return "FORBIDDEN_MODULE_CONSTRUCTION_OR_FORWARD"
    if file.endswith("/torch/optim/optimizer.py") and name == "__init__":
        return "FORBIDDEN_OPTIMIZER_CONSTRUCTION"
    if file.endswith("/torch/serialization.py") and name == "load":
        return "FORBIDDEN_TENSOR_LOAD"
    if "/safetensors/" in file and name in ("load", "load_file", "load_model"):
        return "FORBIDDEN_TENSOR_LOAD"
    if "/vendor/patchcore-inspection/src/patchcore/" in file and name in {
            "__init__", "load", "fit", "predict", "_predict", "_embed"}:
        return "FORBIDDEN_DETECTOR_CALL"
    return None


def static_contract():
    compile(Path(__file__).read_text(encoding="utf-8"), str(__file__), "exec")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    virtual = {
        "flags_only_external_write": classify_open(ROOT / "outside_virtual", None, flags),
        "own_JSON_write": classify_open(OUT / "virtual.json", "x", 0),
        "real_image_read": classify_open(ROOT / "virtual.png", "r", 0),
        "weight_read": classify_open(ROOT / "virtual.pth", "rb", 0),
        "array_read": classify_open(ROOT / "virtual.npy", "rb", 0),
        "CUDA_call": forbidden_call("D:/x/torch/cuda/__init__.py", "is_available"),
        "model_call": forbidden_call("D:/x/torch/nn/modules/module.py", "__init__")}
    require(virtual == {
        "flags_only_external_write": "OUTSIDE_WRITE", "own_JSON_write": None,
        "real_image_read": "FORBIDDEN_PAYLOAD_OPEN", "weight_read": "FORBIDDEN_PAYLOAD_OPEN",
        "array_read": "FORBIDDEN_PAYLOAD_OPEN", "CUDA_call": "FORBIDDEN_CUDA_CALL",
        "model_call": "FORBIDDEN_MODULE_CONSTRUCTION_OR_FORWARD"}, "Virtual static guard contract differs")
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [n.name for node in top if isinstance(node, ast.Import) for n in node.names]
    require(not any(n.split(".")[0] in HEAVY for n in names), "Heavy top-level import")
    loaded = [n for n in sys.modules if n.split(".")[0] in HEAVY]
    require(not loaded, "Static parent already loaded diagnostic dependencies")
    return {"syntax_compiled_only": True, "virtual_events": virtual,
            "actual_file_mutation_or_payload_read_in_scalar_checks": False,
            "model_imports": 0, "diagnostic_children_launched": 0,
            "actual_import_compatibility_verified": False}


def protocol(binding):
    return {"schema": "mura-capability-import-diagnosis-v1", "binding": binding,
            "diagnostic_import_sequence": list(SEQUENCE),
            "lookup_only_not_call": "torchvision.models.wide_resnet50_2",
            "author_path_added_before_patchcore_import": str(ROOT / "vendor/patchcore-inspection/src"),
            "omitted_original_calls": [
                "torch.cuda.is_available/mem_get_info/reset_peak_memory_stats",
                "random.seed/np.random.seed/torch.manual_seed/torch.cuda.manual_seed_all",
                "torch.set_num_threads", "TF32/cudnn property setters",
                "all backbone/detector/optimizer construction, parameter loading and forwards"],
            "environment": "inherit launching process TEMP/TMP/TMPDIR exactly; record, never modify or prewarm",
            "first_outside_write": "capture Python stack before syscall, reject, stop sequence; never allow/retry",
            "additional_scope_guards": "reject payload/network/process launch/mutations outside owned root and operational CUDA/model calls",
            "one_child_maximum": 1, "child_timeout_seconds": 120,
            "partial_transaction": "started without complete is not resumable; preserve and stop",
            "no_target_event": "not reproduced in this environment; not cleared for model construction",
            "completed_transcript": "read-only verification; no new child imports",
            "static_contract": static_contract()}


def prepare():
    if OUT.exists():
        return verify_preparation(full=True)
    binding = day1_binding(full=True)
    snapshot = resources()
    contract = protocol(binding)
    require(not OUT.exists(), "Unknown existing diagnostic directory")
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "sources.json", binding)
    write(OUT / "protocol.json", contract)
    write(OUT / "preparation_complete.json", {
        "outputs": {n: digest(OUT / n) for n in ("sources.json", "protocol.json")},
        "resource_snapshot": snapshot})
    return {"status": "IMPORT_DIAGNOSIS_PREPARED", "output": str(OUT),
            "preparation_sha256": digest(OUT / "preparation_complete.json"),
            "static_contract": contract["static_contract"], "resources": snapshot,
            "diagnostic_children_launched": 0}


def verify_preparation(full=False):
    require(OUT.is_dir() and not OUT.is_symlink() and not OUT.is_junction(),
            "Diagnostic preparation absent/redirected")
    c = read(OUT / "preparation_complete.json")
    require(set(c["outputs"]) == {"sources.json", "protocol.json"}
            and all(digest(OUT / n) == h for n, h in c["outputs"].items()),
            "Preparation metadata differs")
    current = day1_binding(full=full)
    require(read(OUT / "sources.json") == current
            and read(OUT / "protocol.json") == protocol(current),
            "Bound source/implementation/static contract changed")
    return {"status": "IMPORT_DIAGNOSIS_PREPARATION_VERIFIED",
            "preparation_sha256": digest(OUT / "preparation_complete.json"),
            "static_contract": static_contract(), "diagnostic_children_launched": 0}


class StopImport(BaseException):
    pass


def stack_rows():
    rows = []
    frame = sys._getframe(2)
    while frame is not None and len(rows) < 160:
        rows.append({"file": frame.f_code.co_filename,
                     "line": frame.f_lineno, "function": frame.f_code.co_name})
        frame = frame.f_back
    return list(reversed(rows))


def capture(kind, details):
    global FIRST_EVENT
    if FIRST_EVENT is None:
        FIRST_EVENT = {"kind": kind, "during_import": CURRENT_IMPORT,
                       "details": details, "stack_root_to_leaf": stack_rows(),
                       "temporary_environment": env()}
        write(OUT / "first_event.json", FIRST_EVENT)
    raise StopImport(kind)


def child_audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        reason = classify_open(path, mode, flags)
        if reason:
            capture(reason, {"audit_event": event, "target": str(Path(path).resolve()),
                             "mode": mode, "flags": flags})
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink"):
        paths = args[:2] if event in ("os.rename", "os.link", "os.symlink") else args[:1]
        if any(isinstance(p, (str, bytes)) and not Path(os.fsdecode(p)).resolve().is_relative_to(OUT.resolve())
               for p in paths):
            capture("OUTSIDE_MUTATION", {"audit_event": event,
                                        "path_arguments": [os.fsdecode(p) for p in paths if isinstance(p, (str, bytes))]})
    elif event in ("os.putenv", "os.unsetenv") and args:
        key = os.fsdecode(args[0]).upper()
        if key in ENV_NAMES:
            capture("FORBIDDEN_TEMP_ENV_CHANGE", {"audit_event": event, "key": key})
    elif event in ("socket.connect", "socket.getaddrinfo", "pickle.find_class",
                   "subprocess.Popen", "os.system", "os.posix_spawn", "os.chdir"):
        capture("FORBIDDEN_SIDE_EFFECT", {"audit_event": event})
    elif event == "sys.setprofile" and PROFILE_LOCK:
        capture("PROFILE_GUARD_REPLACEMENT", {"audit_event": event})


class ChildImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + ".") for n in FORBIDDEN_IMPORTS):
            capture("FORBIDDEN_PROJECT_IMPORT", {"module": fullname})
        return None


def child_profile(frame, event, arg):
    if event == "call":
        reason = forbidden_call(frame.f_code.co_filename, frame.f_code.co_name)
        if reason:
            capture(reason, {"file": frame.f_code.co_filename, "function": frame.f_code.co_name})
    elif event == "c_call":
        name = getattr(arg, "__name__", "")
        module = getattr(arg, "__module__", "") or ""
        if module.startswith("torch") and (name.startswith("_cuda") or name == "cuda"):
            capture("FORBIDDEN_CUDA_C_CALL", {"module": module, "function": name})


def child(token):
    global CURRENT_IMPORT, PROFILE_LOCK
    require(not (OUT / "complete.json").exists()
            and not (OUT / "child_started.json").exists(), "Diagnostic child already attempted")
    started = read(OUT / "started.json")
    require(started["token"] == token and started["parent_pid"] == os.getppid(),
            "Child launch identity differs")
    require(read(OUT / "sources.json")["implementation"] == own_implementation(),
            "Child implementation differs")
    require(env() == started["temporary_environment"], "Inherited temporary environment differs")
    before_env = env()
    write(OUT / "child_started.json", {"token": token, "pid": os.getpid(),
          "parent_pid": os.getppid(), "temporary_environment": before_env,
          "sequence": list(SEQUENCE), "actual_detector_actions_not_started": True})
    sys.meta_path.insert(0, ChildImportGuard())
    sys.addaudithook(child_audit)
    sys.setprofile(child_profile)
    PROFILE_LOCK = True
    imports = []
    began = time.perf_counter()
    error = None
    outcome = None
    try:
        for name in SEQUENCE:
            CURRENT_IMPORT = name
            if name == "patchcore.common":
                sys.path.insert(0, str(ROOT / "vendor/patchcore-inspection/src"))
            entry = {"module": name, "completed": False}
            imports.append(entry)
            module = importlib.import_module(name)
            if name == "torchvision.models":
                symbol = getattr(module, "wide_resnet50_2")
                require(callable(symbol), "Expected backbone symbol absent")
                entry["symbol_retrieved_not_called"] = "wide_resnet50_2"
            entry["completed"] = True
        outcome = "IMPORT_SEQUENCE_COMPLETED_NO_TARGET_EVENT"
    except StopImport:
        outcome = "OUTSIDE_WRITE_CAPTURED" if FIRST_EVENT and FIRST_EVENT["kind"] == "OUTSIDE_WRITE" else "IMPORT_SCOPE_BLOCKED"
    except BaseException as exc:
        outcome = "IMPORT_ERROR"
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        PROFILE_LOCK = False
        sys.setprofile(None)
    require(env() == before_env, "Diagnostic TEMP/TMP/TMPDIR changed")
    source_files = {}
    for row in FIRST_EVENT["stack_root_to_leaf"] if FIRST_EVENT else []:
        p = Path(row["file"])
        if p.suffix == ".py" and p.is_file():
            source_files[str(p.resolve())] = digest(p)
    model_modules = sorted(n for n in sys.modules if n.split(".")[0] in HEAVY)
    write(OUT / "child_result.json", {
        "token": token, "outcome": outcome, "error": error, "imports": imports,
        "first_event": FIRST_EVENT, "elapsed_seconds": time.perf_counter() - began,
        "temporary_environment_before": before_env, "temporary_environment_after": env(),
        "loaded_dependency_module_names": model_modules,
        "loaded_dependency_module_count": len(model_modules),
        "package_metadata": versions(), "stack_source_sha256": source_files,
        "diagnostic_limits": {
            "not_a_replay_of_omitted_GPU_or_RNG_operations": True,
            "historic_terminal_environment_not_reconstructed": True,
            "no_target_event_does_not_clear_old_failure": True,
            "profile_and_audit_cover_Python_events_not_all_native_internals": True},
        "explicit_actions": {"build_model_calls": 0, "detector_or_optimizer_constructions": 0,
            "weight_loads": 0, "CUDA_calls": 0, "model_forwards": 0,
            "RGB_GT_array_probability_reads": 0, "generation_training_calibration_fitting": 0},
        "forbidden_attempts_are_in_first_event_not_successful_actions": True})
    return {"status": "IMPORT_DIAGNOSTIC_CHILD_STOPPED", "outcome": outcome}


def complete_files(result):
    names = PREP_FILES | {"started.json", "child_started.json", "child_result.json",
                          "stdout.txt", "stderr.txt", "result.json", "verification.json"}
    if result["first_event"] is not None:
        names.add("first_event.json")
    return names


def verify():
    verify_preparation(full=True)
    require((OUT / "complete.json").is_file(), "Partial/unknown diagnostic; do not resume")
    complete = read(OUT / "complete.json")
    result = read(OUT / "child_result.json")
    expected = complete_files(result)
    require(set(complete["outputs"]) == expected
            and {p.name for p in OUT.iterdir()} == expected | {"complete.json"}
            and all(p.is_file() and not p.is_symlink() and not p.is_junction() for p in OUT.iterdir())
            and all(digest(OUT / n) == h for n, h in complete["outputs"].items()),
            "Strict diagnostic transcript identity differs")
    started = read(OUT / "started.json")
    child_start = read(OUT / "child_started.json")
    require(started["token"] == child_start["token"] == result["token"],
            "Completed child token differs")
    require(started["diagnostic_child_launch_attempts"] == 1
            and result["temporary_environment_before"] == result["temporary_environment_after"]
                == started["temporary_environment"],
            "One-child/environment contract differs")
    if result["first_event"] is not None:
        require(read(OUT / "first_event.json") == result["first_event"], "First-event copy differs")
    for p, h in result["stack_source_sha256"].items():
        require(digest(p) == h, "Stack source identity changed")
    require(read(OUT / "verification.json")["protected_sources_verified_after_child"]
            and read(OUT / "result.json")["outcome"] == result["outcome"],
            "Final source/status proof differs")
    return {"status": "IMPORT_DIAGNOSIS_CACHE_VERIFIED", "output": str(OUT),
            "complete_sha256": digest(OUT / "complete.json"), "outcome": result["outcome"],
            "saved_diagnostic_child_launch_attempts": 1, "new_diagnostic_children": 0,
            "event_kind": result["first_event"]["kind"] if result["first_event"] else None,
            "explicit_detector_actions": result["explicit_actions"],
            "old_failure_remains_failed": True, "repair_or_model_compatibility_proven": False}


def run():
    verify_preparation(full=True)
    if (OUT / "complete.json").exists():
        return verify()
    require({p.name for p in OUT.iterdir()} == PREP_FILES,
            "Partial/unknown diagnostic transaction; preserve, do not retry")
    snapshot = resources()
    source_binding = read(OUT / "sources.json")
    token = uuid.uuid4().hex
    write(OUT / "started.json", {"token": token, "parent_pid": os.getpid(),
          "temporary_environment": env(), "diagnostic_child_launch_attempts": 1,
          "preparation_sha256": digest(OUT / "preparation_complete.json")})
    began = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m",
            "mura_capability.import_diagnosis", "_child", "--token", token],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120)
    except BaseException as exc:
        write(OUT / "parent_failure.json", {"type": type(exc).__name__, "message": str(exc),
              "diagnostic_child_launch_attempts": 1, "unknown_child_state_no_retry": True})
        raise
    for name, value in (("stdout.txt", completed.stdout), ("stderr.txt", completed.stderr)):
        with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    require(completed.returncode == 0 and (OUT / "child_result.json").is_file(),
            "Child incomplete/error; preserve transaction, no retry")
    result = read(OUT / "child_result.json")
    require(result["token"] == token, "Returned child token differs")
    require(day1_binding(full=True) == source_binding,
            "Protected sources changed during diagnostic")
    write(OUT / "result.json", {"outcome": result["outcome"], "diagnostic_only": True,
          "first_event": result["first_event"], "old_failure_not_repaired": True,
          "model_scoring_not_ready": True, "no_automatic_next_stage": True})
    write(OUT / "verification.json", {
          "protected_sources_verified_after_child": True,
          "source_binding_unchanged": True, "diagnostic_child_launch_attempts": 1,
          "child_returncode": completed.returncode, "resources_before_launch": snapshot,
          "elapsed_seconds": time.perf_counter() - began,
          "explicit_detector_actions": result["explicit_actions"],
          "interpretation": "complete means diagnostic transcript sealed, not repair/model compatibility"})
    names = complete_files(result)
    require({p.name for p in OUT.iterdir()} == names, "Unlisted diagnostic artifact")
    write(OUT / "complete.json", {"schema": "mura-import-diagnosis-transcript-complete-v1",
          "outputs": {n: digest(OUT / n) for n in sorted(names)}})
    return verify()


def main():
    require(sys.dont_write_bytecode, "Use python -B; bytecode writes are outside diagnosis")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "static-check", "run", "verify", "_child"))
    parser.add_argument("--token")
    args = parser.parse_args()
    if args.command == "_child":
        require(args.token is not None, "Internal child requires owned launch token")
        result = child(args.token)
    else:
        require(args.token is None, "Token is only for owned internal child")
        if args.command == "prepare":
            result = prepare()
        elif args.command == "static-check":
            result = verify_preparation(full=False)
        elif args.command == "run":
            result = run()
        else:
            result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
