"""Independent launch-identity preparation and one-shot import diagnosis.

Only prepare/inspect/static-check/verify are authorized by stages 1-2.
run-once additionally requires --allow-diagnostic-imports. No detector is built.
"""
from __future__ import annotations
import argparse
import ast
import copy
import ctypes
from ctypes import wintypes
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent.parent
BASE = ROOT / "results/mura_capability/week1_v1"
DAY1 = BASE / "day1_admission_v1"
OLD = ROOT / "results/mura_response/semu_applicability_v1"
LEGACY = BASE / "import_diagnosis_v1"
OUT = BASE / "import_diagnosis_launch_v1"
SPEC = PROJECT / "计划/Mura第2天A独立启动身份适配与导入诊断补充方案.md"
DAY1_SHA = "be272e26ff988e192d2d834bdb40b9ed2a9b6b2bd4c1e99e9f114c7730e2da1e"
OLD_CODE_SHA = "32e6a3972fba7c5b16a0495277d337873797783601b8fb05946269d15cbe59ed"
INIT_SHA = "1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa"
LEGACY_HASHES = {
 "preparation_complete.json":"14216ceeaaa4495f3e3f61f3204351f140cac086bc5e61e5c68b93c2faa31433",
 "protocol.json":"a92623429e2fe8295cf26fd40e6ac2c832b8d539f7832fd84506698ef82a6655",
 "sources.json":"c8897856743260a7673b5aa36dd9e6de6474392f97324aa88d51e7f2cc179942",
 "started.json":"3fbeabb030553ae0f067c255fbed70d070a48544b56a261927056347c5a5ce98",
 "stderr.txt":"9ed91183a756955080bc280ec68ca50cbba7edb34a83120034c005ded7f92de2",
 "stdout.txt":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
SEQUENCE = ("numpy","torch","faiss","torchvision.models","patchcore.common","patchcore.patchcore","patchcore.sampler")
HEAVY = ("numpy","scipy","PIL","torch","torchvision","faiss","timm","patchcore","huggingface_hub","safetensors","httpx2")
CORE_NAMES = ("Pillow","faiss-cpu","numpy","scipy","timm","torch","torchvision","tqdm")
EXTRA_NAMES = ("huggingface-hub","safetensors","httpx","httpx2","hf-xet")
ENV_NAMES = ("TEMP","TMP","TMPDIR")
PAYLOAD_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".gif",".webp",".npy",".npz",".pth",".pt",".pkl",".pickle",".faiss",".safetensors"}
FORBIDDEN_IMPORTS = ("mura_response","controlled_generation","detector","localization_training","localization_calibration","pod_probe","mura_capability.import_diagnosis")
PREP_FILES = {"sources.json","protocol.json","preparation_complete.json"}
FULL_FILES = PREP_FILES | {"started.json","popen_observation.json","child_boot.json",
 "child_identity.json","parent_identity.json","decision.json","import_permission.json",
 "child_result.json","stdout.txt","stderr.txt","result.json","verification.json"}
MODEL_ASSETS = {str((ROOT/p).resolve()) for p in (
 "models/checkpoints/wide_resnet50_2-95faca4d.pth",
 "models/weak_localization/patchcore/nnscorer_search_index.faiss",
 "models/weak_localization/patchcore/patchcore_params.pkl",
 "models/controlled_generation/formal_draem/G1_20261030/reconstruction.pth",
 "models/controlled_generation/formal_draem/G1_20261030/segmentation.pth")}
SCHEMA = "mura-launch-handshake-v1"
TOTAL_SECONDS = 120.
HELLO_SECONDS = 15.
GO_SECONDS = 20.
MAX_LINE = 65536
MAX_STREAM = 4*1024*1024
MODE = None
CURRENT_IMPORT = None
FIRST_EVENT = None
PROFILE_LOCK = False
DIAG_DEPENDENCIES = None


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


def own_implementation():
    return {str(Path(__file__).resolve()): digest(__file__),
            str(ROOT / "mura_capability/__init__.py"): digest(ROOT / "mura_capability/__init__.py")}


def _day1_binding(full=False):
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
                       "temporary_environment": env(), "dependency_metadata": DIAG_DEPENDENCIES}
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


def norm(path):
    return os.path.normcase(str(Path(path).resolve())) if isinstance(path,str) and path else None

def doc_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def mapping(value):
    return value if isinstance(value,dict) else {}

def known_equal(a,b):
    return None if a is None or b is None else a==b

def all_true(checks):
    return bool(checks) and all(v is True for v in checks.values())

def interpreter_identity():
    base = str(Path(sys._base_executable).resolve()) if getattr(sys,"_base_executable",None) else None
    launcher = Path(sys.base_prefix)/"Lib/venv/scripts/nt/python.exe"
    return {"executable":norm(sys.executable),"executable_sha256":digest(sys.executable),
      "base_executable":norm(base),"base_executable_sha256":digest(base) if base else None,
      "prefix":norm(sys.prefix),"base_prefix":norm(sys.base_prefix),
      "python_version":sys.version.split()[0],"pyvenv_cfg_sha256":digest(ROOT/".venv/pyvenv.cfg"),
      "launcher":norm(str(launcher)),"launcher_sha256":digest(launcher),
      "base_binary_hash_is_current_observation_not_historical_certificate":True}

def binding(full=False):
    result = _day1_binding(full=full)
    require(digest(ROOT/"mura_capability/import_diagnosis.py")==OLD_CODE_SHA,"Legacy diagnostic source differs")
    require(digest(ROOT/"mura_capability/__init__.py")==INIT_SHA,"Legacy initialization differs")
    require({p.name for p in LEGACY.iterdir()}==set(LEGACY_HASHES),"Legacy six-file inventory differs")
    require(all(digest(LEGACY/n)==h for n,h in LEGACY_HASHES.items()),"Legacy diagnostic field/hash differs")
    require(read(LEGACY/"started.json")["diagnostic_child_launch_attempts"]==1
      and not (LEGACY/"complete.json").exists(),"Legacy budget/state differs")
    text=SPEC.read_text(encoding="utf-8")
    core=text.split("## 二、",1)[1].split("## 五、",1)[0]
    result.update({"legacy_diagnostic_sha256":LEGACY_HASHES,"legacy_code_sha256":OLD_CODE_SHA,
      "interpreter_identity":interpreter_identity(),"spec_path":str(SPEC),
      "spec_rules_sha256":hashlib.sha256(core.encode()).hexdigest(),
      "mutable_progress_not_an_identity_lock":True})
    return result

def kernel_allowed(meta,interpreter):
    m=mapping(meta)
    if m.get("image_path") is None or m.get("image_sha256") is None:return None
    pairs={(interpreter.get("executable"),interpreter.get("executable_sha256")),
           (interpreter.get("base_executable"),interpreter.get("base_executable_sha256"))}
    return (m["image_path"],m["image_sha256"]) in pairs

def child_checks(boot,launch,source,received_token,expected_started_sha):
    boot,launch,source=map(mapping,(boot,launch,source))
    actual_interpreter=mapping(boot.get("interpreter"))
    wanted=mapping(source.get("interpreter_identity"))
    token_hash=hashlib.sha256(received_token.encode()).hexdigest() if isinstance(received_token,str) else None
    checks={"token_match":known_equal(token_hash,launch.get("token_sha256")),
      "implementation_match":known_equal(boot.get("implementation"),source.get("implementation")),
      "source_file_match":known_equal(boot.get("sources_sha256"),launch.get("sources_sha256")),
      "protocol_file_match":known_equal(boot.get("protocol_sha256"),launch.get("protocol_sha256")),
      "started_file_match":known_equal(boot.get("started_sha256"),expected_started_sha),
      "temporary_environment_match":known_equal(boot.get("environment"),launch.get("environment"))}
    for key in wanted:
        checks["interpreter_"+key]=known_equal(actual_interpreter.get(key),wanted[key])
    kernel=mapping(boot.get("kernel"))
    checks["self_kernel_pid_match"]=known_equal(kernel.get("pid"),boot.get("pid"))
    checks["self_kernel_image_allowed"]=kernel_allowed(kernel,wanted)
    checks["self_kernel_alive"]=known_equal(kernel.get("alive"),True)
    return checks

def evaluate_identity(observations,source):
    """Pure comparison. Alternative relation checks are OR, not universal AND."""
    o,source=mapping(observations),mapping(source)
    boot=mapping(o.get("boot"));launch=mapping(o.get("launch"));p=mapping(o.get("popen"))
    child=mapping(o.get("parent_view_child"));controller=mapping(o.get("controller"))
    interp=mapping(source.get("interpreter_identity"));self_kernel=mapping(boot.get("kernel"))
    common=dict(mapping(o.get("child_checks")))
    common.update({"controller_pid_match":known_equal(controller.get("pid"),launch.get("controller_pid")),
      "popen_handle_pid_match":known_equal(p.get("pid"),p.get("returned_pid")),
      "popen_alive":known_equal(p.get("alive"),True),
      "popen_image_allowed":kernel_allowed(p,interp),
      "reported_child_pid_matches_handle":known_equal(boot.get("pid"),child.get("pid")),
      "child_alive":known_equal(child.get("alive"),True),
      "child_kernel_image_allowed":kernel_allowed(child,interp),
      "same_child_creation_time":known_equal(child.get("created_100ns"),self_kernel.get("created_100ns")),
      "same_child_image_path":known_equal(child.get("image_path"),self_kernel.get("image_path")),
      "same_child_image_hash":known_equal(child.get("image_sha256"),self_kernel.get("image_sha256")),
      "launch_record_unchanged":known_equal(o.get("launch_record_unchanged"),True)})
    if not mapping(o.get("child_checks")):common["child_checks_available"]=None
    for name,value in (("controller_pid_positive",controller.get("pid")),
                       ("popen_pid_positive",p.get("pid")),("child_pid_positive",boot.get("pid")),
                       ("child_ppid_positive",boot.get("ppid")),
                       ("popen_creation_known",p.get("created_100ns")),
                       ("child_creation_known",child.get("created_100ns"))):
        common[name]=None if value is None else type(value) is int and value>0
    direct={"child_pid_equals_popen":known_equal(boot.get("pid"),p.get("pid")),
      "child_ppid_equals_controller":known_equal(boot.get("ppid"),controller.get("pid")),
      "same_creation_time":known_equal(child.get("created_100ns"),p.get("created_100ns"))}
    distinct=None if boot.get("pid") is None or p.get("pid") is None else boot["pid"]!=p["pid"]
    ordering=None if child.get("created_100ns") is None or p.get("created_100ns") is None else child["created_100ns"]>=p["created_100ns"]
    launcher={"child_pid_distinct":distinct,"child_ppid_equals_popen":known_equal(boot.get("ppid"),p.get("pid")),
      "popen_is_bound_venv_launcher":None if p.get("image_path") is None else (
        p["image_path"]==interp.get("executable") and p.get("image_sha256")==interp.get("launcher_sha256")
        and interp.get("launcher_sha256")==interp.get("executable_sha256")),
      "child_is_bound_base":None if child.get("image_path") is None else (
        child["image_path"]==interp.get("base_executable") and child.get("image_sha256")==interp.get("base_executable_sha256")),
      "creation_order_known":ordering}
    mode="DIRECT" if all_true(common) and all_true(direct) else (
         "LAUNCHER_ONE_HOP" if all_true(common) and all_true(launcher) else "REJECT")
    return {"common_checks":common,"DIRECT":direct,"LAUNCHER_ONE_HOP":launcher,
      "relation":mode,"identity_passed":mode!="REJECT",
      "rule":"all common AND (one complete DIRECT OR one complete LAUNCHER_ONE_HOP)",
      "failed_or_unknown_common":[k for k,v in common.items() if v is not True]}

def validate_message(message,kind,session,protocol_sha,elapsed,deadline):
    m=mapping(message)
    return {"schema":known_equal(m.get("schema"),SCHEMA),"kind":known_equal(m.get("kind"),kind),
      "session":known_equal(m.get("session"),session),"protocol":known_equal(m.get("protocol_sha256"),protocol_sha),
      "body":isinstance(m.get("body"),dict),
      "deadline":None if elapsed is None else isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and 0<=elapsed<=deadline}

def permission_gate(identity,message_checks,records_saved,hello_count=1):
    return records_saved is True and hello_count==1 and mapping(identity).get("identity_passed") is True and all_true(message_checks)

def record_identity(path,observation,source):
    evaluated=evaluate_identity(observation,source)
    write(path,{"observations":observation,"evaluation":evaluated})
    return evaluated

def control(kind,session,protocol_sha,body):
    return {"schema":SCHEMA,"kind":kind,"session":session,"protocol_sha256":protocol_sha,"body":body}

def static_contract():
    """Artificial observations/events only. No process API, pipes or imports."""
    interp={"executable":"v:/venv/python.exe","executable_sha256":"1"*64,
      "base_executable":"v:/base/python.exe","base_executable_sha256":"2"*64,
      "prefix":"v:/venv","base_prefix":"v:/base","python_version":"3.12.7",
      "pyvenv_cfg_sha256":"3"*64,"launcher":"v:/stdlib/python.exe","launcher_sha256":"1"*64,
      "base_binary_hash_is_current_observation_not_historical_certificate":True}
    source={"interpreter_identity":interp,"implementation":{"virtual.py":"4"*64}}
    nonce="virtual-nonce-not-a-live-token";session=hashlib.sha256(nonce.encode()).hexdigest()
    launch={"controller_pid":100,"controller_ppid":50,"token_sha256":session,
      "environment":{"TEMP":"virtual","TMP":"virtual","TMPDIR":None},
      "sources_sha256":"5"*64,"protocol_sha256":"6"*64}
    kernel={"pid":200,"image_path":interp["executable"],"image_sha256":"1"*64,"created_100ns":20,"alive":True,"error":None}
    boot={"pid":200,"ppid":100,"kernel":copy.deepcopy(kernel),"implementation":source["implementation"],
      "interpreter":interp,"environment":launch["environment"],"sources_sha256":"5"*64,
      "protocol_sha256":"6"*64,"started_sha256":"7"*64}
    checks=child_checks(boot,launch,source,nonce,"7"*64)
    o={"controller":{"pid":100,"ppid":50},"launch":launch,
      "popen":dict(kernel,returned_pid=200),"boot":boot,"parent_view_child":copy.deepcopy(kernel),
      "child_checks":checks,"launch_record_unchanged":True}
    cases={}
    cases["DIRECT"]=evaluate_identity(o,source)["relation"]
    l=copy.deepcopy(o);l["boot"].update(pid=300,ppid=200)
    lk={"pid":300,"image_path":interp["base_executable"],"image_sha256":"2"*64,"created_100ns":21,"alive":True,"error":None}
    l["boot"]["kernel"]=lk;l["parent_view_child"]=copy.deepcopy(lk)
    l["child_checks"]=child_checks(l["boot"],launch,source,nonce,"7"*64)
    cases["LAUNCHER_ONE_HOP"]=evaluate_identity(l,source)["relation"]
    bad=copy.deepcopy(l);bad["child_checks"]["token_match"]=False
    cases["wrong_token"]=evaluate_identity(bad,source)["relation"]
    bad=copy.deepcopy(l);bad["boot"]["ppid"]=400
    cases["unknown_ancestor"]=evaluate_identity(bad,source)["relation"]
    bad=copy.deepcopy(l);bad["parent_view_child"]["created_100ns"]=None
    cases["unknown_handle_identity"]=evaluate_identity(bad,source)["relation"]
    bad=copy.deepcopy(l);bad["child_checks"]["interpreter_prefix"]=False
    cases["wrong_interpreter"]=evaluate_identity(bad,source)["relation"]
    good=evaluate_identity(l,source);msg=control("HELLO",session,"6"*64,{})
    mc=validate_message(msg,"HELLO",session,"6"*64,1,HELLO_SECONDS)
    cases["permission_requires_saved_log"]=permission_gate(good,mc,False)
    cases["permission_rejects_duplicate_HELLO"]=permission_gate(good,mc,True,2)
    cases["permission_GO_valid"]=permission_gate(good,mc,True)
    for label,message,elapsed in (
      ("malformed_HELLO",None,1),("wrong_session",control("HELLO","bad","6"*64,{}),1),
      ("late_HELLO",msg,16),("ABORT_is_not_GO",control("ABORT",session,"6"*64,{}),1),
      ("missing_GO",None,1)):
        kind="GO" if "GO" in label else "HELLO"
        cases[label]=all_true(validate_message(message,kind,session,"6"*64,elapsed,15))
    flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,"O_BINARY",0)
    cases["external_flags_write"]=classify_open(ROOT/"virtual_outside",None,flags)
    cases["image_read"]=classify_open(ROOT/"virtual.png","r",0)
    cases["array_read"]=classify_open(ROOT/"virtual.npy","rb",0)
    cases["weight_read"]=classify_open(ROOT/"virtual.pth","rb",0)
    cases["CUDA_call"]=forbidden_call("v:/torch/cuda/__init__.py","is_available")
    cases["model_call"]=forbidden_call("v:/torch/nn/modules/module.py","__init__")
    cases["legacy_PID_predicate_would_reject_launcher"]=(200!=100)
    expected={"DIRECT":"DIRECT","LAUNCHER_ONE_HOP":"LAUNCHER_ONE_HOP","wrong_token":"REJECT",
      "unknown_ancestor":"REJECT","unknown_handle_identity":"REJECT","wrong_interpreter":"REJECT",
      "permission_requires_saved_log":False,"permission_rejects_duplicate_HELLO":False,
      "permission_GO_valid":True,"malformed_HELLO":False,"wrong_session":False,"late_HELLO":False,
      "ABORT_is_not_GO":False,"missing_GO":False,"external_flags_write":"OUTSIDE_WRITE",
      "image_read":"FORBIDDEN_PAYLOAD_OPEN","array_read":"FORBIDDEN_PAYLOAD_OPEN",
      "weight_read":"FORBIDDEN_PAYLOAD_OPEN","CUDA_call":"FORBIDDEN_CUDA_CALL",
      "model_call":"FORBIDDEN_MODULE_CONSTRUCTION_OR_FORWARD","legacy_PID_predicate_would_reject_launcher":True}
    require(cases==expected,"Artificial identity/permission/event case differs")
    # Additional refusal and review regressions use dictionaries/source AST only.
    extra={}
    for field in ("source_file_match","temporary_environment_match"):
        bad=copy.deepcopy(l);bad["child_checks"][field]=False
        extra[field]=evaluate_identity(bad,source)["relation"]=="REJECT"
    extra["unknown_field_refused"]=not all_true({"identity":None})
    virtual={"boot":{},"child_identity":{"boot":{},"checks":{"ok":True}},
      "parent":{"observations":{"boot":{}},"evaluation":{"identity_passed":True}},
      "decision":{"action":"GO","identity":{"identity_passed":True},"message_checks":{"ok":True},"unexpected_queued_message":False,"elapsed_seconds":1},
      "permission":{"permitted":True,"checks":{"ok":True},"own_checks":{"ok":True}},
      "launch":{"submitted_monotonic_ns":0,"entry_deadline_ns":20},
      "child_result":{"entered_imports":True,"import_entry_monotonic_ns":10,"first_event":None,"status":"IMPORTS_FINISHED_NO_EVENT","imports":[]},
      "result":{"elapsed_seconds":100},"event":None,"event_exists":False}
    virtual["result"]["decision"]=copy.deepcopy(virtual["decision"])
    extra["closed_virtual_transcript"]=all_true(transcript_consistency(virtual))
    for field,value in (("import_entry_monotonic_ns",21),("first_event",{"incomplete":True})):
        bad=copy.deepcopy(virtual);bad["child_result"][field]=value
        extra["reject_"+field]=not all_true(transcript_consistency(bad))
    bad=copy.deepcopy(virtual);bad["permission"]["permitted"]=False
    extra["reject_permission_contradiction"]=not all_true(transcript_consistency(bad))
    bad=copy.deepcopy(virtual);bad["result"]["elapsed_seconds"]=121
    extra["reject_excess_total_budget"]=not all_true(transcript_consistency(bad))
    extra["existing_launch_refused"]=not ({"sources.json","protocol.json","preparation_complete.json","started.json"}==PREP_FILES)
    require(all_true(extra),"Additional artificial refusal/consistency case differs")
    code=Path(__file__).read_text(encoding="utf-8");compile(code,str(__file__),"exec")
    tree=ast.parse(code);popen_calls=[];bad_calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if node.func.value.id=="subprocess" and node.func.attr=="Popen":popen_calls.append(node.lineno)
            if node.func.value.id=="subprocess" and node.func.attr in {"run","call","check_call","check_output"}:bad_calls.append(node.lineno)
    require(len(popen_calls)==1 and not bad_calls,"One-Popen source structure differs")
    require(not any(n.split(".")[0] in HEAVY for n in sys.modules),"Static process imported a prohibited package")
    return {"artificial_case_count":len(cases)+len(extra),"cases":cases,"additional_cases":extra,"syntax_compiled":True,
      "AST_Popen_call_sites":popen_calls,"actual_process_API_calls":0,"new_diagnostic_children":0,
      "heavy_imports":0,"real_file_operations_in_artificial_cases":0,
      "dynamic_handshake_and_import_compatibility_verified":False}

def kernel_api():
    require(MODE in ("run-once","_child"),"Process API is dynamic-stage only")
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    signatures={
      "GetCurrentProcess":([],wintypes.HANDLE),
      "GetProcessId":([wintypes.HANDLE],wintypes.DWORD),
      "QueryFullProcessImageNameW":([wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)],wintypes.BOOL),
      "GetProcessTimes":([wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4,wintypes.BOOL),
      "GetExitCodeProcess":([wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)],wintypes.BOOL),
      "OpenProcess":([wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
      "CloseHandle":([wintypes.HANDLE],wintypes.BOOL),
      "TerminateProcess":([wintypes.HANDLE,wintypes.UINT],wintypes.BOOL)}
    for name,(args,ret) in signatures.items():
        f=getattr(k,name);f.argtypes=args;f.restype=ret
    return k

def observe_process(handle,interpreter):
    k=kernel_api()
    result={"pid":None,"image_path":None,"image_sha256":None,"created_100ns":None,
      "alive":None,"exit_code":None,"error":[]}
    pid=k.GetProcessId(handle)
    if pid:result["pid"]=int(pid)
    else:result["error"].append("GetProcessId:"+str(ctypes.get_last_error()))
    size=wintypes.DWORD(32768);buf=ctypes.create_unicode_buffer(size.value)
    if k.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)):
        result["image_path"]=norm(buf.value)
        if result["image_path"] in (interpreter["executable"],interpreter["base_executable"]):
            try:result["image_sha256"]=digest(buf.value)
            except OSError as e:result["error"].append("image hash:"+str(e))
        else:result["error"].append("Image outside the two bound executable paths")
    else:result["error"].append("QueryFullProcessImageNameW:"+str(ctypes.get_last_error()))
    times=[wintypes.FILETIME() for _ in range(4)]
    if k.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):
        result["created_100ns"]=(int(times[0].dwHighDateTime)<<32)|times[0].dwLowDateTime
    else:result["error"].append("GetProcessTimes:"+str(ctypes.get_last_error()))
    exit_code=wintypes.DWORD()
    if k.GetExitCodeProcess(handle,ctypes.byref(exit_code)):
        result["exit_code"]=int(exit_code.value);result["alive"]=exit_code.value==259
    else:result["error"].append("GetExitCodeProcess:"+str(ctypes.get_last_error()))
    return result

class PipeReader:
    """One thread per owned pipe, bounded transcript and control queue."""
    def __init__(self,stream,parse_lines):
        self.stream=stream;self.parse_lines=parse_lines;self.data=bytearray()
        self.messages=queue.Queue(maxsize=128);self.error=None;self.done=False
        self.thread=threading.Thread(target=self._read,daemon=True)
    def start(self):self.thread.start()
    def _read(self):
        pending=bytearray()
        try:
            while True:
                block=self.stream.read(1 if self.parse_lines else 4096)
                if not block:break
                if len(self.data)+len(block)>MAX_STREAM:
                    self.error="stream over 4MiB";break
                self.data.extend(block)
                if self.parse_lines:
                    pending.extend(block)
                    if len(pending)>MAX_LINE:
                        self.error="control/output line over 64KiB";break
                    if block==b"\n":
                        line=bytes(pending);pending.clear()
                        try:obj=json.loads(line.decode("utf-8"))
                        except (ValueError,UnicodeError):obj={"unstructured_stdout":line.decode("utf-8",errors="replace")}
                        try:self.messages.put_nowait(obj)
                        except queue.Full:self.error="control queue overflow";break
            if pending:self.error=self.error or "unterminated stdout line"
        except BaseException as e:self.error=type(e).__name__+":"+str(e)
        finally:self.done=True

def send_message(stream,message):
    payload=(json.dumps(message,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode()
    require(len(payload)<=MAX_LINE,"Control message over limit")
    stream.write(payload);stream.flush()

def receive_line(stream,seconds):
    """Child bounded stdin wait; no additional process."""
    q=queue.Queue(maxsize=1)
    def reader():
        try:
            data=stream.readline(MAX_LINE+1)
            if not data or len(data)>MAX_LINE or not data.endswith(b"\n"):
                q.put({"error":"EOF/oversize/unterminated control"});return
            q.put({"message":json.loads(data.decode("utf-8"))})
        except BaseException as e:q.put({"error":type(e).__name__+":"+str(e)})
    threading.Thread(target=reader,daemon=True).start()
    try:return q.get(timeout=max(0.001,seconds))
    except queue.Empty:return {"error":"control wait timeout"}

def child_entry():
    global CURRENT_IMPORT,PROFILE_LOCK,DIAG_DEPENDENCIES
    child_began=time.monotonic()
    startup_errors={}
    def observed(name,fn):
        try:return fn()
        except Exception as e:
            startup_errors[name]=type(e).__name__+":"+str(e);return None
    boot={"pid":os.getpid(),"ppid":os.getppid()}
    source=mapping(observed("sources",lambda:read(OUT/"sources.json")))
    protocol=mapping(observed("protocol",lambda:read(OUT/"protocol.json")))
    launch=mapping(observed("launch",lambda:read(OUT/"started.json")))
    started_sha=observed("started_hash",lambda:digest(OUT/"started.json"))
    for name,fn in {
      "kernel":lambda:observe_process(kernel_api().GetCurrentProcess(),source["interpreter_identity"]),
      "interpreter":interpreter_identity,"implementation":own_implementation,"environment":env,
      "sources_sha256":lambda:digest(OUT/"sources.json"),
      "protocol_sha256":lambda:digest(OUT/"protocol.json")}.items():
        boot[name]=observed(name,fn)
    boot.update(started_sha256=started_sha,startup_errors=startup_errors)
    write(OUT/"child_boot.json",boot)
    require(not startup_errors,"Child startup observations incomplete; boot retained")
    DIAG_DEPENDENCIES=source["package_metadata"]
    session=launch["token_sha256"];psha=digest(OUT/"protocol.json")
    received=receive_line(sys.stdin.buffer,max(.001,HELLO_SECONDS-(time.monotonic()-child_began)))
    challenge=mapping(received.get("message"))
    cm=validate_message(challenge,"CHALLENGE",session,psha,0,HELLO_SECONDS)
    body=mapping(challenge.get("body"))
    checks=child_checks(boot,launch,source,body.get("token"),started_sha)
    checks.update({"challenge_"+n:v for n,v in cm.items()})
    checks["challenge_controller_pid"]=known_equal(body.get("controller_pid"),launch.get("controller_pid"))
    write(OUT/"child_identity.json",{"boot":boot,"checks":checks,
      "challenge_error":received.get("error"),"raw_token_saved":False})
    send_message(sys.stdout.buffer,control("HELLO",session,psha,{"boot":boot,"checks":checks}))
    received_go=receive_line(sys.stdin.buffer,max(.001,GO_SECONDS-(time.monotonic()-child_began)))
    go=mapping(received_go.get("message"));gb=mapping(go.get("body"))
    gc=validate_message(go,"GO",session,psha,gb.get("elapsed_seconds"),GO_SECONDS)
    for name in ("decision.json","parent_identity.json"):
        gc[name+"_hash"]=known_equal(digest(OUT/name) if (OUT/name).is_file() else None,
                                      gb.get(name+"_sha256"))
    decision=read(OUT/"decision.json") if (OUT/"decision.json").is_file() else {}
    parent=read(OUT/"parent_identity.json") if (OUT/"parent_identity.json").is_file() else {}
    gc["parent_decision_GO"]=known_equal(decision.get("action"),"GO")
    gc["parent_identity_passed"]=known_equal(mapping(parent.get("evaluation")).get("identity_passed"),True)
    gc["launch_still_identical"]=known_equal(digest(OUT/"started.json"),started_sha)
    entry_deadline_ns=launch["entry_deadline_ns"]
    gc["entry_deadline_live"]=time.monotonic_ns()<=entry_deadline_ns
    permitted=all_true(checks) and all_true(gc)
    write(OUT/"import_permission.json",{"checks":gc,"own_checks":checks,
      "permitted":permitted,"error":received_go.get("error"),"before_any_dependency_import":True,
      "entry_deadline_ns":entry_deadline_ns})
    result={"status":"IDENTITY_OR_PERMISSION_REJECTED","entered_imports":False,
      "imports":[],"first_event":None,"omitted":["CUDA queries","GPU/RNG seeds","thread/TF32 settings","build_model"],
      "prohibited_model_forward_training_or_data_calls_requested":0}
    entry_ns=time.monotonic_ns()
    result["import_entry_monotonic_ns"]=entry_ns if permitted and entry_ns<=entry_deadline_ns else None
    if permitted and entry_ns<=entry_deadline_ns:
        # Guards have already been installed at main entry, before all non-stdlib imports.
        sys.meta_path.insert(0,ChildImportGuard());sys.setprofile(child_profile);PROFILE_LOCK=True
        result["entered_imports"]=True;result["status"]="IMPORTS_FINISHED_NO_EVENT"
        try:
            for module_name in SEQUENCE:
                CURRENT_IMPORT=module_name
                if module_name=="patchcore.common":
                    sys.path.insert(0,str(ROOT/"vendor/patchcore-inspection/src"))
                module=importlib.import_module(module_name)
                if module_name=="torchvision.models":
                    getattr(module,"wide_resnet50_2")  # symbol only, never instantiate
                result["imports"].append(module_name)
        except StopImport as e:
            result["status"]="FIRST_GUARD_EVENT";result["stop_reason"]=str(e)
        except BaseException as e:
            result["status"]="IMPORT_EXCEPTION";result["exception"]=type(e).__name__+":"+str(e)
        finally:
            PROFILE_LOCK=False;sys.setprofile(None);CURRENT_IMPORT=None
        result["first_event"]=FIRST_EVENT
    result["implementation"]=own_implementation();result["environment_unchanged"]=env()==launch["environment"]
    write(OUT/"child_result.json",result)
    send_message(sys.stdout.buffer,control("DONE",session,psha,{"status":result["status"]}))
    return result

def read_preparation(full=False):
    require((OUT/"preparation_complete.json").is_file(),"Preparation not sealed")
    seal=read(OUT/"preparation_complete.json")
    require(all((OUT/n).is_file() and digest(OUT/n)==h for n,h in seal["outputs"].items()),
            "Preparation file identity differs")
    require(read(OUT/"sources.json")==binding(full=full),"Current source/implementation identity differs")
    names={p.name for p in OUT.iterdir()}
    require(names<=FULL_FILES|{"first_event.json","complete.json"},"Unknown output member")
    require(all(p.is_file() and not p.is_symlink() for p in OUT.iterdir()),"Output redirect/directory found")
    return read(OUT/"sources.json"),read(OUT/"protocol.json")

def prepare_protocol():
    source=binding(full=True);snapshot=resources();static=static_contract()
    if OUT.exists():
        read_preparation(full=True)
        require({p.name for p in OUT.iterdir()}==PREP_FILES,"Prepare cannot consume a started transaction")
        return {"status":"PREPARATION_VERIFIED_REUSED","new_files":0,"static":static,"resources":snapshot}
    require(not OUT.is_symlink(),"Redirected output root")
    OUT.mkdir()
    protocol={"schema":SCHEMA,"implementation":source["implementation"],
      "single_new_child_budget":1,"budget_consumed_by_started_record":True,
      "legacy_budget_consumed":1,"legacy_process_chain":"UNKNOWN",
      "identity_rule":"common AND (DIRECT OR LAUNCHER_ONE_HOP)",
      "message_schema":SCHEMA,"max_line_bytes":MAX_LINE,"max_stream_bytes":MAX_STREAM,
      "total_seconds_from_Popen_submission":TOTAL_SECONDS,"HELLO_deadline_seconds":HELLO_SECONDS,
      "GO_permission_deadline_seconds":GO_SECONDS,"dependency_order":list(SEQUENCE),
      "environment_inherited_no_TEMP_TMP_change":True,"current_environment":env(),
      "legacy_environment":read(LEGACY/"started.json").get("temporary_environment"),
      "no_model_or_CUDA_or_data_permission":True,"resources":snapshot,
      "preparation_static_rules":static,"expected_preparation_files":sorted(PREP_FILES),
      "expected_runtime_files":sorted(FULL_FILES),"optional_runtime_file":"first_event.json",
      "complete_only_after_known_owned_exits_and_all_records":True,
      "dynamic_compatibility_verified":False}
    write(OUT/"sources.json",source);write(OUT/"protocol.json",protocol)
    write(OUT/"preparation_complete.json",{"status":"PREPARATION_ONLY",
      "outputs":{n:digest(OUT/n) for n in ("sources.json","protocol.json")},
      "dynamic_child_launches":0,"dependency_imports":0})
    return {"status":"PREPARATION_COMPLETE","directory":str(OUT),"static":static,"resources":snapshot,
      "preparation_complete_sha256":digest(OUT/"preparation_complete.json")}

def static_check():
    source,protocol=read_preparation(full=True)
    require(not (OUT/"started.json").exists(),"Static preparation check will not reinterpret a started transaction")
    return {"status":"STATIC_PREPARATION_VERIFIED","complete_sha256":digest(OUT/"preparation_complete.json"),
      "implementation":source["implementation"],"static":static_contract(),"resources":resources(),
      "new_diagnostic_children":0,"dependency_imports":0,"actual_process_identity_observations":0}

def run_once(allow_diagnostic_imports=False):
    require(allow_diagnostic_imports is True,"Explicit diagnostic-import flag required")
    source,protocol=read_preparation(full=True);snapshot=resources()
    if (OUT/"complete.json").exists():return verify_transcript()
    require({p.name for p in OUT.iterdir()}==PREP_FILES,"Budget/partial/unknown transaction; no second Popen")
    token=os.urandom(32).hex();session=hashlib.sha256(token.encode()).hexdigest()
    submitted_ns=time.monotonic_ns()
    launch={"controller_pid":os.getpid(),"controller_ppid":os.getppid(),
      "submitted_monotonic_ns":submitted_ns,"entry_deadline_ns":submitted_ns+int(GO_SECONDS*1e9),
      "clock_info":str(time.get_clock_info("monotonic")),
      "token_sha256":session,"sources_sha256":digest(OUT/"sources.json"),
      "protocol_sha256":digest(OUT/"protocol.json"),"environment":env(),
      "diagnostic_child_launch_attempts":1,"time_ns":time.time_ns(),"resources":snapshot}
    write(OUT/"started.json",launch);started_sha=digest(OUT/"started.json")
    proc=None;child_handle=None;child_handle_separate=False;child_owned=False
    out_reader=None;err_reader=None;failure=None;decision=None;exit_observations={}
    began=submitted_ns/1e9;deadline=began+TOTAL_SECONDS;hello_count=0
    try:
        proc=subprocess.Popen([str(Path(sys.executable).resolve()),"-B","-X","utf8",
          "-m","mura_capability.import_diagnosis_launch","_child"],
          cwd=str(ROOT),env=None,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
          close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
        popen=observe_process(proc._handle,source["interpreter_identity"])
        popen["returned_pid"]=proc.pid
        write(OUT/"popen_observation.json",{"controller":{"pid":os.getpid(),"ppid":os.getppid()},
          "popen":popen,"elapsed_seconds":time.monotonic()-began})
        out_reader=PipeReader(proc.stdout,True);err_reader=PipeReader(proc.stderr,False)
        out_reader.start();err_reader.start()
        require(time.monotonic()-began<HELLO_SECONDS,"Popen returned beyond HELLO deadline")
        send_message(proc.stdin,control("CHALLENGE",session,launch["protocol_sha256"],
          {"token":token,"controller_pid":os.getpid()}))
        hello=None
        while time.monotonic()-began<=HELLO_SECONDS:
            require(not out_reader.error and not err_reader.error,"Pipe overflow/read failure")
            try:message=out_reader.messages.get(timeout=min(.1,max(.001,HELLO_SECONDS-(time.monotonic()-began))))
            except queue.Empty:
                require(proc.poll() is None,"Child exited before HELLO");continue
            if mapping(message).get("kind")!="HELLO":
                hello=message;break  # record failed message comparisons before rejection
            hello_count+=1;hello=message;break
        boot=mapping(mapping(hello).get("body")).get("boot")
        checks=mapping(mapping(hello).get("body")).get("checks")
        mc=validate_message(hello,"HELLO",session,launch["protocol_sha256"],time.monotonic()-began,HELLO_SECONDS)
        # Independently recompute nonce/source checks from saved boot. Never trust booleans in HELLO alone.
        saved_child=read(OUT/"child_identity.json") if (OUT/"child_identity.json").is_file() else {}
        independent=child_checks(boot,launch,source,token,started_sha);token=None
        independent["HELLO_checks_match_saved"]=checks==saved_child.get("checks")
        independent["HELLO_matches_saved_child"]=mapping(saved_child).get("boot")==boot and mapping(saved_child).get("checks")==checks
        independent["saved_child_checks_all_true"]=all_true(mapping(saved_child.get("checks")))
        independent["launch_file_unchanged"]=digest(OUT/"started.json")==started_sha
        own_boot=mapping(boot);candidate=own_boot.get("pid")
        prelim=(all_true(mc) and all_true(independent) and type(candidate) is int and candidate>0
          and (candidate==proc.pid or own_boot.get("ppid")==proc.pid))
        if candidate==proc.pid and prelim:
            child_handle=proc._handle
        elif prelim:
            child_handle=kernel_api().OpenProcess(0x1000|0x100000|0x0001,False,candidate)
            child_handle_separate=bool(child_handle)
        child_view=observe_process(child_handle,source["interpreter_identity"]) if child_handle else {
          "pid":None,"image_path":None,"image_sha256":None,"created_100ns":None,"alive":None,
          "error":["No finite, source-bound candidate manageable handle"]}
        observation={"controller":{"pid":os.getpid(),"ppid":os.getppid()},"launch":launch,"popen":popen,
          "boot":boot,"parent_view_child":child_view,"child_checks":independent,
          "launch_record_unchanged":digest(OUT/"started.json")==started_sha}
        evaluated=record_identity(OUT/"parent_identity.json",observation,source)
        child_owned=evaluated["identity_passed"]
        # A second queued HELLO cannot silently become dependency output.
        duplicate=not out_reader.messages.empty()
        go=permission_gate(evaluated,mc,True,hello_count+int(duplicate)) and time.monotonic()-began<=GO_SECONDS
        decision={"action":"GO" if go else "ABORT","identity":evaluated,
          "message_checks":mc,"hello_count":hello_count,"unexpected_queued_message":duplicate,
          "elapsed_seconds":time.monotonic()-began,"all_logs_saved_before_decision":True}
        write(OUT/"decision.json",decision)
        send_message(proc.stdin,control(decision["action"],session,launch["protocol_sha256"],
          {"decision.json_sha256":digest(OUT/"decision.json"),
           "parent_identity.json_sha256":digest(OUT/"parent_identity.json"),
           "elapsed_seconds":time.monotonic()-began}))
        def owned_child_alive_independent():
            return bool(child_owned and child_handle and observe_process(child_handle,source["interpreter_identity"])["alive"] is True)
        while time.monotonic()<deadline and (proc.poll() is None or owned_child_alive_independent()):
            require(not out_reader.error and not err_reader.error,"Pipe overflow/read failure")
            try:message=out_reader.messages.get(timeout=.05)
            except queue.Empty:continue
            if mapping(message).get("kind")=="HELLO":
                hello_count+=1;raise RuntimeError("Duplicate HELLO after decision")
        require(proc.poll() is not None and not owned_child_alive_independent(),"Total 120-second budget exhausted")
    except BaseException as e:
        failure={"type":type(e).__name__,"message":str(e)}
    finally:
        if proc is not None:
            if child_owned and child_handle:
                observed_child=observe_process(child_handle,source["interpreter_identity"])
                if observed_child["alive"] is True:
                    kernel_api().TerminateProcess(child_handle,125)
            if proc.poll() is None:
                proc.kill()  # owned Popen only; no tree/name enumeration
            try:proc.wait(timeout=max(.001,min(2.,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:pass
            if child_handle:
                while child_owned and time.monotonic()<deadline and observe_process(child_handle,source["interpreter_identity"])["alive"] is True:
                    time.sleep(.01)
                exit_observations["python_child"]=observe_process(child_handle,source["interpreter_identity"])
            exit_observations["popen"]=observe_process(proc._handle,source["interpreter_identity"])
            for reader in (out_reader,err_reader):
                if reader:reader.thread.join(timeout=max(.001,min(.5,deadline-time.monotonic())))
            for name,reader in (("stdout.txt",out_reader),("stderr.txt",err_reader)):
                if reader:
                    with (OUT/name).open("xb") as stream:stream.write(reader.data)
            if child_handle_separate and child_handle:kernel_api().CloseHandle(child_handle)
        result={"status":"PRESERVED_DIAGNOSTIC_ATTEMPT","failure":failure,"decision":decision,
          "new_Popen_submissions":1,"actual_Popen_returned":proc is not None,"hello_count":hello_count,
          "exit_observations":exit_observations,"elapsed_seconds":time.monotonic()-began,
          "pipe_errors":{"stdout":getattr(out_reader,"error",None),"stderr":getattr(err_reader,"error",None)},
          "pipes_closed":bool(out_reader and err_reader and out_reader.done and err_reader.done),
          "raw_token_saved":False,"legacy_state_not_repaired":True}
        write(OUT/"result.json",result)
    return seal_transcript()

def transcript_consistency(bundle):
    b=bundle;result=b["result"];child=b["child_result"];permission=b["permission"]
    parent=b["parent"];decision=b["decision"];launch=b["launch"]
    entered=child.get("entered_imports") is True
    entry=child.get("import_entry_monotonic_ns")
    event=child.get("first_event")
    return {
      "boot_matches_child_identity":b["boot"]==b["child_identity"].get("boot"),
      "boot_matches_parent":b["boot"]==parent["observations"].get("boot"),
      "own_checks_match":permission.get("own_checks")==b["child_identity"].get("checks"),
      "decision_matches_result":result.get("decision")==decision,
      "decision_matches_identity":decision.get("identity")==parent.get("evaluation"),
      "GO_has_valid_identity":decision.get("action")!="GO" or (parent["evaluation"]["identity_passed"] is True and all_true(decision.get("message_checks",{})) and decision.get("unexpected_queued_message") is False and decision.get("elapsed_seconds",1e9)<=GO_SECONDS),
      "permission_derived_from_checks":permission.get("permitted")== (all_true(permission.get("checks",{})) and all_true(permission.get("own_checks",{}))),
      "entry_requires_GO_permission":not entered or (decision.get("action")=="GO" and permission.get("permitted") is True),
      "entry_deadline":not entered or (type(entry) is int and launch["submitted_monotonic_ns"]<=entry<=launch["entry_deadline_ns"]),
      "no_entry_time_without_entry":entered or entry is None,
      "total_budget":isinstance(result.get("elapsed_seconds"),(int,float)) and 0<=result["elapsed_seconds"]<=TOTAL_SECONDS,
      "event_content_matches":event==b.get("event"),
      "event_presence_matches":(event is not None)==b.get("event_exists"),
      "guard_event_status":(child.get("status")=="FIRST_GUARD_EVENT")== (event is not None),
      "event_requires_import_entry":event is None or entered,
      "dependency_order_prefix":child.get("imports")==list(SEQUENCE[:len(child.get("imports",[]))])}

def transcript_checks():
    source,protocol=read_preparation(full=True)
    names={p.name for p in OUT.iterdir()}
    require(FULL_FILES-{"verification.json"}<=names,"Transcript partial; missing observations/results; preserve without completion")
    result=read(OUT/"result.json");child=read(OUT/"child_result.json")
    require(result["new_Popen_submissions"]==1 and read(OUT/"started.json")["diagnostic_child_launch_attempts"]==1,
            "Single budget differs")
    require(result["pipes_closed"] and not any(result["pipe_errors"].values()),"Unknown/incomplete stream transaction")
    exits=result["exit_observations"]
    require(mapping(exits.get("popen")).get("alive") is False,"Popen exit unknown")
    # Rejected cases can seal only if the recorded Python process also has a verified known exit.
    require(mapping(exits.get("python_child")).get("alive") is False,"Reported Python exit unknown")
    parent=read(OUT/"parent_identity.json")
    require(parent["evaluation"]==evaluate_identity(parent["observations"],source),"Identity comparison differs")
    consistency=transcript_consistency({"result":result,"child_result":child,
      "parent":parent,"decision":read(OUT/"decision.json"),"launch":read(OUT/"started.json"),
      "boot":read(OUT/"child_boot.json"),"child_identity":read(OUT/"child_identity.json"),
      "permission":read(OUT/"import_permission.json"),"event_exists":(OUT/"first_event.json").is_file(),
      "event":read(OUT/"first_event.json") if (OUT/"first_event.json").is_file() else None})
    require(all_true(consistency),"Substantive transcript consistency incomplete")
    require(child["environment_unchanged"] is True,"Child temporary environment changed")
    require(env()==read(OUT/"started.json")["environment"],"Controller environment changed")
    require(result["hello_count"]==1,"Duplicate/missing HELLO transcript")
    return {"status":"TRANSCRIPT_ONLY_VERIFIED","consistency":consistency,"identity_passed":parent["evaluation"]["identity_passed"],
      "child_status":child["status"],"entered_imports":child["entered_imports"],
      "old_process_chain_still_unknown":True,"old_failure_not_repaired":True,
      "model_compatibility_or_forward_verified":False,"single_new_attempt":1}

def seal_transcript():
    checks=transcript_checks()
    write(OUT/"verification.json",checks)
    write(OUT/"complete.json",{"status":"DIAGNOSTIC_TRANSCRIPT_COMPLETE",
      "outputs":{n:digest(OUT/n) for n in sorted(FULL_FILES|({"first_event.json"} if (OUT/"first_event.json").exists() else set()))},
      "identity_passed":checks["identity_passed"],"model_ready":False})
    return checks

def verify_transcript():
    complete=read(OUT/"complete.json")
    require({p.name for p in OUT.iterdir()}==set(complete["outputs"])|{"complete.json"},"Transcript inventory differs")
    require(all(digest(OUT/n)==h for n,h in complete["outputs"].items()),"Transcript hash differs")
    checks=transcript_checks()
    require(read(OUT/"verification.json")==checks,"Saved transcript verification differs")
    return checks

def parent_audit(event,args):
    if event=="import" and args:
        require(not any(args[0]==n or args[0].startswith(n+".") for n in HEAVY),
                "Heavy import prohibited in CPU/controller entry")
    if event=="open" and args and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        reason=classify_open(p,mode,flags)
        if reason=="FORBIDDEN_PAYLOAD_OPEN" and str(p) in MODEL_ASSETS:
            writing=any(c in (mode or "") for c in "wax+") or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            require(not writing,"Model asset may only be byte-hashed")
        else:require(reason is None,"CPU/controller prohibited file access:"+str(p))
    elif event=="subprocess.Popen":require(MODE=="run-once","No subprocess in preparation/static entry")
    elif event in ("socket.connect","socket.getaddrinfo","pickle.find_class","os.system","os.posix_spawn"):
        require(False,"Forbidden CPU/controller side effect:"+event)
    elif event in ("os.putenv","os.unsetenv") and args:
        require(os.fsdecode(args[0]).upper() not in ENV_NAMES,"TEMP/TMP changes prohibited")
    elif event in ("os.mkdir","os.remove","os.rmdir","os.rename","os.link","os.symlink"):
        paths=args[:2] if event in ("os.rename","os.link","os.symlink") else args[:1]
        require(all(not isinstance(p,(str,bytes)) or Path(os.fsdecode(p)).resolve().is_relative_to(OUT.resolve()) for p in paths),
                "Mutation outside independent output")

def main():
    global MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("inspect","prepare","static-check","run-once","verify","_child"))
    parser.add_argument("--allow-diagnostic-imports",action="store_true")
    args=parser.parse_args();MODE=args.command
    sys.dont_write_bytecode=True
    sys.addaudithook(child_audit if MODE=="_child" else parent_audit)
    require(not any(n.split(".")[0] in HEAVY for n in sys.modules),"Entry already has prohibited dependencies")
    if MODE=="inspect":result={"binding":binding(full=True),"resources":resources(),"static":static_contract(),"creates_preparation":False}
    elif MODE=="prepare":result=prepare_protocol()
    elif MODE=="static-check":result=static_check()
    elif MODE=="run-once":result=run_once(args.allow_diagnostic_imports)
    elif MODE=="verify":result=verify_transcript()
    else:child_entry();return
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":
    main()

