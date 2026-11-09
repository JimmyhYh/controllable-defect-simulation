"""Independent stage-one identities and standard-library-only boundaries."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PACKAGE = ROOT / "pretrained_synthetic_pairing"
BOOT = ROOT / "results/pretrained_synthetic_pairing/ksdd2_r5_v1"
OUT = BOOT / "resource_preparation_v1"
LARGE = Path("E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1")
RESULTS, MODELS = LARGE / "results", LARGE / "models"
PLAN = PROJECT / "计划/预训练单头分割可控合成配对实验执行方案.md"
REPORT = PROJECT / "报告/预训练单头分割严格R5基线研究报告.md"
BASE = ROOT / "results/pretrained_segmentation/ksdd2_r5_v1"
OLD = ROOT / "results/fewshot_synthetic/ksdd2_v1/preparation"
MODEL = ROOT / "models/pretrained_segmentation/ksdd2_r5_v1/baseline_2026102202"
GENERIC = ROOT / "models/pretrained_segmentation/resources_v1/encoder/model.safetensors"
ENV = ROOT / ".venv-pretrained-seg"
CPU = ROOT / ".venv"
CODE = ("__init__.py", "common.py", "resources.py", "README_resources.md")
SEEDS = (2026102301, 2026102302, 2026102303)
DATA_SEED, MATERIAL_SEED = 2026102300, 2026101903
R5 = ["11670", "11884", "12070", "12245", "12303"]
ROLES = {"R_real5": 5, "N_train1885": 1885, "Q_normal200": 200,
         "U_anomaly241": 241, "E_official1004": 1004}
TYPES = ["bright_mura", "dark_mura", "bright_line", "dark_line"]
BLOCKED = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "safetensors",
           "numpy", "PIL", "cv2", "scipy", "sklearn", "pickle", "joblib", "huggingface_hub"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OPAQUE_SUFFIXES = IMAGE_SUFFIXES | {".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".safetensors", ".uint32"}
_READ_SCOPES = set()
_GUARD_INSTALLED = False


def require(value, message):
    if not value:
        raise RuntimeError(message + "; preserve state; no automatic retry or substitution")


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, "Unsafe relative name")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name, "Unsafe relative path")
    return path


def no_redirect(path):
    path = Path(path).absolute()
    for member in (path, *path.parents):
        if member.exists():
            info = member.lstat()
            require(not member.is_symlink() and not (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                    "Redirect/reparse point forbidden: " + str(member))
    return path


def opaque_hash_allowed(path):
    path = Path(path).absolute()
    if path == GENERIC or path.is_relative_to(MODEL) and path.suffix in {".pt", ".safetensors"}:
        return True
    folder = OLD / str(MATERIAL_SEED) / "synthetic"
    return (path.parent == folder and path.suffix == ".npz" and len(path.stem) == 4
            and path.stem.isdigit() and 0 <= int(path.stem) < 2500)


@contextmanager
def opaque_read_scope(path):
    path = Path(path).absolute()
    require(opaque_hash_allowed(path), "Opaque file outside fixed hash/header allowlist: " + str(path))
    _READ_SCOPES.add(path)
    try:
        yield
    finally:
        _READ_SCOPES.discard(path)


def sha(path):
    path = no_redirect(path)
    def digest():
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    if path.suffix.lower() in OPAQUE_SUFFIXES:
        with opaque_read_scope(path):
            return digest()
    return digest()


def load(path):
    path = no_redirect(path)
    require(path.suffix == ".json", "Only JSON metadata may be parsed")
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path = no_redirect(path)
    require(path.is_relative_to(OUT), "Write outside independent stage-one output")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    with path.open("xb") as stream:
        stream.write(payload.encode("utf-8")); stream.flush(); os.fsync(stream.fileno())


def tree(folder):
    folder = no_redirect(folder)
    require(folder.is_dir(), "Missing directory: " + str(folder))
    files, dirs = set(), set()
    for member in folder.rglob("*"):
        no_redirect(member)
        name = member.relative_to(folder).as_posix()
        if member.is_dir():
            dirs.add(name)
        else:
            require(member.is_file(), "Unexpected non-file member")
            files.add(name)
    return files, dirs


def expected_dirs(names):
    result = set()
    for name in names:
        parent = safe_relative(name).parent
        while parent.as_posix() != ".":
            result.add(parent.as_posix()); parent = parent.parent
    return result


def verify_closed_tree(path, names, allowed_empty_dirs=()):
    path = Path(path); names = set(names)
    require(tree(path) == (names, expected_dirs(names) | set(allowed_empty_dirs)), "Unknown/missing sealed file or directory: " + str(path))
    for name in allowed_empty_dirs:
        require(tree(path / str(safe_relative(name))) == (set(), set()), "Declared legacy temporary directory is not empty")


def verify_seal(path, allowed_empty_dirs=()):
    path = Path(path)
    complete = load(path / "complete.json")
    names = set(complete["files"])
    verify_closed_tree(path, names | {"complete.json"}, allowed_empty_dirs)
    for name, expected in complete["files"].items():
        require(sha(path / str(safe_relative(name))) == expected, "Sealed hash differs: " + str(path / name))
    return complete


def implementation():
    return {name: sha(PACKAGE / name) for name in CODE}


def plan_contract():
    text = PLAN.read_text(encoding="utf-8")
    scientific = text.split("## 一、", 1)[1].split("## 二、", 1)[0]
    rules = text.split("## 四、", 1)[1].split("## 五、", 1)[0]
    return {"path": str(PLAN), "scientific_sections_1_and_4_sha256": hashlib.sha256((scientific + rules).encode("utf-8")).hexdigest(),
            "large_output_root": str(LARGE), "stage1_output": str(OUT),
            "progress_sections_mutable": True}


def install_guard():
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    require(not any(x.split(".")[0] in BLOCKED for x in sys.modules), "Forbidden package already imported")
    def audit(event, args):
        if event == "import" and args[0].split(".")[0] in BLOCKED:
            raise RuntimeError("Stage one forbids import: " + args[0])
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("No network/pickle operation authorized")
        if event in {"os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink"}:
            raise RuntimeError("Delete/move/link forbidden")
        if event == "os.mkdir":
            path = Path(args[0]).absolute()
            require(path.is_relative_to(BOOT) or path == BOOT.parent, "Directory creation outside new D stage-one root")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return
            path = Path(name).absolute()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                require(path.is_relative_to(OUT), "Write outside new stage-one output")
            elif path.suffix.lower() in OPAQUE_SUFFIXES:
                require(path in _READ_SCOPES, "Pixels/arrays/tensors cannot be opened except explicit stream hash/header scope")
            elif path.is_relative_to(ROOT / "data"):
                require(path == ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json", "Raw data/GT access forbidden")
    sys.addaudithook(audit)
    _GUARD_INSTALLED = True


def require_resources(ram, disk, gpu=False):
    snapshot = resource_snapshot()
    require(snapshot["available_RAM_GiB"] >= ram and snapshot["D_free_GiB"] >= disk, "CPU preparation resources below requested minimum")
    if gpu:
        require(snapshot["GPU"]["free_MiB"] >= 6144, "Free GPU memory below 6GiB")
    return snapshot


def resource_snapshot():
    no_redirect(LARGE)
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in
                    ["total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "available_extended"]]
    mem = Memory(); mem.length = ctypes.sizeof(mem)
    fn = ctypes.windll.kernel32.GlobalMemoryStatusEx
    fn.argtypes = [ctypes.POINTER(Memory)]; fn.restype = ctypes.c_int
    require(fn(ctypes.byref(mem)), "GlobalMemoryStatusEx failed")
    command = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    rows = [line.split(",") for line in command.stdout.strip().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 4, "Expected one identifiable private GPU")
    name, total, free, driver = [x.strip() for x in rows[0]]
    return {"total_RAM_GiB": mem.total / 2**30, "available_RAM_GiB": mem.available / 2**30,
            "D_free_GiB": shutil.disk_usage(ROOT).free / 2**30, "E_free_GiB": shutil.disk_usage(LARGE.anchor).free / 2**30,
            "planned_E_target_exists": LARGE.exists(), "planned_E_path_and_ancestors_have_no_redirect": True,
            "GPU": {"name": name, "total_MiB": int(total), "free_MiB": int(free), "driver": driver},
            "method": "GlobalMemoryStatusEx; disk_usage; nvidia-smi metadata only; no CIM or model imports"}
