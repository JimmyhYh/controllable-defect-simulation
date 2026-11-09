"""Independent stage-3 identities, scalar schedules and process boundaries."""
from __future__ import annotations
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/pretrained_segmentation/ksdd2_r5_v1"
INPUT = BASE / "input_preparation"
PREP = BASE / "pilot_preparation_v1"
RUN = BASE / "probe_v1"
SCORES = BASE / "probe_scores_v1"
METRICS = BASE / "probe_metrics_v1"
MODEL = ROOT / "models/pretrained_segmentation/ksdd2_r5_v1/probe_2026102201"
DATA = ROOT / "data/fewshot_synthetic/ksdd2_v1"
ENV = ROOT / ".venv-pretrained-seg"
RESOURCE = ROOT / "models/pretrained_segmentation/resources_v1"
R5 = ["11670", "11884", "12070", "12245", "12303"]
N16 = ["10056", "10149", "10326", "10425", "10468", "10481", "10538", "10548", "10842", "10880", "11145", "11311", "12024", "12025", "12126", "12206"]
SEED = 2026102201
CODE = ["common.py", "data.py", "preparation.py", "model.py", "training.py", "scoring.py", "metrics.py", "cli.py", "README_pilot.md"]
BLOCKED = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "safetensors", "huggingface_hub", "diffusers", "transformers"}

def require(value, message):
    if not value:
        raise RuntimeError(message + "; preserve state, diagnose; no automatic rerun")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with p.open("x", encoding="utf-8", newline="\n") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())

def paths(folder):
    folder = Path(folder)
    for p in [folder, *folder.parents]:
        require(not p.is_symlink() and not p.is_junction(), "Redirected scientific path")
        if p == ROOT:
            break
    all_paths = list(folder.rglob("*"))
    require(not any(p.is_symlink() or p.is_junction() for p in all_paths), "Redirected cache member")
    return {p.relative_to(folder).as_posix() for p in all_paths if p.is_file()}

def seal(folder, identity):
    save(Path(folder) / "complete.json", {"identity": identity, "files": {n: sha(Path(folder) / n) for n in sorted(paths(folder))}})

def verify_seal(folder):
    folder = Path(folder)
    require((folder / "complete.json").is_file(), "Incomplete/unknown transaction: " + str(folder))
    p = load(folder / "complete.json")
    require(paths(folder) == set(p["files"]) | {"complete.json"}, "Unexpected/missing cache files")
    for name, expected in p["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and ":" not in name and "\\" not in name, "Unsafe manifest path")
        require(sha(folder / name) == expected, "Saved identity changed: " + name)
    return p

def code_identity(names=None):
    return {n: sha(Path(__file__).parent / n) for n in (CODE if names is None else names)}

def lr_at(update, total=2000):
    require(total in (2000, 10000) and type(update) is int and 1 <= update <= total, "Invalid scalar schedule request")
    return [1e-5, 1e-4] if update <= int(.8 * total) else ([2e-6, 2e-5] if update <= int(.9 * total) else [4e-7, 4e-6])

def upstream_files():
    """Only already bound text/source metadata, not original image or GT bytes."""
    p = load(BASE / "resource_preparation/protocol.json")
    env = load(BASE / "environment_v1/environment.json")
    old_input = ROOT / "results/fewshot_synthetic/ksdd2_v1/input"
    old_completion = load(old_input / "completion.json")
    require(sha(old_input / "preparation_complete.json") == old_completion["preparation_sha256"], "Original export completion changed")
    original_export = load(old_input / "preparation_complete.json")
    require(sha(old_input / "export_inventory.json") == original_export["files"]["export_inventory.json"], "Canonical RGB/GT export inventory changed")
    result = {}
    for rel in ["resource_preparation", "environment_preparation_v1", "environment_v1"]:
        folder = BASE / rel
        complete = load(folder / "complete.json")
        for n, expected in complete["files"].items():
            require(sha(folder / n) == expected, "Upstream text seal changed")
            result[(folder / n).relative_to(ROOT).as_posix()] = expected
        result[(folder / "complete.json").relative_to(ROOT).as_posix()] = sha(folder / "complete.json")
    require(sha(BASE / "environment_v1/complete.json") == "a34e351d6ee17f5006a714ba8568239998dc3c475ba2e9b7e47bf466900ade30", "Stage2 fixed completion changed")
    for mapping in [p["implementation"], p["old_identities"]["anchored_metadata_sha256"], p["old_identities"]["protected_branch_sources"], p["old_identities"]["anchored_core_sources"]]:
        for n, expected in mapping.items():
            require(sha(ROOT / n) == expected, "Protected source/metadata changed: " + n)
            result[n] = expected
    for n, expected in env["installed_SMP_source_subset_sha256"].items():
        rel = ".venv-pretrained-seg/Lib/site-packages/" + n
        require(sha(ROOT / rel) == expected, "Installed SMP source identity changed")
        result[rel] = expected
    for n in ["models/pretrained_segmentation/resources_v1/complete.json", "models/pretrained_segmentation/resources_v1/safetensors_header.json",
              "models/pretrained_segmentation/resources_v1/encoder/config.json", "data/fewshot_synthetic/ksdd2_v1/split.json",
              "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json", ".venv-pretrained-seg/pyvenv.cfg", ".venv-pretrained-seg/Scripts/python.exe",
              ".venv-pretrained-seg/Lib/site-packages/torch/nn/modules/batchnorm.py", "fewshot_synthetic/metrics.py", "fewshot_synthetic/common.py", "fewshot_synthetic/data.py"]:
        result[n] = sha(ROOT / n)
    require(sha(RESOURCE / "encoder/model.safetensors") == "b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19", "Encoder resource changed")
    return result

def verify_upstream(expected):
    for n, digest in expected.items():
        require(sha(ROOT / n) == digest, "Upstream binding changed: " + n)
    require(sha(RESOURCE / "encoder/model.safetensors") == "b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19", "Encoder resource changed")

def resources(ram=2, disk=1, gpu=False):
    from .resources import snapshot
    p = snapshot()
    require(p["free_RAM_GiB"] >= ram and p["free_disk_GiB"] >= disk, "Insufficient CPU RAM/disk; no model action started")
    if gpu:
        require(len(p["GPUs"]) == 1 and p["GPUs"][0]["free_MiB"] >= 6144, "Need one GPU with at least6GiB free VRAM")
    return p

def runtime_environment():
    require(Path(sys.executable).resolve() == (ENV / "Scripts/python.exe").resolve(), "Use independent .venv-pretrained-seg for manual model operation")
    actual = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in md.distributions()}
    saved = load(BASE / "environment_v1/environment.json")["metadata"]["distributions"]
    expected = {x["name"].lower().replace("_", "-"): x["version"] for x in saved}
    require(actual == expected, "Runtime dependency set changed")
    # Only in a future explicit model entry; avoid library caches outside this branch.
    import tempfile
    temp = BASE / "runtime_temp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update({"TEMP": str(temp), "TMP": str(temp), "TORCH_HOME": str(temp / "torch"),
                       "HF_HOME": str(temp / "hf"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    tempfile.tempdir = str(temp)

def guard(cpu=True):
    require(not cpu or not any(n.split(".")[0] in BLOCKED or n == "pretrained_segmentation.model" for n in sys.modules), "CPU-only process contains model imports")
    split = load(DATA / "split.json")
    allowed = {(DATA / row[k]).resolve() for row in split["R_real5"] + [r for r in split["N_train1885"] if r["id"] in N16] for k in ("image", "mask")}
    def audit(event, args):
        if cpu and event == "import" and (args[0].split(".")[0] in BLOCKED or args[0] == "pretrained_segmentation.model"):
            raise RuntimeError("CPU input/scalar stage forbids model imports")
        if event == "socket.connect" or cpu and event.startswith("pickle."):
            raise RuntimeError("No network/pickle execution authorized")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.fsdecode(args[0]).lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return  # Windows null device, not a filesystem destination.
            p = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                require(p.is_relative_to(BASE) or p.is_relative_to(MODEL), "Write outside independent scientific outputs")
            elif p.is_relative_to(ROOT / "data"):
                require(p in allowed or p == (DATA / "split.json").resolve(), "Q/E/U or nonfixed raw input forbidden")
    sys.addaudithook(audit)
