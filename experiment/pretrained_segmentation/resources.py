"""Stage 1: pinned source acquisition and CPU-only resource preparation.

Only acquire uses the network, for an explicit allowlist of UTF-8 source and
metadata URLs. No weight, image, GT, array or model package is opened/imported.
Existing packaging is used solely to evaluate dependency metadata and markers.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
VENDOR = ROOT / "vendor/SMP_0_5_0"
OUT = ROOT / "results/pretrained_segmentation/ksdd2_r5_v1/resource_preparation"
DESIGN = PROJECT / "计划/预训练单头分割严格R5基线实现设计.md"
WHEELS = PROJECT / "实验/手动资源/wheels"
COMMIT = "420ce84b0c2df0286fa9bb2bd1499eea625c9b33"
TREE = "2d28cabe067bfacbb40879e98a1f4459ddc3cb6d"
HF_REV = "3f2325ff978283d47aa6a1d6878ca20565622683"
WEIGHT_BYTES = 46805552
WEIGHT_HASH = "b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19"
AUTHOR = {
    "LICENSE": "0f186bfe986e1659183c4d683f67298ee3161249",
    "README.md": "c3df5718f5ee7fe06f687ed75028688a1840779e",
    "pyproject.toml": "f3e55a9631571d8d100b883748f97748e8a03570",
    "segmentation_models_pytorch/__init__.py": "37c64ef66336f723363a6cc95e54b572a12ec1ad",
    "segmentation_models_pytorch/__version__.py": "3d187266f146fb216a04537da37dc5df0811acc4",
    "segmentation_models_pytorch/base/__init__.py": "0df076d9802cc9de24092a20b729f53042285459",
    "segmentation_models_pytorch/base/heads.py": "fbc939cad427fb6b9edfb02c0e841a68de84a03a",
    "segmentation_models_pytorch/base/hub_mixin.py": "a18380d10889749b0da79c61b015d4dc1d1fa7fd",
    "segmentation_models_pytorch/base/initialization.py": "cf518edd499e509678eada95ba017c2196a1532b",
    "segmentation_models_pytorch/base/model.py": "9b0db714a34607a2ed11ebc75d70c89c88cbac56",
    "segmentation_models_pytorch/base/modules.py": "15cfdb124e6ef5d90e196125efccbe91e6616884",
    "segmentation_models_pytorch/base/utils.py": "a0d41943ddca8db3ea6c12b31cfdecd6928b2125",
    "segmentation_models_pytorch/decoders/unet/__init__.py": "bcae4e0fbb8647fcfe9f2bd481964bbbe9ef0290",
    "segmentation_models_pytorch/decoders/unet/decoder.py": "cfeb267eedc002c7fae64ffa570a0d43ee628614",
    "segmentation_models_pytorch/decoders/unet/model.py": "3df36e32b72a6b6acdf12a41722dd8e34d8b094e",
    "segmentation_models_pytorch/encoders/__init__.py": "287a921a7318596743e5119fb34190fb6e1dc8d7",
    "segmentation_models_pytorch/encoders/_base.py": "98c431fbafd2285c57c59830ed243a123a96af68",
    "segmentation_models_pytorch/encoders/_preprocessing.py": "57c7e6d2d566aca14c963adda7ee318853419d9d",
    "segmentation_models_pytorch/encoders/_utils.py": "86128099a550986ca51a3ff263748b4e76a46911",
    "segmentation_models_pytorch/encoders/resnet.py": "d4f2db4e0d581629015e36789d6e18829bca2412",
}
API_BASE = "https://api.github.com/repos/qubvel-org/segmentation_models.pytorch/git/"
HF_RAW = f"https://huggingface.co/smp-hub/resnet18.imagenet/raw/{HF_REV}/"
UPSTREAM = {
    "upstream/github_tag.json": API_BASE + "ref/tags/v0.5.0",
    "upstream/github_commit.json": API_BASE + "commits/" + COMMIT,
    "upstream/github_tree.json": API_BASE + "trees/" + TREE + "?recursive=1",
    "upstream/hf_model.json": f"https://huggingface.co/api/models/smp-hub/resnet18.imagenet/revision/{HF_REV}?blobs=true",
    "upstream/resnet18-README.md": HF_RAW + "README.md",
    "upstream/resnet18-config.json": HF_RAW + "config.json",
    "upstream/smp_pypi.json": "https://pypi.org/pypi/segmentation-models-pytorch/0.5.0/json",
    "upstream/timm_pypi.json": "https://pypi.org/pypi/timm/1.0.20/json",
}
SOURCE_URLS = {n: f"https://raw.githubusercontent.com/qubvel-org/segmentation_models.pytorch/{COMMIT}/{n}" for n in AUTHOR}
LOCAL_NOTICES = {
    "upstream/torchvision-v0.23.0-LICENSE.txt": "vendor/SuperSimpleNet_98ab4d5/provenance/torchvision-v0.23.0-LICENSE.txt",
    "upstream/torchvision-v0.23.0-README.md": "vendor/SuperSimpleNet_98ab4d5/provenance/torchvision-v0.23.0-README.md",
}
ROLES = {"R_real5": 5, "Q_normal200": 200, "N_train1885": 1885, "U_anomaly241": 241, "E_official1004": 1004}
R5 = ["11670", "11884", "12070", "12245", "12303"]
N16 = ["10056", "10149", "10326", "10425", "10468", "10481", "10538", "10548", "10842", "10880", "11145", "11311", "12024", "12025", "12126", "12206"]
PINS = {
    "segmentation-models-pytorch": "0.5.0", "timm": "1.0.20",
    "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "numpy": "1.26.4", "pillow": "12.3.0",
    "huggingface-hub": "0.35.3", "safetensors": "0.6.2", "pyyaml": "6.0.3", "tqdm": "4.70.1",
    "filelock": "4.2.0", "fsspec": "2026.9.0", "packaging": "26.3", "requests": "2.34.2",
    "typing-extensions": "4.16.0", "sympy": "1.14.0", "networkx": "3.7", "jinja2": "3.1.6",
    "setuptools": "84.0.0", "mpmath": "1.3.0", "markupsafe": "3.0.4", "colorama": "0.4.6",
    "charset-normalizer": "3.5.2", "idna": "3.20", "urllib3": "2.8.0", "certifi": "2026.7.22",
}
CANDIDATE = {
    "library": "segmentation-models-pytorch==0.5.0", "source_commit": COMMIT,
    "architecture": "Unet", "encoder": "resnet18", "encoder_depth": 5,
    "decoder_channels": [256, 128, 64, 32, 16], "decoder_interpolation": "nearest",
    "classes": 2, "activation": None, "aux_params": None, "decoder_attention_type": None,
    "encoder_weights": {"repo": "smp-hub/resnet18.imagenet", "revision": HF_REV,
                        "filename": "model.safetensors", "expected_bytes": WEIGHT_BYTES, "expected_sha256": WEIGHT_HASH,
                        "locally_acquired": False, "loaded": False},
    "adaptation": "encoder BN running buffers fixed, affine trainable; decoder BN replaced by GN8, eps1e-5; not native SMP GN option",
    "window": 256, "stride": 128, "tail_window": True, "padding": "right reflection, excluded from loss and metrics",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "loss": "valid-domain CE + soft Dice (smoothing1), empty GT included",
    "train_roles": ["R_real5", "N_train1885"], "Q": "normal calibration only; not training or configuration selection",
    "U": "unused", "E": "already observed; exploratory reuse only; not training or selection",
    "synthesis": "none in this baseline", "training": "manual user initiation only",
    "pilot": {"seed": 2026102201, "steps": 2000, "gate_global_AP": .95, "gate_each_R_window_AP": .85, "prepared": False},
    "formal": {"seed": 2026102202, "steps": 10000, "prepared": False},
    "use": "private device; public materials; personal noncommercial research; company materials isolated",
}
NOTICES = """# Fixed source subset and resource conditions

20 unmodified SMP source/document/license files are pinned to v0.5.0 commit
420ce84b0c2df0286fa9bb2bd1499eea625c9b33. Their actual SHA256 and Git blob
identities are recorded. This is an inspection subset, not an importable clone.
No author source, model or package initializer is executed at this stage.

SMP source is MIT. The separately pinned ResNet18 model card says license: other
and points to the original resnet18-5c106cde.pth artifact. Neither that label nor
SMP's source license establishes an unrestricted grant for ImageNet weights.
The upstream torchvision README warns that pretrained models can have training-
data-derived terms. Its already archived v0.23.0 README/LICENSE are copied as
notices only; they do not prove an exact training lineage or new weight grant.
Use remains personal learning / noncommercial research, not commercial clearance.

Timm 1.0.20 is a declared SMP dependency, chosen explicitly as a fixed candidate
satisfying the published lower bound, not an automatic upgrade. PyPI metadata
reports Apache-2.0. Actual future wheels and their license files still require
verification. There is no claim of runtime compatibility or paper replication.

Only UTF-8 code, documentation, configuration and package/repository metadata
are acquired. The approximately 44.64MiB safetensors file and two missing wheels
are NOT downloaded here. No task weights, dataset, pixels, GT or arrays are read.
"""
FORBIDDEN_IMPORTS = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "huggingface_hub", "safetensors", "numpy", "PIL", "cv2", "pickle", "joblib"}
FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle", ".safetensors"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message + "; preserve state and diagnose; no automatic retry")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = obj if isinstance(obj, (str, bytes)) else json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    with path.open("xb") as f:
        f.write(payload.encode("utf-8") if isinstance(payload, str) else payload)
        f.flush()
        os.fsync(f.fileno())


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name, "Unsafe manifest name")
    return p


def no_redirected_path(path):
    for p in [path, *path.parents]:
        require(not p.is_symlink() and not p.is_junction(), f"Redirected path: {p}")
        if p == ROOT or p == PROJECT:
            break


def file_set(folder):
    no_redirected_path(folder)
    paths = list(folder.rglob("*"))
    require(not any(p.is_symlink() or p.is_junction() for p in paths), "Redirected cache member")
    return {p.relative_to(folder).as_posix() for p in paths if p.is_file()}


def seal(folder, schema):
    write(folder / "complete.json", {"schema": schema, "files": {n: sha(folder / n) for n in sorted(file_set(folder))}})


def verify_seal(folder):
    require((folder / "complete.json").is_file(), f"Partial/unknown cache {folder}")
    m = load(folder / "complete.json")
    require(file_set(folder) == set(m["files"]) | {"complete.json"}, f"Strict cache listing mismatch: {folder}")
    for n, expected in m["files"].items():
        safe_name(n)
        require(sha(folder / n) == expected, f"Cache hash mismatch: {folder / n}")
    return m


def old_identities():
    # Parse literal constants as text, never import/execute an old module.
    source = ast.parse((ROOT / "ssn_baseline/resources.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OLD" for t in n.targets)]
    require(len(values) == 1, "Cannot identify original SSN OLD anchors")
    old = values[0]
    require(old["data/fewshot_synthetic/ksdd2_v1/split.json"] == "b8c2fb376bfd04072baaaf173eb0fccb1ecc5f1339074a6829fe147d0b6210bf", "Split trust anchor changed")
    for n, expected in old.items():
        safe_name(n)
        require(sha(ROOT / n) == expected, f"Existing metadata identity mismatch: {n}")
    common = load(ROOT / "results/fewshot_synthetic/ksdd2_v1/preparation/common_config.json")
    inp = load(ROOT / "results/fewshot_synthetic/ksdd2_v1/input/input_protocol.json")
    sources = {**{f"fewshot_synthetic/{k}": v for k, v in common["implementation"].items()},
               **common["input_identity"]["old_sources"], **{f"fewshot_synthetic/{k}": v for k, v in inp["implementation"].items()}}
    for n, expected in sources.items():
        safe_name(n)
        require(sha(ROOT / n) == expected, f"Existing source mismatch: {n}")
    split = load(ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json")
    seen = set()
    for role, count in ROLES.items():
        rows = split[role]
        ids = [r["id"] for r in rows]
        require(len(ids) == count and len(set(ids)) == count and not seen.intersection(ids), f"Role count/overlap mismatch: {role}")
        require(ids == sorted(ids), f"Role original order mismatch: {role}")
        seen.update(ids)
        for row in rows:
            for field in ["image", "mask"]:
                safe_name(row[field])  # Metadata only; these paths are never opened.
            require(row["part"] == ("test" if role == "E_official1004" else "train"), "Role part changed")
            require(2 <= row["width"] < 256 and row["height"] >= 256, "Unexpected native dimension metadata")
    require([r["id"] for r in split["R_real5"]] == R5 and split["visual_N16_ids"] == N16, "Fixed R5/N16 changed")
    require(load(ROOT / "results/fewshot_synthetic/ksdd2_v1/input/completion.json")["input_ready"] is True, "Original input_ready missing")
    closed = ROOT / "results/ssn_baseline/ksdd2_r5_v1"
    refs = [closed / "pilot_dtype_acceptance_verification_v1", closed / "pilot_cls_bn_sensitivity_summary_v1"]
    for folder in refs:
        verify_seal(folder)  # These seals contain text/JSON only, not probability files.
    acceptance = load(refs[0] / "verification.json")
    summary = load(refs[1] / "summary.json")
    require(acceptance["gate_passed"] is False and acceptance["classification"]["FP"] == 10, "Original SSN gate/FP changed")
    require(summary["row_label"] == "A" and summary["decision_flips"] == 4 and summary["fp_old"] == 10 and summary["fp_new"] == 8, "Closed BN conclusion changed")
    metadata_files = {**old, **{p.relative_to(ROOT).as_posix(): sha(p) for f in refs for p in f.iterdir() if p.is_file()}}
    report = closed / "pilot_cls_bn_sensitivity_report_v1/分类BN统计来源敏感性诊断结果.md"
    metadata_files[report.relative_to(ROOT).as_posix()] = sha(report)
    all_sources = {p.relative_to(ROOT).as_posix(): sha(p) for folder in [ROOT / "fewshot_synthetic", ROOT / "ssn_baseline"] for p in folder.rglob("*.py")}
    return {"anchored_metadata_sha256": metadata_files, "anchored_core_sources": sources,
            "protected_branch_sources": all_sources, "roles": ROLES, "R5": R5, "N16": N16,
            "SSN_original_gate": False, "SSN_BN_row": "A", "content_scope": "metadata and source only; no full old artifact/weight/array verification"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"Unexpected source redirect {code}: {req.full_url}; stop for review")


def fetch(url):
    require(url in set(UPSTREAM.values()) | set(SOURCE_URLS.values()), "URL not in fixed text-only allowlist")
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(urllib.request.Request(url, headers={"User-Agent": "pretrained-segmentation-stage1-static"}), timeout=35) as r:
        require(r.status == 200 and r.geturl() == url, "Unexpected HTTP identity")
        raw = r.read(3 * 1024 * 1024 + 1)
    require(len(raw) <= 3 * 1024 * 1024, "Text response exceeds limit")
    raw.decode("utf-8")
    return raw


def blob_id(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def upstream_checks():
    tag = load(VENDOR / "upstream/github_tag.json")
    commit = load(VENDOR / "upstream/github_commit.json")
    tree = load(VENDOR / "upstream/github_tree.json")
    require(tag["object"] == {"sha": COMMIT, "type": "commit", "url": API_BASE + "commits/" + COMMIT}, "Pinned tag mismatch")
    require(commit["sha"] == COMMIT and commit["tree"]["sha"] == TREE, "Pinned commit/tree mismatch")
    require(tree["sha"] == TREE and tree.get("truncated") is False, "Incomplete Git tree")
    entries = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"}
    for n, expected in AUTHOR.items():
        data = (VENDOR / n).read_bytes()
        require(blob_id(data) == expected == entries[n]["sha"] and len(data) == entries[n]["size"], f"Pinned source blob mismatch: {n}")
        if n.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=n)
    hf = load(VENDOR / "upstream/hf_model.json")
    require(hf["sha"] == HF_REV and hf["id"] == "smp-hub/resnet18.imagenet" and hf["gated"] is False, "HF source identity changed")
    hf_files = {x["rfilename"]: x for x in hf["siblings"]}
    require(set(hf_files) == {".gitattributes", "README.md", "config.json", "model.safetensors"}, "Unexpected HF resource list")
    weight = hf_files["model.safetensors"]
    require(weight["size"] == WEIGHT_BYTES and weight["lfs"]["sha256"] == WEIGHT_HASH, "HF weight metadata identity changed")
    for local, remote in [("resnet18-README.md", "README.md"), ("resnet18-config.json", "config.json")]:
        raw = (VENDOR / "upstream" / local).read_bytes()
        require(blob_id(raw) == hf_files[remote]["blobId"] and len(raw) == hf_files[remote]["size"], "HF text blob mismatch")
    cfg = load(VENDOR / "upstream/resnet18-config.json")
    require(cfg["mean"] == CANDIDATE["normalization"]["mean"] and cfg["std"] == CANDIDATE["normalization"]["std"] and cfg["input_space"] == "RGB", "Preprocessing metadata mismatch")
    require(hf["cardData"]["license"] == "other" and "license: other" in (VENDOR / "upstream/resnet18-README.md").read_text(encoding="utf-8"), "Weight license observation changed")
    for key, name in [("smp", "segmentation-models-pytorch"), ("timm", "timm")]:
        p = load(VENDOR / f"upstream/{key}_pypi.json")
        require(canonical(p["info"]["name"]) == name and p["info"]["version"] == PINS[name], "PyPI metadata identity changed")
    return {"commit": COMMIT, "tree": TREE, "author_files": len(AUTHOR), "HF_revision": HF_REV,
            "weight_remote_bytes": WEIGHT_BYTES, "weight_remote_sha256": WEIGHT_HASH,
            "weight_locally_present": False, "weight_license": "other; unrestricted grant not established"}


def verify_vendor():
    verify_seal(VENDOR)
    expected = set(AUTHOR) | set(UPSTREAM) | set(LOCAL_NOTICES) | {"SOURCE_MANIFEST.json", "THIRD_PARTY_NOTICES.md", "transaction.json", "complete.json"}
    require(file_set(VENDOR) == expected, "Vendor fixed listing mismatch")
    info = upstream_checks()
    m = load(VENDOR / "SOURCE_MANIFEST.json")
    require(m["commit"] == COMMIT and m["tree"] == TREE and set(m["files"]) == set(AUTHOR) | set(UPSTREAM) | set(LOCAL_NOTICES), "Source manifest mismatch")
    for n, row in m["files"].items():
        p = VENDOR / n
        require(sha(p) == row["sha256"] and p.stat().st_size == row["bytes"], "Source manifest bytes mismatch")
        if n in SOURCE_URLS or n in UPSTREAM:
            require(row["url"] == (SOURCE_URLS | UPSTREAM)[n], "Source URL mismatch")
        else:
            require(row["source_local"] == LOCAL_NOTICES[n] and sha(ROOT / LOCAL_NOTICES[n]) == row["sha256"], "Existing notice changed")
    require((VENDOR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == NOTICES, "Resource notice changed")
    return {**info, "manifest_sha256": sha(VENDOR / "SOURCE_MANIFEST.json"), "complete_sha256": sha(VENDOR / "complete.json")}


def acquire():
    old_identities()
    if VENDOR.exists():
        return {"reused": True, **verify_vendor()}
    VENDOR.mkdir(parents=True, exist_ok=False)
    write(VENDOR / "transaction.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "action": "allowlisted source and metadata only", "commit": COMMIT})
    try:
        records = {}
        for n, url in (UPSTREAM | SOURCE_URLS).items():
            raw = fetch(url)
            if n in AUTHOR:
                require(blob_id(raw) == AUTHOR[n], f"Source blob mismatch before saving {n}")
            write(VENDOR / n, raw)
            records[n] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for n, original in LOCAL_NOTICES.items():
            raw = (ROOT / original).read_bytes()
            write(VENDOR / n, raw)
            records[n] = {"source_local": original, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        upstream_checks()
        write(VENDOR / "SOURCE_MANIFEST.json", {"commit": COMMIT, "tree": TREE, "files": records,
              "network_payload_bytes": sum(r["bytes"] for r in records.values() if "url" in r),
              "weights_packages_datasets_downloaded": 0})
        write(VENDOR / "THIRD_PARTY_NOTICES.md", NOTICES)
        seal(VENDOR, "pretrained-segmentation-source-subset-v1")
    except Exception as exc:
        write(VENDOR / "failure.json", {"error": repr(exc), "rule": "retain partial source transaction; do not auto retry or change pin"})
        raise
    return {"reused": False, **verify_vendor()}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_record(path):
    no_redirected_path(path)
    with ZipFile(path) as z:
        # Nested vendored .dist-info/METADATA is not the wheel's own identity.
        ns = [n for n in z.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
        require(len(ns) == 1, f"Ambiguous root wheel METADATA: {path.name}")
        raw = z.read(ns[0])
        msg = BytesParser().parsebytes(raw)
        wheel_raw = z.read(ns[0].rsplit("/", 1)[0] + "/WHEEL")
    return {"path": str(path), "filename": path.name, "bytes": path.stat().st_size,
            "sha256": sha(path), "name": canonical(msg["Name"]), "version": msg["Version"],
            "requires_python": msg["Requires-Python"], "requires_dist": msg.get_all("Requires-Dist", []),
            "metadata_sha256": hashlib.sha256(raw).hexdigest(), "wheel_tags": BytesParser().parsebytes(wheel_raw).get_all("Tag", [])}


def dependencies():
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import parse_tag, sys_tags
    env = default_environment()
    env["extra"] = ""
    require(env["sys_platform"] == "win32" and env["python_version"] == "3.12" and platform.machine().lower() in {"amd64", "x86_64"}, "Candidate targets CPython3.12 Windows x64 only")
    wheels = {}
    # Inspect only names selected in the frozen candidate; unrelated wheels are excluded.
    for path in sorted(WHEELS.glob("*.whl")):
        name = canonical(path.name.split("-", 1)[0])
        if name in PINS and name not in {"segmentation-models-pytorch", "timm"}:
            record = wheel_record(path)
            require(record["name"] not in wheels and record["version"] == PINS[record["name"]], "Ambiguous or conflicting local wheel version")
            wheels[record["name"]] = record
    records = dict(wheels)
    missing = []
    for key, name in [("smp", "segmentation-models-pytorch"), ("timm", "timm")]:
        obj = load(VENDOR / f"upstream/{key}_pypi.json")
        files = [x for x in obj["urls"] if x["filename"].endswith("-py3-none-any.whl")]
        require(len(files) == 1, "Missing/ambiguous remote pure Python wheel")
        f = files[0]
        record = {"name": name, "version": PINS[name], "requires_python": obj["info"]["requires_python"],
                  "requires_dist": obj["info"]["requires_dist"] or [], "filename": f["filename"],
                  "bytes": f["size"], "sha256": f["digests"]["sha256"], "url": f["url"],
                  "wheel_tags": ["py3-none-any"], "local": False, "identity": "remote metadata only; actual wheel not acquired"}
        records[name] = record
        missing.append(record)
    require(set(records) == set(PINS), "Missing pre-existing wheel; do not substitute version")
    tags = set(sys_tags())
    edges, excluded = [], []
    for name, record in sorted(records.items()):
        require(not record["requires_python"] or SpecifierSet(record["requires_python"]).contains(platform.python_version()), f"Python incompatibility for {name}")
        require(any(parse_tag(tag) & tags for tag in record["wheel_tags"]), f"Wheel target incompatible: {name}")
        for text in record["requires_dist"]:
            req = Requirement(text)
            active = req.marker is None or req.marker.evaluate(environment=env)
            row = {"from": name, "requirement": text, "active": active}
            if not active:
                excluded.append(row)
                continue
            target = canonical(req.name)
            require(not req.extras and req.url is None, "Unreviewed extras/direct URL dependency")
            require(target in PINS and req.specifier.contains(PINS[target]), f"Unresolved/incompatible dependency {name}: {text}")
            row["to"] = target
            edges.append(row)
    reached, pending = set(), ["segmentation-models-pytorch"]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(e["to"] for e in edges if e["from"] == current)
    require(reached == set(PINS), "Candidate contains extra/missing transitive runtime pins")
    return {"pins": PINS, "marker_environment": env, "records": records, "active_edges": edges,
            "inactive_requirements": excluded, "local_wheels": len(wheels), "missing_wheels": missing,
            "additional_package_download_bytes": sum(x["bytes"] for x in missing),
            "additional_weight_download_bytes": WEIGHT_BYTES,
            "additional_known_total_bytes": sum(x["bytes"] for x in missing) + WEIGHT_BYTES,
            "scope": "base runtime closure only; docs/test/extras and CPU later-stage image/metrics dependencies excluded; not installation/runtime proof"}


def environment():
    packages = sorted([{"name": d.metadata["Name"], "version": d.version} for d in metadata.distributions()], key=lambda x: canonical(x["name"]))
    return {"executable": sys.executable, "executable_sha256": sha(sys.executable), "python": sys.version,
            "platform": platform.platform(), "packaging_version": metadata.version("packaging"), "packages_metadata_only": packages}


def snapshot():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ["total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended"]]
    mem = Memory()
    mem.length = ctypes.sizeof(mem)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)), "RAM metadata query failed")
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    gpus = []
    for line in r.stdout.strip().splitlines():
        name, total, free, driver = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "total_MiB": float(total), "free_MiB": float(free), "driver": driver})
    return {"utc": datetime.now(timezone.utc).isoformat(), "total_RAM_GiB": mem.total / 2**30,
            "free_RAM_GiB": mem.available / 2**30, "free_disk_GiB": shutil.disk_usage(ROOT).free / 2**30,
            "GPUs": gpus, "interpretation": "snapshot only; not runtime admission or measured training peak"}


def implementation():
    return {n: sha(ROOT / n) for n in ["pretrained_segmentation/__init__.py", "pretrained_segmentation/resources.py", "pretrained_segmentation/README.md"]}


def verify():
    verify_seal(OUT)
    require(file_set(OUT) == {"transaction.json", "protocol.json", "dependency_candidate.json", "requirements-candidate.txt", "verification.json", "第1阶段静态来源与资源准备.md", "complete.json"}, "Resource cache listing changed")
    p = load(OUT / "protocol.json")
    require(p["candidate"] == CANDIDATE and p["vendor"] == verify_vendor(), "Candidate/vendor identity changed")
    require(p["old_identities"] == old_identities() and p["design_sha256"] == sha(DESIGN), "Old or design identity changed")
    require(p["implementation"] == implementation() and p["environment"] == environment(), "Implementation/interpreter/package metadata changed")
    require(load(OUT / "dependency_candidate.json") == dependencies(), "Candidate material/dependency identity changed")
    reqs = "".join(f"{n}=={v}\n" for n, v in sorted(PINS.items()))
    require((OUT / "requirements-candidate.txt").read_text(encoding="utf-8") == reqs, "Candidate pin text changed")
    require(not (ROOT / ".venv-pretrained-seg").exists() and not (ROOT / "models/pretrained_segmentation").exists(), "Later-stage artifacts appeared; verify stage identity before continuation")
    return {"stage1_ready": True, "scope": "source and CPU metadata only", "cache_reused": True,
            "source_commit": COMMIT, "source_files": len(AUTHOR), "roles": ROLES,
            "local_runtime_wheels": len(PINS) - 2, "missing_runtime_wheels": 2,
            "known_future_download_bytes": load(OUT / "dependency_candidate.json")["additional_known_total_bytes"],
            "protocol_sha256": sha(OUT / "protocol.json"), "old_protected_files_unchanged": True,
            "images_GT_arrays_read": 0, "model_operations": 0, "weights_or_wheels_downloaded": 0,
            "installed": False, "runtime_compatibility_verified": False}


def prepare():
    if OUT.exists():
        return verify()
    vendor = verify_vendor()
    old = old_identities()
    env = environment()
    require(platform.python_version() == "3.12.7", "Unexpected preparation Python version")
    require(not (ROOT / ".venv-pretrained-seg").exists() and not (ROOT / "models/pretrained_segmentation").exists(), "Unexpected new runtime/model directory")
    dep = dependencies()
    resource = snapshot()
    require(resource["free_RAM_GiB"] >= 2 and resource["free_disk_GiB"] >= 1, "CPU preparation needs 2GiB RAM and 1GiB disk")
    protocol = {"schema": "pretrained-segmentation-r5-static-resource-v1", "candidate": CANDIDATE,
                "vendor": vendor, "old_identities": old, "design_sha256": sha(DESIGN),
                "mutable_plan_excluded": "plan progress changes; immutable design, candidate, source and input identities are bound instead",
                "implementation": implementation(), "environment": env, "resource_snapshot": resource,
                "new_environment": {"path": str(ROOT / ".venv-pretrained-seg"), "created": False},
                "future_resource_location": str(ROOT / "models/pretrained_segmentation/resources_v1"),
                "license_observation": "SMP MIT; pinned model card license other; no unrestricted weight grant established; personal noncommercial use only",
                "remaining": ["two runtime wheels and fixed generic weight not acquired", "new environment installation and static identity", "weight header and actual loading compatibility", "CPU training inputs and wrapper", "manual probe", "manual pure-real baseline", "separately authorized Q calibration/E exploratory evaluation"],
                "debugging_findings": ["Sandbox socket access failed with WinError10013; exact official metadata URLs worked with permitted network access; no version changes", "setuptools contains nested vendored METADATA; identify wheel metadata only at root .dist-info, not by broad suffix"]}
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "transaction.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "action": "CPU resource preparation only"})
    try:
        write(OUT / "protocol.json", protocol)
        write(OUT / "dependency_candidate.json", dep)
        write(OUT / "requirements-candidate.txt", "".join(f"{n}=={v}\n" for n, v in sorted(PINS.items())))
        forbidden = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN_IMPORTS)
        require(not forbidden, "Forbidden module imported")
        write(OUT / "verification.json", {"stage1_ready": True, "source_subset_files": len(AUTHOR), "source_git_blob_checks": True,
              "fixed_roles": ROLES, "closed_SSN_original_gate": False, "closed_BN_row": "A", "dependency_closure_metadata_only": True,
              "CPU_preparation_gate": {"min_free_RAM_GiB": 2, "min_free_disk_GiB": 1, "passed": True},
              "forbidden_imports": forbidden, "images_GT_probability_reads": 0, "models_or_optimizers_created": 0,
              "parameter_loads_or_forwards_or_updates": 0, "datasets_weights_wheels_downloaded": 0,
              "new_environment_created_or_old_environments_modified": False,
              "old_metadata_and_selected_branch_sources_unchanged": old == old_identities(),
              "new_source_AST_only": True, "runtime_or_tensor_compatibility_verified": False})
        write(OUT / "第1阶段静态来源与资源准备.md", report(protocol, dep))
        seal(OUT, "pretrained-segmentation-resource-preparation-v1")
    except Exception as exc:
        write(OUT / "failure.json", {"error": repr(exc), "rule": "retain partial directory; diagnose, do not auto prepare again"})
        raise
    return verify()


def report(p, dep):
    s = p["resource_snapshot"]
    local_bytes = sum(r["bytes"] for r in dep["records"].values() if r.get("local", True))
    old_count = len(set(p["old_identities"]["anchored_metadata_sha256"]) | set(p["old_identities"]["protected_branch_sources"]) | set(p["old_identities"]["anchored_core_sources"]))
    return f"""# 第1阶段：静态来源与资源准备

日期：2026年10月22日。本阶段完成固定来源、安装材料／依赖元信息、私人设备快照及独立CPU准备／核验入口；没有完成新环境、权重取得、实际加载兼容、输入缓存或训练。

## 来源、许可与既有身份

SMP v0.5.0固定commit `{COMMIT}`、tree `{TREE}`；20份必要源码／说明／许可按原字节保存，逐文件SHA256和Git blob与官方固定树核对。见[来源清单]({VENDOR.as_posix()}/SOURCE_MANIFEST.json)。这是审阅子集，不是完整可运行克隆，未执行任何作者代码。[固定官方来源](https://github.com/qubvel-org/segmentation_models.pytorch/tree/{COMMIT})。

官方ResNet18 ImageNet revision `{HF_REV}`的完整清单为.gitattributes／README.md／config.json／model.safetensors；模型权重{WEIGHT_BYTES}字节（{WEIGHT_BYTES/2**20:.4f}MiB），预期LFS SHA256 `{WEIGHT_HASH}`。只取得README／config和官方文件元信息，没有下载或读取权重／header／tensor。[官方模型卡](https://huggingface.co/smp-hub/resnet18.imagenet/blob/{HF_REV}/README.md)。模型卡指向原resnet18-5c106cde.pth，不将旧WideResNet权重视为可替代资源。

SMP代码是MIT；模型卡标license: other，未取得不受限制的权重授权，不能把代码许可套到模型。既有torchvision官方README／LICENSE快照保留预训练资源条件说明；这不确定原始训练数据的全部权利。用途仍私人设备、个人学习／非商业研究，不作商用许可确认。未来资源取得发现额外冲突时保留停止，不自动换权重。

旧split／输入完成标记及关键封印、冻结核心源码、SSN验收与已收束BN摘要共{old_count}个去重保护路径的hash已核对，范围见protocol。R5五张固定不变，N1885训练；Q200不训练／调配置，U241不用，E1004已见，只允许未来探索性评价。SSN原gate=false、原FP10/16和BN行A保持原含义，本阶段不重新评分、重算AP或解释为修复。没有遍历读取旧模型权重、数组或全部旧成果。

## 候选依赖与实际流量边界

新环境拟为`.venv-pretrained-seg`，尚未创建。核心候选Torch2.8.0+cu128／torchvision0.23.0+cu128、SMP0.5.0、timm1.0.20、HF Hub0.35.3、safetensors0.6.2；完整26项基本运行时闭包与全部实际markers见[dependency_candidate.json](dependency_candidate.json)，固定pin见[requirements-candidate.txt](requirements-candidate.txt)。这是依据声明的静态一致候选，不是安装或运行兼容证明。timm1.0.20是本轮明确选择、满足SMP下界的候选；并非宣称唯一支持版本或跟随latest。

本地已核验{dep['local_wheels']}份所需wheel的文件大小／实际SHA256、顶层METADATA与WHEEL标签，合计{local_bytes/2**30:.4f}GiB；不安装、不解压执行。已有手动材料可复用，不计新增网络流量。未选择diffusers、transformers、accelerate等无关包。Windows／Python markers依实际目标环境求值，不将Linux CUDA包或未选择extras当必需，HF的hf-xet条件排除与原因见完整依赖记录；不手改markers规避依赖。

缺SMP0.5.0 wheel {dep['missing_wheels'][0]['bytes']}字节与timm1.0.20 wheel {dep['missing_wheels'][1]['bytes']}字节，共{dep['additional_package_download_bytes']/2**20:.4f}MiB；加模型权重，未来已知新增文件载荷合计{dep['additional_known_total_bytes']/2**20:.4f}MiB。远程包名／URL／SHA256只据官方PyPI元信息，尚未取得实际wheel。HTTP开销、索引元信息、venv引导和后续CPU图像／指标依赖不计入该数；后续发现缺项先审阅，不自行换pin或扩大安装范围。

## 设备、实现与实际入口

解释器`{sys.executable}`，Python{platform.python_version()}，只用标准库及现有packaging解析依赖；未导入Torch／SMP／NumPy／图像／pickle包。完整现有环境包元信息与解释器hash见protocol，未修改旧环境。SMP源码语法通过AST静态核对，不等于可导入运行。

本次可用RAM{s['free_RAM_GiB']:.6f}GiB／总{s['total_RAM_GiB']:.6f}GiB，磁盘空闲{s['free_disk_GiB']:.6f}GiB；GPU {json.dumps(s['GPUs'], ensure_ascii=False)}。本阶段CPU门槛2GiB RAM／1GiB磁盘通过；未来训练／评分启动4.5GiB RAM／6GiB空闲显存／20GiB磁盘、精确AP 3GiB RAM／80GiB磁盘均需届时重核。本次不是峰值、训练时间或未来运行准入证明。

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.resources verify
```

`verify`实际已实现，严格只读、无联网，核对来源／协议／设计／旧身份／全部本地所选wheel／依赖；会流式hash约{local_bytes/2**30:.2f}GiB已有安装材料，耗时依磁盘。`prepare`仅本阶段CPU准备，已有完整目录只核验复用；`acquire`只取得允许清单的源码／文本，完整vendor只核验复用。未知文件、部分事务或hash冲突保留停止，不自动覆盖或重新下载。新环境或模型目录出现后，本阶段入口会要求按新阶段边界核对，不假装仍是取得前状态。

首次沙箱socket连接在接触远程数据前报WinError10013，相同官方地址在允许网络权限后成功；没有换main、镜像或pin。只读安装材料扫描还发现setuptools内有嵌套vendor METADATA；现只识别根级.dist-info/METADATA，身份与依赖不能取错。两项排查未写入或改变任何旧材料。

## 剩余工作

第2阶段另行授权取得两份wheel及固定权重、建立独立环境和静态header／依赖身份。权重真实可加载性、GN包装／固定BN行为、输入坐标、模型资源与可学习性都尚未验证；没有模型／优化器构造、前向、合成、训练、校准或评价。训练持续由用户手动启动，只提供已实现入口。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "prepare", "verify", "verify-vendor"])
    args = parser.parse_args()
    require(not any(n.split(".")[0] in FORBIDDEN_IMPORTS for n in sys.modules), "Forbidden module already imported")

    def audit(event, values):
        if event == "import" and values[0].split(".")[0] in FORBIDDEN_IMPORTS:
            raise RuntimeError("Static resource command forbids model/image/pickle imports")
        if event.startswith("pickle."):
            raise RuntimeError("Pickle execution forbidden")
        if event == "socket.connect" and args.command != "acquire":
            raise RuntimeError("prepare/verify cannot use network")
        if event == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(values[0]))
            if p.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise RuntimeError(f"Forbidden content access: {p.suffix}")
    sys.addaudithook(audit)
    for p in [VENDOR, OUT, WHEELS]:
        no_redirected_path(p)
    actions = {"acquire": acquire, "prepare": prepare, "verify": verify, "verify-vendor": verify_vendor}
    result = actions[args.command]()
    require(not any(n.split(".")[0] in FORBIDDEN_IMPORTS for n in sys.modules), "Forbidden import detected")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
