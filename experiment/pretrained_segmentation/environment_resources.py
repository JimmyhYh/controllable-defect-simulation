"""Stage 2: fixed downloads, offline isolated installation, static verification.

No model package is imported, and no tensor is deserialized. Stage 1 stays
immutable; its source/seal identities are read independently of its old
pre-acquisition admission conditions. Run with the original .venv interpreter.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
import venv
from zipfile import ZipFile

from . import resources as r

PREP = r.ROOT / "results/pretrained_segmentation/ksdd2_r5_v1/environment_preparation_v1"
RES = r.ROOT / "models/pretrained_segmentation/resources_v1"
OUT = r.ROOT / "results/pretrained_segmentation/ksdd2_r5_v1/environment_v1"
ENV = r.ROOT / ".venv-pretrained-seg"
PYTHON = ENV / "Scripts/python.exe"
BUNDLE = Path(sys.base_prefix) / "Lib/ensurepip/_bundled/pip-24.2-py3-none-any.whl"
STAGE1_PROTOCOL = "15a682b1d50339c8efbd4fe847328d8176ab0c1c512d8e73a61fb9a904e31c08"
WEIGHT_URL = f"https://huggingface.co/smp-hub/resnet18.imagenet/resolve/{r.HF_REV}/model.safetensors"
BLOCKED = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "huggingface_hub", "safetensors", "numpy", "PIL", "cv2", "joblib"}
GUARD = '''
import sys, pickle, _pickle
blocked = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "huggingface_hub", "safetensors", "numpy", "PIL", "cv2", "joblib"}
assert not any(n.split('.')[0] in blocked for n in sys.modules)
def no_pickle(*a, **kw):
    raise RuntimeError('No pickle serialization/deserialization authorized')
class NoPickle:
    def __init__(self, *a, **kw): no_pickle()
for module in (pickle, _pickle):
    for name in ('load','loads','dump','dumps'): setattr(module, name, no_pickle)
    module.Unpickler = module.Pickler = NoPickle
def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in blocked:
        raise RuntimeError('Static install/metadata forbids ML/model imports')
    if event == 'socket.connect' or event.startswith('pickle.'):
        raise RuntimeError('Offline subprocess forbids network/pickle')
sys.addaudithook(audit)
'''
META_SCRIPT = GUARD + '''
import importlib.metadata as m, json, hashlib
from pathlib import Path
rows=[]
for d in m.distributions():
    fields={}
    for f in d.files or []:
        text=str(f).replace('\\\\','/')
        if text.count('/') == 1 and text.endswith(('.dist-info/METADATA','.dist-info/RECORD','.dist-info/WHEEL','.dist-info/INSTALLER')):
            p=Path(d.locate_file(f)); fields[text]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({'name':d.metadata['Name'],'version':d.version,'requires_dist':d.requires or [],'metadata_files_sha256':fields})
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':sys.version,
 'distributions':sorted(rows,key=lambda x:x['name'].lower()),'blocked_imports':[n for n in sys.modules if n.split('.')[0] in blocked],
 'pickle_operations':'disabled; stdlib pickle module may be imported by pip internals'}))
'''


def implementation():
    return {n: r.sha(r.ROOT / n) for n in ["pretrained_segmentation/environment_resources.py", "pretrained_segmentation/README_environment_resources.md"]}


def stage1():
    """Source/metadata identity, not stage-1's now-obsolete absence gates."""
    r.verify_seal(r.OUT)
    r.require(r.sha(r.OUT / "protocol.json") == STAGE1_PROTOCOL, "Fixed stage1 protocol changed")
    p = r.load(r.OUT / "protocol.json")
    r.require(p["candidate"] == r.CANDIDATE and p["vendor"] == r.verify_vendor(), "Fixed source/candidate changed")
    r.require(p["old_identities"] == r.old_identities() and p["design_sha256"] == r.sha(r.DESIGN), "Old metadata or design changed")
    r.require(p["implementation"] == r.implementation() and p["environment"] == r.environment(), "Stage1 code/old environment metadata changed")
    d = r.load(r.OUT / "dependency_candidate.json")
    r.require(d == r.dependencies(), "Frozen local materials or dependency candidate changed")
    r.require(d["pins"] == r.PINS and d["additional_known_total_bytes"] == 49464388, "Fixed dependency/download scope changed")
    r.require(Path(sys.executable).resolve() == Path(p["environment"]["executable"]).resolve(), "Use original .venv interpreter for static stages")
    return {"stage1_complete_sha256": r.sha(r.OUT / "complete.json"), "protocol_sha256": STAGE1_PROTOCOL,
            "dependency_sha256": r.sha(r.OUT / "dependency_candidate.json"), "vendor_complete_sha256": r.sha(r.VENDOR / "complete.json"),
            "design_sha256": r.sha(r.DESIGN), "old_identities": p["old_identities"], "old_environment": p["environment"]}


def old_environment_metadata():
    result = {}
    for name in [".venv", ".venv-ssn", ".venv-reference-interface"]:
        folder = r.ROOT / name
        rows = []
        for d in md.distributions(path=[str(folder / "Lib/site-packages")]):
            rows.append({"name": d.metadata["Name"], "version": d.version,
                         "metadata_sha256": hashlib.sha256(d.read_text("METADATA").encode()).hexdigest()})
        result[name] = {"python_sha256": r.sha(folder / "Scripts/python.exe"),
                       "pyvenv_cfg_sha256": r.sha(folder / "pyvenv.cfg"), "distributions": sorted(rows, key=lambda x: x["name"].lower())}
    return result


def resource_requests():
    dep = r.load(r.OUT / "dependency_candidate.json")
    result = [{"path": "packages/" + x["filename"], "url": x["url"], "bytes": x["bytes"], "sha256": x["sha256"], "kind": "wheel"} for x in dep["missing_wheels"]]
    result.append({"path": "encoder/model.safetensors", "url": WEIGHT_URL, "bytes": r.WEIGHT_BYTES, "sha256": r.WEIGHT_HASH, "kind": "generic ImageNet encoder"})
    return result


def verify_preparation():
    r.verify_seal(PREP)
    r.require(r.file_set(PREP) == {"binding.json", "requirements.lock", "complete.json"}, "Preparation strict listing changed")
    p = r.load(PREP / "binding.json")
    r.require(p["stage1"] == stage1() and p["implementation"] == implementation(), "Stage2 preparation identity changed")
    r.require(p["old_environment_metadata"] == old_environment_metadata(), "Old environment metadata changed")
    r.require(p["requests"] == resource_requests() and p["bootstrap_pip"]["sha256"] == r.sha(BUNDLE), "Fixed acquisition/bootstrap identity changed")
    expected = lock_text()
    r.require((PREP / "requirements.lock").read_text(encoding="utf-8") == expected, "Offline requirements changed")
    return p


def lock_text():
    dep = r.load(r.OUT / "dependency_candidate.json")
    rows = [f"{n}=={v} --hash=sha256:{dep['records'][n]['sha256']}" for n, v in sorted(r.PINS.items())]
    rows.append(f"pip==24.2 --hash=sha256:{r.sha(BUNDLE)}")
    return "\n".join(rows) + "\n"


def prepare():
    if PREP.exists():
        verify_preparation()
        return {"stage2_prepared": True, "reused": True}
    bound = stage1()
    r.require(not RES.exists() and not ENV.exists() and not OUT.exists(), "Unknown stage2 target exists")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Static environment preparation needs RAM2GiB/disk20GiB")
    bootstrap = r.wheel_record(BUNDLE)
    r.require(bootstrap["name"] == "pip" and bootstrap["version"] == "24.2", "Unexpected bundled pip")
    p = {"schema": "pretrained-segmentation-environment-preparation-v1", "stage1": bound,
         "implementation": implementation(), "requests": resource_requests(), "bootstrap_pip": bootstrap,
         "old_environment_metadata": old_environment_metadata(), "resource_snapshot": snap,
         "environment_path": str(ENV), "resource_path": str(RES), "pins": r.PINS,
         "installation": "offline wheels only; --require-hashes --no-deps --no-index; builtin pip module bootstrap; workspace TEMP; no package imports",
         "license_observation": "SMP MIT; pinned encoder card license other; no unrestricted ImageNet-weight grant established; personal noncommercial research only",
         "runtime_compatibility_verified": False}
    PREP.mkdir(parents=True, exist_ok=False)
    r.write(PREP / "binding.json", p)
    r.write(PREP / "requirements.lock", lock_text())
    r.seal(PREP, "pretrained-segmentation-environment-preparation-v1")
    verify_preparation()
    return {"stage2_prepared": True, "reused": False, "known_network_file_bytes": 49464388, "requests": 3}


class FixedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        # HF resolve may use its signed CDN. Never print/store signed query URLs.
        allowed = parsed.scheme == "https" and parsed.username is None and parsed.password is None and (parsed.hostname == "huggingface.co" or parsed.hostname.endswith(".huggingface.co") or parsed.hostname.endswith(".xethub.hf.co") or parsed.hostname.endswith(".hf.co"))
        r.require(allowed and "huggingface" in req.full_url or allowed and "hf.co" in urllib.parse.urlparse(req.full_url).hostname, "Unexpected acquisition redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(row):
    path = RES / row["path"]
    r.safe_name(row["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    r.require(not path.exists() and not part.exists(), "Unknown or partial download; do not retry")
    allowed = {x["url"] for x in resource_requests()}
    r.require(row["url"] in allowed, "URL not in frozen resource requests")
    opener = urllib.request.build_opener(FixedRedirect)
    h, count, reported = hashlib.sha256(), 0, 0
    with opener.open(urllib.request.Request(row["url"], headers={"User-Agent": "pretrained-segmentation-fixed-resource-acquisition"}), timeout=45) as response, part.open("xb") as f:
        r.require(response.status == 200, "Unexpected download HTTP status")
        if response.headers.get("Content-Length"):
            r.require(int(response.headers["Content-Length"]) == row["bytes"], "Unexpected response byte count")
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            r.require(count <= row["bytes"], "Response exceeds fixed resource size")
            f.write(block)
            h.update(block)
            if count - reported >= 8 * 1024 * 1024:
                print(json.dumps({"resource": row["path"], "received_bytes": count, "expected_bytes": row["bytes"]}), flush=True)
                reported = count
        f.flush()
        os.fsync(f.fileno())
    r.require(count == row["bytes"] and h.hexdigest() == row["sha256"], "Downloaded bytes/hash mismatch")
    part.rename(path)
    return {**row, "actual_bytes": count, "actual_sha256": h.hexdigest()}


def unique_json(pairs):
    out = {}
    for k, v in pairs:
        r.require(k not in out, "Duplicate JSON header key")
        out[k] = v
    return out


def header_identity():
    path = RES / "encoder/model.safetensors"
    r.require(path.stat().st_size == r.WEIGHT_BYTES and r.sha(path) == r.WEIGHT_HASH, "Generic weight identity mismatch")
    with path.open("rb") as f:
        raw = f.read(8)
        r.require(len(raw) == 8, "Missing safetensors header length")
        length = struct.unpack("<Q", raw)[0]
        r.require(2 <= length <= 1024 * 1024 and 8 + length < path.stat().st_size, "Unexpected safetensors header size")
        raw = f.read(length)
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json)
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    entries, counts, cursor = [], {}, 0
    for name, row in header.items():
        if name == "__metadata__":
            r.require(isinstance(row, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in row.items()), "Invalid header metadata")
            continue
        r.require(isinstance(row, dict) and set(row) == {"dtype", "shape", "data_offsets"}, "Invalid tensor descriptor")
        dtype, shape, offsets = row["dtype"], row["shape"], row["data_offsets"]
        r.require(dtype in sizes and isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), "Unsupported descriptor dtype/shape")
        r.require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets) and 0 <= offsets[0] <= offsets[1], "Invalid descriptor offsets")
        n = math.prod(shape)
        r.require(offsets[1] - offsets[0] == n * sizes[dtype], "Descriptor byte range mismatch")
        counts[dtype] = counts.get(dtype, 0) + n
        entries.append({"name": name, **row})
    for row in sorted(entries, key=lambda x: (x["data_offsets"][0], x["data_offsets"][1])):
        r.require(row["data_offsets"][0] == cursor, "Header payload gaps/overlap")
        cursor = row["data_offsets"][1]
    r.require(cursor + length + 8 == path.stat().st_size, "Header does not cover full payload")
    by_name = {x["name"]: x for x in entries}
    required_shapes = {"conv1.weight": [64, 3, 7, 7], "bn1.running_mean": [64], "bn1.running_var": [64],
                       "layer1.0.conv1.weight": [64, 64, 3, 3], "layer2.0.conv1.weight": [128, 64, 3, 3],
                       "layer3.0.conv1.weight": [256, 128, 3, 3], "layer4.0.conv1.weight": [512, 256, 3, 3]}
    for name, shape in required_shapes.items():
        r.require(name in by_name and by_name[name]["shape"] == shape, "ResNet18 descriptor family mismatch")
    return {"header_bytes": length, "header_sha256": hashlib.sha256(raw).hexdigest(), "metadata": header.get("__metadata__", {}),
            "tensor_descriptors": entries, "tensor_count": len(entries), "dtype_element_counts": counts,
            "classification_fc_keys": sorted(n for n in by_name if n.startswith("fc.")),
            "num_batches_tracked_keys": sorted(n for n in by_name if n.endswith("num_batches_tracked")),
            "scope": "JSON descriptors and full-file hash only; no tensor payload decoded, no parameter loaded; strict model-load compatibility still pending"}


def wheel_static_checks(path, expected, source_subset=False):
    row = r.wheel_record(path)
    r.require(row["name"] == expected["name"] and row["version"] == expected["version"] and row["bytes"] == expected["bytes"] and row["sha256"] == expected["sha256"], "Acquired wheel identity mismatch")
    from packaging.requirements import Requirement
    actual = sorted(str(Requirement(x)) for x in row["requires_dist"])
    remote = sorted(str(Requirement(x)) for x in expected["requires_dist"])
    r.require(actual == remote and row["requires_python"] == expected["requires_python"], "Actual wheel declaration differs from frozen PyPI candidate")
    licenses, compared = [], []
    with ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("segmentation_models_pytorch/") and n in r.AUTHOR and source_subset:
                r.require(hashlib.sha256(z.read(n)).hexdigest() == r.sha(r.VENDOR / n), "Published wheel/source commit differs")
                compared.append(n)
            if ".dist-info/" in n and not n.endswith("/") and Path(n).name.lower().startswith(("license", "copying", "notice")):
                raw = z.read(n)
                r.require(len(raw) < 256 * 1024, "Unexpected license text size")
                text = raw.decode("utf-8")
                licenses.append({"path": n, "sha256": hashlib.sha256(raw).hexdigest(), "text": text})
    if source_subset:
        r.require(set(compared) == {n for n in r.AUTHOR if n.endswith(".py")}, "Missing SMP source subset in wheel")
    r.require(licenses, "Wheel license text unavailable; retain and review")
    return {**row, "license_texts": licenses, "source_subset_matches": compared}


def acquire():
    verify_preparation()
    if RES.exists():
        return {"reused": True, **verify_acquisition()}
    RES.mkdir(parents=True, exist_ok=False)
    r.write(RES / "transaction.json", {"action": "three fixed resources; no automatic retries", "preparation_sha256": r.sha(PREP / "complete.json")})
    try:
        rows = [download(x) for x in resource_requests()]
        for dest, original in {"encoder/config.json": "upstream/resnet18-config.json", "encoder/README.md": "upstream/resnet18-README.md", "encoder/torchvision-resource-conditions.md": "upstream/torchvision-v0.23.0-README.md"}.items():
            r.write(RES / dest, (r.VENDOR / original).read_bytes())
        dep = r.load(r.OUT / "dependency_candidate.json")
        wheels = {name: wheel_static_checks(RES / "packages" / dep["records"][name]["filename"], dep["records"][name], name == "segmentation-models-pytorch") for name in ["segmentation-models-pytorch", "timm"]}
        r.write(RES / "wheel_verification.json", wheels)
        r.write(RES / "safetensors_header.json", header_identity())
        r.write(RES / "acquisition.json", {"preparation_sha256": r.sha(PREP / "complete.json"), "implementation": implementation(),
                "requests": rows, "network_file_payload_bytes": sum(x["actual_bytes"] for x in rows),
                "HTTP_and_metadata_overhead_not_included": True, "local_install_wheels_reused": 24,
                "tensor_payload_decoded": False, "license_observation": r.load(PREP / "binding.json")["license_observation"]})
        r.seal(RES, "pretrained-segmentation-acquired-resources-v1")
    except Exception as exc:
        r.write(RES / "failure.json", {"error_type": type(exc).__name__, "error": safe_error(exc), "rule": "retain bytes/part files; do not retry, replace or change pin"})
        raise RuntimeError("Resource acquisition stopped; preserved failure.json; diagnose without automatic retry") from None
    return {"reused": False, **verify_acquisition()}


def safe_error(exc):
    # Some CDN exceptions embed signed URLs; retain type/status without secrets.
    return re_safe_url(str(exc))[:2000]


def re_safe_url(text):
    import re
    return re.sub(r"https://[^\s'\"]+", "[URL omitted; see fixed request identity]", text)


def verify_acquisition():
    verify_preparation()
    r.verify_seal(RES)
    expected = {x["path"] for x in resource_requests()} | {"transaction.json", "encoder/config.json", "encoder/README.md", "encoder/torchvision-resource-conditions.md", "wheel_verification.json", "safetensors_header.json", "acquisition.json", "complete.json"}
    r.require(r.file_set(RES) == expected, "Acquisition strict file list changed")
    a = r.load(RES / "acquisition.json")
    r.require(a["implementation"] == implementation() and a["preparation_sha256"] == r.sha(PREP / "complete.json"), "Acquisition binding changed")
    r.require(a["requests"] == [{**x, "actual_bytes": x["bytes"], "actual_sha256": x["sha256"]} for x in resource_requests()] and a["network_file_payload_bytes"] == 49464388, "Acquisition request identity changed")
    for dest, original in {"encoder/config.json": "upstream/resnet18-config.json", "encoder/README.md": "upstream/resnet18-README.md", "encoder/torchvision-resource-conditions.md": "upstream/torchvision-v0.23.0-README.md"}.items():
        r.require(r.sha(RES / dest) == r.sha(r.VENDOR / original), "Resource configuration/notice changed")
    r.require(r.load(RES / "safetensors_header.json") == header_identity(), "Saved header verification changed")
    dep = r.load(r.OUT / "dependency_candidate.json")
    for name, saved in r.load(RES / "wheel_verification.json").items():
        r.require(saved == wheel_static_checks(RES / "packages" / dep["records"][name]["filename"], dep["records"][name], name == "segmentation-models-pytorch"), "Wheel static verification changed")
    return {"resources_ready_static": True, "network_file_payload_bytes": 49464388, "generic_weight_sha256": r.WEIGHT_HASH,
            "complete_sha256": r.sha(RES / "complete.json"), "tensor_loads": 0, "runtime_compatibility_verified": False}


def child_environment(temp=None):
    env = os.environ.copy()
    env.update({"PIP_CONFIG_FILE": os.devnull, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
    if temp is not None:
        env.update({"TEMP": str(temp), "TMP": str(temp)})
    return env


def metadata_snapshot():
    r.require(PYTHON.is_file(), "New environment interpreter missing")
    p = subprocess.run([str(PYTHON), "-I", "-B", "-X", "utf8", "-c", META_SCRIPT], capture_output=True, text=True, encoding="utf-8", env=child_environment(), timeout=90)
    r.require(p.returncode == 0, "Guarded metadata subprocess failed: " + p.stderr[-1500:])
    info = json.loads(p.stdout)
    r.require(Path(info["prefix"]).resolve() == ENV.resolve() and Path(info["base_prefix"]).resolve() != ENV.resolve() and not info["blocked_imports"], "Environment isolation/import guard failed")
    versions = {r.canonical(x["name"]): x["version"] for x in info["distributions"]}
    r.require(len(versions) == len(info["distributions"]) and versions == {**r.PINS, "pip": "24.2"}, "Unexpected installed distribution set/version")
    from packaging.requirements import Requirement
    candidate = r.load(r.OUT / "dependency_candidate.json")
    for row in info["distributions"]:
        name = r.canonical(row["name"])
        if name in r.PINS:
            r.require(sorted(str(Requirement(t)) for t in row["requires_dist"]) == sorted(str(Requirement(t)) for t in candidate["records"][name]["requires_dist"]), "Installed dependency metadata differs from frozen wheel")
            for text in row["requires_dist"]:
                req = Requirement(text)
                if req.marker is None or req.marker.evaluate(environment=candidate["marker_environment"]):
                    r.require(r.canonical(req.name) in versions and req.specifier.contains(versions[r.canonical(req.name)]), "Installed dependency closure conflict")
    return info


def installed_source_identity():
    site = ENV / "Lib/site-packages"
    matched = {}
    for name in r.AUTHOR:
        if name.endswith(".py"):
            r.require(r.sha(site / name) == r.sha(r.VENDOR / name), "Installed SMP source subset differs from fixed commit")
            matched[name] = r.sha(site / name)
    return matched


def install():
    verify_acquisition()
    if OUT.exists():
        return {"reused": True, **verify()}
    r.require(not ENV.exists(), "Unknown/partial new environment; do not reinstall or repair")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Offline installation needs RAM2GiB/disk20GiB")
    OUT.mkdir(parents=True, exist_ok=False)
    temp = OUT / "workspace_temp"
    temp.mkdir(exist_ok=False)
    r.write(OUT / "transaction.json", {"action": "one offline isolated install, no package/model imports", "preparation_sha256": r.sha(PREP / "complete.json"), "acquisition_sha256": r.sha(RES / "complete.json")})
    try:
        venv.EnvBuilder(with_pip=False, clear=False, symlinks=False, upgrade_deps=False).create(ENV)
        r.require("include-system-site-packages = false" in (ENV / "pyvenv.cfg").read_text(encoding="utf-8").lower(), "System packages not isolated")
        script = GUARD + '''
import runpy, json
sys.path.insert(0, sys.argv.pop(1))
sys.argv=['pip']+sys.argv[1:]
try:
    runpy.run_module('pip',run_name='__main__',alter_sys=True)
except SystemExit as exc:
    if exc.code not in (None,0): raise
assert not any(n.split('.')[0] in blocked for n in sys.modules)
print('STATIC_INSTALL_GUARD_OK')
'''
        cmd = [str(PYTHON), "-I", "-B", "-X", "utf8", "-c", script, str(BUNDLE), "--isolated", "--disable-pip-version-check", "install", "--no-input", "--no-cache-dir", "--no-compile", "--no-index", "--no-deps", "--find-links", str(r.WHEELS), "--find-links", str(RES / "packages"), "--find-links", str(BUNDLE.parent), "--require-hashes", "-r", str(PREP / "requirements.lock")]
        process = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=child_environment(temp), timeout=1800)
        r.write(OUT / "offline_install.txt", process.stdout + "\n" + process.stderr)
        r.require(process.returncode == 0 and "STATIC_INSTALL_GUARD_OK" in process.stdout, "Offline guarded installation failed")
        info = metadata_snapshot()
        sources = installed_source_identity()
        after = r.snapshot()
        r.write(OUT / "environment.json", {"schema": "pretrained-segmentation-isolated-environment-static-v1", "metadata": info,
                "python_sha256": r.sha(PYTHON), "pyvenv_cfg_sha256": r.sha(ENV / "pyvenv.cfg"),
                "installed_SMP_source_subset_sha256": sources, "preparation_sha256": r.sha(PREP / "complete.json"),
                "resource_complete_sha256": r.sha(RES / "complete.json"), "implementation": implementation(),
                "resources_before": snap, "resources_after": after, "runtime_compatibility_verified": False,
                "ML_imports": 0, "model_or_optimizer_or_forward_or_training": 0, "offline_install_network_bytes": 0,
                "old_environments_metadata_unchanged": old_environment_metadata() == r.load(PREP / "binding.json")["old_environment_metadata"]})
        r.write(OUT / "verification.json", {"stage2_ready_static": True, "distributions": 27, "runtime_candidate_packages": 26,
                "bootstrap_pip": "24.2", "acquired_network_file_bytes": 49464388, "generic_weight_bytes": r.WEIGHT_BYTES,
                "generic_weight_sha256": r.WEIGHT_HASH, "header_only_no_tensor_parse": True,
                "stage1_and_old_protected_identities_unchanged": stage1() == r.load(PREP / "binding.json")["stage1"],
                "actual_installed_versions_and_metadata_checked": True, "ML_imports_or_model_operations": 0,
                "RGB_GT_probability_accesses": 0, "static_not_runtime_compatibility": True})
        r.write(OUT / "第2阶段独立环境与资源静态核验.md", report(info, after))
        r.seal(OUT, "pretrained-segmentation-isolated-environment-v1")
    except Exception as exc:
        r.write(OUT / "failure.json", {"error_type": type(exc).__name__, "error": safe_error(exc), "rule": "retain new environment and transaction; no automatic repair/reinstall/pin change"})
        raise
    return {"reused": False, **verify()}


def verify():
    verify_acquisition()
    r.verify_seal(OUT)
    r.require(r.file_set(OUT) == {"transaction.json", "offline_install.txt", "environment.json", "verification.json", "第2阶段独立环境与资源静态核验.md", "complete.json"}, "Environment result strict file list changed")
    p = r.load(OUT / "environment.json")
    r.require(p["metadata"] == metadata_snapshot() and p["installed_SMP_source_subset_sha256"] == installed_source_identity(), "Installed metadata/source identity changed")
    r.require(p["implementation"] == implementation() and p["preparation_sha256"] == r.sha(PREP / "complete.json") and p["resource_complete_sha256"] == r.sha(RES / "complete.json"), "Environment binding changed")
    r.require(p["python_sha256"] == r.sha(PYTHON) and p["pyvenv_cfg_sha256"] == r.sha(ENV / "pyvenv.cfg"), "Interpreter/isolation identity changed")
    r.require(p["old_environments_metadata_unchanged"] is True, "Old environment protection failed")
    return {"stage2_ready_static": True, "cache_reused": True, "runtime_candidate_packages": 26, "total_distributions_including_pip": 27,
            "generic_weight_bytes": r.WEIGHT_BYTES, "generic_weight_sha256": r.WEIGHT_HASH,
            "network_file_payload_bytes": 49464388, "model_imports_or_parameter_loads_or_forwards": 0,
            "tensor_payload_parsed": False, "old_and_stage1_identities_unchanged": True,
            "runtime_compatibility_verified": False, "environment_complete_sha256": r.sha(OUT / "complete.json")}


def report(info, snap):
    h = r.load(RES / "safetensors_header.json")
    return f"""# 第2阶段：独立环境与通用资源静态核验

2026年10月22日。本阶段实际完成独立环境安装、三份指定文件取得及CPU静态核验；没有导入模型包、加载参数、解析tensor数值、运行前向或训练，不是运行兼容通过。

## 来源与取得

第1阶段protocol SHA256 `{STAGE1_PROTOCOL}`，固定SMP commit `{r.COMMIT}`、ResNet18 revision `{r.HF_REV}`及旧split／源码／已收束SSN元信息独立核对不变。没有调用原第1阶段的“新环境不存在”准入冒充第2阶段核验，未改其源码／README／vendor／配置／结果。

复用24份固定本地安装包，只联网取得SMP0.5.0、timm1.0.20 wheel和46805552字节model.safetensors；实际三个文件载荷合计49464388字节，约47.1729MiB，HTTP等开销另计。请求、实际全文件hash与封印见[资源清单]({RES.as_posix()}/acquisition.json)。无自动重试、镜像或pin替换；HF固定resolve地址允许转向官方HTTPS签名CDN，不保存签名查询参数。

编码器SHA256 `{r.WEIGHT_HASH}`，原README／配置复制自封存来源。只解析8字节长度与JSON header：{h['tensor_count']}个tensor描述，dtype元素数{json.dumps(h['dtype_element_counts'])}；分类fc键{h['classification_fc_keys']}，num_batches_tracked键{len(h['num_batches_tracked_keys'])}个。完整描述与偏移范围见[safetensors_header.json]({RES.as_posix()}/safetensors_header.json)。全文件hash读取原始字节，未解码tensor值，未做dtype转换或参数加载；后续必须单独核对严格加载与BN缺项处理，不能拿静态header等同模型加载成功。

SMP wheel中既有Python源码子集逐文件匹配固定commit，timm及SMP实际wheel元信息、声明与许可文本已核验保存。SMP源码MIT、timm wheel许可按实际文本保留；骨干模型卡仍是license: other，未建立不受限制的权重授权。私人设备、个人学习／非商业研究边界保持，公司材料隔离，不作商业授权确认。

## 独立安装与身份

新环境`{ENV.as_posix()}`，Python3.12.7。26个固定候选包加本机Python捆绑pip24.2，共{len(info['distributions'])}项；完整实际传递依赖、METADATA／RECORD／WHEEL／INSTALLER摘要和解释器身份见[environment.json](environment.json)，固定hash安装清单见[requirements.lock]({PREP.as_posix()}/requirements.lock)。CPU后续图像／指标依赖未扩装。

安装为离线wheel、no-index／no-deps／require-hashes／no-cache／no-compile，无源码构建或新依赖解析选型；实际安装后逐项核对版本与依赖闭包。先建立不含pip的隔离venv，再用Python自带pip wheel的模块语义引导，同一受保护新解释器一次安装。独立项目TEMP避免既有ensurepip临时目录权限问题，不注入旧site-packages、不升级旧环境。pip内部可导入标准库pickle模块，但load／dump函数及Pickler／Unpickler与pickle审计操作被禁用，没有pickle反序列化。

禁止Torch／torchvision／SMP／timm／HF模型包等导入，安装子进程有完成guard标记；元信息核验子进程独立检查0模型包导入。旧三个环境的解释器、pyvenv.cfg与包元信息身份不变；这不是遍历hash全部旧已安装二进制。新环境只核实际包元信息／RECORD身份与SMP来源子集，不虚称全库运行验证。

## 资源与下一阶段

结束快照可用RAM{snap['free_RAM_GiB']:.6f}GiB、磁盘{snap['free_disk_GiB']:.6f}GiB，GPU {json.dumps(snap['GPUs'],ensure_ascii=False)}。本阶段CPU／安装门槛2GiB RAM／20GiB磁盘；后续训练／评分仍需4.5GiB RAM／6GiB空闲显存／20GiB磁盘重新准入，当前快照不证明模型峰值或训练耗时。

严格R5／N1885不变，Q不训练、U不用、E已见不选型。没有读取任何RGB／GT／概率内容；真实图像输入、GN包装／固定BN、严格加载、可学习性及GPU资源均待后续阶段。所有训练继续由用户手动启动，当前不提供训练命令。

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.environment_resources verify
```

这是已实现的第2阶段只读核验入口，核对原来源／环境身份、取得资源、header与新环境元信息，不联网或安装，不导入模型。完整同身份缓存只读复用，部分／未知事务或hash冲突保留停止，不自动补下载、修环境或重跑。第3阶段只准备CPU输入／训练包装，不隐含模型前向，另行指定执行。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "acquire", "install", "verify-acquisition", "verify"])
    args = parser.parse_args()
    r.require(not any(n.split(".")[0] in BLOCKED for n in sys.modules), "Forbidden model package already imported")
    for path in [PREP, RES, OUT, ENV, BUNDLE]:
        r.no_redirected_path(path)

    def audit(event, values):
        if event == "import" and values[0].split(".")[0] in BLOCKED:
            raise RuntimeError("Static stage forbids ML/model imports")
        if event.startswith("pickle.") or event == "socket.connect" and args.command != "acquire":
            raise RuntimeError("No pickle; only acquire may connect to network")
        if event == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(values[0]))
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle"}:
                raise RuntimeError("No image/GT/array/old-weight content access")
            if p.suffix.lower() == ".safetensors" and p.resolve() != (RES / "encoder/model.safetensors").resolve():
                raise RuntimeError("Only fixed new generic resource may be hashed/header-inspected")
    sys.addaudithook(audit)
    actions = {"prepare": prepare, "acquire": acquire, "install": install, "verify-acquisition": verify_acquisition, "verify": verify}
    print(json.dumps(actions[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
