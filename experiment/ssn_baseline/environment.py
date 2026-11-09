"""Stage 2 offline environment metadata preparation; never imports ML packages."""
from __future__ import annotations

import argparse
import email
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile

from . import resources as r

ENV = r.ROOT / ".venv-ssn"
PYTHON = ENV / "Scripts/python.exe"
WHEELS = r.PROJECT / "实验/手动资源/wheels"
AUDIT = r.ROOT / "results/reference_interface_demo/manual_resource_review_v1/manual_resource_verification.json"
BASE = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/environment_v1"
INVENTORY = BASE / "inventory"
INSTALLATION = BASE / "installation"
PINS = {
    "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "numpy": "1.26.4",
    "pillow": "12.3.0", "filelock": "4.2.0", "typing-extensions": "4.16.0",
    "sympy": "1.14.0", "networkx": "3.7", "jinja2": "3.1.6",
    "markupsafe": "3.0.4", "fsspec": "2026.9.0", "mpmath": "1.3.0",
    "setuptools": "84.0.0", "packaging": "26.3",
}
# Executed only in the fresh environment. Metadata is read without importing
# installed targets. The guard also wraps pip itself during the offline install.
GUARD = '''
import sys
blocked = {"torch", "torchvision", "numpy", "PIL", "cv2", "anomalib", "ssn_baseline.model"}
assert not any(n.split(".")[0] in blocked for n in sys.modules)
def guard(event, args):
    if event == "import" and (args[0].split(".")[0] in blocked or args[0] == "ssn_baseline.model"):
        raise RuntimeError("metadata/install subprocess forbids ML/model imports")
    if event.startswith("pickle.") or event == "socket.connect":
        raise RuntimeError("no pickle or network in offline environment preparation")
sys.addaudithook(guard)
'''
METADATA_SCRIPT = GUARD + '''
import importlib.metadata as m, json, hashlib
from pathlib import Path
rows = []
for d in m.distributions():
    records = {}
    for f in d.files or []:
        if str(f).endswith((".dist-info/METADATA", ".dist-info/RECORD")):
            p = Path(d.locate_file(f))
            records[str(f).replace("\\\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({"name": d.metadata["Name"], "version": d.version, "requires": d.requires or [], "records_sha256": records})
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                  "python": sys.version, "distributions": sorted(rows, key=lambda x: x["name"].lower()),
                  "forbidden_imports": [n for n in sys.modules if n.split(".")[0] in blocked]}))
'''


def norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def implementation():
    return r.digest(Path(__file__))


def metadata_snapshot():
    completed = subprocess.run([str(PYTHON), "-I", "-B", "-X", "utf8", "-c", METADATA_SCRIPT], capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
    info = json.loads(completed.stdout)
    r.require(Path(info["prefix"]).resolve() == ENV.resolve() and info["prefix"] != info["base_prefix"], "Not isolated SSN environment")
    r.require(not info["forbidden_imports"], "ML import in metadata child")
    return info


def dependency_check(rows):
    # packaging is a CPU syntax/marker parser, not an ML/model package.
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import sys_tags, parse_tag
    versions = {norm(x["name"]): x["version"] for x in rows}
    environment = default_environment()
    environment["extra"] = ""
    compatible = set(sys_tags())
    active = []
    for row in rows:
        if row.get("requires_python"):
            r.require(SpecifierSet(row["requires_python"]).contains(environment["python_full_version"]), f"Python incompatible: {row['name']}")
        if row.get("tags"):
            r.require(any(parse_tag(t) & compatible for t in row["tags"]), f"Wheel tags incompatible {row['name']}")
        for text in row["requires"]:
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate(environment):
                continue
            name = norm(req.name)
            r.require(not req.url and name in versions and req.specifier.contains(versions[name]), f"Missing/incompatible active requirement {text}")
            active.append({"from": norm(row["name"]), "requirement": text, "resolved": versions[name]})
    return active


def verify_inventory():
    r.verify_seal(INVENTORY)
    p = r.load(INVENTORY / "inventory.json")
    r.require(p["implementation_sha256"] == implementation() and p["pins"] == PINS, "Environment implementation/pins changed")
    r.require(p["stage1_protocol_sha256"] == r.digest(r.OUT / "protocol.json") and p["audit_sha256"] == r.digest(AUDIT), "Inventory source identity changed")
    for row in p["wheels"]:
        file = WHEELS / row["filename"]
        r.require(file.stat().st_size == row["bytes"] and r.digest(file) == row["sha256"], f"Local wheel changed: {file.name}")
    r.require(dependency_check(p["wheels"]) == p["active_dependencies"], "Dependency resolution changed")
    return p


def inventory():
    r.verify()
    if INVENTORY.exists():
        p = verify_inventory()
        return {"inventory_ready": True, "reused": True, "wheels": len(p["wheels"]), "network_bytes_needed": 0}
    r.require(not ENV.exists() and not INSTALLATION.exists(), "Unexpected SSN installation state")
    audit = r.load(AUDIT)
    known = {x["filename"]: x for x in audit["wheels"]}
    rows = []
    for name, version in PINS.items():
        candidates = [p for p in WHEELS.glob("*.whl") if norm(p.name.split("-")[0]) == name and p.name.split("-")[1] == version]
        r.require(len(candidates) == 1, f"Expected one fixed local wheel for {name}=={version}")
        p = candidates[0]
        sha = r.digest(p)
        old = known[p.name]
        r.require(old["sha256"] == sha and old["bytes"] == p.stat().st_size and old["zip_crc_and_record_hash_verified"] is True, f"Prior wheel audit mismatch {p.name}")
        with zipfile.ZipFile(p) as z:
            # setuptools legitimately ships nested _vendor/*.dist-info notices;
            # only the wheel's unique TOP-LEVEL dist-info identifies the wheel.
            meta_names = [x for x in z.namelist() if x.endswith(".dist-info/METADATA") and x.count("/") == 1]
            r.require(len(meta_names) == 1, "Ambiguous wheel metadata")
            m = email.message_from_bytes(z.read(meta_names[0]))
            wheel = z.read(meta_names[0].rsplit("/", 1)[0] + "/WHEEL").decode("utf-8")
            r.require(norm(m["Name"]) == name and m["Version"] == version, "Wheel internal identity mismatch")
            rows.append({"name": m["Name"], "version": version, "filename": p.name, "bytes": p.stat().st_size,
                         "sha256": sha, "requires": m.get_all("Requires-Dist", []), "requires_python": m["Requires-Python"],
                         "tags": [s[5:].strip() for s in wheel.splitlines() if s.startswith("Tag:")],
                         "uncompressed_bytes": sum(x.file_size for x in z.infolist()),
                         "provenance": "user supplied wheel; matches prior local full CRC/RECORD audit, not an independently authenticated release signature"})
    active = dependency_check(rows)
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Offline preparation requires 2GiB RAM/20GiB disk")
    INVENTORY.mkdir(parents=True, exist_ok=False)
    p = {"schema": "ssn-offline-wheel-inventory-v1", "pins": PINS, "wheels": rows, "active_dependencies": active,
         "implementation_sha256": implementation(), "stage1_protocol_sha256": r.digest(r.OUT / "protocol.json"),
         "audit_path": str(AUDIT), "audit_sha256": r.digest(AUDIT), "resources": snap,
         "compressed_bytes": sum(x["bytes"] for x in rows), "uncompressed_bytes": sum(x["uncompressed_bytes"] for x in rows),
         "network_bytes_needed": 0, "all_extras_disabled": True, "old_environment_modified": False}
    r.write(INVENTORY / "inventory.json", p)
    r.write(INVENTORY / "requirements.lock", "\n".join(f"{norm(x['name'])}=={x['version']} --hash=sha256:{x['sha256']}" for x in rows) + "\n")
    r.seal(INVENTORY, "ssn-offline-inventory-v1")
    verify_inventory()
    return {"inventory_ready": True, "reused": False, "wheels": len(rows), "network_bytes_needed": 0,
            "compressed_GiB": p["compressed_bytes"] / 2**30, "uncompressed_GiB": p["uncompressed_bytes"] / 2**30}


def verify():
    r.verify()
    verify_inventory()
    r.verify_seal(INSTALLATION)
    p = r.load(INSTALLATION / "environment.json")
    r.require(p["metadata"] == metadata_snapshot(), "Installed environment metadata/RECORD identity changed")
    r.require(p["inventory_sha256"] == r.digest(INVENTORY / "complete.json") and p["implementation_sha256"] == implementation(), "Installation binding changed")
    r.require(p["python_sha256"] == r.digest(PYTHON) and p["pyvenv_cfg_sha256"] == r.digest(ENV / "pyvenv.cfg"), "Interpreter isolation identity changed")
    rows = p["metadata"]["distributions"]
    versions = {norm(x["name"]): x["version"] for x in rows}
    r.require({k: versions[k] for k in PINS} == PINS and set(versions) == set(PINS) | {"pip"}, "Unexpected/missing distribution")
    dependency_check(rows)
    return {"environment_ready_static": True, "packages": len(rows), "Torch_imports": 0,
            "model_compatibility_verified": False, "seal_sha256": r.digest(INSTALLATION / "complete.json")}


def install():
    r.verify()
    verify_inventory()
    if INSTALLATION.exists():
        return {"reused": True, **verify()}
    r.require(not ENV.exists(), "Unknown or partial .venv-ssn; no reinstall/repair permitted")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Offline install resource gate")
    INSTALLATION.mkdir(parents=True, exist_ok=False)
    r.write(INSTALLATION / "transaction.json", {"action": "offline isolated install; no ML imports", "inventory_sha256": r.digest(INVENTORY / "complete.json")})
    try:
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False, upgrade_deps=False).create(ENV)
        r.require("include-system-site-packages = false" in (ENV / "pyvenv.cfg").read_text(encoding="utf-8").lower(), "System package isolation not enabled")
        script = GUARD + "\nimport runpy\nsys.argv = ['pip'] + sys.argv[1:]\nrunpy.run_module('pip', run_name='__main__')\n"
        cmd = [str(PYTHON), "-I", "-B", "-X", "utf8", "-c", script, "--isolated", "--disable-pip-version-check", "install", "--no-input", "--no-cache-dir", "--no-compile", "--no-index", "--find-links", str(WHEELS), "--require-hashes", "-r", str(INVENTORY / "requirements.lock")]
        env = os.environ.copy()
        env["PIP_CONFIG_FILE"] = os.devnull
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=1800)
        r.write(INSTALLATION / "offline_install.txt", result.stdout + "\n" + result.stderr)
        r.require(result.returncode == 0, "Offline pip install failed; preserve environment and transaction")
        info = metadata_snapshot()
        versions = {norm(x["name"]): x["version"] for x in info["distributions"]}
        r.require({k: versions[k] for k in PINS} == PINS and set(versions) == set(PINS) | {"pip"}, "Unexpected installed dependency set")
        dependency_check(info["distributions"])
        r.write(INSTALLATION / "environment.json", {"schema": "ssn-environment-static-v1", "metadata": info,
            "python_sha256": r.digest(PYTHON), "pyvenv_cfg_sha256": r.digest(ENV / "pyvenv.cfg"),
            "inventory_sha256": r.digest(INVENTORY / "complete.json"), "implementation_sha256": implementation(),
            "network_bytes": 0, "ML_imports": 0, "forward_or_parameter_updates": 0,
            "metadata_not_runtime_compatibility": True, "resources_after": r.snapshot()})
        r.seal(INSTALLATION, "ssn-isolated-installation-v1")
    except Exception as exc:
        r.write(INSTALLATION / "failure.json", {"error": repr(exc), "rule": "retain directory and .venv-ssn; no automatic retries, pin changes or environment repair"})
        raise
    return {"reused": False, **verify()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "install", "verify"])
    args = parser.parse_args()
    # Parent is always the original interpreter to preserve stage-1 identity;
    # child metadata and pip use only the new environment.
    r.require(Path(sys.executable).resolve() == Path(r.load(r.OUT / "protocol.json")["environment"]["identity"]["executable"]).resolve(), "Use original .venv for static preparation")
    for target in [ENV, BASE]:
        for parent in [target, *target.parents]:
            if parent == r.ROOT:
                break
            r.require(not parent.is_symlink() and not parent.is_junction(), "Redirected new environment/output ancestor")
    def guard(event, arguments):
        if event == "import" and arguments[0].split(".")[0] in {"torch", "torchvision", "numpy", "PIL", "cv2", "anomalib"}:
            raise RuntimeError("Static environment preparation forbids ML imports")
        if event.startswith("pickle.") or event == "socket.connect":
            raise RuntimeError("No pickle or network authorized")
    sys.addaudithook(guard)
    print(json.dumps({"inventory": inventory, "install": install, "verify": verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
