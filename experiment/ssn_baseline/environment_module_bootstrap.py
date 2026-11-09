"""Independent module-semantics bootstrap after two known pre-install failures.

No failed original transaction is retried or rewritten. The interpreter and
pyvenv config must be the known empty, isolated environment from that failure.
Use the Python-bundled pip wheel and workspace TEMP, all offline, no ML import.
"""
from __future__ import annotations

import argparse
import email
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from . import resources as r
from . import environment as e

BASE = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/environment_module_bootstrap_v1"
PREP = BASE / "preparation"
INSTALLATION = BASE / "installation"
TEMP = BASE / "runtime_temp"
BUNDLE = Path(sys.base_prefix) / "Lib/ensurepip/_bundled/pip-24.2-py3-none-any.whl"


def implementation():
    return r.digest(Path(__file__))


def original_failure():
    folders = [e.INSTALLATION, r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/environment_bootstrap_continuation_v1/installation"]
    result = {}
    for i, folder in enumerate(folders):
        names = r.files_under(folder)
        expected = {"transaction.json", "failure.json"} | ({"offline_install.txt"} if i else set())
        r.require(names == expected and not (folder / "complete.json").exists(), "Historical failed transaction changed")
        fail = r.load(folder / "failure.json")
        if i == 0:
            r.require("ensurepip" in fail["error"] and "CalledProcessError" in fail["error"], "Not known ensurepip failure")
        else:
            text = (folder / "offline_install.txt").read_text(encoding="utf-8")
            r.require("To modify pip, please run the following command" in text and "Installing collected packages" not in text, "Not known pre-install pip module invocation failure")
        result[folder.relative_to(r.ROOT).as_posix()] = {n: r.digest(folder / n) for n in sorted(names)}
    return result


def verify_preparation():
    r.verify_seal(PREP)
    p = r.load(PREP / "protocol.json")
    r.verify()
    e.verify_inventory()
    r.require(p["original_failure"] == original_failure(), "Original failure identity changed")
    r.require(p["implementation_sha256"] == implementation() and p["inventory_sha256"] == r.digest(e.INVENTORY / "complete.json"), "Continuation implementation/inventory changed")
    r.require(p["bootstrap_wheel_sha256"] == r.digest(BUNDLE), "Bundled bootstrap pip changed")
    r.require(p["python_sha256"] == r.digest(e.PYTHON) and p["pyvenv_cfg_sha256"] == r.digest(e.ENV / "pyvenv.cfg"), "Known isolated interpreter/config changed")
    return p


def prepare():
    if PREP.exists():
        p = verify_preparation()
        return {"continuation_prepared": True, "reused": True, "original_failure_preserved": True}
    r.verify()
    e.verify_inventory()
    fail = original_failure()
    r.require(not INSTALLATION.exists() and not TEMP.exists(), "Unknown continuation state")
    info = e.metadata_snapshot()
    r.require(info["distributions"] == [] and list((e.ENV / "Lib/site-packages").iterdir()) == [], "Continuation limited to the verified empty environment")
    with zipfile.ZipFile(BUNDLE) as z:
        entries = [n for n in z.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
        r.require(len(entries) == 1, "Ambiguous Python-bundled pip metadata")
        m = email.message_from_bytes(z.read(entries[0]))
        r.require(m["Name"] == "pip" and m["Version"] == "24.2", "Unexpected bootstrap pip version")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Continuation resource gate")
    PREP.mkdir(parents=True, exist_ok=False)
    p = {"schema": "ssn-empty-bootstrap-continuation-v1", "original_failure": fail,
         "implementation_sha256": implementation(), "inventory_sha256": r.digest(e.INVENTORY / "complete.json"),
         "empty_environment_metadata": info, "python_sha256": r.digest(e.PYTHON),
         "pyvenv_cfg_sha256": r.digest(e.ENV / "pyvenv.cfg"), "bootstrap_wheel_path": str(BUNDLE),
         "bootstrap_wheel_sha256": r.digest(BUNDLE), "workspace_temp": str(TEMP), "resources": snap,
         "cause": "original ensurepip TEMP permission failure; independent pip call then stopped before install due Windows pip self-modification guard; both confirmed zero installed packages",
         "ruling": "this stage authorizes offline dependency installation; continue in a new explicitly bound transaction after proving empty state, not rerunning or repairing original failed transaction",
         "unchanged": "same interpreter, env path, 14 dependency pins and wheel hashes; no old environment or frozen source changes",
         "stop": "partial/unknown continuation preserves and stops; no retry or alternate pin"}
    r.write(PREP / "protocol.json", p)
    r.write(PREP / "requirements.lock", (e.INVENTORY / "requirements.lock").read_text(encoding="utf-8") + f"pip==24.2 --hash=sha256:{p['bootstrap_wheel_sha256']}\n")
    r.seal(PREP, "ssn-bootstrap-continuation-preparation-v1")
    verify_preparation()
    return {"continuation_prepared": True, "reused": False, "empty_packages": 0, "network_bytes_needed": 0}


def verify():
    p = verify_preparation()
    r.verify_seal(INSTALLATION)
    saved = r.load(INSTALLATION / "environment.json")
    current = e.metadata_snapshot()
    r.require(current == saved["metadata"] and saved["preparation_sha256"] == r.digest(PREP / "complete.json"), "Continuation installed metadata/identity changed")
    versions = {e.norm(row["name"]): row["version"] for row in current["distributions"]}
    r.require(versions == {**e.PINS, "pip": "24.2"}, "Unexpected/missing installed distribution")
    e.dependency_check(current["distributions"])
    r.require(p["original_failure"] == original_failure(), "Original failure overwritten")
    return {"environment_ready_static": True, "packages": len(versions), "ML_imports": 0,
            "runtime_compatibility_verified": False, "original_failure_preserved": True,
            "seal_sha256": r.digest(INSTALLATION / "complete.json")}


def install():
    p = verify_preparation()
    if INSTALLATION.exists():
        return {"reused": True, **verify()}
    r.require(e.metadata_snapshot() == p["empty_environment_metadata"], "Empty-state admission changed before install")
    r.require(not TEMP.exists(), "Unknown workspace TEMP transaction")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 20, "Continuation install resource gate")
    INSTALLATION.mkdir(parents=True, exist_ok=False)
    TEMP.mkdir(parents=True, exist_ok=False)
    r.write(INSTALLATION / "transaction.json", {"action": "new offline bootstrap/install, not original retry", "preparation_sha256": r.digest(PREP / "complete.json")})
    try:
        script = e.GUARD + "\nimport runpy\nsys.path.insert(0, sys.argv.pop(1))\nsys.argv = ['pip'] + sys.argv[1:]\nrunpy.run_module('pip', run_name='__main__', alter_sys=True)\n"
        cmd = [str(e.PYTHON), "-I", "-B", "-X", "utf8", "-c", script, str(BUNDLE), "--isolated", "--disable-pip-version-check", "install", "--no-input", "--no-cache-dir", "--no-compile", "--no-index", "--find-links", str(e.WHEELS), "--find-links", str(BUNDLE.parent), "--require-hashes", "-r", str(PREP / "requirements.lock")]
        child_env = os.environ.copy()
        child_env.update({"TEMP": str(TEMP), "TMP": str(TEMP), "PIP_CONFIG_FILE": os.devnull, "PYTHONDONTWRITEBYTECODE": "1"})
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=child_env, timeout=1800)
        r.write(INSTALLATION / "offline_install.txt", completed.stdout + "\n" + completed.stderr)
        r.require(completed.returncode == 0, "Independent continuation install failed; preserve all state")
        info = e.metadata_snapshot()
        r.require({e.norm(x["name"]): x["version"] for x in info["distributions"]} == {**e.PINS, "pip": "24.2"}, "Installed versions differ from fixed protocol")
        e.dependency_check(info["distributions"])
        r.write(INSTALLATION / "environment.json", {"schema": "ssn-independent-offline-environment-v1", "metadata": info,
            "preparation_sha256": r.digest(PREP / "complete.json"), "resources_after": r.snapshot(),
            "network_bytes": 0, "ML_imports": 0, "runtime_compatibility_verified": False,
            "original_failed_installation_unchanged": original_failure()})
        r.seal(INSTALLATION, "ssn-bootstrap-continuation-installation-v1")
    except Exception as exc:
        r.write(INSTALLATION / "failure.json", {"error": repr(exc), "rule": "retain original and independent transactions; no automatic retry/repair/pin changes"})
        raise
    return {"reused": False, **verify()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "install", "verify"])
    args = parser.parse_args()
    for path in [BASE, e.ENV]:
        for parent in [path, *path.parents]:
            if parent == r.ROOT:
                break
            r.require(not parent.is_symlink() and not parent.is_junction(), "Redirected continuation path")
    def guard(event, arguments):
        if event == "import" and arguments[0].split(".")[0] in {"torch", "torchvision", "numpy", "PIL", "cv2", "anomalib"}:
            raise RuntimeError("No ML imports in static environment continuation")
        if event.startswith("pickle.") or event == "socket.connect":
            raise RuntimeError("No pickle or network")
    sys.addaudithook(guard)
    print(json.dumps({"prepare": prepare, "install": install, "verify": verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
