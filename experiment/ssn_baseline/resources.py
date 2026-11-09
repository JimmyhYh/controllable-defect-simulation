"""Stage 1 only: pinned text acquisition and stdlib-only resource identities.

No author module, image, array, pickle or ML package is imported or executed.
Use Python -B; completed directories are strict read-only caches. Partial or
unknown transactions are retained and rejected, never repaired automatically.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
COMMIT = "98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d"
TREE = "00c9e9c277d364ac2ed4d87f71c8be38abda11a5"
VENDOR = ROOT / "vendor/SuperSimpleNet_98ab4d5"
OUT = ROOT / "results/ssn_baseline/ksdd2_r5_v1/resource_preparation"
SPEC = PROJECT / "报告/SuperSimpleNet独立可行性核验.md"
BACKBONE = ROOT / "models/checkpoints/wide_resnet50_2-95faca4d.pth"
BACKBONE_HASH = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
BACKBONE_BYTES = 138223492
AUTHOR = {
    "LICENSE": "03af0909aba512b9082fce915ae07b84a617cd0e",
    "README.md": "e1b3d54d47a6bc0057c23f203d776a9bdd8cba31",
    "requirements.txt": "78ac94bab19ed5145c62375586c2ec0955513263",
    "model/feature_extractor.py": "4ee752e4f875c3be52ce7d2c5c0aac556e18f703",
    "model/supersimplenet.py": "40b71a8e894a48d3dac130131e217dcedd440c65",
    "common/loss.py": "6cafd7799870434dbc217bb17c8438ca17506547",
    "common/perlin_noise.py": "832febcd5823b96cdcd30cfd9e98383d7c277db8",
    "train.py": "a9c26b4a5d7071f3050a7fdab81622490a5a06d3",
    "datamodules/ksdd2.py": "410f46cb5b66b5931a711ab3ddc24c8b60dc0291",
    "datamodules/base/datamodule.py": "56453a7b05f02979fad6448d5f05b6c9bb913fb2",
    "datamodules/base/dataset.py": "d5e80fd45f54b13a942d15dc814750fa74a60b55",
    "datamodules/base/__init__.py": "d0b65285e65cc88a8cca52b9a086816553707bb2",
}
# Supplemental upstream license snapshots, not a claim about exact helper lineage.
SUPPLEMENTS = {
    "provenance/apple-ml-destseg-LICENSE.txt": "https://raw.githubusercontent.com/apple/ml-destseg/25b7eefb21088e01946d477b75531ccd761da8cf/LICENSE",
    "provenance/DSR-LICENSE.txt": "https://raw.githubusercontent.com/VitjanZ/DSR_anomaly_detection/9932cd29dfd9833d265d859a891f8b1602c7255e/LICENSE",
    "provenance/torchvision-v0.23.0-LICENSE.txt": "https://raw.githubusercontent.com/pytorch/vision/v0.23.0/LICENSE",
    "provenance/torchvision-v0.23.0-README.md": "https://raw.githubusercontent.com/pytorch/vision/v0.23.0/README.md",
}
OLD = {
    "data/fewshot_synthetic/ksdd2_v1/split.json": "b8c2fb376bfd04072baaaf173eb0fccb1ecc5f1339074a6829fe147d0b6210bf",
    "results/fewshot_synthetic/ksdd2_v1/input/completion.json": "8c51dc5e9020e95f03fb84f486ce1d057bac6b60f7724c76fae6882ddf086f52",
    "results/fewshot_synthetic/ksdd2_v1/input/input_protocol.json": "a1eb292009d5b14049d8952a4ac0a791e8858f001fecc4e06a5bffc593d19b56",
    "results/fewshot_synthetic/ksdd2_v1/preparation/common_config.json": "006bea587745d295ab93f4c522eae2f465a4770c18a3a3ff50a34f6b4b57c430",
    "results/fewshot_synthetic/ksdd2_v1/preparation/complete.json": "68f0ae791191dd9341b7ce572cadd80a15e297981410dda202c20569710812a4",
    "results/fewshot_synthetic/ksdd2_v1/real_metrics_v1/complete.json": "b20bd02d02965919c22ab059e324fc7bdd47e4bb1438cd45ce3b409bbaddb538",
    "results/fewshot_synthetic/ksdd2_v1/figures_day7_v1/complete.json": "fe44132413d476fcf1c11ac133670efc785afea385a54be42ea4d504bd7c1ea3",
}
ROLES = {"R_real5": 5, "Q_normal200": 200, "N_train1885": 1885,
         "U_anomaly241": 241, "E_official1004": 1004}
R5 = ["11670", "11884", "12070", "12245", "12303"]
CANDIDATE = {
    "name": "ICPR2024-style SSN strict-R5 adaptation; not paper-exact reproduction",
    "adapt_cls_feat": True, "dt": None, "dilate": None,
    "backbone": "wide_resnet50_2 / IMAGENET1K_V1 / layer2,layer3 / frozen",
    "input_hw": [640, 232], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "precision": "FP32", "batch": 1, "workers": 0,
    "preserve": ["author feature adaptor", "Perlin/Gaussian feature synthesis", "segmentation and classification heads", "author focal/truncated-L1 losses and optimizer groups"],
    "forbidden": ["Q training or selection", "U weak supervision", "E selection or periodic test", "author pickle", "test-pool normalization", "external RGB synthesis", "old model initialization", "automatic weight download"],
    "train_roles": ["R_real5", "N_train1885"],
    "future_model_start_min": {"free_RAM_GiB": 6, "free_VRAM_GiB": 6, "free_disk_GiB": 80},
    "future_sampled_free_RAM_min_GiB": 3,
    "thresholds_are": "engineering candidates, not author requirements or sufficiency proof",
    "training_budget_and_gate": "not yet fixed; requires authorized CPU input preparation and manual pilot",
    "E_status": "previously observed benchmark; future reuse exploratory, not unseen confirmation",
    "use": "private device, personal noncommercial public research; company materials isolated",
}
NOTICES = """# Source and license preservation

The 12 author files are unmodified bytes from blaz-r/SuperSimpleNet at
98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d. LICENSE is the author's MIT notice.
Only model/feature_extractor.py, model/supersimplenet.py, common/loss.py and
common/perlin_noise.py are candidate core sources. train.py, datamodules and
requirements are reference evidence ONLY, not runnable approved entrypoints.
They include dependencies and data/test behavior that this project excludes.

common/loss.py credits Apple ml-destseg/model/losses.py; common/perlin_noise.py
credits VitjanZ/DSR_anomaly_detection/perlin.py. Their original comments remain.
Supplemental upstream LICENSE snapshots are retained separately. Their pinned
license commits do not establish exact copied-code lineage or resolve all rights.
No helper is relabelled as original project code. DSR's upstream Apache-2.0 and
Apple's own license text must be read alongside the author MIT notice.

Torchvision v0.23.0 code LICENSE and README are supplemental snapshots; the tag
URL and actual file hashes are recorded. Its pretrained-model disclaimer may
involve training-data-derived terms; code licensing is not weight licensing.
No separate complete rights grant for this ImageNet V1 artifact was established.
Use remains personal learning / noncommercial research, not commercial clearance.
Only existing generic V1 weight bytes are hashed; no pickle is deserialized.

No task-trained weights, data, images, GT or arrays are acquired here. This is
a source subset, not a complete runnable clone. Never run author train.py or
import the model during this static resource stage.
"""


def require(ok, message):
    if not ok:
        raise RuntimeError(message + "; preserve state and diagnose, no automatic retry")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def files_under(folder):
    require(not folder.is_symlink() and not folder.is_junction(), f"Redirected directory: {folder}")
    paths = list(folder.rglob("*"))
    require(not any(p.is_symlink() or p.is_junction() for p in paths), "Redirected cache path")
    return {p.relative_to(folder).as_posix() for p in paths if p.is_file()}


def verify_seal(folder):
    require((folder / "complete.json").is_file(), f"Partial/unknown transaction {folder}")
    seal = load(folder / "complete.json")
    require(files_under(folder) == set(seal["files"]) | {"complete.json"}, f"Unexpected/missing files in {folder}")
    for name, expected in seal["files"].items():
        require(name in files_under(folder) and ".." not in Path(name).parts and not Path(name).is_absolute(), "Unsafe manifest path")
        require(digest(folder / name) == expected, f"Hash mismatch {folder / name}")
    return seal


def seal(folder, kind):
    write(folder / "complete.json", {"schema": kind, "files": {n: digest(folder / n) for n in sorted(files_under(folder))}})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"Unexpected redirect for {req.full_url}")


def fetch(url, limit=2 * 1024 * 1024):
    require(url.startswith(("https://raw.githubusercontent.com/", "https://api.github.com/")), "Unapproved source host")
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(urllib.request.Request(url, headers={"User-Agent": "SSN-static-resource-audit"}), timeout=45) as r:
        require(r.status == 200 and r.geturl() == url, "Unexpected HTTP result")
        data = r.read(limit + 1)
    require(len(data) <= limit, "Source response exceeds text-only limit")
    data.decode("utf-8")
    return data


def verify_vendor():
    verify_seal(VENDOR)
    m = load(VENDOR / "SOURCE_MANIFEST.json")
    require(m["commit"] == COMMIT and m["tree"] == TREE, "Author revision mismatch")
    require(set(m["files"]) == set(AUTHOR) | set(SUPPLEMENTS), "Vendor source list mismatch")
    expected = set(AUTHOR) | set(SUPPLEMENTS) | {"SOURCE_MANIFEST.json", "THIRD_PARTY_NOTICES.md", "complete.json", "transaction.json"}
    require(files_under(VENDOR) == expected, "Vendor strict file list mismatch")
    for name, row in m["files"].items():
        payload = (VENDOR / name).read_bytes()
        require(len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"], f"Vendor bytes mismatch {name}")
        expected_url = f"https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/{COMMIT}/{name}" if name in AUTHOR else SUPPLEMENTS[name]
        require(row["url"] == expected_url, "Vendor URL mismatch")
        if name in AUTHOR:
            blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            require(blob == AUTHOR[name] == row["git_blob_sha1"], f"Pinned git blob mismatch {name}")
        if name.endswith(".py"):
            ast.parse(payload.decode("utf-8"), filename=name)
    require((VENDOR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == NOTICES, "Notice changed")
    return {"commit": COMMIT, "tree": TREE, "seal_sha256": digest(VENDOR / "complete.json"),
            "manifest_sha256": digest(VENDOR / "SOURCE_MANIFEST.json"), "author_files": len(AUTHOR), "supplemental_texts": len(SUPPLEMENTS)}


def acquire():
    if VENDOR.exists():
        return {"reused": True, **verify_vendor()}
    VENDOR.mkdir(parents=True, exist_ok=False)
    write(VENDOR / "transaction.json", {"action": "text-only source acquisition", "commit": COMMIT, "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        api = "https://api.github.com/repos/blaz-r/SuperSimpleNet/git/"
        commit_bytes = fetch(api + "commits/" + COMMIT)
        commit = json.loads(commit_bytes)
        require(commit["sha"] == COMMIT and commit["tree"]["sha"] == TREE, "Fixed commit/tree mismatch")
        tree_bytes = fetch(api + "trees/" + TREE + "?recursive=1")
        tree = json.loads(tree_bytes)
        require(tree["sha"] == TREE and not tree.get("truncated", False), "Tree identity/incompleteness")
        entries = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"}
        manifest = {"commit": COMMIT, "tree": TREE, "api_commit_sha256": hashlib.sha256(commit_bytes).hexdigest(),
                    "api_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(), "files": {}}
        for name in list(AUTHOR) + list(SUPPLEMENTS):
            url = f"https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/{COMMIT}/{name}" if name in AUTHOR else SUPPLEMENTS[name]
            data = fetch(url)
            row = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if name in AUTHOR:
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                require(blob == AUTHOR[name] == entries[name]["sha"] and len(data) == entries[name]["size"], f"Author source mismatch {name}")
                row["git_blob_sha1"] = blob
            dest = VENDOR / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            manifest["files"][name] = row
        write(VENDOR / "SOURCE_MANIFEST.json", manifest)
        write(VENDOR / "THIRD_PARTY_NOTICES.md", NOTICES)
        seal(VENDOR, "ssn-vendor-subset-v1")
    except Exception as exc:
        write(VENDOR / "failure.json", {"error": repr(exc), "rule": "retain partial directory; do not rerun acquisition or replace revision"})
        raise
    return {"reused": False, **verify_vendor()}


def identities():
    for name, expected in OLD.items():
        require(digest(ROOT / name) == expected, f"Old identity mismatch: {name}")
    common = load(ROOT / "results/fewshot_synthetic/ksdd2_v1/preparation/common_config.json")
    inp = load(ROOT / "results/fewshot_synthetic/ksdd2_v1/input/input_protocol.json")
    old_code = {**{f"fewshot_synthetic/{k}": v for k, v in common["implementation"].items()}, **common["input_identity"]["old_sources"],
                **{f"fewshot_synthetic/{k}": v for k, v in inp["implementation"].items()}}
    for name, expected in old_code.items():
        require(digest(ROOT / name) == expected, f"Old code identity mismatch: {name}")
    split = load(ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json")
    seen = set()
    for role, count in ROLES.items():
        rows = split[role]
        ids = [r["id"] for r in rows]
        require(len(rows) == count and len(set(ids)) == count and not seen.intersection(ids), f"Invalid/disjoint role {role}")
        seen.update(ids)
        for row in rows:
            require(row["part"] == ("test" if role.startswith("E_") else "train"), "Role partition changed")
            require(row["width"] > 0 and row["height"] > 0 and len(row["image_sha256"]) == 64, "Invalid path metadata")
            if not role.startswith("E_"):
                require(row["training_label"] == ("normal" if role in {"Q_normal200", "N_train1885"} else "anomaly"), "Role label changed")
    require([r["id"] for r in split["R_real5"]] == R5, "R5 changed")
    require(split["visual_N16_ids"] == inp["visual_scope"]["N16"] and set(split["visual_N16_ids"]) <= {r["id"] for r in split["N_train1885"]}, "Fixed N16 changed")
    require(load(ROOT / "results/fewshot_synthetic/ksdd2_v1/input/completion.json")["input_ready"] is True, "Old inputs not ready")
    require(BACKBONE.stat().st_size == BACKBONE_BYTES and digest(BACKBONE) == BACKBONE_HASH, "Local V1 backbone changed")
    resnet = Path(sys.executable).parent.parent / "Lib/site-packages/torchvision/models/resnet.py"
    weight_url = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"
    require(weight_url in resnet.read_text(encoding="utf-8"), "Local torchvision V1 URL mapping missing")
    return {"old_metadata_sha256": OLD, "old_source_sha256": old_code, "roles": ROLES, "R5": R5,
            "N16": split["visual_N16_ids"], "split_paths_not_opened": True, "spec_sha256": digest(SPEC),
            "backbone": {"path": str(BACKBONE), "bytes": BACKBONE_BYTES, "sha256": BACKBONE_HASH, "official_url_reference_only": weight_url,
                         "local_resnet_source_sha256": digest(resnet), "parameter_loading_verified": False},
            "implementation": {n: digest(ROOT / n) for n in ["ssn_baseline/resources.py", "ssn_baseline/__init__.py", "ssn_baseline/README.md"]}}


def snapshot():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ["total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended"]]
    mem = MemoryStatus()
    mem.length = ctypes.sizeof(mem)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)), "Windows RAM snapshot failed")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    rows = []
    for line in gpu.stdout.strip().splitlines():
        name, total, free, driver = [x.strip() for x in line.split(",")]
        rows.append({"name": name, "total_MiB": float(total), "free_MiB": float(free), "driver": driver})
    return {"utc": datetime.now(timezone.utc).isoformat(), "total_RAM_GiB": mem.total_phys / 2**30,
            "free_RAM_GiB": mem.avail_phys / 2**30, "free_disk_GiB": shutil.disk_usage(ROOT).free / 2**30,
            "GPUs": rows, "interpretation": "single metadata snapshot; not model peak, compatibility or runtime admission"}


def environment_identity():
    packages = sorted([{"name": d.metadata["Name"], "version": d.version} for d in metadata.distributions()], key=lambda x: ((x["name"] or "").lower(), x["version"]))
    return {"executable": sys.executable, "executable_sha256": digest(sys.executable), "python": sys.version,
            "platform": platform.platform(), "packages_metadata_only": packages}


def verify():
    verify_seal(OUT)
    p = load(OUT / "protocol.json")
    require(p["candidate"] == CANDIDATE and p["identities"] == identities() and p["vendor"] == verify_vendor(), "Protocol identity mismatch")
    require(p["environment"]["identity"] == environment_identity(), "Interpreter/package metadata changed")
    require(files_under(OUT) == {"transaction.json", "protocol.json", "verification.json", "第1阶段静态来源与资源准备.md", "complete.json"}, "Resource cache file set changed")
    return {"stage1_ready": True, "scope": "source/CPU metadata only", "cache_reused": True,
            "vendor_author_files": len(AUTHOR), "read_images_or_arrays": 0, "model_operations": 0,
            "protocol_sha256": digest(OUT / "protocol.json"), "new_environment_created": False}


def prepare():
    if OUT.exists():
        return verify()
    vendor = verify_vendor()
    identity = identities()
    resource = snapshot()
    require(resource["free_RAM_GiB"] >= 2 and resource["free_disk_GiB"] >= 1, "CPU metadata preparation requires 2GiB RAM/1GiB disk; model thresholds separate")
    env = {"identity": environment_identity(),
           "new_ssn_environment_exists": (ROOT / ".venv-ssn").exists(), "compatible_runtime_verified": False}
    require(not env["new_ssn_environment_exists"], "Unexpected pre-existing SSN environment; inspect before preparing")
    core_names = {"torch", "torchvision", "numpy", "scipy", "pillow"}
    core_packages = {r["name"]: r["version"] for r in env["identity"]["packages_metadata_only"] if (r["name"] or "").lower() in core_names}
    protocol = {"schema": "ssn-r5-static-resources-v1", "candidate": CANDIDATE, "vendor": vendor,
                "identities": identity, "environment": env, "resources_snapshot": resource,
                "CPU_preparation_gate": {"free_RAM_GiB": 2, "free_disk_GiB": 1, "status": "passed; metadata only"},
                "mutable_plan_excluded_from_hash": "progress synchronization changes plan; immutable feasibility report and explicit candidate are bound instead",
                "license_status": "author MIT plus preserved upstream notices; no separate complete V1 rights grant established; noncommercial personal use only",
                "remaining": ["isolated environment", "core wrapper", "CPU R5/N16 label geometry", "manual model pilot", "training budget", "unseen real validation"]}
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "transaction.json", {"action": "CPU metadata preparation only", "started_utc": datetime.now(timezone.utc).isoformat()})
    write(OUT / "protocol.json", protocol)
    write(OUT / "verification.json", {"identity_status": "passed", "fixed_role_counts": ROLES, "author_commit": COMMIT,
          "old_metadata_and_selected_critical_seals_unchanged": True, "old_source_identity_unchanged": True,
          "source_python_syntax_only": True, "backbone_bytes_hash_only": True,
          "forbidden_packages_imported": sorted(n for n in sys.modules if n.split(".")[0] in {"torch", "torchvision", "numpy", "PIL", "pickle", "anomalib"}),
          "image_GT_probability_reads": 0, "model_operations": 0, "installs": 0, "weights_or_data_downloads": 0})
    report = f"""# SuperSimpleNet第1阶段：静态来源与资源准备

完成范围：固定源码／许可封存、CPU元信息与已有通用权重身份。未进入隔离环境安装、图像内容核验或模型运行。协议与逐文件清单见[protocol.json](protocol.json)及[vendor来源清单]({VENDOR.as_posix()}/SOURCE_MANIFEST.json)。

## 来源与角色

作者固定commit `{COMMIT}`、tree `{TREE}`。12份作者源码／说明／许可与4份上游许可／说明文本已保存实际SHA256；作者文件另与固定Git树blob身份核对。保留原字节，AST只检查语法，不执行源码。4份模型／helper文件是核心候选；train.py及数据模块只作来源证据，不是批准运行的入口。[作者固定来源](https://github.com/blaz-r/SuperSimpleNet/tree/{COMMIT})。

严格R5={"／".join(R5)}，N1885训练；Q200仅未来正常校准，U241不作弱监督，E1004不作选择／训练期测试。旧split、输入完成标记、共同配置及指标／图件完成封印与旧核心源码hash核对一致；不是遍历核验所有旧成果内容。未打开任何原图、GT或概率数组。

主候选固定为2024式adapt_cls_feat=True、dt=None、dilate=None；WideResNet50-2 V1冻结骨干、layer2/3、作者特征合成与双头／损失／优化器分组。保留作者特征合成，未加外部RGB合成；不能称完全无合成。整图640×232、ImageNet归一化、FP32／batch1／workers0仍待后续内容与模型核验。未固定小试／正式步数或验收门槛，不套旧GN、旧softmax评分或旧阈值。

## 权重、环境与资源

已有V1 `{BACKBONE.as_posix()}`，{BACKBONE_BYTES}字节，SHA256 `{BACKBONE_HASH}`。与旧身份及本地torchvision官方V1 URL定义相符，仅按字节hash，未反序列化、未执行pickle；参数键／dtype／实际兼容均未验证。

解释器 `{sys.executable}`；Python {platform.python_version()}。核心包元信息：{json.dumps(core_packages, ensure_ascii=False)}。完整包名／版本元信息及解释器hash见protocol；未导入ML包，未安装或修改旧环境，未创建.venv-ssn。

本次快照：可用RAM {resource['free_RAM_GiB']:.6f}GiB／总{resource['total_RAM_GiB']:.6f}GiB，D盘空闲{resource['free_disk_GiB']:.6f}GiB；GPU {json.dumps(resource['GPUs'], ensure_ascii=False)}。CPU准备门槛2GiB RAM／1GiB磁盘是本阶段的保守工程检查。后续模型建议起点RAM6GiB／空闲显存6GiB／磁盘80GiB、运行RAM采样3GiB；它们不是充分性证明。本次快照不作未来运行准入，实际显存峰值／训练时间／完整恢复未实测。

MIT作者许可与Apple／DSR上游许可文本分开保存，不重标为项目原创；上游许可版本并不确认helper复制的精确历史。Torchvision v0.23.0 README声明代码与预训练权重条件不同；未取得该V1单独完整权利授予，不能称任意商用许可。用途仍私人设备、个人学习／非商业研究，公司材料隔离。[官方预训练权重说明](https://github.com/pytorch/vision/tree/v0.23.0#pre-trained-model-license)。

## 实际入口与剩余问题

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m ssn_baseline.resources verify
```

verify严格只读核对完整缓存、来源、旧元信息、核心源码、协议及本地V1字节hash，无联网／模型操作。acquire只取得指定文本，完整缓存只读复用；prepare仅CPU准备，完整缓存也只读复用。陌生文件／部分事务／hash变化保留停止，不自动重试或覆盖。

首次沙箱网络读取出现WinError10013；保持同一固定地址后，经允许的网络权限读取成功，未换main或权重。本阶段只准备，无Torch／模型包导入、模型构造、优化器、前向、生成、参数更新、训练、校准或新评价。

下一阶段另行授权隔离环境与核心包装静态准备，先核对可复用安装材料与流量。模型构造本身会触发尺寸探测前向，因此不属于静态准备。训练始终由用户手动启动；未见真实验证来源仍待核验，当前不提供未实现的后续命令。
"""
    write(OUT / "第1阶段静态来源与资源准备.md", report)
    require(not load(OUT / "verification.json")["forbidden_packages_imported"], "Forbidden package imported; stop before seal")
    seal(OUT, "ssn-resource-preparation-v1")
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "prepare", "verify", "verify-vendor"])
    args = parser.parse_args()
    forbidden = {"torch", "torchvision", "numpy", "PIL", "pickle", "anomalib", "model"}
    require(not any(n.split(".")[0] in forbidden for n in sys.modules), "Forbidden package already imported")
    def audit(event, args):
        if event == "import" and args[0].split(".")[0] in forbidden:
            raise RuntimeError("Static-only command forbids ML/model/pickle imports")
        if event.startswith("pickle."):
            raise RuntimeError("No pickle execution authorized")
        if event == "socket.connect" and parser_command != "acquire":
            raise RuntimeError("No network authorized for CPU prepare/verify")
    parser_command = args.command
    sys.addaudithook(audit)
    for folder in [VENDOR, OUT]:
        for parent in [folder, *folder.parents]:
            if parent == ROOT:
                break
            require(not parent.is_symlink() and not parent.is_junction(), "Redirected output ancestor")
    actions = {"acquire": acquire, "prepare": prepare, "verify": verify, "verify-vendor": verify_vendor}
    print(json.dumps(actions[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
