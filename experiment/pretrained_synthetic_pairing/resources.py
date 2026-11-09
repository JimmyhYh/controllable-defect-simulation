"""Explicit CPU source/resource preparation; no input decode, requests or ML imports."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import common as c

ANCHORS = {
    "data/fewshot_synthetic/ksdd2_v1/split.json": "b8c2fb376bfd04072baaaf173eb0fccb1ecc5f1339074a6829fe147d0b6210bf",
    "results/fewshot_synthetic/ksdd2_v1/preparation/complete.json": "68f0ae791191dd9341b7ce572cadd80a15e297981410dda202c20569710812a4",
    "results/fewshot_synthetic/ksdd2_v1/preparation/common_config.json": "006bea587745d295ab93f4c522eae2f465a4770c18a3a3ff50a34f6b4b57c430",
    "results/fewshot_synthetic/ksdd2_v1/preparation/synthetic_manifest.json": "6159a6106a39de372439e88e284030505c9a8fa7768cfa0bfdf8eb8df6ed5966",
    "models/pretrained_segmentation/ksdd2_r5_v1/baseline_2026102202/complete.json": "ea39f6bd69ffdeb870831a37c166c84143ab887683c46948d7e3ed177a87bdea",
    "models/pretrained_segmentation/resources_v1/encoder/model.safetensors": "b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19",
}
EXPECTED_REPORT = "0aab0e9ba52f2ca9774a1e45cb992dac1c50953d7391219001c7c1e935b53a8b"
FINAL_WEIGHT = "f8f4f94029da49704c5d2b194c549dd24ef364a7f7d365ff6fcef9e3f373e454"


def environment_metadata():
    """A clean model-environment process reads distribution metadata only."""
    script = r'''
import sys,os,json,hashlib
from pathlib import Path
blocked={'torch','torchvision','segmentation_models_pytorch','timm','safetensors','numpy','PIL','cv2','scipy','sklearn','pickle','joblib','huggingface_hub'}
def must(value,message):
 if not value: raise RuntimeError(message)
must(not any(n.split('.')[0] in blocked for n in sys.modules),'Forbidden startup import')
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in blocked: raise RuntimeError('Blocked model/data import')
 if event=='socket.connect' or event.startswith('pickle.'): raise RuntimeError('No network/pickle')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  mode,flags=args[1] or '',args[2] or 0
  if any(x in str(mode) for x in 'wax+') or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC): raise RuntimeError('Metadata subprocess is read-only')
sys.addaudithook(guard)
import importlib.metadata as md
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
site=Path(sys.prefix)/'Lib/site-packages'
actual={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions()}
expected={r['name'].lower().replace('_','-'):r['version'] for r in source['metadata']['distributions']}
must(actual==expected,'Dependency set differs')
for row in source['metadata']['distributions']:
 for relative,digest in row['metadata_files_sha256'].items(): must(sha(site/relative)==digest,'Metadata hash differs: '+relative)
for relative,digest in source['installed_SMP_source_subset_sha256'].items(): must(sha(site/relative)==digest,'Installed source differs: '+relative)
must(sha(sys.executable)==source['python_sha256'],'Interpreter hash differs')
must(sha(Path(sys.prefix)/'pyvenv.cfg')==source['pyvenv_cfg_sha256'],'pyvenv.cfg hash differs')
must(sys.version==source['metadata']['python'],'Python version differs')
must(not any(n.split('.')[0] in blocked for n in sys.modules),'Forbidden metadata import')
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':sys.version,'distributions':actual,'python_sha256':sha(sys.executable),'pyvenv_cfg_sha256':sha(Path(sys.prefix)/'pyvenv.cfg'),'distribution_METADATA_RECORD_WHEEL_INSTALLER_hashes_verified':True,'installed_SMP_source_subset_verified':True,'forbidden_imports':[],'writes_or_model_operations':0},ensure_ascii=False))
'''
    run = subprocess.run([str(c.ENV / "Scripts/python.exe"), "-B", "-X", "utf8", "-c", script, str(c.BASE / "environment_v1/environment.json")],
                         capture_output=True, text=True, encoding="utf-8", timeout=90, check=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return json.loads(run.stdout)


def collect_sources():
    sources = {}
    source_trees = []
    def add(path, expected=None):
        path = Path(path)
        digest = c.sha(path)
        c.require(expected is None or digest == expected, "Source hash differs: " + str(path))
        sources[str(path)] = digest
        return digest
    def json_seal(folder):
        # The frozen installer explicitly used this sole empty temporary folder.
        empty = ("workspace_temp",) if folder == c.BASE / "environment_v1" else ()
        complete = c.verify_seal(folder, allowed_empty_dirs=empty)
        source_trees.append({"root": str(folder), "complete_sha256": add(folder / "complete.json"), "allowed_empty_dirs": list(empty)})
        for name, expected in complete["files"].items():
            add(folder / str(c.safe_relative(name)), expected)
        return complete
    for name, expected in ANCHORS.items():
        add(c.ROOT / name, expected)
    add(c.REPORT, EXPECTED_REPORT)
    add(c.PROJECT / "计划/预训练单头分割严格R5基线实现设计.md")
    split_path = c.ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json"
    split = c.load(split_path)
    c.require({role: len(split[role]) for role in c.ROLES} == c.ROLES, "Fixed role counts differ")
    c.require([x["id"] for x in split["R_real5"]] == c.R5, "Fixed R5 differs")
    all_ids = [row["id"] for role in c.ROLES for row in split[role]]
    c.require(len(all_ids) == len(set(all_ids)) == 3335, "Role ID overlap or population differs")
    normals = {row["id"]: row for row in split["N_train1885"]}

    # Existing seals contain only source, dependency or accepted-result metadata here.
    for name in ["environment_v1", "probe_acceptance_verification_v1", "baseline_training_verification_v1"]:
        json_seal(c.BASE / name)
    for name, expected in c.load(c.BASE / "environment_v1/environment.json")["implementation"].items():
        c.require(c.safe_relative(name).parts[0] == "pretrained_segmentation", "Environment source outside expected package")
        add(c.ROOT / str(c.safe_relative(name)), expected)
    model_complete = json_seal(c.MODEL)  # Opaque weights/optimizer are stream-hashed, never parsed.
    c.require(model_complete["identity"]["steps"] == 10000 and model_complete["identity"]["seed"] == 2026102202, "Prior formal model identity differs")
    add(c.MODEL / "final_weights.safetensors", FINAL_WEIGHT)
    proof = c.load(c.BASE / "baseline_training_verification_v1/verification.json")
    c.require(proof["identity"]["final_weight_sha256"] == FINAL_WEIGHT and proof["updates"] == 10000,
              "Saved formal verification identity differs")
    c.require([r["step"] for r in proof["checkpoints"]] == list(range(1000, 10001, 1000)), "Ten checkpoint metadata incomplete")
    c.require(proof["R_exposures"] == proof["N_exposures"] == 5000 and proof["actual_sequence_equal_frozen"]
              and proof["initial_BN_matches_all_saved_checkpoints_and_final"], "Prior training evidence invalid")
    accepted = c.load(c.BASE / "probe_acceptance_verification_v1/verification.json")
    c.require(accepted["gate_passed"] is True and accepted["step"] == 2000 and all(accepted["gate_checks"].values()), "Prior final technical gate not passed")
    for item in [model_complete["identity"], proof["identity"]]:
        for key in ["implementation", "pilot_implementation"]:
            for name, expected in item[key].items():
                add(c.ROOT / "pretrained_segmentation" / str(c.safe_relative(name)), expected)
    for name in ["native_inference.py", "calibration_common.py", "normal_calibration.py", "real_scoring.py", "real_metrics.py"]:
        add(c.ROOT / "pretrained_segmentation" / name)
    formal = c.BASE / "formal_preparation_v1"
    complete = c.load(formal / "complete.json"); add(formal / "complete.json", model_complete["identity"]["formal_complete_sha256"])
    c.verify_closed_tree(formal, set(complete["files"]) | {"complete.json"})
    source_trees.append({"root": str(formal), "complete_sha256": sources[str(formal / "complete.json")], "allowed_empty_dirs": []})
    for name in ["inputs.json", "protocol.json", "requests.json", "transaction.json"]:
        add(formal / name, complete["files"][name])
    native_inputs = c.load(formal / "inputs.json")
    expected_rows = split["R_real5"] + split["N_train1885"]
    c.require(len(native_inputs) == 1890 and [x["id"] for x in native_inputs] == [x["id"] for x in expected_rows], "Prior training input roles/order differ")
    for item, row in zip(native_inputs, expected_rows):
        c.require(all(item[k] == row[k] for k in ["id", "image", "mask", "height", "width"])
                  and item["source_sha256"]["image"] == row["image_sha256"], "Prior native source metadata differs")

    vendor = c.ROOT / "vendor/SMP_0_5_0"
    vendor_complete = json_seal(vendor)
    manifest = c.load(vendor / "SOURCE_MANIFEST.json")
    c.require(manifest["commit"] == "420ce84b0c2df0286fa9bb2bd1499eea625c9b33", "SMP source commit differs")
    for name, record in manifest["files"].items():
        add(vendor / str(c.safe_relative(name)), record["sha256"])
    header_path = c.ROOT / "models/pretrained_segmentation/resources_v1/safetensors_header.json"
    add(header_path)
    add(c.ROOT / "models/pretrained_segmentation/resources_v1/acquisition.json")
    c.require(c.GENERIC.stat().st_size == 46805552, "Generic ImageNet file size differs")
    with c.opaque_read_scope(c.GENERIC):
        with c.GENERIC.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            c.require(0 < length <= 1024 * 1024, "Unexpected safetensors JSON header length")
            raw_header = stream.read(length)
    saved_header = c.load(header_path)
    c.require(len(raw_header) == length and hashlib.sha256(raw_header).hexdigest() == saved_header["header_sha256"], "Generic JSON header identity differs")
    header = json.loads(raw_header)
    descriptors = {k: v for k, v in header.items() if k != "__metadata__"}
    c.require(len(descriptors) == saved_header["tensor_count"] == 102 and all(v["dtype"] == "F32" for v in descriptors.values()), "Generic descriptor set differs")

    old_complete = c.load(c.OLD / "complete.json")
    # Enumerate the entire sealed namespace, but never open the other two NPZ pools.
    c.verify_closed_tree(c.OLD, set(old_complete["files"]) | {"complete.json"})
    source_trees.append({"root": str(c.OLD), "complete_sha256": sources[str(c.OLD / "complete.json")], "allowed_empty_dirs": []})
    c.require(old_complete["input_identity"]["split"] == sources[str(split_path)], "Old synthesis uses different split")
    for name, expected in old_complete["implementation"].items():
        add(c.ROOT / "fewshot_synthetic" / str(c.safe_relative(name)), expected)
    for name, expected in old_complete["input_identity"]["old_sources"].items():
        add(c.ROOT / str(c.safe_relative(name)), expected)
    for name in ["common_config.json", "synthetic_manifest.json", f"{c.MATERIAL_SEED}/sequence.json"]:
        add(c.OLD / name, old_complete["files"][name])
    config = c.load(c.OLD / "common_config.json")
    c.require(config["formal_seeds"] == [2026101903, 2026101904, 2026101905] and min(config["formal_seeds"]) == c.MATERIAL_SEED, "Original minimum material seed differs")
    c.require(config["synthetic_types"] == c.TYPES and config["synthetic_per_seed"] == 2500 and config["amplitude"] == .04
              and config["attempts"] == 5 and config["coverage_min"] == .9, "Original synthesis protocol differs")
    old_sequence = c.load(c.OLD / str(c.MATERIAL_SEED) / "sequence.json")
    c.require(len(old_sequence) == 10000, "Old sequence metadata incomplete")
    rows = [row for row in c.load(c.OLD / "synthetic_manifest.json") if row["seed"] == c.MATERIAL_SEED]
    c.require(len(rows) == 2500 and [r["ordinal"] for r in rows] == list(range(2500)), "Material ordinal order differs")
    selected = []
    for ordinal, row in enumerate(rows):
        name = f"{c.MATERIAL_SEED}/synthetic/{ordinal:04d}.npz"
        request = row["background_request"]
        c.require(row["file"] == name and row["valid"] is True and row["type"] == c.TYPES[ordinal % 4], "Material request identity/type differs")
        c.require(request == old_sequence[row["sequence_step"] - 1] and request["step"] == 4 * (ordinal + 1)
                  and request["role"] == "intervention" and request["synthetic_ordinal"] == ordinal, "Material is not from frozen original sequence")
        c.require(request["id"] in normals, "Material background outside N1885")
        normal = normals[request["id"]]
        ys = sorted(set(range(0, normal["height"] - 256 + 1, 128)) | {normal["height"] - 256})
        c.require(request["y"] in ys and type(request["flip_x"]) is bool and type(request["flip_y"]) is bool, "Invalid original coordinate/flip metadata")
        attempts = row["attempts"]
        c.require(1 <= len(attempts) <= 5 and [a["attempt"] for a in attempts] == list(range(len(attempts)))
                  and [a["accepted"] for a in attempts] == [False] * (len(attempts) - 1) + [True], "Accepted attempt chronology differs")
        accepted_attempt = attempts[-1]; spec = accepted_attempt["spec"]; stats = accepted_attempt["stats"]
        c.require(spec["amplitude"] == (.04 if ordinal % 2 == 0 else -.04)
                  and spec["kind"] == ("mura_like" if ordinal % 4 < 2 else "line_like"), "Original kind/intensity differs")
        c.require(stats["valid"] is True and stats["effective_coverage"] >= .9 and stats["outside_ROI_max_float_residual"] == 0
                  and accepted_attempt["outside_valid_max_residual"] == 0 and row["positive_pixels"] > 0
                  and row["valid_pixels"] == 256 * normal["width"], "Saved effective-domain metadata invalid")
        c.require(row["sha256"] == old_complete["files"][name], "Material hash not in original seal")
        add(c.OLD / name, row["sha256"])
        selected.append({"ordinal": ordinal, "type": row["type"], "file": str(c.OLD / name), "sha256": row["sha256"],
                         "background_request": request, "accepted_attempt": accepted_attempt["attempt"],
                         "positive_pixels_recorded": row["positive_pixels"], "valid_pixels_recorded": row["valid_pixels"]})
    c.require(Counter(r["type"] for r in selected) == Counter(dict.fromkeys(c.TYPES, 625)), "Four material type counts differ")
    folder = c.OLD / str(c.MATERIAL_SEED) / "synthetic"
    c.require(c.tree(folder) == ({f"{k:04d}.npz" for k in range(2500)}, set()), "Unknown/missing member in selected material folder")
    environment = environment_metadata()
    model_bytes = sum((c.MODEL / name).stat().st_size for name in c.tree(c.MODEL)[0])
    disk = {"old_model_tree_bytes": model_bytes, "six_model_tree_estimate_bytes": 6 * model_bytes,
            "six_Q_E_probability_bytes": 6 * 4 * (29202876 + 146411528),
            "one_exact_table_pair_bytes": 8522825736, "six_exact_table_pairs_bytes": 51136954416,
            "five_prior_exact_table_pairs_bytes": 5 * 8522825736, "new_metadata_evidence_reserve_bytes": 2 * 2**30}
    disk["required_free_before_round_bytes"] = max(130 * 2**30, 80 * 2**30 + disk["six_model_tree_estimate_bytes"]
        + disk["six_Q_E_probability_bytes"] + disk["five_prior_exact_table_pairs_bytes"] + disk["new_metadata_evidence_reserve_bytes"])
    disk["formula"] = "max(130GiB,80GiB+six old-sized model trees+six Q/E float32 maps+five retained AP table pairs+2GiB evidence reserve); E drive; not measured future peak"
    return sources, selected, environment, disk, source_trees


def protocol():
    return {"schema": "pretrained-single-head-paired-composition-resource-v1", "plan_contract": c.plan_contract(),
            "initialization_seeds": list(c.SEEDS), "data_seed": c.DATA_SEED, "fixed_old_material_seed": c.MATERIAL_SEED,
            "old_material_ordinals": [0, 2499], "material_items": 2500, "material_type_counts": dict.fromkeys(c.TYPES, 625),
            "same_data_requests_all_three_pairs": True, "training_requests_generated": 0,
            "A_exposures": {"real": 5000, "synthetic": 0, "normal": 5000, "real_per_R5": 1000},
            "B_exposures": {"real": 2500, "synthetic": 2500, "normal": 5000, "real_per_R5": 500},
            "shared_slots": 7500, "intervention_slots": 2500, "updates": 10000,
            "four_slot_cycle": ["shared_R", "shared_N", "A_extra_R_or_B_fixed_synthetic", "shared_N"],
            "data_rng": "PCG64 SeedSequence([2026102300,namespace]);0 shared R,1 extra A R,2 N,3 flips; no generation at stage one",
            "model": {"SMP": "0.5.0", "commit": "420ce84b0c2df0286fa9bb2bd1499eea625c9b33", "architecture": "Unet",
                      "encoder": "ImageNet ResNet18", "revision": "3f2325ff978283d47aa6a1d6878ca20565622683",
                      "head": "single two-class softmax segmentation head; no classification head",
                      "encoder_BN": "20 running-stat buffers fixed, affine and other encoder parameters trainable", "decoder": "10 BN replaced by GN8",
                      "loss": "valid-domain unweighted CE + Dice smoothing1 including empty GT", "initialization": "generic encoder plus new decoder/head; never probe/task/final weights"},
            "Adam": {"lr": [1e-5, 1e-4], "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0},
            "lr_intervals": [[1, 8000, 1e-5, 1e-4], [8001, 9000, 2e-6, 2e-5], [9001, 10000, 4e-7, 4e-6]],
            "precision": "FP32", "batch": 1, "workers": 0, "checkpoint_updates": list(range(1000, 10001, 1000)),
            "input_policy": {"window": 256, "stride": 128, "tail": "sorted unique H-256", "pad": "right reflect; excluded",
                             "real": "uint8/255 then ImageNet normalization", "synthetic": "already flipped/padded float32[0,1]; no /255, quantization or second flip; original M_eff/valid",
                             "float_label_threshold": 0.003921568859368563, "coverage_min": .9},
            "roles": c.ROLES, "R5": c.R5, "Q": "not training/synthesis/config selection; future normal calibration only", "U": "unused",
            "E": "historically seen; exploratory only; not model/seed/material/config selection",
            "threshold": {"old_tau_unchanged": 7.264203168233501e-06, "future_new_model_thresholds": "each model's Q200 quantile independently, all frozen before E",
                          "q": .95, "method": "linear", "comparison": "strict >", "score": "native float32 stitched anomaly softmax max; convert maxima to float64 for quantile"},
            "resources": {"stage1_RAM_GiB": 2, "stage1_D_disk_GiB": 2, "future_model_RAM_GiB": 4.5, "future_free_GPU_GiB": 6,
                          "future_model_disk_GiB": 20, "future_AP_RAM_GiB": 3, "future_AP_disk_GiB": 80, "large_drive_minimum_GiB": 130},
            "paths": {"code": str(c.PACKAGE), "stage1": str(c.OUT), "future_results": str(c.RESULTS), "future_models": str(c.MODELS), "large_output_root": str(c.LARGE)},
            "user_E_drive_steering": "User stated E has ample space during stage one; only new future outputs planned on E; old D artifacts stay unchanged",
            "interpretation": "composition strategy; reduced real-anomaly repetition and changed normal-derived backgrounds/pixel-label exposure remain mixed; no single-factor realism/pretraining/screen claim",
            "stops": "unknown/partial/hash mismatch/failure/locks preserve; no replacement pool, automatic retry, generation, approximate metrics or lower resource gate",
            "training": "always manually started by user; no stage-two or runtime authorization implied"}


def readiness(snapshot, disk):
    required = disk["required_free_before_round_bytes"] / 2**30
    reasons = []
    if snapshot["available_RAM_GiB"] < 2: reasons.append("CPU available RAM below 2GiB")
    if snapshot["D_free_GiB"] < 2: reasons.append("D metadata disk below 2GiB")
    if snapshot["E_free_GiB"] < required: reasons.append("E free disk below full-round forecast")
    if snapshot["planned_E_target_exists"]: reasons.append("Planned E target is already occupied; stage one has no validated existing transaction there")
    return {"stage1_resource_conditions_met": not reasons, "blocking_reasons": reasons, "required_E_free_GiB": required,
            "snapshot": snapshot, "large_output_writability_tested": False, "E_directory_created": False,
            "future_model_start_snapshot_conditions_met": snapshot["available_RAM_GiB"] >= 4.5 and snapshot["GPU"]["free_MiB"] >= 6144,
            "future_resources_must_be_rechecked": True, "stage2_or_training_started": False}


def brief(result):
    return {k: result[k] for k in ["stage1_metadata_verified", "selected_materials_stream_hash_verified", "material_type_counts", "resource_state", "payload_decodes_or_model_operations", "complete_cache_reused"]}


def inspect():
    if c.OUT.exists():
        return verify()
    sources, selected, environment, disk, source_trees = collect_sources()
    state = readiness(c.resource_snapshot(), disk)
    return {"stage1_metadata_verified": True, "selected_materials_stream_hash_verified": len(selected),
            "material_type_counts": dict(Counter(r["type"] for r in selected)), "resource_state": state,
            "payload_decodes_or_model_operations": 0, "complete_cache_reused": False,
            "sources": sources, "selected": selected, "environment": environment, "disk": disk, "source_trees": source_trees}


def prepare():
    if c.OUT.exists():
        return verify()
    result = inspect()
    c.OUT.mkdir(parents=True, exist_ok=False)
    try:
        return save_preparation(result)
    except Exception as error:
        # Only this invocation's freshly created transaction may receive a failure record.
        if not (c.OUT / "complete.json").exists() and not (c.OUT / "failure.json").exists():
            c.write(c.OUT / "failure.json", {"error": repr(error), "rule": "preserve; no automatic retry or substitution"})
        raise


def save_preparation(result):
    c.write(c.OUT / "transaction.json", {"scope": "CPU identity/resource only; no data decode or model", "started_at_UTC": datetime.now(timezone.utc).isoformat()})
    c.write(c.OUT / "protocol.json", protocol())
    c.write(c.OUT / "sources.json", result["sources"])
    c.write(c.OUT / "source_trees.json", result["source_trees"])
    c.write(c.OUT / "selected_materials.json", result["selected"])
    c.write(c.OUT / "environment_metadata.json", result["environment"])
    c.write(c.OUT / "resource_assessment.json", {"state": result["resource_state"], "disk_forecast": result["disk"],
        "diagnosis": {"D_only_capacity_GiB": result["resource_state"]["snapshot"]["D_free_GiB"], "resolution": "new large results/models planned on E per user steering; no E writes or old moves", "CIM": "CIM access was unavailable; direct GlobalMemoryStatusEx confirmed current RAM"}})
    summary = brief(result)
    summary.update({"eligible_for_future_CPU_input_preparation": result["resource_state"]["stage1_resource_conditions_met"],
                    "NPZ_payloads_or_RGB_or_GT_or_probabilities_decoded": 0, "training_requests_generated": 0,
                    "tensor_or_model_loaded": 0, "future_code_not_implemented": ["inputs", "training", "scoring", "metrics", "figures"]})
    c.write(c.OUT / "verification.json", summary)
    c.write(c.OUT / "第1阶段CPU来源资源与旧合成复用核验.md", report(result))
    files, dirs = c.tree(c.OUT)
    expected = {"transaction.json", "protocol.json", "sources.json", "source_trees.json", "selected_materials.json", "environment_metadata.json", "resource_assessment.json", "verification.json", "第1阶段CPU来源资源与旧合成复用核验.md"}
    c.require(files == expected and not dirs, "Unknown stage-one output before sealing")
    c.write(c.OUT / "complete.json", {"identity": {"implementation": c.implementation(), "plan_contract": c.plan_contract(),
             "meaning": "CPU stage-one identity/resource assessment only; no stage-two implementation or model execution"},
             "files": {name: c.sha(c.OUT / name) for name in sorted(files)}})
    return summary


def verify():
    complete = c.verify_seal(c.OUT)
    c.require(complete["identity"]["implementation"] == c.implementation(), "Stage-one implementation differs")
    c.require(complete["identity"]["plan_contract"] == c.plan_contract(), "Frozen scientific contract/output paths differ")
    c.require(c.load(c.OUT / "protocol.json") == protocol(), "Frozen protocol differs")
    for entry in c.load(c.OUT / "source_trees.json"):
        root = Path(entry["root"])
        c.require(c.sha(root / "complete.json") == entry["complete_sha256"], "Upstream tree seal identity differs")
        saved = c.load(root / "complete.json")
        c.verify_closed_tree(root, set(saved["files"]) | {"complete.json"}, entry["allowed_empty_dirs"])
    for path, expected in c.load(c.OUT / "sources.json").items():
        c.require(c.sha(path) == expected, "Frozen upstream source differs: " + path)
    c.require(environment_metadata() == c.load(c.OUT / "environment_metadata.json"), "Installed environment metadata differs")
    selected = c.load(c.OUT / "selected_materials.json")
    c.require(len(selected) == 2500 and [r["ordinal"] for r in selected] == list(range(2500)), "Selected material population differs")
    summary = c.load(c.OUT / "verification.json")
    disk = c.load(c.OUT / "resource_assessment.json")["disk_forecast"]
    summary["resource_state"] = readiness(c.resource_snapshot(), disk)
    summary["eligible_for_future_CPU_input_preparation"] = summary["resource_state"]["stage1_resource_conditions_met"]
    summary["complete_cache_reused"] = True
    return summary


def report(result):
    state = result["resource_state"]; snap = state["snapshot"]; disk = result["disk"]
    return f'''# 第1阶段：CPU来源／资源及固定旧合成复用核验

仅完成元信息、逐文件stream hash和资源核验。固定旧种子2026101903的2500份NPZ按原ordinal0—2499全部核hash，四类各625；N来源、坐标、原sequence、原接受尝试顺序、保存有效性及原完整清单一致。其余两套旧合成不读取；未解析任何NPZ、RGB、GT、概率或tensor载荷，未生成新请求／图像，未导入模型或训练。实际残差与标签逐像素相符仍待第2阶段CPU核验，不将保存统计当实际重验。

单头原模型10000步／10保存点／最终权重、已有小试gate、vendor／通用ImageNet文件hash及JSON header、完整27项环境包／解释器／dist-info元信息和SMP安装源码子集已核验。旧optimizer和模型文件仅stream hash，无pickle执行或参数加载。旧报告、数据角色与实现保持不变。

## 来源适配及独立审阅

初次只读检查发现旧environment_v1含空workspace_temp；原封存安装源码明确使用该路径，故只精确允许这个有来源的空目录，其他未知文件或目录仍拒绝。独立环境implementation键本来已含包前缀，新的来源适配按原相对路径解析，没有修改旧代码或旧身份。

一次独立只读代码审阅给出3项Important与1项目标占用Minor；目标占用会影响新目录隔离，执行方按Important处理。封存前同一轮修正为：只在prepare本次新建事务后可记失败；inspect／verify及已有部分事务绝不写入；准备与缓存复核都检查全部上游目录闭合（另两池仅枚举名称）；环境身份采用显式条件而非可被优化模式移除的assert；只读核E目标未占用及祖先无重解析点。没有新增软件测试、过程表或提交流程。

审阅未执行实际hash或资源查询，实际值以本入口直接运行结果为证。float适配、实际残差、模型前向／恢复及未来各自新阈值均未在此阶段验证；common的写入／审计入口仅限阶段1，未来模块须独立实现E输出事务，不修改此冻结scope。未把尚未实现的bind_sources便利接口或未来E盘准入当作当前运行能力。

## 冻结的新研究协议

3个初始化seed2026102301／02／03，一套数据seed2026102300，三对共用固定请求和2500份材料。A=真实异常5000＋正常5000；B=真实异常2500＋合成2500＋相同正常5000。共同异常／正常7500槽与干预2500槽、10000步、模型／损失／Adam／8000与9000后衰减均明确，当前新请求仍为0。B真实异常重复减半和背景／像素曝光组成混杂保留；E已见仅探索性。所有训练由用户手动启动。

旧τ=7.264203168233501e-06保留；未来六模型分别按Q200的q0.95／linear／严格大于规则冻结新τ后才能评价E。当前没有读Q／E图像或概率、校准、评价或AP计算。

## 资源及路径决定

首次D盘可用约{snap['D_free_GiB']:.3f}GiB，小于130GiB整轮规划；用户指出E盘有空间后，E盘实测{snap['E_free_GiB']:.3f}GiB。新大输出固定{c.LARGE.as_posix()}，其results／models为独立后续根；本阶段协议保存在D盘。本次未创建／写入E目录、移动旧文件、建立链接或修改旧全局路径；E实际写入能力待第2阶段显式核验。

旧完整模型目录{disk['old_model_tree_bytes']}字节，六组按同大小估算{disk['six_model_tree_estimate_bytes']}字节；六组Q／E float32概率净载荷{disk['six_Q_E_probability_bytes']}字节，六套精确AP表{disk['six_exact_table_pairs_bytes']}字节。为第六次AP启动前仍有80GiB，按六模型＋全部概率＋已保留五套表＋2GiB新元信息余量估算，E启动空闲至少{state['required_E_free_GiB']:.3f}GiB，高于130GiB基本线；这不是未来实际峰值保证。

当前RAM可用{snap['available_RAM_GiB']:.3f}GiB／总{snap['total_RAM_GiB']:.3f}GiB，满足与否以state记录为准；GPU空闲{snap['GPU']['free_MiB']}MiB／总{snap['GPU']['total_MiB']}MiB。CPU准备RAM≥2GiB／D磁盘≥2GiB，未来模型启动RAM≥4.5GiB／空闲显存≥6GiB需届时重新核对，不能把本阶段通过当训练资源通过。CIM查询不可访问，已通过直接GlobalMemoryStatusEx读取RAM，无模型导入。

阶段1资源条件满足：{state['stage1_resource_conditions_met']}；阻塞项：{state['blocking_reasons']}。规划E目标当时存在：{snap['planned_E_target_exists']}，根及祖先已只读核无重解析点；未测试写入。该准入针对首次建立独立新目录，后续E目录已经创建时必须由后续阶段自身完整事务核验，不能用本阶段复核冒充恢复。本阶段不授权自动进入后续CPU输入准备。来源不明、未知目录／锁或部分事务停止保留，不换池、重生成或降低资源门槛。

## 已实现入口

在项目目录用CPU环境运行：

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.resources verify
```

verify严格只读核原完整目录／hash、冻结来源与当前资源，不解码载荷或改写封存结果。prepare在完整同身份缓存下走同一核验；inspect只做只读来源／资源检查且不写文件。未知／部分目录不当作恢复。inputs／training／scoring／metrics／figures尚未实现，不提供其终端命令。

唯一进度入口为原配对执行方案；下一项仅按新授权实施第2阶段，当前训练／前向／合成／校准／指标均0。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare", "verify"])
    args = parser.parse_args()
    c.install_guard()
    for name in c.CODE:
        if name.endswith(".py"):
            ast.parse((c.PACKAGE / name).read_text(encoding="utf-8"), filename=name)
    result = {"inspect": inspect, "prepare": prepare, "verify": verify}[args.command]()
    print(json.dumps(brief(result), ensure_ascii=False, indent=2))
    return 0 if result["resource_state"]["stage1_resource_conditions_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
