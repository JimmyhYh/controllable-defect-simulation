"""Stage two: train-only CPU cache validation and one fixed paired sequence."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import sys
import time

from . import common as c

OUT = c.RESULTS / "inputs_v1"
NATIVE = c.BASE / "formal_preparation_v1"
PILOT = c.BASE / "input_preparation"
CODE = ("inputs.py", "README_inputs.md")
OUTPUT_FILES = {"transaction.json", "protocol.json", "native_inputs.json", "source_refs.json", "requests.json",
                "synthetic_verification.json", "compatibility.json", "summary.json", "第2阶段CPU共同请求与实际合成标签核验.md"}
STAGE1_SHA = "a6153cd01a361df59818b42a861260b82751cbd8815a9d22a0a3b92257a8e898"
BLOCKED = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "safetensors", "huggingface_hub", "joblib"}
_GUARD = False
_ALLOWED_NPZ = set()
_REFERENCES = {}
_ITEMS = {}
_SELECTED = []
_COUNTERS = Counter()
_DECODED_NATIVE = set()
_TRAINING_ROWS = None
np = None
rasterize = None


def digest(path):
    path = c.no_redirect(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity():
    return {name: digest(c.PACKAGE / name) for name in CODE}


def write(name, obj):
    path = c.no_redirect(OUT / str(c.safe_relative(name)))
    c.require(path.parent == OUT, "Stage-two output is a flat sealed namespace")
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("xb") as stream:
        stream.write(payload.encode("utf-8")); stream.flush(); os.fsync(stream.fileno())


def guard():
    """Own stage-two scope: never change frozen stage-one globals or audit hook."""
    global _GUARD
    if _GUARD:
        return
    c.require(not any(n.split(".")[0] in BLOCKED for n in sys.modules), "Model package already imported")
    sys.dont_write_bytecode = True
    creation_dirs = {OUT, *[x for x in OUT.parents if x != Path(OUT.anchor)]}
    def audit(event, args):
        if event == "import" and (args[0].split(".")[0] in BLOCKED or args[0] == "controlled_generation.render"):
            raise RuntimeError("Stage two forbids model/renderer import: " + args[0])
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("Network/pickle execution forbidden")
        if event in {"os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink"}:
            raise RuntimeError("Delete/move/link forbidden")
        if event == "os.mkdir":
            c.require(Path(args[0]).absolute() in creation_dirs, "Directory creation outside explicit E target lineage")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return
            path = Path(name).absolute()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(x in str(mode) for x in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                c.require(path.parent == OUT, "Write outside exact independent E inputs directory")
            elif path.suffix.lower() == ".npz":
                c.require(path in _ALLOWED_NPZ, "NPZ outside R5/N1885 or fixed old03 pool")
            elif path.suffix.lower() in c.OPAQUE_SUFFIXES:
                c.require(path in c._READ_SCOPES, "Raw RGB/GT/probability/tensor payload access forbidden")
            elif path.is_relative_to(c.ROOT / "data"):
                c.require(path == c.ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json", "Raw data access forbidden")
    sys.addaudithook(audit)
    _GUARD = True


def bind_stage1(rehash=True):
    """Validate immutable stage one without using its E-target-absent admission after creation."""
    c.require(digest(c.OUT / "complete.json") == STAGE1_SHA, "Stage-one completion anchor differs")
    complete = c.verify_seal(c.OUT)
    c.require(complete["identity"]["implementation"] == c.implementation(), "Frozen common/resources implementation differs")
    c.require(complete["identity"]["plan_contract"] == c.plan_contract(), "Frozen scientific plan/output roots differ")
    proof = c.load(c.BOOT / "resource_verification_v1/verification.json")
    c.verify_seal(c.BOOT / "resource_verification_v1")
    c.require(proof["stage1_complete_sha256"] == STAGE1_SHA and proof["exit_code"] == 0, "Independent stage-one verification missing")
    sources = c.load(c.OUT / "sources.json")
    closures = c.load(c.OUT / "source_trees.json")
    c.require(len(sources) == 2655 and len(closures) == 7, "Stage-one source population differs")
    for entry in closures:
        root = Path(entry["root"])
        c.require(digest(root / "complete.json") == entry["complete_sha256"], "Upstream closure identity differs")
        c.verify_closed_tree(root, set(c.load(root / "complete.json")["files"]) | {"complete.json"}, entry["allowed_empty_dirs"])
    if rehash:
        for path, expected in sources.items():
            c.require(c.sha(path) == expected, "Stage-one source changed: " + path)
    # The frozen metadata reader uses a separate stdlib-only interpreter, never a model import.
    from .resources import environment_metadata
    c.require(environment_metadata() == c.load(c.OUT / "environment_metadata.json"), "Frozen independent model environment changed")
    return sources


def source_metadata():
    global _REFERENCES, _ITEMS, _SELECTED, _ALLOWED_NPZ
    complete = c.load(NATIVE / "complete.json")
    refs = {str(NATIVE / "complete.json"): digest(NATIVE / "complete.json")}
    c.require(refs[str(NATIVE / "complete.json")] == "ff6a797fee6622e1c90b7f4ee23e8497ec2c8e1f758de5faf850c5673439a6e1", "Native input seal differs")
    items = c.load(NATIVE / "inputs.json")
    split = c.load(c.ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json")
    c.require(len(items) == 1890 and [r["id"] for r in items] == [r["id"] for r in split["R_real5"] + split["N_train1885"]], "Native role/order differs")
    for row in items:
        c.require(row["cache"] == f"inputs/{row['id']}.npz" and row["kind"] == ("real" if row["id"] in c.R5 else "normal"), "Native cache role/path differs")
        path = NATIVE / row["cache"]
        refs[str(path)] = complete["files"][row["cache"]]
    refs[str(NATIVE / "inputs.json")] = complete["files"]["inputs.json"]
    c.require(digest(NATIVE / "inputs.json") == refs[str(NATIVE / "inputs.json")], "Native item metadata hash differs")
    selected = c.load(c.OUT / "selected_materials.json")
    c.require(len(selected) == 2500 and [r["ordinal"] for r in selected] == list(range(2500)), "Fixed material order differs")
    for row in selected:
        path = c.OLD / str(c.MATERIAL_SEED) / "synthetic" / f"{row['ordinal']:04d}.npz"
        c.require(row["file"] == str(path) and row["background_request"]["id"] in {r["id"] for r in split["N_train1885"]}, "Material source outside fixed N")
        refs[str(path)] = row["sha256"]
    pilot_seal = c.load(PILOT / "complete.json")
    refs[str(PILOT / "complete.json")] = digest(PILOT / "complete.json")
    c.require(refs[str(PILOT / "complete.json")] == "d33ddbde7b0b4cf585ea8aa407903c28dc885d4730607b94188c8de346b9d396", "Original 21-window seal differs")
    refs[str(PILOT / "protocol.json")] = pilot_seal["files"]["protocol.json"]
    c.require(digest(PILOT / "protocol.json") == refs[str(PILOT / "protocol.json")], "Original 21-window metadata differs")
    for name in ["controlled_generation/__init__.py", "controlled_generation/data.py", "controlled_generation/geometry.py"]:
        refs[str(c.ROOT / name)] = digest(c.ROOT / name)
    _REFERENCES = refs; _ITEMS = {r["id"]: r for r in items}; _SELECTED = selected
    _ALLOWED_NPZ = {Path(name) for name in refs if Path(name).suffix == ".npz"}
    return items, selected, refs


def cpu_modules():
    global np, rasterize
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(), "Use existing CPU .venv")
    c.require(metadata.version("numpy") == "1.26.4" and metadata.version("opencv-python") == "4.11.0.86", "Frozen CPU numerical dependency differs")
    import numpy as numpy
    from controlled_generation.geometry import rasterize as geometry_rasterize
    np, rasterize = numpy, geometry_rasterize
    c.require("controlled_generation.render" not in sys.modules and not any(n.split(".")[0] in BLOCKED for n in sys.modules), "Unexpected model/renderer import")


def resources():
    snapshot = c.resource_snapshot()
    required = c.load(c.OUT / "resource_assessment.json")["disk_forecast"]["required_free_before_round_bytes"] / 2**30
    c.require(snapshot["available_RAM_GiB"] >= 2 and snapshot["D_free_GiB"] >= 2 and snapshot["E_free_GiB"] >= required,
              "CPU RAM/D metadata space/E full-round forecast not satisfied")
    return snapshot


def snapshot_npz(path):
    path = c.no_redirect(path)
    c.require(path in _ALLOWED_NPZ and str(path) in _REFERENCES, "Unbound NPZ")
    c.require(path.stat().st_size <= 16 * 2**20, "Unexpected cache size")
    raw = path.read_bytes()
    c.require(hashlib.sha256(raw).hexdigest() == _REFERENCES[str(path)], "Same-snapshot NPZ hash differs")
    return io.BytesIO(raw)


@lru_cache(maxsize=64)
def native(sample_id):
    row = _ITEMS[sample_id]
    with np.load(snapshot_npz(NATIVE / row["cache"]), allow_pickle=False) as pack:
        c.require(set(pack.files) == {"rgb", "mask"}, "Native NPZ keys differ")
        rgb, mask = pack["rgb"], pack["mask"]
    h, w = row["height"], row["width"]
    c.require(rgb.dtype == np.uint8 and rgb.shape == (h, w, 3) and mask.dtype == np.uint8 and mask.shape == (h, w), "Native dtype/shape differs")
    c.require(np.all((mask == 0) | (mask == 1)) and bool(mask.any()) == (row["kind"] == "real"), "Native target is not original binary role")
    c.require([int(mask[r["y"]:r["y"] + 256].sum()) for r in row["windows"]] == [r["positive_pixels"] for r in row["windows"]], "Native legal-window positivity differs")
    rgb.flags.writeable = False; mask.flags.writeable = False
    _COUNTERS["native_npz_decode_operations"] += 1; _DECODED_NATIVE.add(sample_id)
    return rgb, mask


def normalize_float(image):
    c.require(image.dtype == np.float32 and image.shape == (256, 256, 3) and np.isfinite(image).all()
              and image.min() >= 0 and image.max() <= 1, "Expected unquantized RGB float32[0,1]")
    mean = np.asarray([.485, .456, .406], np.float32)
    std = np.asarray([.229, .224, .225], np.float32)
    return np.ascontiguousarray(((image - mean) / std).transpose(2, 0, 1))


def native_pack(ref, flips):
    row = _ITEMS[ref["id"]]
    rgb, mask = native(row["id"])
    y = ref["y"]
    c.require(y in [r["y"] for r in row["windows"]], "Window not in fixed native grid")
    w = row["width"]
    tile = np.pad(rgb[y:y + 256], ((0, 0), (0, 256-w), (0, 0)), mode="reflect")
    target = np.pad(mask[y:y + 256], ((0, 0), (0, 256-w)), mode="constant")
    valid = np.zeros((256, 256), dtype=bool); valid[:, :w] = True
    for axis, active in [(1, flips[0]), (0, flips[1])]:
        if active:
            tile, target, valid = (np.flip(x, axis=axis).copy() for x in [tile, target, valid])
    image = normalize_float(tile.astype(np.float32) / np.float32(255))
    c.require(int(valid.sum()) == w * 256 and not target[~valid].any(), "Native padding enters labels/effective domain")
    return {"image": image, "target": np.ascontiguousarray(target), "valid": np.ascontiguousarray(valid)}


def synthetic_pack(ordinal):
    row = _SELECTED[ordinal]
    with np.load(snapshot_npz(Path(row["file"])), allow_pickle=False) as arrays:
        c.require(set(arrays.files) == {"image", "target", "valid"}, "Synthetic NPZ keys differ")
        image, target, valid_raw = arrays["image"], arrays["target"], arrays["valid"]
    c.require(target.dtype == valid_raw.dtype == np.uint8 and target.shape == valid_raw.shape == (256, 256)
              and np.all((target == 0) | (target == 1)) and np.all((valid_raw == 0) | (valid_raw == 1)), "Synthetic mask/valid is not original binary cache")
    valid = valid_raw.astype(bool)
    c.require(target.any() and not target[~valid].any() and int(target.sum()) == row["positive_pixels_recorded"]
              and int(valid.sum()) == row["valid_pixels_recorded"], "Synthetic effective target/domain differs")
    _COUNTERS["synthetic_npz_decode_operations"] += 1
    return {"image": normalize_float(image), "target": target, "valid": valid}, image


def pack_identity(pack):
    result = {}
    h = hashlib.sha256()
    for key in ["image", "target", "valid"]:
        array = np.ascontiguousarray(pack[key])
        label = f"{key}:{array.dtype}:{array.shape}".encode()
        value = hashlib.sha256(array.tobytes()).hexdigest()
        result[key] = value; h.update(label); h.update(bytes.fromhex(value))
    result["pack"] = h.hexdigest()
    return result


def accepted_records():
    rows = c.load(c.OLD / "synthetic_manifest.json")
    selected = [r for r in rows if r["seed"] == c.MATERIAL_SEED]
    c.require([r["ordinal"] for r in selected] == list(range(2500)), "Original accepted record order differs")
    return selected


def verify_synthetic(ordinal, pack, image, accepted):
    row = _SELECTED[ordinal]; request = row["background_request"]
    native_row = _ITEMS[request["id"]]
    c.require(native_row["kind"] == "normal", "Synthetic source is not N")
    rgb, mask = native(native_row["id"])
    w, y = native_row["width"], request["y"]
    background = np.pad(rgb[y:y + 256].astype(np.float32) / np.float32(255), ((0,0),(0,256-w),(0,0)), mode="reflect")
    valid = np.zeros((256, 256), dtype=bool); valid[:, :w] = True
    for axis, active in [(1, request["flip_x"]), (0, request["flip_y"])]:
        if active:
            background, valid = (np.flip(x, axis=axis).copy() for x in [background, valid])
    c.require(np.array_equal(valid, pack["valid"]), "Cached flip/padding effective domain differs")
    attempt = accepted["attempts"][-1]
    c.require(attempt["accepted"] and attempt["attempt"] == row["accepted_attempt"], "Accepted attempt differs")
    spec = attempt["spec"]
    c.require(spec["shape"] in {"ellipse", "line"}, "Only original accepted ellipse/line geometry authorized")
    roi, geometry = rasterize(spec, seed=c.MATERIAL_SEED)
    c.require(geometry["raster"] == attempt["stats"]["geometry"]["raster"], "Original raster geometry metadata differs")
    d = np.max(np.abs(image - background), axis=-1)
    effective = d > np.float32(1/255)
    coverage = float((effective & roi).sum() / roi.sum())
    outside_valid = float(d[~valid].max(initial=0))
    outside_roi = float(d[~roi].max(initial=0))
    c.require(np.array_equal(effective, pack["target"].astype(bool)) and effective.any()
              and not (roi & ~valid).any() and coverage >= .9 and outside_valid == outside_roi == 0,
              "Actual residual/effective mask/ROI/domain invalid; no resampling or repair")
    residual_stats = [float(d[roi].min()), float(np.median(d[roi])), float(d[roi].max())]
    c.require(residual_stats == attempt["stats"]["roi_residual_min_median_max"] and coverage == attempt["stats"]["effective_coverage"], "Actual residual statistics differ from original accepted result")
    return {"ordinal": ordinal, "type": row["type"], "background_id": request["id"], "source_sha256": row["sha256"],
            "accepted_attempt": row["accepted_attempt"], "effective_pixels": int(effective.sum()), "ROI_pixels": int(roi.sum()),
            "valid_pixels": int(valid.sum()), "effective_coverage": coverage, "roi_residual_min_median_max": residual_stats,
            "outside_valid_max_residual": outside_valid, "outside_ROI_max_residual": outside_roi,
            "ROI_sha256": hashlib.sha256(roi.tobytes()).hexdigest(), "D_float32_sha256": hashlib.sha256(d.tobytes()).hexdigest(),
            "target_equals_actual_D_gt_float32_1over255": True, "cached_orientation_preserved": True,
            "normalization": "float32[0,1] minus mean divided std; no second /255/uint8/flip",
            "normalized_min": float(pack["image"].min()), "normalized_max": float(pack["image"].max()),
            "original_clip_fraction": attempt["stats"]["roi_clipped_channel_fraction"], "pack_identity": pack_identity(pack)}


def requests(items):
    """Exactly the frozen scalar RNG contract; never use model/global RNG."""
    generators = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.DATA_SEED, k]))) for k in range(4)]
    shared_rng, extra_rng, normal_rng, flip_rng = generators
    shared_order = [int(x) for _ in range(500) for x in shared_rng.permutation(5)]
    extra_order = [int(x) for _ in range(500) for x in extra_rng.permutation(5)]
    normal_ids = normal_rng.integers(1885, size=5000)
    flips = flip_rng.random((10000, 2)) < .5
    rows = []; normal_index = 0
    for k in range(2500):
        for slot in range(4):
            update = 4*k + slot + 1
            flags = [bool(x) for x in flips[update-1]]
            row = {"update": update, "slot": "shared_R" if slot == 0 else "intervention" if slot == 2 else "shared_N",
                   "flip_h": flags[0], "flip_v": flags[1], "flip_origin": "namespace3 pre-generated row-major random(10000,2)"}
            if slot == 0:
                item = items[shared_order[k]]
                choices = [w for w in item["windows"] if w["positive_pixels"] > 0]
                choice = choices[int(shared_rng.integers(len(choices)))]
                row["common_input"] = {"kind": "real", "id": item["id"], "y": choice["y"]}
            elif slot in (1, 3):
                item = items[5 + int(normal_ids[normal_index])]; normal_index += 1
                choice = item["windows"][int(normal_rng.integers(len(item["windows"])))]
                row["common_input"] = {"kind": "normal", "id": item["id"], "y": choice["y"]}
            else:
                item = items[extra_order[k]]
                choices = [w for w in item["windows"] if w["positive_pixels"] > 0]
                choice = choices[int(extra_rng.integers(len(choices)))]
                row["A_input"] = {"kind": "real", "id": item["id"], "y": choice["y"]}
                row["B_input"] = {"kind": "synthetic", "ordinal": k, "type": _SELECTED[k]["type"]}
                old = _SELECTED[k]["background_request"]
                row.update(flip_h=old["flip_x"], flip_v=old["flip_y"], flip_origin="old accepted cached background_request; B already flipped")
            rows.append(row)
    c.require(len(rows) == 10000 and normal_index == 5000, "Common request count differs")
    return rows


def actual_pack(ref, row):
    if ref["kind"] == "synthetic":
        return synthetic_pack(ref["ordinal"])[0]
    return native_pack(ref, (row["flip_h"], row["flip_v"]))


def verify_requests(rows):
    counts = {g: Counter() for g in "AB"}; r5 = {g: Counter() for g in "AB"}
    pixel_counts = {g: {k: {"updates": 0, "valid_pixels": 0, "positive_pixels": 0} for k in ["real", "normal", "synthetic"]} for g in "AB"}
    synthetics = []; shared = 0; identities = []
    old = accepted_records()
    for row in rows:
        packs = {}
        if "common_input" in row:
            pack = actual_pack(row["common_input"], row)
            identity = pack_identity(pack)
            row["input_identity_A"] = row["input_identity_B"] = identity
            packs = {"A": pack, "B": pack}; shared += 1
        else:
            packs["A"] = actual_pack(row["A_input"], row)
            packs["B"], raw = synthetic_pack(row["B_input"]["ordinal"])
            syn = verify_synthetic(row["B_input"]["ordinal"], packs["B"], raw, old[row["B_input"]["ordinal"]])
            synthetics.append(syn)
            for group in "AB": row["input_identity_"+group] = pack_identity(packs[group])
        for group in "AB":
            ref = row.get("common_input", row.get(group+"_input")); kind = ref["kind"]; pack = packs[group]
            counts[group][kind] += 1
            if kind == "real": r5[group][ref["id"]] += 1
            number = pixel_counts[group][kind]
            number["updates"] += 1; number["valid_pixels"] += int(pack["valid"].sum())
            number["positive_pixels"] += int(pack["target"][pack["valid"]].sum())
        if row["update"] % 1000 == 0:
            resources(); print(json.dumps({"CPU_requests_verified": row["update"], "of": 10000, "model_operations": 0}), flush=True)
    c.require(counts["A"] == Counter(real=5000, normal=5000) and counts["B"] == Counter(real=2500, normal=5000, synthetic=2500), "Group exposure differs")
    c.require(r5["A"] == Counter(dict.fromkeys(c.R5,1000)) and r5["B"] == Counter(dict.fromkeys(c.R5,500)), "R5 exposure differs")
    c.require(shared == 7500 and len(synthetics) == 2500 and Counter(x["type"] for x in synthetics) == Counter(dict.fromkeys(c.TYPES,625)), "Shared/material count differs")
    normal_ids = {r["common_input"]["id"] for r in rows if r["slot"] == "shared_N"}
    background_ids = {r["background_id"] for r in synthetics}
    return synthetics, {"A": dict(counts["A"]), "B": dict(counts["B"]), "R5_by_group": {g: dict(r5[g]) for g in "AB"},
            "shared_slots": shared, "all_shared_input_array_hashes_equal": True, "intervention_slots": 2500,
            "common_N_distinct_images": len(normal_ids), "synthetic_background_distinct_images": len(background_ids),
            "normal_and_synthetic_background_overlap": len(normal_ids & background_ids), "pixel_exposure_by_group_role": pixel_counts,
            "horizontal_flips": sum(r["flip_h"] for r in rows), "vertical_flips": sum(r["flip_v"] for r in rows)}


def compatibility(synthetics):
    pilot = c.load(PILOT / "protocol.json"); rows = []
    for item in pilot["items"]:
        ref = {"kind": item["kind"], "id": item["id"], "y": item["y"]}
        pack = native_pack(ref, (False, False))
        c.require(int(pack["valid"].sum()) == item["valid_pixels"] and int(pack["target"].sum()) == item["positive_pixels"], "Original pilot window differs")
        rows.append({"kind": item["kind"], "id": item["id"], "y": item["y"], "flip_h": False, "flip_v": False,
                     "valid_pixels": int(pack["valid"].sum()), "positive_pixels": int(pack["target"].sum()), "input_identity": pack_identity(pack)})
    c.require(len(rows) == 21 and sum(x["valid_pixels"] for x in rows) == 1225984 and sum(x["positive_pixels"] for x in rows) == 24482, "Original21 denominator differs")
    for syn in synthetics[:4]:
        rows.append({"kind": "synthetic", "ordinal": syn["ordinal"], "type": syn["type"], "already_flipped": True,
                     "valid_pixels": syn["valid_pixels"], "positive_pixels": syn["effective_pixels"], "input_identity": syn["pack_identity"]})
    c.require(len(rows) == 25, "Fixed compatibility inputs incomplete")
    return {"inputs": rows, "training_image_sources_in_original21": 21, "synthetic_sources": 4,
            "valid_pixels": sum(x["valid_pixels"] for x in rows), "positive_pixels": sum(x["positive_pixels"] for x in rows),
            "meaning": "CPU interface metadata only; not training, forward, AP or learnability acceptance"}


def preparation_protocol():
    return {"schema": "pretrained-paired-inputs-v1", "stage1_complete_sha256": STAGE1_SHA, "stage1_implementation": c.implementation(),
            "implementation": code_identity(), "plan_contract": c.plan_contract(), "output": str(OUT),
            "data_seed": c.DATA_SEED, "initialization_seeds": list(c.SEEDS), "single_shared_sequence_all_pairs": True,
            "updates_per_group": 10000, "shared_slots": 7500, "synthetic_count": 2500, "material_seed": c.MATERIAL_SEED,
            "normalization": {"real": "uint8/255 then mean/std", "synthetic": "original float32[0,1] minus mean/std; no /255 or uint8 or second flip",
                              "mean": [.485,.456,.406], "std": [.229,.224,.225]},
            "CPU_environment": {k: metadata.version(k) for k in ["numpy","Pillow","opencv-python"]},
            "geometry": "original accepted ellipse/line spec via frozen geometry.rasterize only; render not imported/called",
            "source_caches_copied_or_rewritten": False, "actual_residual_label": "max abs RGB difference > float32(1/255)",
            "coverage_min": .9, "outside_valid_and_ROI_residual": "exact zero", "empty_or_invalid": "preserve and stop; no pool/sample/seed/amplitude replacement",
            "scope": "R5/N1885 native cached inputs and old03 NPZ only; no raw RGB/GT, Q/E/U, probability, model or training",
            "E_historically_seen": True, "interpretation": "composition strategy includes reduced R5 and changed background/pixel exposure",
            "training": "manual user startup only; stage3 not authorized"}


def prepare():
    if OUT.exists():
        return verify(decode=False)
    bind_stage1()
    items, selected, refs = source_metadata()
    guard(); snapshot = resources()
    c.no_redirect(c.LARGE)
    c.require(not c.LARGE.exists(), "E experiment root already exists; do not adopt unknown directory")
    cpu_modules()
    c.LARGE.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        transaction = {"stage1_complete_sha256": STAGE1_SHA, "started_at_UTC": datetime.now(timezone.utc).isoformat(),
                       "scope": "new exact E input directory, CPU cache preparation only", "resources": snapshot}
        write("transaction.json", transaction)
        c.require(c.load(OUT / "transaction.json") == transaction, "E exclusive-create/fsync/readback writeability check failed")
        write("protocol.json", preparation_protocol())
        for index, item in enumerate(items):
            native(item["id"])
            if (index+1) % 100 == 0: resources()
        c.require(len(_DECODED_NATIVE) == 1890, "Not all fixed training caches decoded")
        rows = requests(items)
        synthetics, exposures = verify_requests(rows)
        compat = compatibility(synthetics)
        write("native_inputs.json", items)
        write("source_refs.json", refs)
        write("requests.json", rows)
        write("synthetic_verification.json", {"items": synthetics, "effective": len(synthetics), "types": dict(Counter(x["type"] for x in synthetics)),
              "render_or_new_generation_calls": 0, "coverage_min_observed": min(x["effective_coverage"] for x in synthetics),
              "actual_residual_max_range": [min(x["roi_residual_min_median_max"][2] for x in synthetics),max(x["roi_residual_min_median_max"][2] for x in synthetics)]})
        write("compatibility.json", compat)
        # Recheck source identities/closed namespaces; do not use stage1's absent-E admission after our creation.
        bind_stage1()
        for path, expected in refs.items():
            c.require(digest(path) == expected, "CPU input source changed during preparation")
        c.require("controlled_generation.render" not in sys.modules and not any(n.split(".")[0] in BLOCKED for n in sys.modules), "Forbidden execution detected")
        result = {"CPU_inputs_ready": True, "native_source_images_verified": 1890, "synthetic_materials_verified": 2500,
                  "fixed_request_rows": len(rows), "three_pairs_reference_same_requests": True, "exposures": exposures,
                  "compatibility_inputs": 25, "compatibility_valid_pixels": compat["valid_pixels"], "compatibility_positive_pixels": compat["positive_pixels"],
                  "decode_operations": dict(_COUNTERS), "original_RGB_GT_probability_reads": 0, "model_tensor_forward_training_calibration_metrics_or_render": 0,
                  "resources_before": snapshot, "resources_after": resources(), "E_exclusive_create_fsync_readback_verified": True,
                  "elapsed_seconds": time.monotonic()-started, "cache_reuse": False}
        write("summary.json", result)
        write("第2阶段CPU共同请求与实际合成标签核验.md", report(result))
        expected = OUTPUT_FILES
        c.verify_closed_tree(OUT, expected)
        write("complete.json", {"identity": {"implementation": code_identity(), "stage1_complete_sha256": STAGE1_SHA, "plan_contract": c.plan_contract()},
                                "files": {name: digest(OUT/name) for name in sorted(expected)}})
        return result
    except Exception as error:
        if not (OUT/"complete.json").exists() and not (OUT/"failure.json").exists():
            write("failure.json", {"error": repr(error), "rule": "preserve exact transaction; no retry/pool/seed/amplitude repair"})
        raise
    finally:
        native.cache_clear()


def verify(decode=False):
    bind_stage1()
    items, selected, refs = source_metadata()
    guard(); resources()
    c.require(set(c.load(OUT/"complete.json")["files"]) == OUTPUT_FILES, "Missing or unknown output in completion file list")
    c.verify_closed_tree(OUT, OUTPUT_FILES | {"complete.json"})
    seal = c.verify_seal(OUT)
    c.require(seal["identity"] == {"implementation": code_identity(), "stage1_complete_sha256": STAGE1_SHA, "plan_contract": c.plan_contract()}, "Stage-two implementation/input contract differs")
    c.require(c.load(OUT/"protocol.json") == preparation_protocol(), "Frozen input protocol differs")
    c.require(c.load(OUT/"native_inputs.json") == items and c.load(OUT/"source_refs.json") == refs, "Frozen source references differ")
    for path, expected in refs.items(): c.require(digest(path) == expected, "Frozen CPU source hash differs")
    rows = c.load(OUT/"requests.json"); summary = c.load(OUT/"summary.json")
    c.require(len(rows) == 10000 and [r["update"] for r in rows] == list(range(1,10001)), "Request sequence incomplete")
    c.require(summary["CPU_inputs_ready"] and summary["exposures"]["shared_slots"] == 7500, "Saved exposure summary differs")
    if decode:
        cpu_modules()
        expected = requests(items)
        for actual, base in zip(rows, expected):
            c.require({k:v for k,v in actual.items() if not k.startswith("input_identity_")} == base, "RNG request differs")
        saved_identities = [(r["input_identity_A"], r["input_identity_B"]) for r in rows]
        synthetics, exposures = verify_requests(rows)
        c.require(saved_identities == [(r["input_identity_A"], r["input_identity_B"]) for r in rows], "Normalized input bytes differ from saved request identities")
        c.require(exposures == summary["exposures"] and synthetics == c.load(OUT/"synthetic_verification.json")["items"], "CPU input verification differs")
        c.require(compatibility(synthetics) == c.load(OUT/"compatibility.json"), "Compatibility metadata differs")
        native.cache_clear()
    summary["cache_reuse"] = True
    summary["verification_decoded_cache_payloads"] = bool(decode)
    return summary


def training_pack(group, update):
    """CPU loader only. Future training must separately authorize and verify its own run."""
    global _TRAINING_ROWS
    c.require(group in {"A","B"} and type(update) is int and 1 <= update <= 10000, "Invalid group/update")
    if _TRAINING_ROWS is None:
        verify(decode=False)
        _TRAINING_ROWS = c.load(OUT/"requests.json")
    guard(); cpu_modules()
    row = _TRAINING_ROWS[update-1]
    ref = row.get("common_input",row.get(group+"_input"))
    pack = actual_pack(ref,row)
    c.require(pack_identity(pack) == row["input_identity_"+group], "Runtime input bytes differ from CPU preparation")
    pack["input_identity"] = row["input_identity_"+group]
    return pack


def report(r):
    return f'''# 第2阶段：共同请求与实际合成标签CPU核验

第1阶段完整协议／2655来源／7闭合目录、固定单头训练侧原生缓存与旧03池身份已核对。仅首次独占创建本轮E根及results/inputs_v1，transaction.json排他创建、fsync及读回通过；未认领已有未知目录或改动D盘旧成果。全部结果都是元信息、请求及核验证据，原图／缓存未复制、量化或重写。

实际1890份R5/N1885原生缓存完整核hash／二值GT／合法窗；固定2500份旧NPZ核float32[0,1]、已有翻转／有效域、原背景实际D及M_eff、ROI／域外严格零变化、原接受spec几何与覆盖≥0.90，四类各625。原接受几何仅重建ROI，无render或新合成。合成直接ImageNet归一化，没有再除255、转uint8、重新补边或二次翻转。

数据seed2026102300冻结一套10000共同请求，三初始化seed仅作未来协议值。A真实5000／正常5000，B真实2500／合成2500／正常5000；共享7500槽使用完全相同输入引用与数组hash，每R分别1000／500次，干预2500槽来源差异明确。实际曝光及像素阳性／有效域总数见summary.json，不能称像素级正负比例匹配或合成单因素因果。

固定25项兼容输入仅元信息，原21训练窗＋四类各首个旧合成；实际有效{r['compatibility_valid_pixels']}像素／阳性{r['compatibility_positive_pixels']}。没有前向、训练或可学习性验收。

本次新请求{r['fixed_request_rows']}行，不是模型更新。CPU保存段{r['elapsed_seconds']:.3f}秒；启动RAM{r['resources_before']['available_RAM_GiB']:.3f}GiB／E空闲{r['resources_before']['E_free_GiB']:.3f}GiB，仅快照，不是连续峰值。数据解码事件{r['decode_operations']}，LRU最多64份原生缓存；1890是不同源图数，重复CPU读取不增加独立样本。未读取原RGB／原GT／Q/E/U或概率，模型／tensor／训练／校准／指标／渲染均0，旧阈值／权重／源码／环境不变。

## 已实现的只读核验入口

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.inputs verify
```

verify只核完整同身份结果／来源，默认不解码缓存、不写入。verify-decode是显式完整CPU复核，无模型或生成；prepare已有完整结果只走默认verify。未知／部分事务保留停止，不自动补样／重试。training_pack仅CPU取样包装，不是训练入口；training/scoring/metrics仍未实现。下一项仅另行授权第3阶段准备，所有训练继续由用户手动启动。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare","verify","verify-decode"])
    args = parser.parse_args()
    for name in CODE:
        if name.endswith(".py"): ast.parse((c.PACKAGE/name).read_text(encoding="utf-8"),filename=name)
    result = prepare() if args.command == "prepare" else verify(decode=args.command == "verify-decode")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
