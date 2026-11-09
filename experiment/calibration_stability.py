"""Week 3 frozen C calibration and held-out H evaluation; no fitting or E inference."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import winreg

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "calibration_stability" / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "cb96f342ab3d2ff7fd5086587de1248f6f1ae5fbb7419f3f11c95a0b531cf7b9"
SCORES_PATH = OUTPUT.parent / "scores.npz"
SUMMARY_PATH = OUTPUT.parent / "summary.json"
H_SCORES_PATH = OUTPUT.parent / "h_scores.npz"
H_RESULT_PATH = OUTPUT.parent / "h_evaluation.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resources():
    # Read-only metadata: no network, tensor operations or model initialization.
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    import torch
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.strip()
    require(torch.cuda.is_available(), "Saved CUDA configuration unavailable; investigate before changing protocol")
    return {"python": sys.version.split()[0], "cpu": cpu,
            "logical_threads": os.cpu_count(), "windows": str(sys.getwindowsversion()),
            "ram_total_gib": memory.total / 2**30, "ram_available_gib": memory.available / 2**30,
            "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30,
            "nvidia_smi_columns": ["name", "driver_version", "memory_total_mib", "memory_free_mib"],
            "nvidia_smi": gpu, "cuda_available": True,
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "numpy", "faiss-cpu", "timm", "scipy", "scikit-learn")},
            "scope": "Read-only metadata; free resources are a snapshot, not a runtime guarantee",
            "cim_limitation": "Session Get-CimInstance failed to access CIM; RAM via GlobalMemoryStatusEx, CPU via registry"}


def expected_protocol():
    split_path = ROOT / "data" / "split.json"
    config_path = ROOT / "models" / "formal_patchcore" / "configuration.json"
    split, config = read_json(split_path), read_json(config_path)
    settings = config["configuration"]
    groups = split["splits"]
    require({k: len(groups[k]) for k in ("R", "V", "P", "E")} ==
            {"R": 176, "V": 49, "P": 20, "E": 124}, "Original split counts differ")
    require(settings["selected_paths"] == groups["R"][:80] and settings["R_selected"] == 80,
            "Formal selected paths are not the frozen R80")
    require(config["formal_bank_built"] and config["bank_vectors"] == 627, "Formal R80 bank incomplete")
    require(settings["device"] == "cuda" and settings["faiss_device"] == "cpu", "Saved devices differ")
    require(config["source_commit"] == "fcaa92f124fb1ad74a7acf56726decd4b27cbcad", "Author version differs")
    require(config["threshold"] == 1.5751205444335934 and
            config["threshold_quantile"] == 0.95 and config["quantile_method"] == "linear",
            "Original calibration differs")
    day1 = read_json(ROOT / "results" / "day1.json")
    require(day1["formal_configuration"]["selected_paths"] == settings["selected_paths"], "Day 1 selection differs")
    identities = {}
    def identity(path, expected=None):
        value = digest(path)
        if expected is not None:
            require(value == expected, "Source identity mismatch: " + str(path))
        identities[str(path.relative_to(ROOT).as_posix())] = value
    identity(split_path, config["split_sha256"])
    identity(config_path)
    identity(ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth", config["weights_sha256"])
    index_path = ROOT / "models" / "formal_patchcore" / "nnscorer_search_index.faiss"
    identity(index_path, config["index_sha256"])
    identity(ROOT / "models" / "formal_patchcore" / "patchcore_params.pkl")
    index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
    require(index.ntotal == 627 and index.d == 1024 and index.metric_type == faiss.METRIC_L2
            and isinstance(index, faiss.IndexFlatL2), "FAISS bank type/dimension/count differ")
    del index
    for name in ("data.py", "detector.py", "evaluation.py", "demo.py", "requirements-lock.txt"):
        identity(ROOT / name)
    source_root = ROOT / "vendor" / "patchcore-inspection"
    require((source_root / "LICENSE").is_file(), "Author license missing")
    for path in sorted((source_root / "src" / "patchcore").rglob("*.py")):
        identity(path)
    identity(source_root / "LICENSE")
    category = Path(split["category_root"]).resolve()
    require(category == (ROOT / "data" / "mvtec_ad" / "leather").resolve(), "Unexpected public data root")
    normal_groups = [groups[k] for k in ("R", "V", "P")]
    normals = [name for group in normal_groups for name in group]
    require(len(normals) == len(set(normals)) == 245, "Original normal groups overlap")
    require(set(normals) == set(split["train_sha256"]), "Training identities do not cover original normals")
    hashes = {}
    for name in normals:
        path = (category / name).resolve()
        require(path.parent == category / "train" / "good" and path.suffix == ".png", "Non-normal path in split")
        value = digest(path)  # Bytes only, no image decoding or model scoring.
        require(value == split["train_sha256"][name], "Normal image identity changed: " + name)
        hashes[name] = value
    require(len(set(hashes.values())) == 245, "Duplicate normal content identities")
    for name in ("license.txt", "readme.txt"):
        path = category / name
        require(path.is_file(), "Public data license/readme missing")
        identities["data/mvtec_ad/leather/" + name] = digest(path)
    unused = groups["R"][80:]
    order = np.random.default_rng(20261010).permutation(96)
    c = [unused[int(i)] for i in order[:48]]
    h = [unused[int(i)] for i in order[48:]]
    excluded = set(groups["R"][:80] + groups["V"] + groups["P"] +
                   [item["image"] for item in groups["E"]])
    require(not (set(c) & set(h)) and not ((set(c) | set(h)) & excluded)
            and set(c + h) == set(unused), "New split overlaps or does not cover U96")
    return {"schema_version": 1, "mode": "main_C48_H48", "source_commit": config["source_commit"],
            "source_sha256": identities, "configuration": settings, "category_root": str(category),
            "splits": {"R80": groups["R"][:80], "U96": unused, "C": c, "H": h},
            "normal_image_sha256": hashes,
            "counts": {"R80": 80, "U96": 96, "C": 48, "H": 48},
            "split_rule": "Preserve original R[80:] order; PCG64 permutation; first48 C, last48 H",
            "split_seed": 20261010, "sampling_seed": 20261011, "rng": "numpy.random.default_rng (PCG64)",
            "sample_sizes": [8, 16, 32, 48], "repetitions": 500, "primary_comparison": [16, 48],
            "sampling_rule": "One RNG stream; each repetition permutes C48; nested prefixes; no replacement within repetition",
            "threshold_quantile": 0.95, "quantile_method": "linear", "decision": "score > threshold",
            "original_V_threshold": config["threshold"], "canonical_new_threshold": "Full C48 quantile; not yet calculated",
            "score_rule": "Raw FAISS squared L2, nearest neighbour1, max patch; no score normalization",
            "preprocessing": split["preprocessing"], "batch_size": 1,
            "freeze_order": "C48 scores and all thresholds frozen before H48 image inference",
            "metrics": {"threshold": ["median", "q05", "q95", "q95-q05", "median absolute deviation from full C48 threshold"],
                        "H": "Each repetition FP/48; canonical C48 and original V threshold integer FP/48",
                        "interval": "Finite-pool empirical middle90%, not population confidence interval",
                        "all48": "Zero spread by construction, not absence of population uncertainty"},
            "scope": "Personal learning and noncommercial public research; no company materials",
            "day1": "Protocol only; no image scores, thresholds, new fit or evaluation",
            "fallback": "If resources unavailable, saved V49 scores only; H evaluation deferred"}


def prepare():
    expected = expected_protocol()
    if OUTPUT.exists():
        saved = read_json(OUTPUT)
        stable = {k: v for k, v in saved.items() if k not in ("prepared_at", "resource_check")}
        require(stable == expected, "Existing protocol/source differs; do not replace frozen protocol")
        # Resource snapshot can change; report current metadata without rewriting frozen data.
        current = resources()
        state = "existing protocol verified; not rewritten"
    else:
        current = resources()
        saved = dict(expected, prepared_at=datetime.now().astimezone().isoformat(), resource_check=current)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        state = "protocol prepared; no model inference"
    print(json.dumps({"status": state, "protocol": str(OUTPUT), "counts": saved["counts"],
                      "resource_check_current": current}, ensure_ascii=True, indent=2))


def c_protocol():
    # Do not call expected_protocol(): it hashes all normal images including H/P.
    require(digest(OUTPUT) == FROZEN_PROTOCOL_SHA256, "Day 1 protocol changed")
    protocol = read_json(OUTPUT)
    for relative, expected in protocol["source_sha256"].items():
        require(digest(ROOT / relative) == expected, "Frozen source changed: " + relative)
    require(protocol["sample_sizes"] == [8, 16, 32, 48] and protocol["repetitions"] == 500
            and protocol["sampling_seed"] == 20261011 and protocol["threshold_quantile"] == 0.95
            and protocol["quantile_method"] == "linear" and protocol["decision"] == "score > threshold",
            "Frozen sampling settings differ")
    category = Path(protocol["category_root"]).resolve()
    require(category == (ROOT / "data/mvtec_ad/leather").resolve(), "Unexpected data root")
    require(len(protocol["splits"]["C"]) == len(set(protocol["splits"]["C"])) == 48, "Invalid C paths")
    for relative in protocol["splits"]["C"]:
        path = (category / relative).resolve()
        require(path.parent == category / "train/good", "C path outside public normal directory")
        require(digest(path) == protocol["normal_image_sha256"][relative], "C image changed: " + relative)
    return protocol


def sampling(scores, protocol):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (48,) and np.isfinite(scores).all(), "Need 48 finite C scores")
    rng = np.random.default_rng(protocol["sampling_seed"])
    permutations = np.stack([rng.permutation(48) for _ in range(protocol["repetitions"])])
    thresholds = np.column_stack([
        np.quantile(scores[permutations[:, :m]], 0.95, axis=1, method="linear")
        for m in protocol["sample_sizes"]])
    canonical = float(np.quantile(scores, 0.95, method="linear"))
    require(np.array_equal(thresholds[:, -1], np.full(500, canonical)), "All48 thresholds should be identical")
    stats = []
    for column, m in enumerate(protocol["sample_sizes"]):
        values = thresholds[:, column]
        q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95], method="linear")
        stats.append({"m": m, "repetitions": 500, "q05": float(q05), "median": float(median),
                      "q95": float(q95), "q95_minus_q05": float(q95-q05),
                      "median_absolute_deviation_from_C48": float(np.median(np.abs(values-canonical)))})
    return permutations, thresholds, canonical, stats


def verify_c_cache(protocol):
    summary = read_json(SUMMARY_PATH)
    require(summary["protocol_sha256"] == digest(OUTPUT) and summary["scores_sha256"] == digest(SCORES_PATH),
            "Frozen results/protocol identity differs")
    with np.load(SCORES_PATH, allow_pickle=False) as arrays:
        require(arrays["C_paths"].tolist() == protocol["splits"]["C"], "Cached C order differs")
        permutations, thresholds, canonical, stats = sampling(arrays["C_scores"], protocol)
        require(np.array_equal(arrays["permutations"], permutations)
                and np.array_equal(arrays["thresholds"], thresholds)
                and float(arrays["canonical_C48_threshold"]) == canonical, "Cached sampling differs")
    require(summary["threshold_statistics"] == stats and summary["canonical_C48_threshold"] == canonical,
            "Cached statistics differ")
    return summary


def calibrate():
    protocol = c_protocol()
    if SUMMARY_PATH.exists() or SCORES_PATH.exists():
        require(SUMMARY_PATH.exists() and SCORES_PATH.exists(), "Partial result; investigate, do not overwrite")
        print(json.dumps({"status": "frozen C results verified; no model inference",
                          "summary": verify_c_cache(protocol)}, ensure_ascii=True, indent=2))
        return
    current = resources()
    import torch
    from evaluation import load_formal, infer, process_memory
    from demo import synchronize
    config = read_json(ROOT / "models/formal_patchcore/configuration.json")
    torch.cuda.reset_peak_memory_stats()
    synchronize("cuda")
    start = time.perf_counter()
    model = load_formal(config)  # Loads saved bank only; never calls fit.
    synchronize("cuda")
    load_seconds = time.perf_counter()-start
    scores, durations = [], []
    with torch.no_grad():
        for relative in protocol["splits"]["C"]:
            score, heatmap, elapsed = infer(model, Path(protocol["category_root"]) / relative)
            scores.append(score)
            durations.append(elapsed)
            del heatmap
    sample_start = time.perf_counter()
    permutations, thresholds, canonical, stats = sampling(scores, protocol)
    sampling_seconds = time.perf_counter()-sample_start
    # Freeze C arrays before any H evaluation; no H data is opened by this command.
    require(not SCORES_PATH.exists() and not SUMMARY_PATH.exists(), "Another result appeared; do not overwrite")
    with SCORES_PATH.open("xb") as stream:
        np.savez_compressed(stream, C_paths=np.asarray(protocol["splits"]["C"]),
                            C_scores=np.asarray(scores, dtype=np.float64),
                            permutations=permutations, thresholds=thresholds,
                            canonical_C48_threshold=np.asarray(canonical))
    summary = {"mode": "main_C48_H48", "stage": "C_calibration_frozen_before_H",
               "frozen_at": datetime.now().astimezone().isoformat(), "protocol_sha256": digest(OUTPUT),
               "scores_sha256": digest(SCORES_PATH), "calibration_source_sha256": digest(Path(__file__)),
               "C_count": 48, "H_evaluated": False, "C_score_min": float(min(scores)),
               "C_score_median": float(np.median(scores)), "C_score_max": float(max(scores)),
               "canonical_C48_threshold": canonical, "original_V_threshold": config["threshold"],
               "threshold_statistics": stats, "primary_comparison": [16, 48],
               "statistics_method": "linear empirical quantiles, middle90% finite-pool range; not population CI",
               "all48_caveat": "Zero variation by finite-pool construction, not population certainty",
               "future_H_statistics": {"denominator": 48, "fields": ["FP", "FPR_median", "FPR_q05", "FPR_q95"],
                                       "quantile_method": "linear", "status": "not evaluated"},
               "timings": {"model_load_seconds": load_seconds, "C_read_inference_seconds": float(sum(durations)),
                           "C_per_image_seconds": durations, "sampling_seconds": sampling_seconds},
               "timing_scope": "Load excludes startup/imports; C includes read/resize/normalization and synchronized author inference; sampling CPU only; file saves excluded",
               "resources_before_model_load": current, "resources_after_C": process_memory(),
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
               "scope": "C only; no H/E/P images, no fit, no abnormal training, no old threshold or results writes"}
    with SUMMARY_PATH.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verify_c_cache(protocol)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def h_metrics(scores, thresholds, canonical, original):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (48,) and np.isfinite(scores).all(), "Need 48 finite H scores")
    require(thresholds.shape == (500, 4), "Frozen thresholds must be 500x4")
    fp = np.sum(scores[None, None, :] > thresholds[:, :, None], axis=2, dtype=np.int64)
    stats = []
    for j, m in enumerate((8, 16, 32, 48)):
        q05, median, q95 = np.quantile(fp[:, j], [.05, .5, .95], method="linear")
        stats.append({"m": m, "repetitions": 500, "denominator_per_repetition": 48,
                      "FP_min": int(fp[:, j].min()), "FP_max": int(fp[:, j].max()),
                      "FP_q05": float(q05), "FP_median": float(median), "FP_q95": float(q95),
                      "FPR_q05": float(q05/48), "FPR_median": float(median/48), "FPR_q95": float(q95/48)})
    refs = []
    for name, threshold in (("canonical_C48", canonical), ("original_V49", original)):
        count = int(np.sum(scores > threshold))
        refs.append({"name": name, "threshold": float(threshold), "FP": count,
                     "normal_count": 48, "FPR": count/48, "TN": 48-count})
    delta = (fp[:, 1]-fp[:, 3])/48*100
    primary = {"comparison": "m16 minus m48, same repetition and same H48",
               "unit": "percentage points", "q05": float(np.quantile(delta, .05, method="linear")),
               "median": float(np.median(delta)), "q95": float(np.quantile(delta, .95, method="linear"))}
    return fp, stats, refs, primary


def verify_h_cache(protocol, c_summary, thresholds):
    result = read_json(H_RESULT_PATH)
    require(result["protocol_sha256"] == digest(OUTPUT)
            and result["C_scores_sha256"] == digest(SCORES_PATH)
            and result["C_summary_sha256"] == digest(SUMMARY_PATH)
            and result["H_scores_sha256"] == digest(H_SCORES_PATH), "H/C frozen identity differs")
    with np.load(H_SCORES_PATH, allow_pickle=False) as arrays:
        require(arrays["H_paths"].tolist() == protocol["splits"]["H"], "H order differs")
        fp, stats, refs, primary = h_metrics(arrays["H_scores"], thresholds,
                                          c_summary["canonical_C48_threshold"], c_summary["original_V_threshold"])
        require(np.array_equal(arrays["FP_by_repetition"], fp), "H FP arrays differ")
    require(result["H_statistics"] == stats and result["reference_results"] == refs
            and result["primary_paired_difference"] == primary, "H descriptive results differ")
    return result


def evaluate_h():
    protocol = c_protocol()  # Checks source metadata and C identities only; no E/P image reads.
    c_summary = verify_c_cache(protocol)
    require(digest(SCORES_PATH) == "4d89d357c5e97c42f37c6276a06c396226de2fb53cf243697037f3d2823dfe91"
            and digest(SUMMARY_PATH) == "c395f592e4d69be4ceea66bfd4fd9d4f0f88ad81792c7e857025888de836ceaa",
            "Day 2 frozen C artifacts changed")
    with np.load(SCORES_PATH, allow_pickle=False) as arrays:
        thresholds = arrays["thresholds"].copy()  # Use saved thresholds; no H-dependent calibration.
    category = Path(protocol["category_root"]).resolve()
    h_paths = protocol["splits"]["H"]
    require(len(h_paths) == len(set(h_paths)) == 48 and not set(h_paths) & set(protocol["splits"]["C"]),
            "H split size/overlap differs")
    for relative in h_paths:
        path = (category / relative).resolve()
        require(path.parent == category / "train/good" and digest(path) == protocol["normal_image_sha256"][relative],
                "H source identity differs: " + relative)
    if H_RESULT_PATH.exists() or H_SCORES_PATH.exists():
        require(H_RESULT_PATH.exists() and H_SCORES_PATH.exists(), "Partial H result; investigate, do not overwrite")
        print(json.dumps({"status": "saved H results verified; no model inference",
                          "result": verify_h_cache(protocol, c_summary, thresholds)}, ensure_ascii=True, indent=2))
        return
    current = resources()
    import torch
    from evaluation import load_formal, infer, process_memory
    from demo import synchronize
    config = read_json(ROOT / "models/formal_patchcore/configuration.json")
    torch.cuda.reset_peak_memory_stats()
    synchronize("cuda")
    start = time.perf_counter()
    model = load_formal(config)
    synchronize("cuda")
    load_seconds = time.perf_counter()-start
    scores, durations = [], []
    with torch.no_grad():
        for relative in h_paths:
            score, heatmap, elapsed = infer(model, category / relative)
            scores.append(score)
            durations.append(elapsed)
            del heatmap
    metric_start = time.perf_counter()
    fp, stats, refs, primary = h_metrics(scores, thresholds,
                                       c_summary["canonical_C48_threshold"], c_summary["original_V_threshold"])
    metric_seconds = time.perf_counter()-metric_start
    require(not H_SCORES_PATH.exists() and not H_RESULT_PATH.exists(), "H result appeared; do not overwrite")
    with H_SCORES_PATH.open("xb") as stream:
        np.savez_compressed(stream, H_paths=np.asarray(h_paths), H_scores=np.asarray(scores, dtype=np.float64),
                            FP_by_repetition=fp)
    result = {"stage": "H48_evaluated_with_frozen_C_thresholds", "H_evaluated": True,
              "completed_at": datetime.now().astimezone().isoformat(), "C_frozen_at": c_summary["frozen_at"],
              "protocol_sha256": digest(OUTPUT), "C_scores_sha256": digest(SCORES_PATH),
              "C_summary_sha256": digest(SUMMARY_PATH), "H_scores_sha256": digest(H_SCORES_PATH),
              "evaluation_source_sha256": digest(Path(__file__)), "H_count": 48,
              "H_score_min": float(min(scores)), "H_score_median": float(np.median(scores)), "H_score_max": float(max(scores)),
              "H_statistics": stats, "reference_results": refs, "primary_paired_difference": primary,
              "decision": "score > frozen threshold", "FPR_resolution_percentage_points": 100/48,
              "descriptive_quantile_method": "linear", "empirical_range": "finite C pool repeated sampling on the same H48; not population CI",
              "timings": {"model_load_seconds": load_seconds, "H_read_inference_seconds": float(sum(durations)),
                          "H_per_image_seconds": durations, "metrics_seconds": metric_seconds},
              "timing_scope": "Load excludes startup/imports; H includes read/resize/normalization and synchronized inference; metrics CPU; saves excluded",
              "resources_before_model_load": current, "resources_after_H": process_memory(),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "scope": "H scored once each; no fit, no E/P inference, no H-dependent selection, no AUROC/recall/pixelAP claims",
              "artifact_design": "Separate H scientific arrays/result preserve all Day2 C arrays and summary byte-for-byte"}
    with H_RESULT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verify_h_cache(protocol, c_summary, thresholds)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "calibrate", "evaluate"], help="Prepare, calibrate C or evaluate H")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "calibrate":
        calibrate()
    else:
        evaluate_h()
