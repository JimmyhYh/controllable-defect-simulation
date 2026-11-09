"""Independent CPU metrics for the paired pretrained segmentation experiment.

``prepare`` and ``verify`` are metadata-only.  They never open normalized GT
arrays or E probability arrays and never calculate a real metric.  ``run`` is
the only entry that authorizes those frozen arrays and the actual CPU metrics.
No model package, model forward, training, generation or recalibration is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

from . import real_scoring as r

t, c, a = r.t, r.c, r.a
PREP = c.RESULTS / "real_metrics_preparation_v1"
OUTPUT = c.RESULTS / "real_metrics_v1"
E_PROOF = c.BOOT / "real_scoring_verification_v1/all_six_v1"
E_PROOF_SHA = "3f0e7765f34f0df53281a67caa5f4fdfe85208ad94b7e0b56289807198b6744f"

OLD_RESULT_ROOT = c.ROOT / "results/fewshot_synthetic/ksdd2_v1"
LABEL_SOURCE = OLD_RESULT_ROOT / "real_metrics_v1"
LABEL_PROOF = OLD_RESULT_ROOT / "real_metrics_verifications_v1/all_six_v1"
LABEL_SOURCE_SHA = "b20bd02d02965919c22ab059e324fc7bdd47e4bb1438cd45ce3b409bbaddb538"
LABEL_PROOF_SHA = "3b5d2277a75134b0ab21f39780b02cd59ed6cc7a6129e2bd559acbfedc66639b"

PROBE_METRICS = OLD_RESULT_ROOT / "probe_retry_metrics_v1"
PROBE_PROOF = OLD_RESULT_ROOT / "probe_retry_metrics_verification_v1"
PROBE_METRICS_SHA = "eb082002d5a286de1815455326d068d9ad5e52bfe0a381c9c792d350b505c37b"
PROBE_PROOF_SHA = "70851d792ff4c6e9b4f82979e75ac4f26cfcc2c1947d8efb77d3b2b359f4ae09"

EXACT_SOURCE = c.ROOT / "fewshot_synthetic/metrics.py"
EXACT_SOURCE_SHA = "a65281962dcce26e68c99234cd11c28d47c1713184486e05244c486825168948"
EXACT_BINS = 0x3F800001
EXACT_CHUNK = 1048576

CODE = ("real_metrics.py", "README_real_metrics.md")
REPORT = "第6阶段独立CPU真实指标准备与运行边界.md"
PREP_FILES = {
    "transaction.json", "protocol.json", "sources.json", "GT_requests.json",
    "models.json", "environment.json", "scalar_verification.json", REPORT,
}
MODEL_ORDER = [f"{group}_{seed}" for group, seed in t.ORDER]
_GUARDED = False


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def implementation():
    return {name: t.sha(c.PACKAGE / name) for name in CODE}


def _inside(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


def install_guard(phase, destinations=()):
    """Process-local guard.  ``prepare`` forbids every array/image payload."""
    global _GUARDED
    c.require(not _GUARDED, "One explicit metric operation per process")
    c.require(phase in {"prepare", "run"}, "Unknown metric guard phase")
    destinations = tuple(c.no_redirect(path) for path in destinations)
    ancestors = {parent for root in destinations for parent in root.parents if parent != Path(root.anchor)}
    model_packages = {"torch", "torchvision", "segmentation_models_pytorch", "timm", "safetensors"}
    prepare_blocked = c.BLOCKED | {"scipy", "sklearn", "cv2"}
    blocked = prepare_blocked if phase == "prepare" else model_packages
    c.require(not any(name.split(".")[0] in blocked or a.model_module(name) for name in sys.modules),
              "Forbidden package already imported in CPU metric process")

    probability_roots = tuple(r.paths(group, seed)[0] / "probabilities" for group, seed in t.ORDER)
    label_root = LABEL_SOURCE / "labels"

    def audit(event, args):
        if event == "import" and (args[0].split(".")[0] in blocked or a.model_module(args[0])):
            raise RuntimeError("Forbidden model/data package import in CPU metric operation")
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("Network and pickle execution are forbidden")
        if event in {"os.link", "os.symlink", "os.remove", "os.rename", "os.rmdir"}:
            raise RuntimeError("Link, delete, move and implicit cleanup are forbidden")
        if event == "os.mkdir":
            path = Path(args[0]).absolute()
            c.require(path in ancestors or _inside(path, destinations),
                      "Directory creation outside independent metric destinations")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return
            path = Path(name).absolute()
            mode, flags = args[1] or "", args[2] or 0
            writing = any(char in str(mode) for char in "wax+") or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                c.require(_inside(path, destinations), "Write outside independent metric destination")
                return
            if path.is_relative_to(c.ROOT / "data"):
                c.require(path == a.SPLIT, "Raw RGB/GT byte read is forbidden")
                return
            suffix = path.suffix.lower()
            if suffix == ".npy":
                allowed = phase == "run" and (
                    _inside(path, probability_roots) or path.is_relative_to(label_root)
                    or _inside(path, destinations))
                c.require(allowed, "GT/probability array payload is forbidden in metric preparation")
                return
            if suffix == ".uint32":
                c.require(phase == "run" and _inside(path, destinations),
                          "Exact AP table is outside the authorized metric result")
                return
            if suffix in c.OPAQUE_SUFFIXES:
                c.require(path in t._HASH_SCOPE or path in c._READ_SCOPES
                          or phase == "run" and _inside(path, destinations),
                          "Unbound image/tensor/archive payload")

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    _GUARDED = True


def resource_snapshot(min_ram=2, min_d=2, min_e=80):
    snapshot = c.resource_snapshot()
    c.require(snapshot["available_RAM_GiB"] >= min_ram, f"Need at least {min_ram}GiB available RAM")
    c.require(snapshot["D_free_GiB"] >= min_d, f"Need at least {min_d}GiB free on D")
    c.require(snapshot["E_free_GiB"] >= min_e, f"Need at least {min_e}GiB free on E")
    return snapshot


def exact_core_reference():
    c.require(t.sha(EXACT_SOURCE) == EXACT_SOURCE_SHA, "Frozen exact AP source changed")
    tree = ast.parse(EXACT_SOURCE.read_text(encoding="utf-8"), filename=str(EXACT_SOURCE))
    constants = {}
    signature = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"BINS", "CHUNK"}:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "exact_ap":
            signature = [arg.arg for arg in node.args.args]
    c.require(constants == {"BINS": EXACT_BINS, "CHUNK": EXACT_CHUNK}
              and signature == ["block_factory", "destination", "identity", "expected_n", "expected_p"],
              "Frozen exact AP constants/signature changed")

    probe_complete = t.load(PROBE_METRICS / "complete.json")
    c.require(t.sha(PROBE_METRICS / "complete.json") == PROBE_METRICS_SHA,
              "Frozen probe metric completion changed")
    proof_complete = t.load(PROBE_PROOF / "complete.json")
    t.verify_seal(PROBE_PROOF, set(proof_complete["files"]))
    c.require(t.sha(PROBE_PROOF / "complete.json") == PROBE_PROOF_SHA,
              "Frozen probe metric proof changed")
    proof = t.load(PROBE_PROOF / "verification.json")
    summary = t.load(PROBE_METRICS / "metrics.json")
    exact = t.load(PROBE_METRICS / "exact/metrics.json")
    c.require(proof["source_complete_sha256"] == PROBE_METRICS_SHA
              and probe_complete["files"]["metrics.json"] == t.sha(PROBE_METRICS / "metrics.json")
              and probe_complete["files"]["exact/metrics.json"] == t.sha(PROBE_METRICS / "exact/metrics.json")
              and summary["gate_passed"] is True and summary["AP"] == exact["AP"] == proof["AP"]
              and summary["absolute_difference"] <= 1e-10
              and summary["independent_distinct_scores"] == exact["distinct_scores"] == proof["distinct_scores"]
              and summary["independent_tied_score_values"] == exact["tied_score_values"] == proof["tied_score_values"],
              "Frozen sklearn/ties crosscheck differs")
    return {
        "source_path": str(EXACT_SOURCE), "source_sha256": EXACT_SOURCE_SHA,
        "function": "fewshot_synthetic.metrics.exact_ap",
        "signature": signature, "BINS": EXACT_BINS, "CHUNK": EXACT_CHUNK,
        "probe_metric_complete_sha256": PROBE_METRICS_SHA,
        "probe_verification_complete_sha256": PROBE_PROOF_SHA,
        "probe_AP_sklearn_absolute_difference": summary["absolute_difference"],
        "probe_distinct_scores": exact["distinct_scores"],
        "probe_tied_score_values": exact["tied_score_values"],
        "probe_arrays_reopened_now": 0,
    }


def label_reference(req, metric_sources):
    c.require(t.sha(LABEL_SOURCE / "complete.json") == LABEL_SOURCE_SHA,
              "Frozen normalized-label source completion changed")
    proof_complete = t.load(LABEL_PROOF / "complete.json")
    t.verify_seal(LABEL_PROOF, set(proof_complete["files"]))
    c.require(t.sha(LABEL_PROOF / "complete.json") == LABEL_PROOF_SHA,
              "Frozen normalized-label proof changed")
    proof = t.load(LABEL_PROOF / "verification.json")
    complete = t.load(LABEL_SOURCE / "complete.json")
    summary = t.load(LABEL_SOURCE / "labels/summary.json")
    c.require(proof["verified"] is True and proof["source_metric_complete_sha256"] == LABEL_SOURCE_SHA
              and proof["source_GT_summary_sha256"] == t.sha(LABEL_SOURCE / "labels/summary.json")
              == complete["files"]["labels/summary.json"],
              "Frozen normalized-label evidence differs")
    c.verify_closed_tree(LABEL_SOURCE, set(complete["files"]) | {"complete.json"})
    c.require(len(summary["rows"]) == len(req) == len(metric_sources) == 1004,
              "Normalized-label metadata count differs")

    rows = []
    for request, source, old in zip(req, metric_sources, summary["rows"]):
        old_request = old["request"]
        c.require(source == {"index": request["index"], "id": request["id"],
                             "mask": old_request["mask"], "mask_sha256": old_request["mask_sha256"],
                             "native_shape": request["native_shape"]}
                  and old_request["ordinal"] == request["index"]
                  and old_request["id"] == request["id"]
                  and [old_request["height"], old_request["width"]] == request["native_shape"],
                  "Normalized-label order/source/native shape differs")
        name = f"labels/{request['id']}.npy"
        path = LABEL_SOURCE / name
        c.require(complete["files"][name] == old["normalized_sha256"]
                  and path.is_file() and path.stat().st_size > 0,
                  "Normalized-label file identity metadata differs")
        rows.append({
            "index": request["index"], "id": request["id"],
            "native_shape": request["native_shape"],
            "mask": source["mask"], "mask_sha256": source["mask_sha256"],
            "normalized_path": str(path), "normalized_sha256": old["normalized_sha256"],
            "normalized_bytes": path.stat().st_size,
        })
    c.require(sum(row["normalized_bytes"] for row in rows) > 146411528,
              "Normalized-label NPY metadata size is implausible")
    return rows, {
        "source_complete_sha256": LABEL_SOURCE_SHA,
        "source_verification_complete_sha256": LABEL_PROOF_SHA,
        "source_summary_sha256": t.sha(LABEL_SOURCE / "labels/summary.json"),
        "normalized_arrays": 1004,
        "array_payloads_opened_now": 0,
        "historical_counts_only_until_run": {
            "anomaly_images": summary["anomaly_images"],
            "normal_images": summary["normal_images"],
            "positive_pixels": summary["positive_pixels"],
        },
    }


def _metadata_names(req):
    probabilities = {r.probability_path(row) for row in req}
    return r.run_files(req) - probabilities, probabilities


def E_references(p, req, models):
    proof_complete = t.load(E_PROOF / "complete.json")
    t.verify_seal(E_PROOF, set(proof_complete["files"]), proof_complete["identity"])
    c.require(t.sha(E_PROOF / "complete.json") == E_PROOF_SHA,
              "Six-model E scoring proof changed")
    proof = t.load(E_PROOF / "verification.json")
    c.require(proof["real_scoring_verified"] is True and proof["model_order"] == MODEL_ORDER
              and proof["totals"]["models"] == 6
              and proof["totals"]["model_image_scores"] == 6024
              and proof["totals"]["window_forwards"] == 25620
              and proof["totals"]["GT_reads_or_metrics_during_verification"] == 0,
              "Six-model E scoring proof is incomplete")
    metadata_names, probability_names = _metadata_names(req)
    references = []
    for model, row in zip(models, proof["entries"]):
        group, seed = model["group"], model["seed"]
        label = f"{group}_{seed}"
        c.require(row["model"] == label, "E proof/model order differs")
        root, runtime = r.paths(group, seed)
        bound = r.binding(p, model)
        complete = t.load(root / "complete.json")
        c.require(t.sha(root / "complete.json") == row["result_complete_sha256"]
                  and complete["identity"] == bound
                  and set(complete["files"]) == r.run_files(req),
                  "E completion identity/namespace differs")
        c.verify_closed_tree(root, r.run_files(req) | {"complete.json"})
        for name in metadata_names:
            c.require(t.sha(root / name) == complete["files"][name],
                      "E scoring metadata changed: " + label + "/" + name)
        c.require(all(len(complete["files"][name]) == 64 for name in probability_names),
                  "E probability manifest digest is invalid")
        a.verify_runtime(root, runtime, bound)
        scores = t.load(root / "scores.json")
        transaction = t.load(root / "transaction.json")
        c.require(scores["binding"] == transaction["binding"] == bound
                  and scores["images_scored"] == scores["raw_E_image_decodes"] == len(scores["items"]) == 1004
                  and scores["window_forwards"] == transaction["intended_windows"] == 4270
                  and scores["native_pixels"] == 146411528 and scores["constructor_forwards"] == 0
                  and scores["state_before"] == scores["state_after"] == model["expected_state_hash"]
                  and scores["buffers_before"] == scores["buffers_after"] == model["expected_buffers_hash"]
                  and scores["GT_read_by_predictor"] is False and scores["metrics_computed"] is False
                  and scores["training_or_parameter_updates"] == 0
                  and scores["original_threshold"] == model["calibration"]["threshold"],
                  "E saved scoring metadata differs")
        expected_journal = []
        for request in req:
            expected_journal.append({"event": "decode_start", "index": request["index"], "id": request["id"]})
            expected_journal.extend({"event": "window_start", "index": request["index"],
                                     "id": request["id"], "window_index": index, "y": y}
                                    for index, y in enumerate(request["ys"]))
        actual_journal = [json.loads(line) for line in (root / "window_journal.jsonl").read_text(encoding="utf-8").splitlines()]
        c.require(actual_journal == expected_journal, "E window intention journal differs")
        threshold_path = Path(model["calibration"]["threshold_path"])
        threshold = t.load(threshold_path)
        c.require(t.sha(threshold_path) == model["calibration"]["threshold_sha256"]
                  and threshold["threshold"] == scores["original_threshold"] == row["original_threshold"],
                  "Original Q threshold identity differs")
        c.require(t.sha(model["final_weight_path"]) == model["final_weight_sha256"]
                  == row["final_weight_sha256"], "Final model weight stream hash differs")
        c.require(row["probability_files"] == 1004 and row["windows"] == 4270
                  and row["native_pixels"] == 146411528 and row["state_unchanged"] is True
                  and row["BN_buffers_unchanged"] is True and row["metrics_computed"] is False,
                  "Independent E proof fields differ")
        references.append({
            "group": group, "seed": seed, "label": label,
            "result_root": str(root), "runtime_root": str(runtime),
            "result_complete_sha256": row["result_complete_sha256"],
            "scores_sha256": row["scores_sha256"],
            "probability_manifest_digest": stable({name: complete["files"][name] for name in sorted(probability_names)}),
            "probability_files": 1004,
            "final_weight_path": model["final_weight_path"],
            "final_weight_sha256": model["final_weight_sha256"],
            "threshold_path": str(threshold_path),
            "threshold_sha256": model["calibration"]["threshold_sha256"],
            "threshold": threshold["threshold"],
            "expected_state_hash": model["expected_state_hash"],
            "expected_buffers_hash": model["expected_buffers_hash"],
        })
    return references, {
        "preparation_complete_sha256": t.sha(r.PREP / "complete.json"),
        "all_six_scoring_verification_complete_sha256": E_PROOF_SHA,
        "model_order": MODEL_ORDER,
        "probability_payloads_opened_now": 0,
    }


def source_context():
    p, req, models = r.verify_preparation()
    metric_sources = t.load(r.PREP / "metric_sources.json")
    references, scoring_source = E_references(p, req, models)
    gt_rows, gt_source = label_reference(req, metric_sources)
    exact_source = exact_core_reference()
    c.require([row["label"] for row in references] == MODEL_ORDER
              and len(gt_rows) == len(req) == 1004, "Metric source order differs")
    sources = {
        "E_scoring": scoring_source,
        "normalized_GT": gt_source,
        "exact_AP": exact_source,
        "scientific_plan": c.plan_contract(),
    }
    return p, req, metric_sources, references, gt_rows, sources


def protocol(context):
    _, req, _, refs, gt_rows, sources = context
    return {
        "schema": "paired-pretrained-E1004-exact-CPU-metrics-v1",
        "implementation": implementation(),
        "scientific_plan": c.plan_contract(),
        "model_order": MODEL_ORDER,
        "images_per_model": 1004,
        "unique_E_images": 1004,
        "native_pixels_per_model": 146411528,
        "model_image_scores": 6024,
        "GT": {
            "source": "frozen normalized uint8 labels from prior independently verified real_metrics_v1",
            "requests_digest": stable(gt_rows),
            "native_shape": "exact saved HxW; no resize, dilation or soft labels",
            "run_validation": "hash before np.load; uint8; values subset {0,1}; exact native shape",
            "original_mask_path_and_existing_hash_bound": True,
            "raw_original_GT_read": False,
            "actual_counts": "unknown until explicit run opens all1004 normalized arrays",
            "historical_values_are_only_run_checks": sources["normalized_GT"]["historical_counts_only_until_run"],
        },
        "pixel_AP": {
            "function": "fewshot_synthetic.metrics.exact_ap",
            "source_sha256": EXACT_SOURCE_SHA,
            "scope": "global native pixels; every original pixel once; no subsampling or per-image AP averaging",
            "chunk_max": EXACT_CHUNK,
            "histogram_bins": EXACT_BINS,
            "two_uint32_tables_bytes_per_model": EXACT_BINS * 8,
            "six_models_table_bytes": EXACT_BINS * 8 * 6,
            "ties": "identical float32 values grouped exactly; negative zero normalized by frozen routine",
            "crosscheck": sources["exact_AP"],
        },
        "image_metrics": {
            "AUROC": "all1004 native maxima; single actual class -> NA",
            "threshold": "each model original Q200 tau; score strictly greater; no recalibration",
            "counts": "TP/FN on actual anomalies; FP/TN on actual normals; empty denominator -> NA rate",
        },
        "paired_summary": {
            "pairs": [["A_2026102301", "B_2026102301"],
                      ["A_2026102302", "B_2026102302"],
                      ["A_2026102303", "B_2026102303"]],
            "primary_delta": "100*(pixel_AP_B-pixel_AP_A) percentage points",
            "primary_mean": "arithmetic mean of three paired AP deltas",
            "primary_sample_SD": "statistics.stdev, ddof=1",
            "supplementary_deltas": "AUROC/recall/normal false-positive-rate percentage points and TP/FN/FP/TN counts",
            "no_population_significance": True,
        },
        "selected_examples": {
            "reference_pair": ["A_2026102301", "B_2026102301"],
            "rules": ["first A01 TN", "first A01 FP", "first A01 TP", "first A01 FN",
                      "first anomaly A_FN_to_B_TP", "first anomaly A_TP_to_B_FN",
                      "first normal A_FP_to_B_TN", "first normal A_TN_to_B_FP"],
            "deduplicate_and_keep_all_reasons": True,
            "empty_group": "NA; no replacement",
            "RGB_opened_during_selection": False,
        },
        "transaction": {
            "partial_or_failure": "preserve and stop; no resume/reaccumulate/substitute",
            "complete_same_identity": "read-only verification; no metric recomputation",
            "successful_intersection": False,
        },
        "resources": {
            "minimum_available_RAM_GiB": 3,
            "minimum_D_free_GiB": 80,
            "minimum_E_free_GiB": 80,
            "reason": "frozen exact_ap retains two uint32 tables per model; old helper also enforces D80GiB",
        },
        "preparation": {
            "GT_array_reads": 0, "probability_array_reads": 0, "actual_metrics": False,
            "Torch_or_model_operations": 0,
        },
        "meaning": "E historically seen; exploratory paired composition evidence only; no unseen or screen claim",
        "request_count": len(req), "reference_count": len(refs),
    }


def safe_rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def judgement(values):
    mean = statistics.fmean(values)
    if all(value > 0 for value in values):
        return "three paired initialization deltas are positive in this fixed setting"
    if mean > 0:
        return "positive mean but direction is inconsistent across initialization pairs"
    return "no positive mean gain in this fixed setting"


def scalar_verification():
    c.require(not (0.5 > 0.5) and 0.5000001 > 0.5, "Strict greater-than rule differs")
    dummy = [1.0, 2.0, 3.0]
    c.require(statistics.fmean(dummy) == 2.0 and statistics.stdev(dummy) == 1.0,
              "Paired mean/sample SD convention differs")
    c.require(abs(100 * (0.73 - 0.70) - 3.0) < 1e-12
              and safe_rate(1, 0) is None and safe_rate(1, 2) == 0.5,
              "Percentage-point or NA convention differs")
    c.require(judgement([1.0, 2.0, 3.0]).startswith("three")
              and judgement([1.0, -0.5, 1.0]).startswith("positive mean")
              and judgement([-1.0, 0.0, 0.0]).startswith("no positive"),
              "Frozen conclusion rule differs")
    return {
        "fictitious_scalars_only": True,
        "strict_greater_tie": True,
        "paired_mean_sample_SD_ddof1": True,
        "percentage_points": True,
        "empty_denominator_NA": True,
        "exact_BINS": EXACT_BINS,
        "exact_CHUNK": EXACT_CHUNK,
        "GT_array_reads": 0,
        "probability_array_reads": 0,
        "actual_GT_counts_or_positive_pixels": "not computed",
        "actual_AP_AUROC_counts_or_pair_conclusions": False,
        "Torch_imported": "torch" in sys.modules,
        "model_operations": 0,
    }


def environment(snapshot):
    return {
        "python": sys.version,
        "executable": sys.executable,
        "resource_snapshot": snapshot,
        "stdlib_only_preparation": True,
        "exact_AP_source_sha256": EXACT_SOURCE_SHA,
    }


def report(snapshot):
    fence = "`" * 3
    return f"""# 第6阶段独立CPU真实指标准备与运行边界

本次只完成指标模块、协议、来源hash、GT路径／既有hash、六模型顺序和虚构标量准备。六组E1004评分完整缓存、最终权重及各自原Q200阈值均以已封存完成身份绑定；没有打开6024份概率数组。规范GT复用旧实验中已独立核验的1004份uint8二值标签缓存，本次只读取其JSON身份、路径、已有hash与文件大小，不打开标签数组，也不重新读取原始GT。

实际计算入口将逐份核规范标签hash、uint8、{{0,1}}、原生尺寸和实际异常／正常／阳性像素分母；历史110／894与486461只作运行时一致性检查，当前不宣称重新核实。每模型按原生146,411,528像素全局计算精确像素AP，不抽像素、不平均逐图AP、不改用近似分箱；复用冻结`fewshot_synthetic.metrics.exact_ap`及既有sklearn／实际ties交叉证据。整图AUROC、各自原tau严格大于的TP／FN／FP／TN、召回和正常误报率同时保存，空分母或单类别写NA。

三对全部报告B−A：像素AP百分点差为主终点并汇总均值与样本SD（ddof=1），同时报告AUROC／召回／误报率百分点差及错误计数差。固定样例规则只使用第一对A01/B01，按E原序取A01首个TN／FP／TP／FN及四种A→B翻转，去重保留全部原因，空组NA、不替补；冻结ID时不打开RGB。

实际输出为`{OUTPUT.as_posix()}`。任何部分／失败事务均原样保留并停止，不自动恢复、重累计、取六模型成功交集或替换指标。完整同身份结果只读核验，不重复AP。E曾用于历史实验，结果只作本固定单类别／固定R5／固定划分／三初始化的探索性训练组成证据，不称新未见验证、统计显著、合成机理单因素因果或屏幕收益。

本次资源快照：RAM {snapshot['available_RAM_GiB']:.3f}GiB，D盘 {snapshot['D_free_GiB']:.3f}GiB，E盘 {snapshot['E_free_GiB']:.3f}GiB。未来实际入口要求RAM≥3GiB、D≥80GiB、E≥80GiB；六模型精确表理论字节数为{EXACT_BINS * 8 * 6}，运行时间尚未实测。所有训练仍由用户手动启动，本CPU指标不训练模型。

只读准备核验：

{fence}powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics verify
{fence}

实际CPU指标入口（已实现，本次未运行）：

{fence}powershell
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics run
{fence}
"""


def _prepared_context():
    context = source_context()
    complete = t.verify_seal(PREP, PREP_FILES,
                             {"implementation": implementation(), "plan_contract": c.plan_contract(),
                              "E_scoring_proof_sha256": E_PROOF_SHA})
    c.require(t.load(PREP / "protocol.json") == protocol(context)
              and t.load(PREP / "sources.json") == context[5]
              and t.load(PREP / "GT_requests.json") == context[4]
              and t.load(PREP / "models.json") == context[3]
              and t.load(PREP / "scalar_verification.json") == scalar_verification()
              and complete["identity"]["E_scoring_proof_sha256"] == E_PROOF_SHA,
              "Frozen real metric preparation differs")
    return context


def prepare():
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(),
              "Use the CPU .venv interpreter")
    snapshot = resource_snapshot(2, 80, 80)
    install_guard("prepare", destinations=(PREP,))
    if PREP.exists():
        _prepared_context()
        return {"metric_preparation_verified": True, "reused": True, **scalar_verification()}
    c.require(not OUTPUT.exists(), "Unknown existing metric output; preserve and diagnose")
    context = source_context()
    for name in CODE:
        if name.endswith(".py"):
            ast.parse((c.PACKAGE / name).read_text(encoding="utf-8"), filename=name)
    PREP.mkdir(parents=True, exist_ok=False)
    values = {
        "transaction.json": {
            "scope": "CPU metadata/hash/fictitious scalars only",
            "resources": snapshot,
            "GT_array_reads": 0,
            "probability_array_reads": 0,
            "actual_metrics": False,
            "model_operations": 0,
        },
        "protocol.json": protocol(context),
        "sources.json": context[5],
        "GT_requests.json": context[4],
        "models.json": context[3],
        "environment.json": environment(snapshot),
        "scalar_verification.json": scalar_verification(),
        REPORT: report(snapshot),
    }
    for name, value in values.items():
        t.save(PREP / name, value)
    t.seal(PREP, PREP_FILES,
           {"implementation": implementation(), "plan_contract": c.plan_contract(),
            "E_scoring_proof_sha256": E_PROOF_SHA})
    _prepared_context()
    return {
        "metric_preparation_ready": True,
        "models": 6,
        "unique_E_images": 1004,
        "native_pixels_per_model": 146411528,
        "GT_array_reads": 0,
        "probability_array_reads": 0,
        "actual_metrics": False,
        "model_operations": 0,
        "resources": snapshot,
    }


def verify_preparation():
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(),
              "Use the CPU .venv interpreter")
    install_guard("prepare")
    _prepared_context()
    return {"metric_preparation_verified": True, **scalar_verification()}


def binding():
    return {
        "preparation_complete_sha256": t.sha(PREP / "complete.json"),
        "protocol_sha256": t.sha(PREP / "protocol.json"),
        "GT_requests_sha256": t.sha(PREP / "GT_requests.json"),
        "models_sha256": t.sha(PREP / "models.json"),
        "implementation": implementation(),
        "E_scoring_proof_sha256": E_PROOF_SHA,
    }


def _read_bound_array(path, expected_sha256):
    import io
    import numpy as np
    payload = Path(path).read_bytes()
    c.require(hashlib.sha256(payload).hexdigest() == expected_sha256,
              "Metric array input bytes changed: " + str(path))
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _append_journal(event):
    path = OUTPUT / "journal.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_labels(gt_rows, token):
    import numpy as np
    folder = OUTPUT / "labels"
    folder.mkdir()
    rows = []
    for request in gt_rows:
        resource_snapshot(3, 80, 80)
        _append_journal({"token": token, "phase": "before_normalized_GT_read", "index": request["index"], "id": request["id"]})
        target = _read_bound_array(request["normalized_path"], request["normalized_sha256"])
        c.require(target.dtype == np.uint8 and list(target.shape) == request["native_shape"]
                  and np.all((target == 0) | (target == 1)),
                  "Normalized GT must be native uint8 binary {0,1}")
        name = f"labels/{request['index']:04d}_{request['id']}.npy"
        path = OUTPUT / name
        with path.open("xb") as stream:
            np.save(stream, target, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        positive = int(target.sum())
        rows.append({"request": request, "cache": name, "cache_sha256": t.sha(path),
                     "positive_pixels": positive, "anomaly": bool(positive)})
    positives = sum(row["positive_pixels"] for row in rows)
    anomalies = sum(row["anomaly"] for row in rows)
    summary = {"images": 1004, "native_pixels": 146411528,
               "positive_pixels": positives, "anomaly_images": anomalies,
               "normal_images": 1004 - anomalies, "rows": rows}
    c.require((summary["anomaly_images"], summary["normal_images"], summary["positive_pixels"])
              == (110, 894, 486461),
              "Actual normalized GT denominators differ from frozen historical check; stop without subset")
    t.save(OUTPUT / "labels/summary.json", summary)
    return summary


def _verify_labels(gt_rows):
    import numpy as np
    summary = t.load(OUTPUT / "labels/summary.json")
    c.require(summary["images"] == 1004 and summary["native_pixels"] == 146411528
              and (summary["anomaly_images"], summary["normal_images"], summary["positive_pixels"])
              == (110, 894, 486461) and len(summary["rows"]) == 1004,
              "Actual normalized GT summary differs")
    for request, row in zip(gt_rows, summary["rows"]):
        c.require(row["request"] == request and t.sha(OUTPUT / row["cache"]) == row["cache_sha256"],
                  "Normalized GT cache identity/order differs")
        target = np.load(OUTPUT / row["cache"], allow_pickle=False)
        c.require(target.dtype == np.uint8 and list(target.shape) == request["native_shape"]
                  and np.all((target == 0) | (target == 1))
                  and int(target.sum()) == row["positive_pixels"],
                  "Normalized GT cache payload differs")
    return summary


def _exact_identity(ref, labels):
    return {
        "binding": binding(),
        "model": ref["label"],
        "E_complete_sha256": ref["result_complete_sha256"],
        "normalized_GT_summary_sha256": t.sha(OUTPUT / "labels/summary.json"),
        "positive_pixels": labels["positive_pixels"],
        "exact_source_sha256": EXACT_SOURCE_SHA,
        "threshold_sha256": ref["threshold_sha256"],
        "threshold": ref["threshold"],
    }


def _blocks(ref, p, req, models, metric_sources, labels, token):
    import numpy as np
    from . import real_cache
    group, seed = ref["group"], ref["seed"]
    for source, probability, score in real_cache.iter_metric_inputs(group, seed, p, req, models, metric_sources):
        row = labels["rows"][source["index"]]
        request = row["request"]
        c.require(request["id"] == source["id"] and request["native_shape"] == source["native_shape"],
                  "Probability/GT request alignment differs")
        target = np.load(OUTPUT / row["cache"], allow_pickle=False)
        c.require(probability.dtype == np.float32 and target.dtype == np.uint8
                  and list(probability.shape) == list(target.shape) == request["native_shape"]
                  and int(target.sum()) == row["positive_pixels"]
                  and score == float(probability.max()),
                  "Native probability/GT pair differs")
        _append_journal({"token": token, "phase": "AP_input", "model": ref["label"],
                         "index": request["index"], "id": request["id"]})
        scores, truth = probability.reshape(-1), target.reshape(-1)
        for start in range(0, scores.size, EXACT_CHUNK):
            resource_snapshot(3, 80, 80)
            yield scores[start:start + EXACT_CHUNK], truth[start:start + EXACT_CHUNK]


def _image_metrics(ref, labels):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    root = Path(ref["result_root"])
    complete = t.load(root / "complete.json")
    c.require(t.sha(root / "scores.json") == complete["files"]["scores.json"] == ref["scores_sha256"],
              "E image score source differs")
    saved = t.load(root / "scores.json")["items"]
    c.require(len(saved) == len(labels["rows"]) == 1004, "E image score count differs")
    truth = np.asarray([row["anomaly"] for row in labels["rows"]], dtype=bool)
    scores = np.asarray([row["score"] for row in saved], dtype=np.float64)
    c.require(np.isfinite(scores).all(), "Non-finite E image score")
    predicted = scores > ref["threshold"]
    anomaly_n, normal_n = int(truth.sum()), int((~truth).sum())
    tp, fn = int(np.sum(predicted & truth)), int(np.sum(~predicted & truth))
    fp, tn = int(np.sum(predicted & ~truth)), int(np.sum(~predicted & ~truth))
    metric = {
        "AUROC": None if anomaly_n == 0 or normal_n == 0 else float(roc_auc_score(truth, scores)),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "anomaly_denominator": anomaly_n, "normal_denominator": normal_n,
        "recall": safe_rate(tp, anomaly_n),
        "normal_false_positive_rate": safe_rate(fp, normal_n),
        "threshold": ref["threshold"], "threshold_sha256": ref["threshold_sha256"],
        "comparison": "strictly greater",
        "FP_ids": [row["request"]["id"] for row, flag in zip(labels["rows"], predicted & ~truth) if flag],
        "FN_ids": [row["request"]["id"] for row, flag in zip(labels["rows"], ~predicted & truth) if flag],
    }
    return metric, predicted


def _paired(models):
    pairs = []
    for seed in c.SEEDS:
        left = next(row for row in models if row["model"] == f"A_{seed}")
        right = next(row for row in models if row["model"] == f"B_{seed}")
        a_metric, b_metric = left["image_metrics"], right["image_metrics"]
        pairs.append({
            "seed": seed,
            "AP_A": left["pixel_AP"], "AP_B": right["pixel_AP"],
            "delta_AP_percentage_points": 100 * (right["pixel_AP"] - left["pixel_AP"]),
            "AUROC_A": a_metric["AUROC"], "AUROC_B": b_metric["AUROC"],
            "delta_AUROC_percentage_points": None if a_metric["AUROC"] is None or b_metric["AUROC"] is None
                else 100 * (b_metric["AUROC"] - a_metric["AUROC"]),
            "delta_recall_percentage_points": None if a_metric["recall"] is None or b_metric["recall"] is None
                else 100 * (b_metric["recall"] - a_metric["recall"]),
            "delta_normal_false_positive_rate_percentage_points": None
                if a_metric["normal_false_positive_rate"] is None or b_metric["normal_false_positive_rate"] is None
                else 100 * (b_metric["normal_false_positive_rate"] - a_metric["normal_false_positive_rate"]),
            "delta_TP": b_metric["TP"] - a_metric["TP"],
            "delta_FN": b_metric["FN"] - a_metric["FN"],
            "delta_FP": b_metric["FP"] - a_metric["FP"],
            "delta_TN": b_metric["TN"] - a_metric["TN"],
        })
    values = [row["delta_AP_percentage_points"] for row in pairs]
    return {
        "pairs": pairs,
        "mean_delta_AP_percentage_points": statistics.fmean(values),
        "sample_SD_delta_AP_percentage_points": statistics.stdev(values),
        "ddof": 1,
        "judgement": judgement(values),
        "limits": "same fixed1004 data and same synthetic pool; n=3 paired initializations, not independent datasets",
    }


def _selected_examples(labels, decisions):
    a = decisions["A_2026102301"]
    b = decisions["B_2026102301"]
    rules = [
        ("A01_first_TN", lambda y, pa, pb: not y and not pa),
        ("A01_first_FP", lambda y, pa, pb: not y and pa),
        ("A01_first_TP", lambda y, pa, pb: y and pa),
        ("A01_first_FN", lambda y, pa, pb: y and not pa),
        ("first_anomaly_A_FN_to_B_TP", lambda y, pa, pb: y and not pa and pb),
        ("first_anomaly_A_TP_to_B_FN", lambda y, pa, pb: y and pa and not pb),
        ("first_normal_A_FP_to_B_TN", lambda y, pa, pb: not y and pa and not pb),
        ("first_normal_A_TN_to_B_FP", lambda y, pa, pb: not y and not pa and pb),
    ]
    selected, groups = {}, []
    for reason, predicate in rules:
        chosen = None
        for index, row in enumerate(labels["rows"]):
            if predicate(row["anomaly"], bool(a[index]), bool(b[index])):
                chosen = row["request"]["id"]
                selected.setdefault(chosen, []).append(reason)
                break
        groups.append({"reason": reason, "id": chosen, "status": "NA" if chosen is None else "selected"})
    return {
        "reference_pair": ["A_2026102301", "B_2026102301"],
        "groups": groups,
        "examples": [{"id": id_, "reasons": reasons} for id_, reasons in selected.items()],
        "deduplicated": True,
        "replacement_for_empty_group": False,
        "RGB_opened": False,
    }


def _expected_result_files(gt_rows, refs):
    names = {"transaction.json", "journal.jsonl", "labels/summary.json", "metrics.json", "真实指标结果.md"}
    names |= {f"labels/{row['index']:04d}_{row['id']}.npy" for row in gt_rows}
    for ref in refs:
        names |= {f"{ref['label']}/exact/{name}" for name in
                  ("transaction.json", "counts.uint32", "positive.uint32", "metrics.json", "complete.json")}
    return names


def _result_report(result):
    rows = ["# 六模型E1004真实指标结果", "", "本文件由冻结缓存计算生成；完整数值以metrics.json为准。", "",
            "| 模型 | 像素AP | 整图AUROC | TP/FN | FP/TN |", "| --- | ---: | ---: | ---: | ---: |"]
    for model in result["models"]:
        metric = model["image_metrics"]
        auroc = "NA" if metric["AUROC"] is None else f"{metric['AUROC']:.10f}"
        rows.append(f"| {model['model']} | {model['pixel_AP']:.10f} | {auroc} | {metric['TP']}/{metric['FN']} | {metric['FP']}/{metric['TN']} |")
    rows += ["", "三对B−A及均值／样本SD见metrics.json。E历史已见，结果只作探索性配对组成证据。", ""]
    return "\n".join(rows)


def _verify_results(context):
    import numpy as np
    _, _, _, refs, gt_rows, _ = context
    expected = _expected_result_files(gt_rows, refs)
    complete = t.verify_seal(OUTPUT, expected, binding())
    transaction = t.load(OUTPUT / "transaction.json")
    c.require(transaction["binding"] == binding(), "Metric transaction binding differs")
    journal = [json.loads(line) for line in (OUTPUT / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    c.require(journal and journal[-1] == {"token": transaction["token"], "phase": "complete"},
              "Metric transaction journal is incomplete")
    labels = _verify_labels(gt_rows)
    result = t.load(OUTPUT / "metrics.json")
    c.require(result["binding"] == binding() and [row["model"] for row in result["models"]] == MODEL_ORDER
              and result["unique_E_images"] == 1004
              and result["native_pixels_per_model"] == 146411528
              and result["actual_positive_pixels_per_model"] == labels["positive_pixels"]
              and result["actual_anomaly_images"] == labels["anomaly_images"]
              and result["actual_normal_images"] == labels["normal_images"],
              "Final metric denominator/order differs")
    for ref, row in zip(refs, result["models"]):
        folder = OUTPUT / ref["label"] / "exact"
        inner = t.load(folder / "complete.json")
        c.require(t.sha(folder / "complete.json") == complete["files"][f"{ref['label']}/exact/complete.json"]
                  and inner["identity"] == _exact_identity(ref, labels),
                  "Exact AP inner completion identity differs")
        exact = t.load(folder / "metrics.json")
        c.require(exact["AP"] == row["pixel_AP"] and exact["pixels"] == 146411528
                  and exact["positive_pixels"] == labels["positive_pixels"]
                  and exact["exact_float32_scores"] is True and exact["histogram_bins"] == EXACT_BINS
                  and 0 <= exact["AP"] <= 1,
                  "Exact AP value/denominator differs")
        for name in ("counts.uint32", "positive.uint32"):
            c.require((folder / name).stat().st_size == EXACT_BINS * 4,
                      "Exact AP table size differs")
        metric = row["image_metrics"]
        c.require(metric["TP"] + metric["FN"] == labels["anomaly_images"]
                  and metric["FP"] + metric["TN"] == labels["normal_images"]
                  and metric["threshold"] == ref["threshold"]
                  and metric["comparison"] == "strictly greater",
                  "Image metric count/threshold differs")
    c.require(result["paired"] == _paired(result["models"]), "Paired summary differs")
    c.require(set(complete["files"]) == expected, "Unknown/missing final metric files")
    c.require(not any(name.split(".")[0] in {"torch", "torchvision", "segmentation_models_pytorch", "timm"}
                      for name in sys.modules), "Model package entered CPU metric verifier")
    return result


def run():
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(),
              "Use the CPU .venv interpreter")
    install_guard("run", destinations=(OUTPUT,))
    context = _prepared_context()
    if (OUTPUT / "complete.json").exists():
        return {"complete_cache_verified": True, "result": _verify_results(context),
                "AP_AUROC_recomputed": False, "model_operations": 0}
    c.require(not OUTPUT.exists(), "Partial/unknown metric transaction; preserve and diagnose")
    snapshot = resource_snapshot(3, 80, 80)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    t.save(OUTPUT / "transaction.json", {"token": token, "binding": binding(), "resources": snapshot})
    started = time.monotonic()
    try:
        p, req, metric_sources, refs, gt_rows, _ = context
        labels = _prepare_labels(gt_rows, token)
        models, decisions = [], {}
        from fewshot_synthetic import metrics as frozen
        c.require(t.sha(Path(frozen.__file__)) == EXACT_SOURCE_SHA
                  and frozen.BINS == EXACT_BINS and frozen.CHUNK == EXACT_CHUNK,
                  "Runtime exact AP implementation differs")
        for ref in refs:
            resource_snapshot(3, 80, 80)
            _append_journal({"token": token, "phase": "before_exact_AP", "model": ref["label"]})
            exact = frozen.exact_ap(
                lambda ref=ref: _blocks(ref, p, req, t.load(r.PREP / "models.json"),
                                        metric_sources, labels, token),
                OUTPUT / ref["label"] / "exact", _exact_identity(ref, labels),
                146411528, labels["positive_pixels"])
            image_metric, predicted = _image_metrics(ref, labels)
            decisions[ref["label"]] = predicted
            models.append({"model": ref["label"], "pixel_AP": exact["AP"],
                           "exact_distinct_scores": exact["distinct_scores"],
                           "exact_tied_score_values": exact["tied_score_values"],
                           "image_metrics": image_metric,
                           "exact_complete_sha256": t.sha(OUTPUT / ref["label"] / "exact/complete.json")})
            print(json.dumps({"model": ref["label"], "completed_models": len(models), "of": 6},
                             ensure_ascii=False), flush=True)
        result = {
            "schema": "paired-pretrained-E1004-exact-metrics-v1",
            "binding": binding(), "models": models,
            "unique_E_images": 1004, "native_pixels_per_model": 146411528,
            "actual_positive_pixels_per_model": labels["positive_pixels"],
            "actual_anomaly_images": labels["anomaly_images"],
            "actual_normal_images": labels["normal_images"],
            "paired": _paired(models),
            "selected_examples": _selected_examples(labels, decisions),
            "elapsed_seconds_before_final_seal": time.monotonic() - started,
            "new_model_operations": 0, "thresholds_recomputed": 0,
            "raw_original_GT_reads": 0, "Torch_imported": False,
        }
        t.save(OUTPUT / "metrics.json", result)
        t.save(OUTPUT / "真实指标结果.md", _result_report(result))
        _append_journal({"token": token, "phase": "complete"})
        t.seal(OUTPUT, _expected_result_files(gt_rows, refs), binding())
    except BaseException as error:
        if not (OUTPUT / "failure.json").exists():
            t.save(OUTPUT / "failure.json", {"token": token, "error": repr(error),
                                              "action": "preserve partial state; no automatic resume or reaccumulation"})
        raise
    return {"finished": True, "result": _verify_results(_prepared_context()), "model_operations": 0}


def verify_results():
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(),
              "Use the CPU .venv interpreter")
    install_guard("run", destinations=(OUTPUT,))
    return {"complete_cache_verified": True, "result": _verify_results(_prepared_context()),
            "AP_AUROC_recomputed": False, "model_operations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "verify-results"))
    args = parser.parse_args()
    functions = {"prepare": prepare, "verify": verify_preparation,
                 "run": run, "verify-results": verify_results}
    print(json.dumps(functions[args.command](), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    sys.modules.setdefault("pretrained_synthetic_pairing.real_metrics", sys.modules[__name__])
    main()
