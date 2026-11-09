"""Independent Q200 preparation and explicit single-model calibration CLI.

Preparation/verify are stdlib only. Model and probability operations are lazy
and available solely through the explicitly selected runtime/cache commands.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from . import training as t

c = t.c
PREP = c.RESULTS / "calibration_preparation_v1"
ASSEMBLY = c.RESULTS / "training_verifications_v1/all_six_v1"
ASSEMBLY_SHA = "9d1471071147a9723bf36873432b852ff4f98f20e986ff983a05d4e9a9330246"
SPLIT = c.ROOT / "data/fewshot_synthetic/ksdd2_v1/split.json"
SPLIT_SHA = "b8c2fb376bfd04072baaaf173eb0fccb1ecc5f1339074a6829fe147d0b6210bf"
INVENTORY = c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json"
INVENTORY_SHA = "da1dc019307841e24a07f565627808ecdb4ea4d0512e4ae9e5ad9ad4f1221ac8"
CODE = ("calibration.py", "calibration_runtime.py", "calibration_cache.py", "README_calibration.md")
REPORT = "第5阶段正常校准准备与运行.md"
FILES = {"transaction.json", "protocol.json", "requests.json", "models.json", "sources.json", "environment.json", "scalar_verification.json", REPORT}
SCORE_RULE = {
    "window": 256, "stride": 128, "tail": "include H-256; sorted unique",
    "pad": "right reflect; discard before stitching",
    "normalization": "RGB/255 then ImageNet mean/std; no resize or contrast stretch",
    "window_probability": "raw float32 anomaly softmax, no augmentation or smoothing",
    "stitching": "float32 sum in ascending y order, uint16 coverage, float32 equal-weight division; each native pixel once in final map",
    "image_score": "maximum of saved native float32 probability map",
    "precision": "FP32", "batch": 1, "workers": 0,
}
THRESHOLD_RULE = {"q": .95, "method": "linear", "comparison": "strictly greater", "score_dtype_for_quantile": "float64 conversion of native float32 maxima", "normal_images": 200}
_GUARDED = False


def paths(group, seed):
    t.paths(group, seed)  # Validate without changing any frozen global path.
    key = f"{group}_{seed}"
    return c.RESULTS / "calibration_v1" / key, c.LARGE / "runtime_calibration_v1" / key


def implementation():
    return {name: t.sha(c.PACKAGE / name) for name in CODE}


def geometry(height, width):
    c.require(type(height) is int and height >= 256 and type(width) is int and 2 <= width < 256, "Unsupported native Q geometry")
    ys = sorted(set(range(0, height - 256 + 1, 128)) | {height - 256})
    coverage = [0] * height
    for y in ys:
        for index in range(y, y + 256):
            coverage[index] += 1
    c.require(ys[0] == 0 and ys[-1] == height - 256 and min(coverage) > 0 and max(coverage) < 65536, "Invalid window coverage")
    return {"ys": ys, "window_count": len(ys), "native_shape": [height, width], "native_pixels": height * width,
            "coverage_min": min(coverage), "coverage_max": max(coverage), "padded_width": 256}


def q_requests():
    c.require(t.sha(SPLIT) == SPLIT_SHA and t.sha(INVENTORY) == INVENTORY_SHA, "Q metadata source differs")
    split = t.load(SPLIT)
    inventory = t.load(INVENTORY)["files"]
    rows = split["Q_normal200"]
    c.require(len(rows) == len({r["id"] for r in rows}) == 200, "Q200 count differs")
    prohibited = {r["id"] for role in ("R_real5", "N_train1885", "U_anomaly241", "E_official1004") for r in split[role]}
    c.require(not prohibited.intersection(r["id"] for r in rows), "Q role overlap")
    requests = []
    for index, row in enumerate(rows):
        c.require(row["part"] == "train" and row["training_label"] == "normal" and row["image"] == f"train/{row['id']}.png", "Unexpected Q metadata")
        c.require(row["image_sha256"] == inventory[row["image"]], "Q image source metadata differs")
        requests.append({"index": index, "id": row["id"], "image": row["image"], "image_sha256": row["image_sha256"],
                         "height": row["height"], "width": row["width"], **geometry(row["height"], row["width"])})
    c.require(sum(r["window_count"] for r in requests) == 844 and sum(r["native_pixels"] for r in requests) == 29202876, "Q planned denominators differ")
    return requests  # Deliberately omit mask/GT paths and class labels.


def scalar_verification():
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]
    toy = 189 + (.95 * 199 - 189)
    strict = [x > .5 for x in (0., .5, .5, 1.)]
    average = f32(f32(f32(.2) + f32(.8)) / f32(2))
    c.require(abs(toy - 189.05) < 1e-10 and strict == [False, False, False, True] and average == .5, "Scalar rule differs")
    cases = [geometry(256, 229), geometry(512, 229), geometry(635, 229)]
    c.require([r["ys"] for r in cases] == [[0], [0, 128, 256], [0, 128, 256, 379]], "Tail/de-duplication scalar differs")
    states = [new_run_allowed(a, b) for a, b in ((False, False), (True, False), (False, True), (True, True))]
    c.require(states == [True, False, False, False], "Unknown/partial transaction admission differs")
    fake = {"files": {"owner.json": "owner-hash"}, "directories": ["tmp"]}
    c.require(runtime_listing_matches(fake, fake), "Identical runtime metadata rejected")
    changed = [
        {"files": {**fake["files"], "tmp/unknown.lock": "new-hash"}, "directories": ["tmp"]},
        {"files": fake["files"], "directories": ["tmp", "tmp/unknown_empty"]},
        {"files": {"owner.json": "changed-owner"}, "directories": ["tmp"]},
    ]
    c.require(all(not runtime_listing_matches(x, fake) for x in changed), "Unknown/changed runtime metadata accepted")
    c.require(model_module("pretrained_segmentation.model") and model_module("pretrained_segmentation.model.child") and not model_module("pretrained_synthetic_pairing.calibration"), "CPU model-module boundary differs")
    return {"scope": "fictitious scalars and integer window geometry only", "toy_linear_quantile_0_to_199": toy,
            "toy_strict_gt_half": strict, "toy_FP32_overlap_average": average, "tail_cases": cases,
            "new_run_admission_for_output_runtime_absent_present": states,
            "fictitious_runtime_unknown_lock_empty_dir_owner_change_rejected": True,
            "CPU_lazy_model_import_names_rejected_without_importing": True,
            "actual_Q_scores_or_threshold": False, "model_or_optimizer_instantiated": False}


def new_run_allowed(output_present, runtime_present):
    c.require(type(output_present) is bool and type(runtime_present) is bool, "Invalid transaction existence metadata")
    return not output_present and not runtime_present


def model_module(name):
    return name == "pretrained_segmentation.model" or name.startswith("pretrained_segmentation.model.")


def runtime_listing_matches(current, saved):
    return current == saved


def runtime_manifest(runtime):
    runtime = c.no_redirect(runtime)
    files, dirs = c.tree(runtime)
    c.require({p.name for p in runtime.iterdir()} == {"owner.json", "tmp"} and (runtime / "tmp").is_dir(), "Unknown runtime-root member")
    return {"files": {name: t.sha(runtime / str(c.safe_relative(name))) for name in sorted(files)}, "directories": sorted(dirs)}


def verify_runtime(root, runtime, bound):
    saved = t.load(root / "runtime_manifest.json")
    c.require(saved["binding"] == bound and t.load(runtime / "owner.json") == bound, "Runtime owner/binding differs")
    c.require(runtime_listing_matches(runtime_manifest(runtime), saved["manifest"]), "Unknown/missing/changed runtime file or empty directory")


def guard(cpu=True, destinations=(), raw_q=False, cache=None, temporary=None):
    global _GUARDED
    c.require(not _GUARDED, "Start a new process for each explicit operation")
    destinations = tuple(c.no_redirect(p) for p in destinations)
    temporary = c.no_redirect(temporary) if temporary is not None else None
    cache = c.no_redirect(cache) if cache is not None else None
    ancestors = {p for d in destinations for p in d.parents if p != Path(d.anchor)}
    allowed_q = {c.no_redirect(SPLIT.parent / row["image"]) for row in q_requests()} if raw_q else set()
    # NumPy imports the pickle module; only importing it is harmless. Its audit
    # events remain forbidden, and np.load is restricted to allow_pickle=False.
    blocked = c.BLOCKED - ({"numpy", "pickle"} if cache is not None else set())
    c.require(not cpu or not any(n.split(".")[0] in blocked or model_module(n) for n in sys.modules), "Forbidden CPU import already present")
    def audit(event, args):
        if event == "import" and (args[0] == "controlled_generation.render" or cpu and (args[0].split(".")[0] in blocked or model_module(args[0]))):
            raise RuntimeError("Operation forbids model/data import")
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("Network/pickle execution forbidden")
        if event in {"os.link", "os.symlink"}:
            raise RuntimeError("Links forbidden")
        if event in {"os.remove", "os.rename", "os.rmdir"}:
            affected = args[:2] if event == "os.rename" else args[:1]
            c.require(not cpu and temporary is not None and all(Path(x).absolute().is_relative_to(temporary) and Path(x).absolute() != temporary for x in affected), "Delete/move outside own temporary files forbidden")
        if event == "os.mkdir":
            p = Path(args[0]).absolute()
            c.require(p in ancestors or any(p == d or p.is_relative_to(d) for d in destinations), "Directory outside independent output")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), "\\\\.\\nul"}:
                return
            p = Path(name).absolute(); mode, flags = args[1] or "", args[2] or 0
            writing = any(ch in str(mode) for ch in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                c.require(any(p.is_relative_to(d) for d in destinations), "Write outside explicit independent transaction")
            elif p.is_relative_to(c.ROOT / "data"):
                c.require(p == SPLIT or p in allowed_q, "Q/E/U/GT byte read forbidden outside explicit Q runtime")
            elif p.suffix.lower() in c.OPAQUE_SUFFIXES:
                hash_read = p in t._HASH_SCOPE or p in c._READ_SCOPES
                cached = cache is not None and p.is_relative_to(cache / "probabilities") and p.suffix == ".npy"
                model_read = not cpu and (p == c.GENERIC or any(p == t.paths(g, s)[1] / "final_weights.safetensors" for g, s in t.ORDER))
                c.require(hash_read or cached or model_read, "Unbound probability/tensor/array read forbidden")
    sys.dont_write_bytecode = True
    sys.addaudithook(audit); _GUARDED = True


def training_check():
    command = [str(c.CPU / "Scripts/python.exe"), "-B", "-X", "utf8", "-m", "pretrained_synthetic_pairing.training", "verify"]
    run = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=360,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    c.require(run.returncode == 0, "Frozen training/input verifier failed: " + run.stderr[-1800:])
    checked = json.loads(run.stdout)
    c.require(checked["preparation_ready"] and checked["training_updates"] == checked["model_forward_calls"] == 0, "Unexpected preparation evidence")
    return {"command": command, "exit_code": 0, "mode": "clean stdlib CPU child; no pixel or model operations"}


def upstream():
    c.require(t.sha(ASSEMBLY / "complete.json") == ASSEMBLY_SHA, "Six-model reference anchor differs")
    total = t.load(ASSEMBLY / "verification.json")
    identity = {"training_preparation_complete_sha256": t.sha(t.PREP / "complete.json"), "input_complete_sha256": t.INPUT_SHA,
                "plan_contract": c.plan_contract(), "order": [f"{g}_{s}" for g, s in t.ORDER]}
    t.verify_seal(ASSEMBLY, {"sources.json", "verification.json", "第4阶段六组训练与配对技术核验.md"}, identity)
    c.require(total["training_only_technical_complete"] and total["formal_models"] == 6 and total["checkpoints"] == 60 and total["model_order"] == identity["order"], "Six-model evidence incomplete")
    source = t.load(ASSEMBLY / "sources.json")
    c.require(len(source) == 420 and len(total["entries"]) == 6 and len(total["pairs"]) == 3, "Incomplete source references")
    for p, digest in source.items():
        c.require(t.sha(p) == digest, "Frozen training reference changed: " + p)
    models = []
    for entry, (group, seed) in zip(total["entries"], t.ORDER):
        run, model, _ = t.paths(group, seed); bound = t.binding(group, seed)
        result = t.load(run / "result.json"); initial = t.load(model / "initialization.json")
        proof = Path(entry["per_group_verification"]); verified = t.load(proof / "verification.json")
        c.require(entry["group"] == group and entry["seed"] == seed and entry["binding"] == verified["binding"] == result["binding"] == bound, "Model order/binding differs")
        c.require(t.sha(run / "complete.json") == entry["run_complete_sha256"] and t.sha(model / "complete.json") == entry["model_complete_sha256"] and t.sha(proof / "complete.json") == entry["per_group_verification_complete_sha256"], "Per-group completion differs")
        c.require(verified["training_verified"] and result["step"] == 10000 and result["all_finite"] and len(entry["checkpoints"]) == 10, "Final technical evidence missing")
        c.require(entry["final_weight_sha256"] == result["final_weight_sha256"] == t.sha(model / "final_weights.safetensors"), "Final weight differs")
        c.require(initial == entry["initialization"] and result["initial_buffers_hash"] == result["final_buffers_hash"] == initial["initial_buffers_hash"], "Frozen BN identity differs")
        t.verify_seal(run, {"binding.json", "transaction.json", "result.json"}, bound)
        mcomp = t.load(model / "complete.json"); t.verify_seal(model, set(mcomp["files"]), bound)
        for ck, step in zip(entry["checkpoints"], range(1000, 10001, 1000)):
            folder = model / f"checkpoints/step_{step:05d}"
            record = t.load(folder / "checkpoint.json")
            c.require(ck["step"] == step and ck["path"] == str(folder) and t.sha(folder / "complete.json") == ck["complete_sha256"], "Checkpoint reference differs")
            c.require(record["step"] == step and record["next_lr"] == t.next_lr(step) and record["buffers_hash"] == initial["initial_buffers_hash"], "Checkpoint schedule/BN differs")
        models.append({"group": group, "seed": seed, "binding": bound, "final_step": 10000,
                       "final_weight_path": str(model / "final_weights.safetensors"), "final_weight_sha256": result["final_weight_sha256"],
                       "expected_state_hash": result["final_state_hash"], "expected_buffers_hash": result["final_buffers_hash"],
                       "initialization": initial, "run_complete_sha256": entry["run_complete_sha256"], "model_complete_sha256": entry["model_complete_sha256"],
                       "per_group_verification_complete_sha256": entry["per_group_verification_complete_sha256"]})
    for folder in (ASSEMBLY, t.PREP, c.OUT):
        for p in folder.iterdir():
            if p.is_file():
                source[str(p)] = t.sha(p)
    source.update(t.load(t.PREP / "sources.json"))
    for path in (SPLIT, INVENTORY, c.GENERIC, c.BASE / "environment_v1/environment.json"):
        source[str(path)] = t.sha(path)
    for name in ("common.py", "model.py", "data.py", "native_inference.py"):
        path = c.ROOT / "pretrained_segmentation" / name; source[str(path)] = t.sha(path)
    return source, models


def protocol(models):
    training = t.load(t.PREP / "protocol.json")
    return {"schema": "paired-pretrained-Q200-calibration-v1", "implementation": implementation(), "plan_contract": c.plan_contract(),
            "all_six_complete_sha256": ASSEMBLY_SHA, "training_preparation_complete_sha256": t.sha(t.PREP / "complete.json"),
            "split_sha256": SPLIT_SHA, "inventory_sha256": INVENTORY_SHA, "model_order": [f"{m['group']}_{m['seed']}" for m in models],
            "model_compatibility": training["model_compatibility"], "model_and_loss": training["model_and_loss"],
            "model_function": "pretrained_segmentation.model.build_model; generic constructor then strict load of own final weights, no optimizer",
            "score_rule": SCORE_RULE, "threshold_rule": THRESHOLD_RULE, "images_per_model": 200, "windows_per_model": 844,
            "native_pixels_per_model_metadata": 29202876, "unique_normal_images_across_models": 200,
            "total_planned_model_image_scores": 1200, "total_planned_window_forwards": 5064,
            "resources": {"CPU_RAM_GiB": 2, "CPU_D_E_free_GiB": 2, "model_RAM_GiB": 4.5, "model_VRAM_MiB": 6144, "model_E_free_GiB": 20,
                          "per_image_RAM_GiB": 2, "per_image_E_free_GiB": 1},
            "transaction": "new process per explicit model; predecessor complete; append+fsync pre-forward window intentions; any partial/unknown runtime preserve/stop, never retry a window",
            "reuse": "complete same identity only; cache verification loads own saved probabilities and threshold but never quantile/model",
            "BN": "encoder fixed buffers; decoder GN8; eval/inference_mode; final state and buffers unchanged",
            "meaning": "same-pool FP/200 is calibration, not independent validation; no anomaly recall, AP or screen benefit",
            "GPU_runtime_or_native_stitching_measured_now": False, "actual_thresholds": None}


def verify_preparation():
    t.resources()
    c.require(t.sha(ASSEMBLY / "complete.json") == ASSEMBLY_SHA, "Assembly identity differs")
    t.verify_seal(PREP, FILES, {"implementation": implementation(), "all_six_complete_sha256": ASSEMBLY_SHA, "plan_contract": c.plan_contract()})
    p = t.load(PREP / "protocol.json"); req = t.load(PREP / "requests.json"); models = t.load(PREP / "models.json")
    c.require(req == q_requests() and p == protocol(models) and t.load(PREP / "scalar_verification.json") == scalar_verification(), "Frozen preparation differs")
    c.require([(m["group"], m["seed"]) for m in models] == t.ORDER, "Frozen model order differs")
    for path, digest in t.load(PREP / "sources.json").items():
        c.require(t.sha(path) == digest, "Prepared source changed: " + path)
    return p, req, models


def prepare():
    c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(), "Use CPU .venv")
    guard(cpu=True, destinations=(PREP,)); snapshot = t.resources()
    if PREP.exists():
        verify_preparation()
        return {"preparation_ready": True, "cache_reused": True, "actual_calibration": False, "Q_image_reads": 0}
    check = training_check(); sources, models = upstream(); req = q_requests()
    from .resources import environment_metadata
    env = environment_metadata()
    c.require(env == t.load(c.OUT / "environment_metadata.json"), "Frozen environment metadata differs")
    for group, seed in t.ORDER:
        run, runtime = paths(group, seed)
        c.require(not run.exists() and not runtime.exists(), "Unknown calibration output/runtime before preparation")
    for name in CODE:
        if name.endswith(".py"):
            ast.parse((c.PACKAGE / name).read_text(encoding="utf-8"), filename=name)
    # Match original rule literals without importing numpy/data/model modules.
    tree = ast.parse((c.ROOT / "pretrained_segmentation/native_inference.py").read_text(encoding="utf-8"))
    literals = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"SCORE_RULE", "THRESHOLD_RULE"}}
    c.require(literals == {"SCORE_RULE": SCORE_RULE, "THRESHOLD_RULE": THRESHOLD_RULE}, "Original native score rule differs")
    p = protocol(models)
    PREP.mkdir(parents=True, exist_ok=False)
    t.save(PREP / "transaction.json", {"resources": snapshot, "training_source_check": check, "scope": "CPU metadata/hash/scalars; no Q bytes, tensor or model operation"})
    for name, value in {"protocol.json": p, "requests.json": req, "models.json": models, "sources.json": sources, "environment.json": env, "scalar_verification.json": scalar_verification(), REPORT: preparation_report(p, snapshot)}.items():
        t.save(PREP / name, value)
    t.seal(PREP, FILES, {"implementation": implementation(), "all_six_complete_sha256": ASSEMBLY_SHA, "plan_contract": c.plan_contract()})
    verify_preparation()
    return {"preparation_ready": True, "models": 6, "Q_unique_images": 200, "planned_windows_per_model": 844,
            "planned_native_pixels_per_model": 29202876, "actual_calibration": False, "Q_image_reads": 0, "model_operations": 0}


def binding(p, model):
    return {"preparation_complete_sha256": t.sha(PREP / "complete.json"), "protocol_sha256": t.sha(PREP / "protocol.json"),
            "requests_sha256": t.sha(PREP / "requests.json"), "models_sha256": t.sha(PREP / "models.json"),
            "implementation": p["implementation"], "all_six_complete_sha256": ASSEMBLY_SHA, "model": model}


def find_model(models, group, seed):
    t.paths(group, seed)
    return next(m for m in models if (m["group"], m["seed"]) == (group, seed))


def run_files(requests):
    return {"transaction.json", "window_journal.jsonl", "runtime_manifest.json", "scores.json", "threshold.json", "正常校准结果.md"} | {f"probabilities/{r['index']:03d}_{r['id']}.npy" for r in requests}


def verify_saved_metadata(group, seed, p, requests, models):
    """Hash/metadata gate shared by current-cache verification and successors."""
    root, runtime = paths(group, seed); model = find_model(models, group, seed); bound = binding(p, model)
    t.verify_seal(root, run_files(requests), bound)
    verify_runtime(root, runtime, bound)
    scores = t.load(root / "scores.json"); threshold = t.load(root / "threshold.json")
    c.require(scores["binding"] == threshold["binding"] == bound and threshold["score_source_sha256"] == t.sha(root / "scores.json"), "Saved score/threshold binding differs")
    c.require(scores["images_scored"] == scores["raw_Q_image_decodes"] == len(scores["items"]) == 200 and scores["window_forwards"] == 844 and scores["constructor_forwards"] == 0 and scores["native_pixels"] == 29202876, "Saved scoring counts differ")
    c.require(scores["state_before"] == scores["state_after"] == model["expected_state_hash"] and scores["buffers_before"] == scores["buffers_after"] == model["expected_buffers_hash"], "Saved model state/BN differs")
    c.require(threshold["rule"] == THRESHOLD_RULE and threshold["normal_denominator"] == 200 and threshold["independent_validation"] is False and threshold["quantile_evaluations"] == 1, "Saved threshold rule differs")
    return scores, threshold


def predecessor_ready(group, seed, p, requests, models):
    index = t.ORDER.index((group, seed))
    for g, s in t.ORDER[:index]:
        verify_saved_metadata(g, s, p, requests, models)


def preparation_report(p, snapshot):
    fence = chr(96) * 3
    return f"""# 第5阶段正常Q200校准准备与运行

六组最终10000步／60检查点、六组逐项核验证据和all_six_v1引用已只读核对；环境解释器／依赖元信息与封存源一致。仅新增独立calibration.py／calibration_runtime.py／calibration_cache.py／README_calibration.md，不改冻结training／inputs／common／resources／旧模型或原生评分源码。遵从用户约束，不建立过程表、软件测试、提交或工作树；仅进行来源、AST／CLI、整数坐标和虚构标量核验，未执行模型或资源训练试跑。

独立静态审阅发现两项事务缺口：旧初稿只核runtime顶层、未绑定临时文件／空目录，且后组只核前组结果而不核其runtime。根因为目录枚举结果被丢弃、前组准入与当前缓存走不同核验路径；已在封存前同一修正轮新增runtime_manifest.json，绑定owner、全部文件hash及目录清单，并让当前缓存和后组准入共用同一只读元信息／hash核验。CPU对惰性模型模块的导入拦截也按用户明确边界补齐；仅校验模块名字符串，不实际导入模型。虚构锁、未知空目录、owner变化及新运行四态均拒绝不明确状态，AST核对共同路径；未建立测试套件或制造真实失败缓存。无延期项。CUDA／第三方导入、退出清理对严格runtime清单的影响和实际时间资源仍未实测，若变化保留停止，不自动忽略或补推。

固定顺序A01/B01/A02/B02/A03/B03，每组200张相同唯一正常图、844窗、元信息29,202,876原生像素；六组合计1200模型图像评分／5064窗，仍只有200张唯一Q图。只从split和inventory元信息固定请求，不包含GT路径，不读取或重新证明正常GT。

同一ImageNet单头U-Net／encoder固定BN／decoder GN8、FP32／batch1／workers0。预测器仅接收RGB及窗口坐标，不接GT。原生256窗stride128、尾窗去重、右reflect补边在拼回前剔除；y升序float32累加、uint16覆盖、float32等权平均。无拉伸／平滑／测试池归一化。保存原始float32异常softmax拼回图，整图max作为分数。模型从通用构造后严格加载本组最终权重，构造无前向、无optimizer；核最终参数／BN前后及初始BN来源不变。

每组仅一次将200个原生float32 max转float64并作q=.95／linear，严格score>tau。实际新tau未知，不用旧先导tau或旧Q分数。完整缓存核验只读原概率及原tau核FP/TN，不重新分位数；FP/200只是同池校准，不是独立验证、不支持异常召回或屏幕结论。虚构0—199标量189.05／严格大于ties／FP32 .5仅检验规则，不是Q实际数值。

独立准备：{PREP.as_posix()}。未来每组results/calibration_v1/{{group}}_{{seed}}；各组新进程，自有runtime_calibration_v1/{{group}}_{{seed}}。按原序要求前组完整；只提供单模型入口，不自动串行。逐窗前写入fsync意向，发生任何未知／部分输出或runtime残留即保留停止，不自动补窗、重新评分、补阈值或清理标记。含完整缓存时只读核验，不导入Torch或构造模型；未知状态须另行systematic-debugging。

完成时runtime_manifest绑定owner、全部临时文件hash与目录，新增／缺失／变化文件、锁或空目录均停止，前组准入同样核对。科学结果及所有旧成果不删改；即使第三方在进程退出时自行清理临时文件，也不能忽略清单差异，需要另行查因。这是尚未实测的运行边界，不声称静态核验等于实际CUDA兼容。

CPU准备RAM>=2GiB且D/E各>=2GiB；实际评分启动RAM>=4.5GiB、空闲显存>=6GiB、E>=20GiB，逐图RAM>=2GiB／E>=1GiB。准备快照RAM{snapshot['available_RAM_GiB']:.3f}GiB、E{snapshot['E_free_GiB']:.3f}GiB、空闲显存{snapshot['GPU']['free_MiB']}MiB仅当时元信息；本轮校准耗时／显存／原生拼回实际兼容尚未实测，先导单模型约34.75秒只作历史参考，不保证总耗时。完整GPU恢复不涉及本评分入口，训练恢复仍未实测。

只读CPU准备核验（不读Q像素或概率）：

{fence}powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration verify
{fence}

首模型A01实际校准人工入口已实现，本次未执行：

{fence}powershell
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration run --group A --seed 2026102301
{fence}

完整结果存在后，CPU只读核验入口为calibration verify-results --group A --seed 2026102301（使用.venv）。本次没有实际校准、阈值、FP/TN或Q/E/U数据访问；不自动进入E评价。所有训练继续由用户手动启动。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "verify-results"))
    parser.add_argument("--group", choices=("A", "B")); parser.add_argument("--seed", type=int, choices=c.SEEDS)
    args = parser.parse_args()
    if args.command in {"prepare", "verify"}:
        c.require(args.group is None and args.seed is None, "Metadata operations take no model arguments")
        if args.command == "prepare":
            result = prepare()
        else:
            c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(), "Use CPU .venv")
            guard(cpu=True); verify_preparation()
            result = {"preparation_verified": True, "Q_image_reads": 0, "probability_payloads_read": 0, "actual_calibration": False}
    else:
        c.require(args.group is not None and args.seed is not None, "Explicit single-model group and seed required")
        if args.command == "run":
            from .calibration_runtime import run
            result = run(args.group, args.seed)
        else:
            c.require(Path(sys.executable).resolve() == (c.CPU / "Scripts/python.exe").resolve(), "Use CPU .venv")
            root, _ = paths(args.group, args.seed); guard(cpu=True, cache=root)
            p, requests, models = verify_preparation()
            from .calibration_cache import verify_results
            result = verify_results(args.group, args.seed, p, requests, models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.modules.setdefault("pretrained_synthetic_pairing.calibration", sys.modules[__name__])
    main()
