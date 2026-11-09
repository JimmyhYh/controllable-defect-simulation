"""Explicit metadata preparation, once-only normal scoring, and saved-threshold verification."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
from . import common as c
from . import formal_common as f
from . import calibration_common as a
from .native_inference import SCORE_RULE, THRESHOLD_RULE, geometry, scalar_verification

def q_requests():
    split = c.load(c.DATA / "split.json")
    rows = split["Q_normal200"]
    inventory = c.load(c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json")["files"]
    c.require(len(rows) == 200 and len({r["id"] for r in rows}) == 200, "Fixed Q200 count/order identity invalid")
    prohibited = {r["id"] for key in ["R_real5", "N_train1885", "U_anomaly241", "E_official1004"] for r in split[key]}
    c.require(not prohibited & {r["id"] for r in rows}, "Q role overlap")
    result = []
    for index, row in enumerate(rows):
        c.require(row["part"] == "train" and row["training_label"] == "normal" and row["image"] == f"train/{row['id']}.png", "Q normal source metadata differs")
        c.require(row["image_sha256"] == inventory[row["image"]], "Q source metadata SHA differs")
        result.append({"index": index, "id": row["id"], "image": row["image"], "image_sha256": row["image_sha256"],
                       "height": row["height"], "width": row["width"]} | geometry(row["height"], row["width"]))
    return result

def verify_preparation():
    seal = f.strict_seal(a.PREP)
    c.require(c.paths(a.PREP) == {"protocol.json", "requests.json", "第6阶段正常校准准备与运行.md", "complete.json"}, "Unknown calibration preparation member")
    p = c.load(a.PREP / "protocol.json"); req = c.load(a.PREP / "requests.json")
    c.require(p["implementation"] == a.implementation() and p["formal_implementation"] == f.implementation() and p["pilot_implementation"] == c.code_identity(), "Calibration/formal/pilot implementation differs")
    c.verify_upstream(p["upstream"])
    c.require(p["score_rule"] == SCORE_RULE and p["threshold_rule"] == THRESHOLD_RULE and p["scalar_verification"] == scalar_verification(), "Frozen native score/threshold rule differs")
    c.require(req == q_requests() and p["images"] == len(req) == 200 and p["planned_window_forwards"] == sum(x["window_count"] for x in req) and p["planned_native_pixels"] == sum(x["native_pixels"] for x in req), "Calibration fixed metadata/order/denominators differ")
    formal = c.load(f.PREP / "protocol.json"); trained = c.load(f.RUN / "result.json")
    c.require(p["compatibility"] == formal["compatibility"] and p["seed"] == f.SEED and p["final_step"] == 10000, "Calibration model compatibility/budget differs")
    c.require(p["final_weight_sha256"] == c.sha(f.MODEL / "final_weights.safetensors") == trained["final_weight_sha256"] and p["expected_state_hash"] == trained["final_state_hash"] and p["expected_buffers_hash"] == trained["final_buffers_hash"] == trained["initial_buffers_hash"], "Calibration final model identity differs")
    c.require(seal["identity"]["meaning"] == "metadata/scalars only; no Q image, probability or actual threshold read/computed", "Preparation meaning differs")
    return p, req

def prepare():
    if a.PREP.exists():
        p, req = verify_preparation()
        return {"calibration_ready": True, "reused": True, "Q_images": len(req), "actual_calibration_executed": False}
    sources, formal, summary = a.upstream()
    snapshot = c.resources(2, 2)
    c.require(not a.RUN.exists() and not a.RUNTIME.exists(), "Unknown calibration output/runtime exists before preparation")
    for name in a.CODE:
        if name.endswith(".py"): ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"), filename=name)
    req = q_requests(); trained = c.load(f.RUN / "result.json")
    p = {"schema": "pretrained-segmentation-normal-calibration-preparation-v1", "implementation": a.implementation(),
         "formal_implementation": f.implementation(), "pilot_implementation": c.code_identity(), "upstream": sources,
         "seed": f.SEED, "final_step": 10000, "final_weight_sha256": summary["final_weight_sha256"],
         "expected_state_hash": trained["final_state_hash"], "expected_buffers_hash": trained["initial_buffers_hash"],
         "compatibility": formal["compatibility"], "score_rule": SCORE_RULE, "threshold_rule": THRESHOLD_RULE,
         "images": 200, "planned_window_forwards": sum(x["window_count"] for x in req), "planned_native_pixels": sum(x["native_pixels"] for x in req),
         "scalar_verification": scalar_verification(), "resource_snapshot": snapshot, "raw_Q_or_GT_decodes": 0,
         "probability_payloads_read": 0, "actual_threshold": None, "model_imports_or_forwards": 0,
         "cache_policy": "complete identical cache verify only; any partial/unknown transaction preserve/stop, never auto resume or repeat a window",
         "meaning": "normal calibration same-pool FP/200 is not independent validation; no real detection/screen claim"}
    a.PREP.mkdir(parents=True, exist_ok=False)
    c.save(a.PREP / "protocol.json", p); c.save(a.PREP / "requests.json", req)
    c.save(a.PREP / "第6阶段正常校准准备与运行.md", preparation_report(p))
    c.require(c.paths(a.PREP) == {"protocol.json", "requests.json", "第6阶段正常校准准备与运行.md"} and not any(x.is_dir() for x in a.PREP.rglob("*")), "Unknown preparation file/directory before seal")
    c.seal(a.PREP, {"meaning": "metadata/scalars only; no Q image, probability or actual threshold read/computed"})
    verify_preparation()
    return {"calibration_ready": True, "reused": False, "Q_images": 200, "planned_windows": p["planned_window_forwards"], "metadata_native_pixels": p["planned_native_pixels"], "actual_calibration_executed": False}

def score_binding(p):
    return {"preparation_complete_sha256": c.sha(a.PREP / "complete.json"), "protocol_sha256": c.sha(a.PREP / "protocol.json"),
            "formal_model_complete_sha256": c.sha(f.MODEL / "complete.json"), "final_weight_sha256": p["final_weight_sha256"], "implementation": p["implementation"]}

def verify_results():
    p, req = verify_preparation(); bound = score_binding(p)
    c.resources(2, 1)
    c.require(f.strict_seal(a.RUN)["identity"] == bound, "Saved calibration result belongs to a different source")
    expected = {"transaction.json", "scores.json", "threshold.json", "正常校准结果.md", "complete.json"} | {f"probabilities/{x['index']:03d}_{x['id']}.npy" for x in req}
    c.require(c.paths(a.RUN) == expected, "Unknown/missing normal calibration member")
    result = c.load(a.RUN / "scores.json"); threshold = c.load(a.RUN / "threshold.json")
    c.require(result["binding"] == bound and result["images_scored"] == 200 and result["window_forwards"] == p["planned_window_forwards"] and result["constructor_forwards"] == 0 and len(result["items"]) == 200, "Calibration identity/forward count differs")
    c.require(result["state_before"] == result["state_after"] == p["expected_state_hash"] and result["buffers_before"] == result["buffers_after"] == p["expected_buffers_hash"], "Calibration parameter/BN state differs")
    for row, request in zip(result["items"], req):
        name = f"probabilities/{request['index']:03d}_{request['id']}.npy"
        c.require(row["id"] == request["id"] and row["index"] == request["index"] and row["probability"] == name and row["window_forwards"] == request["window_count"] and row["source_image_sha256"] == request["image_sha256"], "Saved calibration input/order differs")
        prob = np.load(a.RUN / name, allow_pickle=False)
        c.require(prob.dtype == np.float32 and list(prob.shape) == request["native_shape"] and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, "Invalid native calibration probability")
        c.require(row["score"] == float(prob.max()), "Saved native max differs")
    c.require(threshold["binding"] == bound and threshold["score_source_sha256"] == c.sha(a.RUN / "scores.json") and threshold["rule"] == THRESHOLD_RULE, "Frozen threshold source/rule differs")
    tau = threshold["threshold"]
    c.require(isinstance(tau, (float, int)) and np.isfinite(tau) and 0 <= tau <= 1, "Invalid saved normal threshold")
    fp = sum(row["score"] > tau for row in result["items"])
    c.require(threshold["FP"] == fp and threshold["TN"] == 200 - fp and threshold["normal_denominator"] == 200, "Saved same-pool FP/TN differs")
    # Deliberately no np.quantile here: frozen threshold is never recalibrated.
    return {"calibration_verified": True, "images": 200, "window_forwards_recorded": p["planned_window_forwards"], "threshold": tau,
            "FP": fp, "TN": 200-fp, "normal_denominator": 200, "additional_forwards": 0, "threshold_recalculated": False,
            "meaning": "same normal calibration pool, not independent validation"}

def run():
    if a.RUN.exists(): return verify_results()  # Any partial/unknown state stops; no recovery.
    p, req = verify_preparation(); bound = score_binding(p)
    a.require_fresh_runtime(a.RUNTIME.exists())
    snapshot = c.resources(4.5, 20, gpu=True)
    a.runtime_environment()
    a.RUN.mkdir(parents=True, exist_ok=False)
    c.save(a.RUN / "transaction.json", {"binding": bound, "intended_images": 200, "intended_window_forwards": p["planned_window_forwards"], "resources_before_model": snapshot,
                                        "rule": "no automatic retry/continuation after partial or uncertain forward"})
    import time
    started = time.monotonic(); rows, resources = [], []
    current_index = None
    try:
        import io, hashlib, torch
        from PIL import Image
        from safetensors.torch import load_file
        from .model import build_model, state_hash, buffers_hash
        from .native_inference import predict_native
        model, _ = build_model(f.SEED, p)
        model.load_state_dict(load_file(str(f.MODEL / "final_weights.safetensors"), device="cpu"), strict=True)
        model.to("cuda").eval()
        before, buffers = state_hash(model.state_dict()), buffers_hash(model)
        c.require(before == p["expected_state_hash"] and buffers == p["expected_buffers_hash"], "Final calibration tensors differ")
        torch.cuda.reset_peak_memory_stats()
        for request in req:
            current_index = request["index"]
            snap = c.resources(2, 1); snap["image_index"] = current_index; resources.append(snap)
            raw = (c.DATA / request["image"]).read_bytes()
            c.require(hashlib.sha256(raw).hexdigest() == request["image_sha256"], "Actual Q image identity differs before decode")
            with Image.open(io.BytesIO(raw)) as im:
                c.require(im.mode in ("RGB", "L") and im.size == (request["width"], request["height"]), "Actual Q RGB geometry/mode differs")
                rgb = np.asarray(im.convert("RGB")).copy()
            prob, forwards = predict_native(model, rgb)
            c.require(forwards == request["window_count"] and list(prob.shape) == request["native_shape"], "Actual window/shape differs")
            name = f"probabilities/{current_index:03d}_{request['id']}.npy"
            (a.RUN / "probabilities").mkdir(exist_ok=True)
            np.save(a.RUN / name, prob, allow_pickle=False)
            rows.append({"index": current_index, "id": request["id"], "source_image_sha256": request["image_sha256"],
                         "probability": name, "window_forwards": forwards, "score": float(prob.max())})
            del raw, rgb, prob
            if (current_index + 1) % 25 == 0: print({"Q_images": current_index+1, "of": 200, "window_forwards": sum(x["window_forwards"] for x in rows)}, flush=True)
        torch.cuda.synchronize()
        result = {"binding": bound, "items": rows, "images_scored": 200, "window_forwards": sum(x["window_forwards"] for x in rows), "constructor_forwards": 0,
                  "state_before": before, "state_after": state_hash(model.state_dict()), "buffers_before": buffers, "buffers_after": buffers_hash(model),
                  "GT_read_by_predictor": False, "raw_Q_image_decodes": 200, "native_pixels": sum(x["native_pixels"] for x in req),
                  "elapsed_seconds": time.monotonic()-started, "sampled_min_free_RAM_GiB": min(x["free_RAM_GiB"] for x in resources),
                  "resource_samples": resources, "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated()/2**20, "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved()/2**20,
                  "score_rule": SCORE_RULE, "training_or_parameter_updates": 0}
        c.require(result["state_before"] == result["state_after"] and result["buffers_before"] == result["buffers_after"], "Scoring changed parameters/BN; preserve")
        c.save(a.RUN / "scores.json", result)
        scores = np.asarray([x["score"] for x in rows], dtype=np.float64)
        tau = float(np.quantile(scores, THRESHOLD_RULE["q"], method=THRESHOLD_RULE["method"]))
        fp = int((scores > tau).sum())
        threshold = {"binding": bound, "score_source_sha256": c.sha(a.RUN / "scores.json"), "threshold": tau, "rule": THRESHOLD_RULE,
                     "FP": fp, "TN": 200-fp, "normal_denominator": 200, "independent_validation": False}
        c.save(a.RUN / "threshold.json", threshold)
        c.save(a.RUN / "正常校准结果.md", f"# 正式模型正常校准\n\n最终10000步权重，固定Q200原序，每图一次、合计{result['window_forwards']}窗；原生概率拼回max，τ={tau:.17g}，q0.95／linear／严格大于。FP={fp}/200，TN={200-fp}/200，只是同池正常校准，非独立验证；未评价异常召回或屏幕效果。完整缓存只核验，不重算阈值；未知事务保留停止。\n")
        expected = {"transaction.json", "scores.json", "threshold.json", "正常校准结果.md"} | {row["probability"] for row in rows}
        c.require(c.paths(a.RUN) == expected and {x.relative_to(a.RUN).as_posix() for x in a.RUN.rglob("*") if x.is_dir()} == {"probabilities"}, "Unknown calibration member before completion")
        c.seal(a.RUN, bound)
    except BaseException as exc:
        c.save(a.RUN / "failure.json", {"error": repr(exc), "completed_image_files": len(rows), "current_request_index": current_index,
                                       "meaning": "may include incomplete/uncertain forward; preserve; no retry or completion"})
        raise
    return verify_results()

def preparation_report(p):
    return f'''# 第6阶段：正常校准CPU准备与运行

本次只完成来源／元信息／hash、窗口坐标、虚构标量和静态代码核验；未导入Torch／SMP／timm／模型、加载参数、前向、训练、校准、计算实际阈值或指标，未读取Q／E／U原图、GT或概率。正式10000步／10保存点与独立训练核验已只读核对。新增独立校准五文件，不改任何已封存小试／正式模块、入口、全局路径、环境、权重或结果。

独立静态审阅指出新运行可能静默复用仅残留的临时目录；根因是只核文件路径而未要求目录不存在。已补齐新运行的临时目录存在即停止、mkdir不得复用，并将资源准入置于创建目录前；虚构布尔存在／不存在状态与源码核验通过，没有创建未知现场、导入模型或做资源前向。完整同身份结果仍仅只读核验。静态核验不等于CUDA运行兼容或实际资源测量。

## 冻结评分与阈值

唯一模型为seed2026102202最终10000步，权重hash `{p['final_weight_sha256']}`。只读复用同一单二分类U-Net、固定编码器BN与decoder GN8；FP32／batch1／workers0，无增强、平滑或池归一化。原生RGB保留对比度，以RGB/255和同一ImageNet均值／标准差归一化；256窗／stride128及尾窗去重，右反射补边在拼回前剔除。按y升序以float32累加、uint16覆盖计数、float32等权除法得到原生float32概率；整图分数只取已保存原生图max。浮点累加次序与精度明确固定，不在结果后另换平滑／max融合／评分头。预测器只接收RGB，不接收或读取GT。

Q200角色和原序仅引用原split元信息，未查看图像或重新验证零GT。200份请求没有GT路径；元信息预定{p['planned_window_forwards']}次窗口前向和{p['planned_native_pixels']:,}原生像素，不是本次实际推理数量。全部图窗口覆盖／首尾／重复／有效宽已用整数坐标核对。虚构0—199标量的q0.95／linear为189.05、相等阈值判正常及FP32等权标量均核验，不是Q分数或实际τ。

未来每图仅评分一次、每窗仅前向一次，保存原始float32拼回图及max；全部完成并核参数／BN前后与正式初始化签名一致后，仅一次以200个max转float64作np.quantile(q=.95,method=linear)，score严格>τ。实际τ、FP／TN当前未知；同池FP/200不是独立正常验证，不支持异常召回、AP或屏幕效果。

## 已实现显式命令

CPU准备核验（不读Q像素／概率，不导入模型）：

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli verify
```

实际正常校准入口已实现，本次未执行；须用户另行明确授权助手运行，或用户自行运行：

```powershell
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli run
```

只读结果核验入口verify-results仅在实际完整结果存在后使用；它读本组缓存与原τ核FP／TN，不重新计算分位数或前向。不存在／未知／部分结果保留停止，不提供隐式续接、删锁、自动补推理或降规格。未来评分中断、OOM或状态不明即保存失败现场；即使前200图已保存但尚未complete，也不能自动重运行或自动补阈值。

准备目录{a.PREP.as_posix()}；未来结果{a.RUN.as_posix()}，运行临时目录独立。CPU准备／核验RAM≥2GiB、磁盘≥2GiB；实际前向启动RAM≥4.5GiB／空闲显存≥6GiB／磁盘≥20GiB，每图开始前RAM≥2GiB／磁盘≥1GiB。任何资源不满足先保留停止，完整同身份缓存只做CPU核验，无模型导入或新增前向。

本次准备快照RAM{p['resource_snapshot']['free_RAM_GiB']:.3f}GiB／磁盘{p['resource_snapshot']['free_disk_GiB']:.3f}GiB，仅当时信息。正式Q校准时间、峰值与输出形状合法性待实际运行，不能把训练11.32分钟当校准耗时。未知身份或代码差异先查因，不用E／旧评价调配置或选择中间权重；不自动进入E评分。
'''
