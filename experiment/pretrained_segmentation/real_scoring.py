"""E1004 metadata preparation and explicit once-only label-free scoring."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
from . import common as c
from . import formal_common as f
from . import real_scoring_common as r
from .native_inference import SCORE_RULE, THRESHOLD_RULE, geometry

REPORT = "第6阶段E1004真实评分准备与运行.md"

def requests():
    split = c.load(c.DATA / "split.json"); rows = split["E_official1004"]
    inv = c.load(c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json")["files"]
    c.require(len(rows) == len({x["id"] for x in rows}) == 1004, "E1004 identity count differs")
    others = {x["id"] for k in ["R_real5", "N_train1885", "Q_normal200", "U_anomaly241"] for x in split[k]}
    c.require(not others & {x["id"] for x in rows}, "E data role overlap")
    req, gt = [], []
    for index, row in enumerate(rows):
        c.require(row["part"] == "test" and row["image"] == f"test/{row['id']}.png" and row["mask"] == f"test/{row['id']}_GT.png" and row["training_label"] is None, "E source metadata differs")
        c.require(row["image_sha256"] == inv[row["image"]], "E RGB hash metadata differs")
        req.append({"index": index, "id": row["id"], "image": row["image"], "image_sha256": row["image_sha256"],
                    "height": row["height"], "width": row["width"]} | geometry(row["height"], row["width"]))
        gt.append({"index": index, "id": row["id"], "mask": row["mask"], "mask_sha256": inv[row["mask"]], "native_shape": [row["height"], row["width"]]})
    c.require(sum(x["native_pixels"] for x in req) == 146411528, "E native metadata pixel count differs")
    return req, gt

def scalar_verification():
    c.require([x > .5 for x in [0., .5, 1.]] == [False, False, True], "Strict > scalar differs")
    c.require(np.float32((np.float32(.2) + np.float32(.8)) / np.float32(2)) == np.float32(.5), "FP32 average scalar differs")
    r.fresh_runtime(False)
    rejected = False
    try: r.fresh_runtime(True)
    except RuntimeError: rejected = True
    c.require(rejected, "Unknown runtime admission must stop")
    return {"strict_greater": [False, False, True], "FP32_average": .5, "unknown_runtime_rejected": True, "real_scores_or_quantiles_used": False}

def verify_preparation():
    c.resources(2, 1)
    seal = f.strict_seal(r.PREP)
    c.require(c.paths(r.PREP) == {"protocol.json", "requests.json", "metric_sources.json", REPORT, "complete.json"}, "Unknown E preparation member")
    p = c.load(r.PREP / "protocol.json"); req, gt = requests()
    c.require(p["implementation"] == r.implementation() and p["formal_implementation"] == f.implementation(), "E/formal implementation differs")
    c.verify_upstream(p["upstream"])
    c.require(c.load(r.PREP / "requests.json") == req and c.load(r.PREP / "metric_sources.json") == gt, "E request/GT metadata original order differs")
    c.require(p["score_rule"] == SCORE_RULE and p["threshold_rule"] == THRESHOLD_RULE and p["threshold"] == r.TAU and p["scalar_verification"] == scalar_verification(), "E frozen rules differ")
    trained = c.load(f.RUN / "result.json"); formal = c.load(f.PREP / "protocol.json")
    c.require(p["compatibility"] == formal["compatibility"] and p["seed"] == f.SEED and p["final_step"] == 10000, "E model configuration differs")
    c.require(p["final_weight_sha256"] == trained["final_weight_sha256"] and p["expected_state_hash"] == trained["final_state_hash"] and p["expected_buffers_hash"] == trained["initial_buffers_hash"] == trained["final_buffers_hash"], "E expected final model differs")
    c.require(p["images"] == len(req) == 1004 and p["planned_window_forwards"] == sum(x["window_count"] for x in req) and p["planned_native_pixels"] == sum(x["native_pixels"] for x in req), "E metadata denominators differ")
    c.require(seal["identity"] == {"meaning": "E metadata/hash/scalars only; no RGB/GT/probability/model operation"}, "E preparation meaning differs")
    return p, req

def prepare():
    if r.PREP.exists():
        p, req = verify_preparation()
        return preparation_summary(p, True)
    snapshot = c.resources(2, 2)
    c.require(not r.RUN.exists() and not r.RUNTIME.exists(), "Existing E score/runtime state before preparation")
    sources, formal, summary = r.upstream()
    for name in r.CODE:
        if name.endswith(".py"): ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"), filename=name)
    req, gt = requests(); trained = c.load(f.RUN / "result.json")
    p = {"schema": "pretrained-segmentation-e1004-scoring-preparation-v1", "implementation": r.implementation(),
         "formal_implementation": f.implementation(), "upstream": sources, "compatibility": formal["compatibility"],
         "seed": f.SEED, "final_step": 10000, "final_weight_sha256": summary["final_weight_sha256"],
         "expected_state_hash": trained["final_state_hash"], "expected_buffers_hash": trained["initial_buffers_hash"],
         "score_rule": SCORE_RULE, "threshold_rule": THRESHOLD_RULE, "threshold": r.TAU,
         "threshold_source": "normal_calibration_v1/threshold.json", "images": len(req),
         "planned_window_forwards": sum(x["window_count"] for x in req), "planned_native_pixels": sum(x["native_pixels"] for x in req),
         "scalar_verification": scalar_verification(), "resource_snapshot": snapshot,
         "Q_evidence_scope": "original complete/JSON/code hashes and prior full independent audit; Q probability payloads not re-read",
         "GT_scope": "paths/source hashes/native shape metadata only; no GT bytes or class/positive counts verified now",
         "cache_policy": "complete identical set read-only verify; partial/unknown/runtime residue preserve and stop; no automatic resume/repeat",
         "actual_E_scoring_or_metrics": False, "meaning": "E historically seen; exploratory only, not new unseen validation or screen benefit"}
    r.PREP.mkdir(parents=True, exist_ok=False)
    c.save(r.PREP / "protocol.json", p); c.save(r.PREP / "requests.json", req); c.save(r.PREP / "metric_sources.json", gt)
    c.save(r.PREP / REPORT, preparation_report(p))
    c.require(c.paths(r.PREP) == {"protocol.json", "requests.json", "metric_sources.json", REPORT} and not any(x.is_dir() for x in r.PREP.rglob("*")), "Unknown preparation member before seal")
    c.seal(r.PREP, {"meaning": "E metadata/hash/scalars only; no RGB/GT/probability/model operation"})
    verify_preparation()
    return preparation_summary(p, False)

def preparation_summary(p, reused):
    return {"E_scoring_ready": True, "reused": reused, "images_metadata": p["images"], "planned_windows": p["planned_window_forwards"],
            "native_pixels_metadata": p["planned_native_pixels"], "original_threshold": p["threshold"], "raw_RGB_GT_probability_reads": 0,
            "model_operations": 0, "E_scoring_or_metrics_executed": False}

def binding(p):
    return {"preparation_complete_sha256": c.sha(r.PREP / "complete.json"), "protocol_sha256": c.sha(r.PREP / "protocol.json"),
            "requests_sha256": c.sha(r.PREP / "requests.json"), "metric_sources_sha256": c.sha(r.PREP / "metric_sources.json"),
            "formal_model_complete_sha256": c.sha(f.MODEL / "complete.json"), "final_weight_sha256": p["final_weight_sha256"],
            "threshold_source_sha256": c.sha(c.BASE / p["threshold_source"]), "threshold": p["threshold"], "implementation": p["implementation"]}

def run():
    from .real_cache import verify_results
    if r.RUN.exists(): return verify_results()
    p, req = verify_preparation(); bound = binding(p)
    r.fresh_runtime(r.RUNTIME.exists()); snapshot = c.resources(4.5, 20, gpu=True)
    r.runtime_environment(); r.RUN.mkdir(parents=True, exist_ok=False)
    c.save(r.RUN / "transaction.json", {"binding": bound, "intended_images": 1004, "intended_windows": p["planned_window_forwards"],
                                        "resources_before_model": snapshot, "rule": "partial/uncertain forward preserve and stop; no automatic retry"})
    import time
    started = time.monotonic(); rows, resources = [], []; current = None
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
        c.require(before == p["expected_state_hash"] and buffers == p["expected_buffers_hash"], "E actual final model/BN differs")
        torch.cuda.reset_peak_memory_stats()
        (r.RUN / "probabilities").mkdir(exist_ok=False)
        for request in req:
            current = request["index"]
            snap = c.resources(2, 1); snap["image_index"] = current; resources.append(snap)
            raw = (c.DATA / request["image"]).read_bytes()
            c.require(hashlib.sha256(raw).hexdigest() == request["image_sha256"], "Actual E RGB identity differs")
            with Image.open(io.BytesIO(raw)) as im:
                c.require(im.mode in ("RGB", "L") and im.size == (request["width"], request["height"]), "E RGB dimensions/mode differ")
                rgb = np.asarray(im.convert("RGB")).copy()
            prob, forwards = predict_native(model, rgb)
            c.require(forwards == request["window_count"] and list(prob.shape) == request["native_shape"], "E actual window/native shape differs")
            name = f"probabilities/{current:04d}_{request['id']}.npy"
            with (r.RUN / name).open("xb") as handle:
                np.save(handle, prob, allow_pickle=False); handle.flush()
                import os
                os.fsync(handle.fileno())
            rows.append({"index": current, "id": request["id"], "source_image_sha256": request["image_sha256"], "probability": name,
                         "window_forwards": forwards, "probability_sha256": c.sha(r.RUN / name), "score": float(prob.max())})
            del raw, rgb, prob
            if (current + 1) % 25 == 0: print({"E_images": current+1, "of": 1004, "window_forwards": sum(x["window_forwards"] for x in rows)}, flush=True)
        torch.cuda.synchronize()
        result = {"binding": bound, "items": rows, "images_scored": len(rows), "window_forwards": sum(x["window_forwards"] for x in rows),
                  "constructor_forwards": 0, "raw_E_image_decodes": len(rows), "native_pixels": p["planned_native_pixels"],
                  "state_before": before, "state_after": state_hash(model.state_dict()), "buffers_before": buffers, "buffers_after": buffers_hash(model),
                  "GT_read_by_predictor": False, "training_or_parameter_updates": 0, "score_rule": SCORE_RULE,
                  "metrics_computed": False, "elapsed_seconds": time.monotonic()-started, "resource_samples": resources,
                  "sampled_min_free_RAM_GiB": min(x["free_RAM_GiB"] for x in resources),
                  "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated()/2**20, "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved()/2**20}
        c.require(result["state_before"] == result["state_after"] and result["buffers_before"] == result["buffers_after"], "E scoring changed parameters/BN")
        c.save(r.RUN / "scores.json", result)
        c.save(r.RUN / "E1004评分结果.md", f"# E1004探索性评分\n\n固定1004图／{result['window_forwards']}窗，原生{p['planned_native_pixels']}像素；同一最终10000步权重和原τ={p['threshold']:.17g}。只保存概率与max，未读GT、计算指标或重校准。E为历史已见探索集，不是新未见验证或屏幕效果。完整同身份缓存仅核验复用；部分／未知事务保留停止。\n")
        expected = {"transaction.json", "scores.json", "E1004评分结果.md"} | {x["probability"] for x in rows}
        c.require(c.paths(r.RUN) == expected and {x.relative_to(r.RUN).as_posix() for x in r.RUN.rglob("*") if x.is_dir()} == {"probabilities"}, "Unknown E result before seal")
        c.seal(r.RUN, bound)
    except BaseException as exc:
        if not (r.RUN / "failure.json").exists():
            c.save(r.RUN / "failure.json", {"error": repr(exc), "current_request_index": current, "completed_image_files": len(rows),
                                           "meaning": "possibly partial/uncertain forward; preserve; no automatic completion"})
        raise
    del model
    return verify_results()

def preparation_report(p):
    return f'''# 第6阶段E1004真实评分：CPU准备与人工运行

本次仅元信息／hash／窗口坐标／虚构标量与资源核验；未读取Q／E／U原图、GT或概率，未导入Torch／SMP／timm／模型、创建tensor、加载参数、前向、训练、生成、校准或计算指标。正式10000步／10保存点及独立核验证据已核对。Q完整校准以原封存JSON／完成清单及既有全量独立核验证据绑定，本次不重新读Q概率或取分位数。原τ={p['threshold']:.17g}，原同池FP10／TN190／200不是独立验证。

四个新增Python模块AST、显式CLI帮助、全部1004请求原序／整数坐标与严格大于、FP32等权、未知runtime拒绝的虚构标量核验通过；未新建软件测试。独立环境另以只查包元信息的进程确认27项版本与冻结清单一致，无模型导入。一次独立只读静态审阅未发现实质问题；静态核验不等于新E入口CUDA兼容或实际运行成功。

## 冻结输入与口径

E原split顺序1004张，计划{p['planned_window_forwards']}窗口前向、{p['planned_native_pixels']:,}原生像素；均是CPU元信息计数，非实际评分。请求仅含RGB路径／身份／原生几何，无GT或类别；metric_sources单独保存未来GT路径／来源hash／shape，不打开GT。当前不重新声称110／894或486461阳性已核验。每图／每预定窗各一次，尾窗去重、全行覆盖、右补边有效宽与原序均核对。

最终seed2026102202／10000步权重SHA `{p['final_weight_sha256']}`；只读复用原生predict_native，不另写拼回算法。单二分类U-Net、固定encoder BN／decoder GN8／FP32／batch1／workers0，RGB原始对比度与ImageNet归一化；256窗／stride128，右reflect补边剔除，y升序FP32累加及等权除法，整图分数只取保存原生float32异常softmax的max，无增强／平滑／拉伸。预测器不接收GT；实际运行前后参数／全部BN buffer须与正式最终／初始签名一致。τ仅作为绑定来源，不在评分时重新计算阈值或阈值计数。

## 已实现命令与输出

CPU准备核验（无模型／图像／概率读取）：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.real_scoring_cli verify
```

实际E评分入口已实现，本次未执行：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.real_scoring_cli run
```

未来完整评分只读核验为同一CLI的verify-results；它只核本组原始概率／max／身份，不读GT或计算AP／AUROC／TP等。独立real_cache提供未来CPU指标读取器iter_metric_inputs，先完整核验缓存后依原序返回概率及GT来源元信息，不打开GT、不计算指标；真实指标计算入口尚未实现，不能提前运行。

准备目录 `{r.PREP.as_posix()}`；实际结果 `{r.RUN.as_posix()}`；运行临时目录 `{r.RUNTIME.as_posix()}`。完整同身份缓存只读核验复用（新增前向0）；任何未知文件／目录／锁、身份差异、部分输出／failure或新运行前残留runtime均保留停止，无补推、恢复、删标记或成功交集缩分母。实际前向开始后出错不自动再次运行。

CPU准备RAM≥2GiB／磁盘≥2GiB；新评分启动RAM≥4.5GiB／空闲显存≥6GiB／磁盘≥20GiB，每图前RAM≥2GiB／磁盘≥1GiB；完整缓存核验RAM≥2GiB／磁盘≥1GiB。准备快照RAM{p['resource_snapshot']['free_RAM_GiB']:.3f}GiB、磁盘{p['resource_snapshot']['free_disk_GiB']:.3f}GiB，未来重新准入，不能用当前快照保证评分。仅以Q已测34.750秒／844窗线性外推，E评分段约{34.750*p['planned_window_forwards']/844:.0f}秒；不含完整前后hash核验等开销，非实测承诺。新E实际时间／峰值待运行。

E已被历史实验见过，未来只作单seed／固定R5与划分／单类别／未知采集分组的探索性评价，不称新未见验证或屏幕收益。不读取旧评价选配置，不自动进入指标或训练；全部训练继续由用户手动启动。旧源码／入口／环境／权重／τ／缓存与成果保持不变。
'''
