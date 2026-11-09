"""Fixed real-data experiment. Labels are used only after image-only inference."""
import ctypes
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from data import ROOT, NormalImages, read_image, read_mask
from detector import make_detector
from demo import memory, synchronize

MODEL_DIR = ROOT / "models" / "formal_patchcore"
RESULT_DIR = ROOT / "results"
CONFIG_PATH = MODEL_DIR / "configuration.json"
RESULT_PATH = RESULT_DIR / "real_evaluation.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def timestamp():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged",
                                                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not query(ctypes.c_void_p(-1), ctypes.byref(values), values.cb):
        raise OSError("Unable to query experiment process memory")
    return {"process_working_set_mib": values.ws / 2**20,
            "process_peak_working_set_mib": values.peak_ws / 2**20}


def fixed_protocol():
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    day1 = json.loads((RESULT_DIR / "day1.json").read_text(encoding="utf-8"))
    config = {key: value for key, value in day1["formal_configuration"].items()
              if key != "formal_bank_built"}
    paths = config["selected_paths"]
    if paths != split["splits"]["R"][:80] or len(paths) != config["R_selected"]:
        raise ValueError("Formal R differs from the already fixed Day 1 selection")
    selected = set(paths)
    if any(not p.startswith("train/good/") for p in paths):
        raise ValueError("Non-normal input in formal bank")
    if selected & (set(split["splits"]["V"]) | set(split["splits"]["P"])):
        raise ValueError("Normal bank overlaps V or P")
    if selected & {e["image"] for e in split["splits"]["E"]}:
        raise ValueError("Normal bank overlaps E")
    category = Path(split["category_root"])
    for p, expected in split["train_sha256"].items():
        if digest(category / p) != expected:
            raise ValueError("Normal source image changed: " + p)
    return split, day1, config


def infer(model, path):
    device = model.device.type
    synchronize(device)
    start = time.perf_counter()
    image = read_image(path).unsqueeze(0)
    scores, maps = model.predict(image)
    synchronize(device)
    elapsed = time.perf_counter() - start
    score = float(scores[0])
    heatmap = np.asarray(maps[0], dtype=np.float32)
    if heatmap.shape != (224, 224) or not np.isfinite(score) or not np.isfinite(heatmap).all():
        raise ValueError("Nonfinite or incorrectly shaped detector output")
    return score, heatmap, elapsed


def render_prediction(path, score, heatmap, threshold, display_upper, output):
    # Same V-derived scale for every image; visualization never changes inference scores.
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
    z = np.clip(heatmap / display_upper, 0, 1)
    colored = (255 * np.stack([z, np.zeros_like(z), 1-z], axis=-1)).astype(np.uint8)
    overlay = (0.6*rgb + 0.4*colored).astype(np.uint8)
    canvas = Image.new("RGB", (672, 304), "white")
    for j, value in enumerate((rgb, colored, overlay)):
        canvas.paste(Image.fromarray(value), (224*j, 0))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for x, label in [(6, "原图"), (230, "异常热图"), (454, "热图叠加")]:
        draw.text((x, 226), label, fill="black", font=font)
    decision = "异常" if score > threshold else "正常"
    draw.text((6, 250), f"判定：{decision}   分数：{score:.5f}   阈值：{threshold:.5f}", fill="black", font=font)
    draw.text((6, 276), "红色表示较高评分；整图分数严格大于阈值判异常。", fill="black", font=font)
    output = Path(output)
    try:
        canvas.save(output)
    except OSError as error:
        # Windows viewers may memory-map a PNG: truncation then raises EINVAL (22).
        # Preserve that displayed file and publish this prediction to a fresh path.
        if not output.exists() or error.errno not in (13, 22):
            raise
        output = output.with_name(f"{output.stem}_{uuid.uuid4().hex[:12]}{output.suffix}")
        canvas.save(output)
        print("原预览暂时无法覆盖，已保存新的结果图：", str(output), flush=True)
    return output


def check_saved_config(config):
    if config["split_sha256"] != digest(ROOT / "data" / "split.json"):
        raise ValueError("Fixed split changed after calibration")
    if config["index_sha256"] != digest(MODEL_DIR / "nnscorer_search_index.faiss"):
        raise ValueError("Formal memory bank changed after calibration")
    if config["weights_sha256"] != digest(ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"):
        raise ValueError("Backbone weight changed after calibration")


def load_formal(config):
    check_saved_config(config)
    model = make_detector(config["configuration"]["device"])
    # Author scorer loads the saved FAISS bank; no fitting, labels or paired backgrounds.
    model.anomaly_scorer.load(str(MODEL_DIR))
    if model.anomaly_scorer.nn_method.search_index.ntotal != config["bank_vectors"]:
        raise ValueError("Saved bank vector count mismatch")
    return model


def evaluate():
    if RESULT_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        check_saved_config(config)
        previous = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if previous["configuration_sha256"] != digest(CONFIG_PATH):
            raise ValueError("Completed evaluation and formal configuration differ")
        print("正式评价已完成；保留固定结果，不重复建库或评价。", flush=True)
        print(json.dumps(previous["metrics"], ensure_ascii=False, indent=2))
        return
    start_all = time.perf_counter()
    split, day1, settings = fixed_protocol()
    category = Path(split["category_root"])
    device = settings["device"]
    before = {**memory(), **process_memory()}
    if CONFIG_PATH.exists():
        # Resume only an already frozen normal bank and V threshold, never recalibrate from E.
        frozen = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        model = load_formal(frozen)
        validation_scores = np.asarray(frozen["validation_scores"], dtype=np.float64)
        fit_seconds = frozen["fit_seconds"]
        v_seconds = frozen["validation_seconds"]
        print("使用已保存且核验的正式库与V阈值。", flush=True)
    else:
        print("BUILD: fixed first80 R normals only", flush=True)
        model = make_detector(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        begin = time.perf_counter()
        with torch.no_grad():
            model.fit(DataLoader(NormalImages(category, settings["selected_paths"]), batch_size=1, num_workers=0))
        synchronize(device)
        fit_seconds = time.perf_counter() - begin
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_to_path(str(MODEL_DIR))
        after_fit = {**memory(), **process_memory()}
        print("CALIBRATE: V only, threshold=95th percentile (linear)", flush=True)
        v_scores, v_maps, v_times = [], [], []
        for p in split["splits"]["V"]:
            score, heatmap, seconds = infer(model, category / p)
            v_scores.append(score); v_maps.append(heatmap); v_times.append(seconds)
        validation_scores = np.asarray(v_scores, dtype=np.float64)
        threshold = float(np.quantile(validation_scores, 0.95, method="linear"))
        display_upper = float(np.quantile(np.stack(v_maps), 0.99, method="linear"))
        if display_upper <= 0 or not np.isfinite(display_upper):
            raise ValueError("Invalid V-derived display scale")
        v_seconds = sum(v_times)
        frozen = {"configuration": settings, "source_commit": day1["source_commit"],
                  "split_sha256": digest(ROOT / "data" / "split.json"),
                  "weights_sha256": digest(ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"),
                  "index_sha256": digest(MODEL_DIR / "nnscorer_search_index.faiss"),
                  "bank_vectors": int(model.anomaly_scorer.nn_method.search_index.ntotal),
                  "threshold": threshold, "threshold_quantile": 0.95, "quantile_method": "linear",
                  "decision": "score > threshold", "validation_paths": split["splits"]["V"],
                  "validation_scores": v_scores, "validation_seconds": v_seconds,
                  "validation_above_threshold": int(np.sum(validation_scores > threshold)),
                  "display_scale": [0.0, display_upper], "display_scale_source": "99th percentile of all V heatmap pixels",
                  "frozen_at": timestamp(), "frozen_before_E_inference": True,
                  "fit_seconds": fit_seconds, "resources_after_fit": after_fit,
                  "formal_bank_built": True}
        # This file is committed to disk BEFORE the first E prediction.
        write_json(CONFIG_PATH, frozen)
        del v_maps
    config_sha = digest(CONFIG_PATH)
    threshold = frozen["threshold"]
    print(f"FROZEN: threshold={threshold:.8f}, bank={frozen['bank_vectors']}; now evaluate all E", flush=True)
    scores, maps, masks, labels, rows, times = [], [], [], [], [], []
    for i, item in enumerate(split["splits"]["E"]):
        # Detector only sees this image. The independent label is read afterwards.
        score, heatmap, elapsed = infer(model, category / item["image"])
        label = int(item["kind"] != "good")
        mask = read_mask(category / item["mask"] if item["mask"] else None)
        scores.append(score); maps.append(heatmap); masks.append(mask); labels.append(label); times.append(elapsed)
        rows.append({**item, "label": label, "score_raw": score,
                     "predicted_anomaly": bool(score > threshold), "inference_seconds": elapsed,
                     "image_sha256": digest(category / item["image"])})
        if (i+1) % 20 == 0 or i+1 == len(split["splits"]["E"]):
            print(f"E inference {i+1}/{len(split['splits']['E'])}", flush=True)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    maps = np.stack(maps); masks = np.stack(masks)
    predicted = scores > threshold
    positives = labels == 1
    n_anomaly, n_good = int(positives.sum()), int((~positives).sum())
    tp = int(np.sum(predicted & positives)); fn = n_anomaly-tp
    fp = int(np.sum(predicted & ~positives)); tn = n_good-fp
    metric_start = time.perf_counter()
    metrics = {"image_auroc": float(roc_auc_score(labels, scores)),
               "anomaly_recall": tp/n_anomaly, "anomaly_detected": tp, "anomaly_total": n_anomaly,
               "anomaly_missed": fn, "normal_false_positive_rate": fp/n_good,
               "normal_false_positives": fp, "normal_total": n_good, "normal_true_negatives": tn,
               "pixel_average_precision": float(average_precision_score(masks.ravel(), maps.ravel())),
               "pixel_total": int(masks.size), "pixel_positive": int(masks.sum()),
               "pixel_negative": int(masks.size-masks.sum()),
               "test_images_total": int(labels.size)}
    metric_seconds = time.perf_counter()-metric_start
    grouped = {}
    for kind in sorted({row["kind"] for row in rows}):
        group = [row for row in rows if row["kind"] == kind]
        grouped[kind] = {"total": len(group), "predicted_anomaly": sum(row["predicted_anomaly"] for row in group)}
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(RESULT_DIR / "real_evaluation.npz", validation_scores=validation_scores,
                        test_scores=scores, test_image_labels=labels, test_maps=maps, test_pixel_labels=masks,
                        test_image_paths=np.array([row["image"] for row in rows]))
    example_dir = RESULT_DIR / "real_examples"
    example_dir.mkdir(exist_ok=True)
    # Fixed path-order examples, not cherry-picked by the resulting scores.
    examples = [next(i for i, row in enumerate(rows) if row["kind"] == kind) for kind in ("good", "color")]
    for i in examples:
        name = "normal.png" if rows[i]["kind"] == "good" else "anomaly.png"
        render_prediction(category / rows[i]["image"], scores[i], maps[i], threshold,
                          frozen["display_scale"][1], example_dir / name)
    result = {"dataset": "MVTec AD leather", "usage": "personal learning and non-commercial research",
              "scope": "public industrial category; does not measure display or Mura detection",
              "completed_at": timestamp(), "configuration_sha256": config_sha,
              "source_commit": day1["source_commit"], "split_sha256": frozen["split_sha256"],
              "data_archive": day1["data_archive"], "configuration": settings,
              "bank_vectors": frozen["bank_vectors"], "bank_dimension": 1024,
              "threshold": threshold, "threshold_protocol": "V only, numpy.quantile(q=0.95,method='linear'); score > threshold",
              "validation_count": len(validation_scores), "validation_above_threshold": frozen["validation_above_threshold"],
              "metrics": metrics, "by_kind": grouped,
              "missed_anomalies": [row["image"] for row in rows if row["label"] and not row["predicted_anomaly"]],
              "false_positive_normals": [row["image"] for row in rows if not row["label"] and row["predicted_anomaly"]],
              "preprocessing": "whole-image RGB bilinear resize 224x224, /255, ImageNet fixed mean/std; no crop or per-image contrast stretch",
              "pixel_label_processing": "whole-image L mask nearest resize 224x224; value>0 positive; normal images all-zero",
              "pixel_ap_protocol": "micro/global AP over every pixel from all 124 E images, including all 32 normal images; sklearn noninterpolated average_precision_score; raw author smoothed 224x224 maps, no min/max normalization",
              "image_score_protocol": "author max nearest-neighbour squared L2 patch distance; raw score, no batch normalization",
              "display_scale": frozen["display_scale"], "display_scale_source": frozen["display_scale_source"],
              "timings": {"formal_fit_seconds": fit_seconds, "validation_inference_seconds": v_seconds,
                          "test_inference_seconds": sum(times), "test_mean_single_image_seconds": float(np.mean(times)),
                          "metric_seconds": metric_seconds, "total_run_seconds": time.perf_counter()-start_all,
                          "scope": "fit includes read/preprocess/extract/coreset/FAISS but excludes backbone initialization and bank serialization; V/E inference includes read/preprocess/extract/query/map and CUDA synchronization; total starts inside evaluate() after Python startup/module imports, includes backbone initialization, metrics and artifact writing before final JSON/report"},
              "resources": {"before": before, "after_fit": frozen["resources_after_fit"],
                            "after_evaluation": {**memory(), **process_memory()}},
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "torchvision", "numpy", "faiss-cpu", "timm", "scipy", "scikit-learn")},
              "test_predictions": rows}
    if device == "cuda":
        result["resources"].update(gpu_name=torch.cuda.get_device_name(),
                                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                                  gpu_memory_scope="PyTorch process allocations/reservations; excludes other apps and some driver memory")
    if digest(CONFIG_PATH) != config_sha:
        raise ValueError("Configuration changed during E evaluation")
    write_json(RESULT_PATH, result)
    write_report(result)
    print(json.dumps({"threshold": threshold, "metrics": metrics, "timings": result["timings"]}, ensure_ascii=False, indent=2))


def write_report(result):
    m = result["metrics"]; t = result["timings"]; r = result["resources"]
    calibration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v_scores = np.asarray(calibration["validation_scores"])
    normal_scores = np.asarray([row["score_raw"] for row in result["test_predictions"] if row["label"] == 0])
    anomaly_scores = np.asarray([row["score_raw"] for row in result["test_predictions"] if row["label"] == 1])
    group_rows = "\n".join(f"| {kind} | {g['total']} | {g['predicted_anomaly']} | {g['total']-g['predicted_anomaly']} |" for kind,g in result["by_kind"].items())
    fn = "、".join(result["missed_anomalies"]) or "无"
    fp = "、".join(result["false_positive_normals"]) or "无"
    text = f"""# MVTec AD leather 真实基线评价

评价完成：{result['completed_at']}。私人设备、个人学习与非商业研究。**结果只支持公开工业 leather 类别，未评价显示屏或 Mura。**

## 固定实验条件

数据包、来源、许可证和第1天资源条件见[Demo运行说明](../README.md)。作者PatchCore固定提交 `{result['source_commit']}`，Apache-2.0；骨干为ImageNet V1 WideResNet50-2。正常训练源245张，R/V/P分别176/49/20张，正式库只用已固定R前80张；P、V、E及所有异常图均未参与建库。单骨干、layer2+layer3、patch3、1024维嵌入、1% approximate greedy coreset（投影128、起始10）、近邻1、批量1，不微调、不集成。正式库有 **{result['bank_vectors']}个1024维向量**。GPU提取特征和coreset，CPU FAISS检索。

初始化口径也沿用固定作者源码：[NetworkFeatureAggregator.feature_dimensions](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py)在正式eval前用全1占位输入查询维度，此时BatchNorm运行统计会更新一次；本机核对bn1的running_mean与原权重最大绝对差约0.363507。之后正式图像建库与推理均eval且无梯度，没有利用R/V/P/E图像更新骨干统计或梯度训练。这里的ImageNet V1指权重来源，不能理解为所有BN缓冲量完整保持原值。本轮第1天配置、正式评价和重新加载均使用同一作者初始化行为，分数可重复；未在看到E后切换初始化方式。该实现细节不支持声称完整论文复现，未来如比较严格保持原始BN状态的初始化，应另定独立实验协议，不能替换当前固定结果。

先建库，再用49张V正常图的原始整图分数固定95%分位数，`numpy.quantile(method='linear')`，得到 **阈值={result['threshold']:.8f}**。严格大于阈值判异常，相等判正常。V中高于阈值 **{result['validation_above_threshold']}/49**；名义5%尾部不等于有限样本或独立E恰有5%误报。库和阈值已在首张E推理前保存，没有用E调参或改动资源方案。

图像由RGB原图整图双线性缩放224×224，像素/255后用固定ImageNet均值和标准差；不裁剪、不逐图增强。该小规模配置与作者示例的256缩放后中心裁剪不同，不称完整论文复现。异常掩码在同一全图范围用最近邻缩放224×224，以非零值为异常；正常图标签为全零。缩小图像可能损失细小缺陷，像素评价在224×224进行，不能等同原分辨率定位精度。

## 完整独立E评价

| 指标 | 结果与分母 |
|---|---|
| 整图AUROC | {m['image_auroc']:.6f}（92张异常对32张正常；E共124张） |
| 固定阈值异常召回 | {m['anomaly_detected']}/{m['anomaly_total']} = {m['anomaly_recall']:.2%} |
| 漏检 | {m['anomaly_missed']}/{m['anomaly_total']} |
| 正常误报 | {m['normal_false_positives']}/{m['normal_total']} = {m['normal_false_positive_rate']:.2%} |
| 正常正确排除 | {m['normal_true_negatives']}/{m['normal_total']} |
| 补充像素AP | {m['pixel_average_precision']:.6f} |

整图分数为作者实现中各patch最近邻平方L2距离的最大值；AUROC用原始整图分数，不用阈值后二值输出。像素AP使用作者上采样／平滑后的原始224×224热图，采用scikit-learn非插值average_precision_score：**全部124张图的像素展平后统一计算**，包括32张正常图，不是逐图AP平均，也不是PR曲线梯形积分。总像素 **{m['pixel_total']:,}**，其中异常 **{m['pixel_positive']:,}**、正常 **{m['pixel_negative']:,}**。未做V/E分批min-max归一化，也未优化像素阈值。

| 类别 | 图数 | 判异常 | 判正常 |
|---|---:|---:|---:|
{group_rows}

漏检路径：{fn}。

正常误报路径：{fp}。

这些样例可供后续解释，当前参数已固定，不据此改动本轮检测器。高AUROC和固定阈值下的召回／误报回答不同问题，不能相互替代。

**本轮主要问题是正常误报较高。**V原始分数中位数为{np.median(v_scores):.6f}，E正常图中位数为{np.median(normal_scores):.6f}，后者高于已固定阈值；V中高于阈值3/49，而E正常图为21/32。E正常最高分{normal_scores.max():.6f}，异常最低分{anomaly_scores.min():.6f}，这解释了排序AUROC为1与当前阈值误报率65.625%同时出现。它说明本次正常验证阈值在独立正常测试图上的泛化不足，不能把AUROC为1描述成总体判定完美。具体差异来自何种纹理、成像或建模因素尚未验证，不归因于已证实的光照变化，也不根据E重新找阈值。

## 单图展示与运行条件

默认示例按路径顺序预先指定正常good与异常color各第一张，不挑成绩最好样例。每张展示原图、热图、叠加以及整图阈值判定。所有图共用V全部热图像素99%分位数确定的显示范围 **[0,{result['display_scale'][1]:.6f}]**；显示裁切不改变原始评分，红色区域不能直接当像素缺陷判定。异常图超过这个正常参考上限的区域会饱和为红色，因此预览可能不能清楚区分高分区域内部强度；完整原始热图保存在NPZ中，像素AP未使用显示裁切后的图。作者热图采用双线性上采样和sigma=4高斯平滑。

![正常示例](real_examples/normal.png)

![异常示例](real_examples/anomaly.png)

设备为i7-14650HX、约15.62GiB物理内存、RTX5060 Laptop约8GiB显存，驱动595.97。软件实际版本：{', '.join(k+'='+v for k,v in result['versions'].items())}。

- 正式80张建库：{t['formal_fit_seconds']:.3f}秒；V共49张读取／推理：{t['validation_inference_seconds']:.3f}秒。
- E共124张读取／推理：{t['test_inference_seconds']:.3f}秒，单张均值{t['test_mean_single_image_seconds']:.4f}秒；指标计算{t['metric_seconds']:.3f}秒。
- 从evaluate函数开始至最终JSON／报告写入前累计：{t['total_run_seconds']:.3f}秒，不含Python启动和模块导入。含骨干初始化、库文件保存、指标及其他产物写入，不能只把推理均值称生产吞吐量。
- 建库计时包含读图／预处理／特征／coreset／FAISS，不含骨干初始化和库文件写入。推理计时包含读图／预处理／特征／CPU检索／热图，CUDA已同步，不含GT读取和指标计算。
- 当前进程峰值工作集：{r['after_evaluation']['process_peak_working_set_mib']:.1f}MiB；这不是整机所有应用占用。运行前可用RAM{r['before']['ram_available_gib']:.2f}GiB，建库后{r['after_fit']['ram_available_gib']:.2f}GiB。
- PyTorch显存峰值allocated {r.get('peak_gpu_allocated_mib',0):.1f}MiB、reserved {r.get('peak_gpu_reserved_mib',0):.1f}MiB，不含其他应用及全部驱动内存。

## 成果和局限

正式库及阈值保存在 `models/formal_patchcore/`；JSON保存协议、分母、逐图最终评分和误判路径，`real_evaluation.npz`保存V分数、E评分／热图／变换后标签，支持复核。这些是最终科学结果，不是每日记录。单图入口见[运行说明](../README.md)。

Windows环境下FAISS原生文件接口无法写入本项目中文路径，目录存在仍报找不到路径，与[FAISS官方仓库已报告的Unicode路径问题](https://github.com/facebookresearch/faiss/issues/3073)一致。封装层只将索引读写改为FAISS serialize_index/deserialize_index加Python二进制文件读写，[官方I/O说明](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)支持内存字节序列化；作者核心和索引类型／距离／抽样未改。保存后实际通过单图入口重新加载索引，预先指定color/000的评分和完整评价时一致。

本轮仅一个工业类别、一个固定配置，没有训练收益对照、置信区间或公司材料。该数据的纹理、成像和缺陷分布与屏幕亮度不均／低对比Mura不同；本轮数据不能证明屏幕检出率、产线速度或真实Mura的适用性。第3天合成尚未执行，也不能据此宣称生成数据提升检测效果。
"""
    (RESULT_DIR / "真实评价结果.md").write_text(text, encoding="utf-8")


def predict_single(image_path, output_dir):
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("Formal bank and V threshold missing; run demo.py evaluate first")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    # Only the explicitly supplied image is opened; no directory scan or label lookup.
    model = load_formal(config)
    score, heatmap, elapsed = infer(model, image_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preview_path = render_prediction(image_path, score, heatmap, config["threshold"],
                                     config["display_scale"][1], output / "preview.png")
    np.save(output / "score_map.npy", heatmap)
    result = {"image": str(image_path), "score_raw": score, "threshold": config["threshold"],
              "predicted_anomaly": bool(score > config["threshold"]), "decision": "score > threshold",
              "inference_seconds": elapsed, "configuration_sha256": digest(CONFIG_PATH),
              "display_scale": config["display_scale"], "preview_path": str(preview_path),
              "note": "single image only; no label or paired background used"}
    write_json(output / "prediction.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PREVIEW", str(preview_path))
