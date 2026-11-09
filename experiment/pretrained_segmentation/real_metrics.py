"""Explicit independent CPU real metrics. Preparation never decodes GT or arrays."""
from __future__ import annotations
import ast
import hashlib
import io
from pathlib import Path
import time
import numpy as np
from . import common as c
from . import formal_common as f
from . import real_scoring_common as r
from . import real_metrics_common as m

REPORT = "第6阶段独立CPU真实指标准备与运行.md"

def rules():
    return {"GT": "fixed source hash before single decode; mode1/L; exact native shape; values subset of {0,1} OR {0,255}; nonzero->1; no resize/dilation",
            "pixel_AP": "frozen fewshot_synthetic.metrics.exact_ap; all native pixels once; no sampling, padding or mean per-image AP",
            "ties": "float32 exact bit-value grouping; -0 normalized to +0; frozen descending scan; native N and actual P plus independent table count/tie checks",
            "image_AUROC": "sklearn roc_auc_score on all1004 original native maxima; only defined with both GT classes",
            "classification": "score strictly greater than original Q threshold; TP/FN on actual anomaly, FP/TN on actual normal; equality is normal",
            "empty": "NA/null with reason for zero denominator/no positives/single-class AUROC; never silently omit images",
            "examples": "first ID in original ascending E order for each TP/FN/FP/TN; empty group NA; deduplicate and retain all reasons",
            "reuse": "complete same identity read-only verification; partial/unknown/failure preserve and stop; no resume/reaccumulation/retry",
            "resource": {"start_RAM_GiB": 3, "start_disk_GiB": 80, "checks": "entry, each GT, each AP chunk, table-check chunks; frozen exact scan has entry gate only, not continuous monitoring"}}

def scalar_validation():
    c.require(m.BINS == 0x3f800001 and m.CHUNK == 1048576, "Frozen AP geometry differs")
    c.require([s > .5 for s in [0., .5, 1.]] == [False, False, True], "Strict > scalar differs")
    # Invented tied-score group arithmetic only, NOT exact_ap/sklearn or real scores.
    c.require(abs((.5*.5 + .5*(2/3)) - 7/12) < 1e-15, "Invented tied-group arithmetic differs")
    c.require(m.BINS*8 == 8522825736 and m.PIXELS < 2**32, "Exact table capacity/size differs")
    m.require_closed_members({"complete.json", "scores.json", "probabilities/toy.npy"}, {"probabilities"}, {"scores.json", "probabilities/toy.npy"})
    rejected = 0
    for extra_file, extra_dir in [("failure.json", None), (None, "unknown_lock")]:
        try:
            m.require_closed_members({"complete.json", "scores.json", "probabilities/toy.npy"} | ({extra_file} if extra_file else set()),
                                     {"probabilities"} | ({extra_dir} if extra_dir else set()), {"scores.json", "probabilities/toy.npy"})
        except RuntimeError: rejected += 1
    c.require(rejected == 2, "Invented unknown file/empty directory must stop")
    return {"strict_greater_and_tied_group_arithmetic": True, "two_table_bytes": m.BINS*8, "table_count_uint32_capacity_sufficient": True,
            "invented_unknown_file_and_empty_directory_rejected": True,
            "actual_GT_decodes": 0, "probability_payloads_parsed": 0, "actual_AP_AUROC_counts": False}

def verify_preparation():
    m.resource(2, 1)
    seal = f.strict_seal(m.PREP)
    c.require(c.paths(m.PREP) == {"protocol.json", "sources.json", "GT_requests.json", REPORT, "complete.json"}, "Unknown metric preparation member")
    p = c.load(m.PREP / "protocol.json")
    c.require(p["implementation"] == m.implementation() and p["rules"] == rules() and p["scalar_validation"] == scalar_validation() and p["cpu_environment"] == m.cpu_environment(), "Metric implementation/rules/CPU environment differs")
    m.source_trees()
    sources = c.load(m.PREP / "sources.json"); c.verify_upstream(sources)
    gt = c.load(m.PREP / "GT_requests.json")
    c.require(gt == c.load(r.PREP / "metric_sources.json") and len(gt) == 1004 and sum(x["native_shape"][0]*x["native_shape"][1] for x in gt) == m.PIXELS, "Metric fixed GT metadata differs")
    c.require(p["threshold"] == r.TAU == c.load(c.BASE / "normal_calibration_v1/threshold.json")["threshold"] and p["native_pixels"] == m.PIXELS, "Original threshold or native population differs")
    c.require(seal["identity"] == {"meaning": "CPU metadata/hash/scalars only; GT not decoded, E probabilities not parsed, metrics not computed"}, "Preparation meaning differs")
    return p, gt

def prepare():
    if m.PREP.exists():
        p, _ = verify_preparation(); return summary(p, True)
    snapshot = m.resource(3, 80)
    c.require(not m.RUN.exists(), "Unknown metric output before preparation")
    ep, req, gt, scores, sources, reference = m.source_context()
    for name in m.CODE:
        if name.endswith(".py"): ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"), filename=name)
    p = {"schema": "pretrained-segmentation-real-cpu-metrics-preparation-v1", "implementation": m.implementation(),
         "cpu_environment": m.cpu_environment(), "threshold": r.TAU, "native_pixels": m.PIXELS, "unique_images": 1004,
         "final_step": 10000, "final_weight_sha256": ep["final_weight_sha256"], "rules": rules(), "scalar_validation": scalar_validation(),
         "probe_crosscheck": reference, "resource_snapshot": snapshot, "GT_class_and_positive_denominators": "unknown until explicit CPU compute decodes all GT",
         "output": str(m.RUN), "scope": "E historically seen, single seed/fixed R5/split/class, unknown acquisition groups; exploratory only, no screen claim"}
    m.PREP.mkdir(parents=True, exist_ok=False)
    c.save(m.PREP / "protocol.json", p); c.save(m.PREP / "sources.json", sources); c.save(m.PREP / "GT_requests.json", gt)
    c.save(m.PREP / REPORT, preparation_report(p))
    c.require(c.paths(m.PREP) == {"protocol.json", "sources.json", "GT_requests.json", REPORT} and not any(x.is_dir() for x in m.PREP.rglob("*")), "Unknown preparation member before seal")
    c.seal(m.PREP, {"meaning": "CPU metadata/hash/scalars only; GT not decoded, E probabilities not parsed, metrics not computed"})
    verify_preparation(); return summary(p, False)

def summary(p, reused):
    return {"CPU_real_metrics_ready": True, "reused": reused, "unique_E_metadata_images": 1004, "native_pixels_metadata": m.PIXELS,
            "original_threshold": p["threshold"], "actual_GT_decodes": 0, "probability_payloads_parsed": 0, "actual_metrics_computed": False}

def binding(p):
    return {"metric_preparation_complete_sha256": c.sha(m.PREP / "complete.json"), "protocol_sha256": c.sha(m.PREP / "protocol.json"),
            "source_manifest_sha256": c.sha(m.PREP / "sources.json"), "GT_requests_sha256": c.sha(m.PREP / "GT_requests.json"),
            "scoring_complete_sha256": c.sha(r.RUN / "complete.json"), "implementation": p["implementation"],
            "final_weight_sha256": p["final_weight_sha256"], "original_threshold_sha256": c.sha(c.BASE / "normal_calibration_v1/threshold.json"), "threshold": p["threshold"]}

def bound_bytes(path, digest):
    payload = Path(path).read_bytes()
    c.require(hashlib.sha256(payload).hexdigest() == digest, "Source byte snapshot changed: " + str(path))
    return payload

def bound_array(path, digest):
    return np.load(io.BytesIO(bound_bytes(path, digest)), allow_pickle=False)

def decode_gt(request):
    from PIL import Image
    raw = bound_bytes(c.DATA / request["mask"], request["mask_sha256"])
    with Image.open(io.BytesIO(raw)) as im:
        h, w = request["native_shape"]
        c.require(im.mode in ("1", "L") and im.size == (w, h), "GT mode/native shape differs; no resizing")
        y = np.asarray(im)
        c.require(y.ndim == 2 and (np.all((y == 0)|(y == 1)) or np.all((y == 0)|(y == 255))), "GT must be exact binary0/1 or0/255")
        return (y != 0).astype(np.uint8)

def save_labels(gt):
    (m.RUN / "labels").mkdir(exist_ok=False); rows = []
    for request in gt:
        m.resource(3, 80)
        y = decode_gt(request)
        name = f"labels/{request['index']:04d}_{request['id']}.npy"
        with (m.RUN / name).open("xb") as handle:
            np.save(handle, y, allow_pickle=False); handle.flush()
            import os
            os.fsync(handle.fileno())
        rows.append(request | {"cache": name, "cache_sha256": c.sha(m.RUN / name), "pixels": int(y.size), "positive_pixels": int(y.sum()), "anomaly": bool(y.any())})
    n = sum(x["pixels"] for x in rows); positive = sum(x["positive_pixels"] for x in rows); anomaly = sum(x["anomaly"] for x in rows)
    c.require(len(rows) == 1004 and n == m.PIXELS, "Actual native GT population differs; no intersection/subset")
    result = {"items": rows, "unique_images": 1004, "native_pixels": n, "positive_pixels": positive, "anomaly_images": anomaly, "normal_images": 1004-anomaly, "original_GT_decodes": 1004}
    c.save(m.RUN / "labels/summary.json", result); return result

def blocks(labels, scores):
    for label, row in zip(labels["items"], scores["items"]):
        m.resource(3, 80)
        c.require(label["id"] == row["id"] and label["index"] == row["index"], "GT/probability order differs")
        y = bound_array(m.RUN / label["cache"], label["cache_sha256"])
        prob = bound_array(r.RUN / row["probability"], row["probability_sha256"])
        c.require(y.dtype == np.uint8 and prob.dtype == np.float32 and list(y.shape) == list(prob.shape) == label["native_shape"] and np.all((y==0)|(y==1)) and int(y.sum()) == label["positive_pixels"], "Native GT/probability pairing differs")
        c.require(np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1 and float(prob.max()) == row["score"], "Saved raw probability/max differs")
        s = prob.reshape(-1); target = y.reshape(-1)
        for start in range(0, s.size, m.CHUNK):
            m.resource(3, 80)
            yield s[start:start+m.CHUNK], target[start:start+m.CHUNK]

def table_counts(exact, labels):
    """Independent counts/ties invariant scan, not another AP calculation."""
    total = np.memmap(m.RUN / "exact/counts.uint32", dtype="<u4", mode="r", shape=(m.BINS,))
    positive = np.memmap(m.RUN / "exact/positive.uint32", dtype="<u4", mode="r", shape=(m.BINS,))
    n = p = distinct = ties = 0
    for start in range(0, m.BINS, m.CHUNK):
        m.resource(2, 1)
        t = np.asarray(total[start:start+m.CHUNK]); y = np.asarray(positive[start:start+m.CHUNK])
        c.require(np.all(y <= t), "Exact positive count exceeds total")
        n += int(t.sum(dtype=np.uint64)); p += int(y.sum(dtype=np.uint64)); distinct += int((t > 0).sum()); ties += int((t > 1).sum())
    del t, y, total, positive
    c.require(n == m.PIXELS == exact["pixels"] and p == labels["positive_pixels"] == exact["positive_pixels"] and distinct == exact["distinct_scores"] and ties == exact["tied_score_values"], "Exact global population/ties invariant differs")
    return {"pixels": n, "positive_pixels": p, "distinct_scores": distinct, "tied_score_values": ties, "positive_not_above_total": True, "AP_recomputed": False}

def image_metrics(labels, scores, tau):
    from sklearn.metrics import roc_auc_score
    rows = []; counts = {key: 0 for key in ["TP", "FN", "FP", "TN"]}; selected = []
    first = {key: None for key in counts}
    for label, score in zip(labels["items"], scores["items"]):
        c.require(label["id"] == score["id"] and label["index"] == score["index"], "Image metric order differs")
        anomaly = label["anomaly"]; predicted = score["score"] > tau
        kind = ("TP" if predicted else "FN") if anomaly else ("FP" if predicted else "TN")
        counts[kind] += 1
        row = {"index": label["index"], "id": label["id"], "anomaly": anomaly, "score": score["score"], "predicted_anomaly": predicted, "outcome": kind}
        rows.append(row)
        if first[kind] is None:
            first[kind] = label["id"]; selected.append({"index": label["index"], "id": label["id"], "reasons": ["first " + kind + " in frozen original ascending E order"]})
    npos, nneg = labels["anomaly_images"], labels["normal_images"]
    c.require(counts["TP"]+counts["FN"] == npos and counts["FP"]+counts["TN"] == nneg and npos+nneg == 1004, "Image denominators differ")
    auc = float(roc_auc_score([x["anomaly"] for x in rows], [x["score"] for x in rows])) if npos and nneg else None
    return {"AUROC": auc, "AUROC_NA_reason": None if auc is not None else "GT contains fewer than two image classes", "threshold": tau,
            **counts, "anomaly_denominator": npos, "normal_denominator": nneg, "anomaly_recall": counts["TP"]/npos if npos else None,
            "normal_false_positive_rate": counts["FP"]/nneg if nneg else None, "zero_denominator_policy": "NA/null", "all_items": rows,
            "first_by_outcome": first, "selected_examples": selected}

def compute():
    if m.RUN.exists(): return verify_results()
    p, gt = verify_preparation(); bound = binding(p); snapshot = m.resource(3, 80)
    m.RUN.mkdir(parents=True, exist_ok=False)
    c.save(m.RUN / "transaction.json", {"binding": bound, "resources": snapshot, "rule": "no resume/reaccumulate; partial/unknown preserve and stop"})
    started = time.monotonic()
    try:
        # A future explicit compute alone authorizes GT decode and probability parsing.
        from .real_cache import verify_results as verify_scores
        verified = verify_scores()
        labels = save_labels(gt)
        score_digest = c.load(m.PREP / "sources.json")[(r.RUN / "scores.json").relative_to(c.ROOT).as_posix()]
        scores = __import__('json').loads(bound_bytes(r.RUN / "scores.json", score_digest).decode("utf-8"))
        exact_identity = bound | {"labels_summary_sha256": c.sha(m.RUN / "labels/summary.json")}
        exact = m.exact_ap(lambda: blocks(labels, scores), m.RUN / "exact", exact_identity, m.PIXELS, labels["positive_pixels"])
        invariant = table_counts(exact, labels)
        images = image_metrics(labels, scores, p["threshold"])
        result = {"binding": bound, "unique_E_images": 1004, "native_pixels": m.PIXELS, "positive_pixels": labels["positive_pixels"],
                  "anomaly_images": labels["anomaly_images"], "normal_images": labels["normal_images"], "global_pixel_AP": exact["AP"],
                  "pixel_AP_NA_reason": None if labels["positive_pixels"] else "no GT-positive native pixels", "exact": exact, "exact_count_tie_verification": invariant,
                  "image_metrics": images, "selected_examples": images["selected_examples"], "score_cache_verification": verified,
                  "elapsed_seconds": time.monotonic()-started, "resources_at_entry": snapshot, "scope": p["scope"],
                  "new_forward_training_or_recalibration": 0}
        c.save(m.RUN / "metrics.json", result)
        c.save(m.RUN / "第6阶段CPU真实指标结果.md", result_report(result))
        expected = {"transaction.json", "metrics.json", "第6阶段CPU真实指标结果.md", "labels/summary.json"} | {x["cache"] for x in labels["items"]} | {"exact/"+x for x in ["transaction.json", "counts.uint32", "positive.uint32", "metrics.json", "complete.json"]}
        c.require(c.paths(m.RUN) == expected and {x.relative_to(m.RUN).as_posix() for x in m.RUN.rglob("*") if x.is_dir()} == {"exact", "labels"}, "Unknown metric member before seal")
        c.seal(m.RUN, bound)
    except BaseException as exc:
        if not (m.RUN / "failure.json").exists(): c.save(m.RUN / "failure.json", {"error": repr(exc), "rule": "preserve; do not retry/reaccumulate/approximate"})
        raise
    return verify_results()

def verify_results():
    p, gt = verify_preparation(); bound = binding(p)
    c.require(f.strict_seal(m.RUN)["identity"] == bound, "Complete metric binding differs")
    result = c.load(m.RUN / "metrics.json"); labels = c.load(m.RUN / "labels/summary.json")
    c.require(result["binding"] == bound and result["unique_E_images"] == labels["unique_images"] == 1004 and result["native_pixels"] == labels["native_pixels"] == m.PIXELS, "Metric complete population differs")
    c.require(result["positive_pixels"] == labels["positive_pixels"] == sum(x["positive_pixels"] for x in labels["items"]), "Metric actual positive denominator differs")
    c.require(result["anomaly_images"] == labels["anomaly_images"] == sum(x["anomaly"] for x in labels["items"]) and result["normal_images"] == labels["normal_images"] == 1004-labels["anomaly_images"], "Metric image denominators differ")
    exact_identity = bound | {"labels_summary_sha256": c.sha(m.RUN / "labels/summary.json")}
    c.require(f.strict_seal(m.RUN / "exact")["identity"] == exact_identity and c.load(m.RUN / "exact/metrics.json") == result["exact"] and result["global_pixel_AP"] == result["exact"]["AP"], "Metric exact result differs")
    expected = {"transaction.json", "metrics.json", "第6阶段CPU真实指标结果.md", "complete.json", "labels/summary.json"} | {f"labels/{x['index']:04d}_{x['id']}.npy" for x in gt} | {"exact/"+x for x in ["transaction.json", "counts.uint32", "positive.uint32", "metrics.json", "complete.json"]}
    c.require(c.paths(m.RUN) == expected and len(labels["items"]) == 1004, "Unknown/missing metric cache members")
    for source, row in zip(gt, labels["items"]):
        c.require(all(row[k] == v for k,v in source.items()) and row["cache"] == f"labels/{source['index']:04d}_{source['id']}.npy", "Normalized GT source/order differs")
        y = bound_array(m.RUN / row["cache"], row["cache_sha256"])
        c.require(y.dtype == np.uint8 and list(y.shape) == source["native_shape"] and np.all((y==0)|(y==1)) and y.size == row["pixels"] and int(y.sum()) == row["positive_pixels"] and bool(y.any()) == row["anomaly"], "Normalized GT cache differs")
    c.require(result["new_forward_training_or_recalibration"] == 0, "Metric scope differs")
    # Saved metric values only: never call exact_ap/AUROC or original GT decoding on reuse.
    return {"CPU_real_metrics_verified": True, "unique_E_images": 1004, "native_pixels": m.PIXELS,
            "global_pixel_AP": result["global_pixel_AP"], "image_metrics": {k:v for k,v in result["image_metrics"].items() if k not in {"all_items", "selected_examples"}},
            "metrics_recomputed": False, "GT_original_decodes": 0, "new_forwards": 0}

def result_report(result):
    im = result["image_metrics"]
    return f"# 独立CPU真实指标\n\n1004张唯一E图，实际{result['anomaly_images']}异常／{result['normal_images']}正常，原生{m.PIXELS}像素／{result['positive_pixels']}阳性。全局精确像素AP={result['global_pixel_AP']}；整图AUROC={im['AUROC']}。原τ={im['threshold']:.17g}、严格大于：TP／FN={im['TP']}／{im['FN']}（异常分母{im['anomaly_denominator']}），FP／TN={im['FP']}／{im['TN']}（正常分母{im['normal_denominator']}）；召回={im['anomaly_recall']}，正常误报率={im['normal_false_positive_rate']}。零分母／无阳性或单类别对应指标写NA/null，不删除样本。\n\nAP是每个原生像素一次的全局排序，不是平均逐图AP；分类分数为保存原生max，未重新校准。全部首个TP／FN／FP／TN（空组NA）与原因见metrics.json，未读RGB挑图。E历史已见，单seed／固定R5与划分／单类别／未知采集组，不支持稳定收益、预训练因果或屏幕效果。本次无模型前向、训练、生成或重校准；完整指标缓存仅核验不重算AP。\n"

def preparation_report(p):
    return f'''# 第6阶段独立CPU真实指标：准备与运行

本次只做CPU元信息／来源hash／整数分母与虚构标量核验，未读取任何GT文件字节、Q／E／U原图、解析概率载荷、计算真实AP／AUROC／计数或结论，未导入Torch／SMP／timm／模型、加载参数、前向、训练、生成或重校准。完整E评分以保存NPY文件hash和既有全量核验证据绑定；哈希读取文件字节不等于解析数组，未调用旧verify-results解析概率。

一次独立静态审阅发现，原新增复用入口只逐项核已登记hash，不能拒绝上游后来新增的未知文件或空目录；已按systematic-debugging以虚构清单确认原因，补齐E准备／E评分／E全量证据／Q校准证据的完整文件和目录集合及hash复核。虚构failure文件／未知空锁目录均被拒绝，实际上游目录未写入任何模拟异常；未改封存评分实现或数据。新增3个Python模块AST、显式CLI、CPU来源／标量与资源核验通过，未新增软件测试或模型运行。审阅后的单次修正由执行方核验，未重复派发审阅。

## 冻结指标口径

单个最终10000步模型，原E原序1004张／146,411,528原生像素、4270窗评分缓存和原τ={p['threshold']:.17g}。规范GT路径／来源hash／shape来自冻结metric_sources；GT类别与阳性分母当前未知，历史110／894或486461仅为历史，不构成本次实际GT核验。未来显式compute先完整核验评分概率，再对1004规范GT各解码一次，校验字节hash、mode1/L、尺寸与逐图二值{{0,1}}或{{0,255}}，非零映射1，不resize／膨胀／软化；另存uint8原生GT缓存及实际分母，RGB始终不打开。

每个原生像素一次，拼回图已剔除右补边，禁止重复计算重叠窗。全局AP只读复用fewshot_synthetic.metrics.exact_ap，不抽样／平均逐图AP；其两个表共{m.BINS*8:,}字节（约{m.BINS*8/2**30:.2f}GiB），为全部非负float32值精确分组，并非近似分箱。输入逐图有界分块；相同值共同处理、-0规范为+0。复用旧21窗实际sklearn差{p['probe_crosscheck']['absolute_difference']:.3g}与643077不同分数／265563重复分数值的封存证据，不重算旧AP；它是算法依据，不是E效果。未来E原生N／实际P、精确扫描计数和独立磁盘表计数／ties不变量均核对，不在E全量拼接作高内存sklearn替代。

整图AUROC用1004个保存原生max和GT是否非空计算；固定阈值strict>τ分别给TP／FN／FP／TN、异常召回／正常误报及真实分母。无阳性像素的AP、单类别AUROC、零分母率写NA/null并注明原因，不减分母。按E原ID升序取首个TP／FN／FP／TN，空组NA，去重且保留理由；不按最好定位选图、不打开RGB。单seed无三对均值／差值／显著性判断。E历史已见，只作探索性评价；不能据与旧架构差异证明预训练因果收益或屏幕效果。

## 已实现入口、缓存与停止规则

CPU准备核验（不解码GT／概率或计算指标）：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.real_metrics_cli verify
```

未来实际CPU指标入口已实现，本次未执行；需用户手动运行或另行明确授权：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.real_metrics_cli compute
```

准备独立real_metrics_preparation_v1；结果独立real_metrics_v1，内含labels／exact及最终metrics。完整同身份结果仅verify-results只读核验保存值和规范缓存，不解码原GT或重算AP／AUROC；任何未知文件／目录／锁、hash／依赖冲突、部分事务、failure或只有exact完整而总事务未完成均停止保留，不自动续接、重新累计、取交集、近似替代或删现场。GT读入与概率读取使用同一字节快照hash后解析，防止两次打开间身份替换。

准备／实际计算准入RAM≥3GiB／磁盘≥80GiB，CPU只读完整复核RAM≥2GiB／磁盘≥1GiB。逐GT和AP输入块同原3／80门槛；独立表计数核验2／1。冻结exact_ap内部排序表扫描只在入口检查资源，不是连续监控；运行中OOM／磁盘错误保留，不改算法。准备快照RAM{p['resource_snapshot']['available_RAM_GiB']:.3f}GiB／磁盘{p['resource_snapshot']['free_disk_GiB']:.3f}GiB，仅快照，未来重新准入。算法有8.52GB表写入和多轮hash／扫描，实际本轮耗时与峰值待测，不把旧22.110秒小试或E评分172.437秒当指标耗时承诺。全部训练仍由用户手动启动；本次不进入实际计算、图件或新研究。
'''
