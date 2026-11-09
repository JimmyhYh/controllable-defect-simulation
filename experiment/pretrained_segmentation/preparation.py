"""CPU-only pilot protocol, compatibility whitelist and scalar schedules."""
from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as c

def strict_directories(folder):
    """Additional strict layer; preserve already sealed CPU input source bytes."""
    folder = Path(folder)
    manifest = c.verify_seal(folder)
    expected = set()
    for name in manifest["files"]:
        parent = Path(name).parent
        while str(parent) != ".":
            expected.add(parent.as_posix()); parent = parent.parent
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_dir()}
    c.require(actual == expected, "Unknown/missing directory or directory-shaped lock; preserve and stop")
    return manifest

def compatibility():
    h = c.load(c.RESOURCE / "safetensors_header.json")
    desc = {x["name"]: {"shape": x["shape"], "dtype": x["dtype"]} for x in h["tensor_descriptors"]}
    c.require(len(desc) == 102 and all(x["dtype"] == "F32" for x in desc.values()), "Actual fixed encoder descriptors changed")
    fc = {n: desc[n] for n in ["fc.weight", "fc.bias"]}
    c.require(fc["fc.weight"]["shape"] == [1000, 512] and fc["fc.bias"]["shape"] == [1000], "Unexpected classification descriptor")
    bn = sorted(n[:-len("running_mean")] for n in desc if n.endswith("running_mean"))
    counters = [prefix + "num_batches_tracked" for prefix in bn]
    c.require(len(bn) == 20 and all(prefix + "running_var" in desc for prefix in bn) and not set(counters) & set(desc), "Unexpected legacy BN descriptor family")
    source_path = c.ENV / "Lib/site-packages/torch/nn/modules/batchnorm.py"
    text = source_path.read_text(encoding="utf-8")
    ast.parse(text)
    c.require("version is None or version < 2" in text and "num_batches_tracked" in text and "default value of 0" in text, "Torch2.8 legacy BN compatibility evidence changed")
    return {"original_descriptors": desc, "remove_exact_keys": ["fc.bias", "fc.weight"],
            "add_exact_zero_int64_scalar_keys": counters, "BN_prefixes": bn,
            "encoder_expected_keys": sorted((set(desc) - set(fc)) | set(counters)),
            "torch_BN_source_path": source_path.relative_to(c.ROOT).as_posix(), "torch_BN_source_sha256": c.sha(source_path),
            "policy": "runtime explicit fc removal plus20 zero-int64 counters; strict remaining keys/shapes/dtypes; no generic strict=False",
            "tensor_created_or_loaded": False, "runtime_compatibility_verified": False}

def schedule_verification():
    result = {}
    for total in [2000, 10000]:
        a, b = int(.8 * total), int(.9 * total)
        updates = [1, a, a + 1, b, b + 1, total]
        expected = [[1e-5, 1e-4], [1e-5, 1e-4], [2e-6, 2e-5], [2e-6, 2e-5], [4e-7, 4e-6], [4e-7, 4e-6]]
        c.require([c.lr_at(u, total) for u in updates] == expected, "Scalar LR boundary mismatch")
        resume = {str(done): c.lr_at(done + 1, total) if done < total else None for done in [0, a - 1, a, b - 1, b, total - 1, total]}
        result[str(total)] = {"updates": dict(zip(map(str, updates), expected)), "resume_next_update_by_completed": resume}
    return result

def requests():
    rg = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.SEED, 0])))
    ng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.SEED, 1])))
    def cycle(indices, count, rng):
        out = []
        while len(out) < count:
            out.extend(int(i) for i in rng.permutation(indices))
        return out[:count]
    real = cycle(list(range(5)), 1000, rg)
    normal = cycle(list(range(5, 21)), 1000, ng)
    return [{"update": i + 1, "item_index": real[i // 2] if i % 2 == 0 else normal[i // 2]} for i in range(2000)]

def verify_pilot(decode=True, source_bytes=True):
    from .data import verify_inputs
    ip = verify_inputs(decode=decode, source_bytes=source_bytes)
    strict_directories(c.INPUT)
    seal = strict_directories(c.PREP)
    p = c.load(c.PREP / "protocol.json")
    c.require(p["implementation"] == c.code_identity() and p["input_complete_sha256"] == c.sha(c.INPUT / "complete.json"), "Pilot implementation/input identity changed")
    c.require(p["compatibility"] == compatibility() and p["schedules"] == schedule_verification(), "Compatibility/scalar protocol changed")
    seq = c.load(c.PREP / "requests.json")
    c.require(seq == requests() and len(seq) == 2000, "Fixed input sequence changed")
    c.require(p["valid_pixels"] == ip["valid_pixels"] and p["positive_pixels"] == ip["positive_pixels"], "Pilot actual denominators changed")
    c.require(p["seed"] == c.SEED and p["steps"] == 2000 and p["gate"] == {"global_AP_min": .95, "each_R_window_AP_min": .85, "classification_0_5_gate": False}, "Pilot gate/budget changed")
    c.require(p["visual_review"]["approved"] is True and set(p["visual_review"]["observations"]) == set(c.R5 + c.N16), "Fixed21 visual review incomplete")
    c.require(seal["identity"]["meaning"] == "CPU preparation only; training and runtime loading not performed", "Preparation meaning changed")
    return p, ip, seq

def prepare_pilot(review):
    if c.PREP.exists():
        p, _, _ = verify_pilot()
        return {"pilot_ready_for_manual_run": True, "reused": True, "valid_pixels": p["valid_pixels"], "positive_pixels": p["positive_pixels"]}
    from .data import verify_inputs
    ip = verify_inputs()
    c.require(review["approved"] is True and set(review["observations"]) == set(c.R5 + c.N16) and all(isinstance(x, str) and x for x in review["observations"].values()), "Need actual full fixed21 visual review, no uncertain/substituted item")
    c.require(review["input_complete_sha256"] == c.sha(c.INPUT / "complete.json"), "Visual review/input identity mismatch")
    resource_snapshot = c.resources(2, 1)
    for name in c.CODE:
        if name.endswith(".py"):
            ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"), filename=name)
    seq = requests()
    exposure = Counter(ip["items"][s["item_index"]]["id"] for s in seq)
    c.require(all(exposure[x] == 200 for x in c.R5) and sum(exposure[x] for x in c.N16) == 1000 and all(exposure[x] in (62, 63) for x in c.N16), "Unexpected fixed exposure")
    p = {"schema": "pretrained-r5-pilot-preparation-v1", "implementation": c.code_identity(), "input_complete_sha256": c.sha(c.INPUT / "complete.json"),
         "visual_review": review, "resource_snapshot": resource_snapshot, "compatibility": compatibility(), "schedules": schedule_verification(), "seed": c.SEED, "steps": 2000,
         "window": 256, "batch": 1, "workers": 0, "precision": "FP32", "augmentation": "none in fixed-window probe", "synthesis": False,
         "valid_pixels": ip["valid_pixels"], "positive_pixels": ip["positive_pixels"], "unique_images_and_windows": 21, "exposure_by_id": dict(exposure),
         "gate": {"global_AP_min": .95, "each_R_window_AP_min": .85, "classification_0_5_gate": False},
         "checkpoints": [400, 800, 1200, 1600, 2000], "Adam": {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0},
         "loss": "valid-only unweighted CE mean plus DiceLoss=1-(2sum(p*y)+1)/(sum(p)+sum(y)+1); empty GT included",
         "formal_scope": "only scalar10000 schedule checked; formal requests/training not implemented or authorized",
         "manual_model_interpreter": str(c.ENV / "Scripts/python.exe"), "GPU_resume_verified": False,
         "stop": "no automatic retry, extra steps, altered gate/sample/seed, best checkpoint or Q/E/U access"}
    c.PREP.mkdir(parents=True, exist_ok=False)
    c.save(c.PREP / "protocol.json", p); c.save(c.PREP / "requests.json", seq)
    c.save(c.PREP / "第3阶段CPU输入与人工小试准备.md", report(p, ip))
    c.seal(c.PREP, {"meaning": "CPU preparation only; training and runtime loading not performed"})
    verify_pilot()
    return {"pilot_ready_for_manual_run": True, "reused": False, "valid_pixels": p["valid_pixels"], "positive_pixels": p["positive_pixels"], "actual_training_performed": False}

def report(p, ip):
    detail = "\n".join(f"- {i['id']}：{i['kind']}，原H×W={i['height']}×{i['width']}，y={i['y']}，有效{i['valid_pixels']}／阳性{i['positive_pixels']}像素。" for i in ip["items"])
    return f"""# 第3阶段：CPU输入与人工小试准备

实际仅完成固定输入、静态实现与标量协议；未导入Torch／SMP／timm、创建tensor、加载参数、实例化模型／优化器、前向、反向、更新、训练、校准或模型评价。源RGB／GT只解码固定R5＋旧N16各21份，不读Q／E／U或额外正常图。

## 实际输入与审阅

共21张独立训练图对应21个工作窗；R5每图首个GT非空合法256窗，N16每图首窗。原始对比度／原始像素不变，无缩放／膨胀／合成／翻转；RGB除255后ImageNet归一化，右反射补边不参与损失和指标。实际有效像素{p['valid_pixels']}、阳性{p['positive_pixels']}；这不是SSN整图或特征分母。

{detail}

5张R完整RGB／GT／选窗与N16全部原RGB的CPU位置总览在[输入目录]({c.INPUT.as_posix()}/previews)。实际逐图视觉观察保存在protocol.visual_review.observations；它只检查身份、坐标和缺陷标签适用，不把窗口框或正常显示内容当新缺陷标注。

11670的预定首个阳性窗只覆盖该缺陷的一部分；12245使用原生尾窗，缺陷接近窗底。它们符合事先规定的首个阳性合法窗规则，未移动窗口或替补样本。N16中的自然纹理和亮点不据视觉重新标注；原正常GT均为零。固定窗口拟合不等于完整原图的缺陷覆盖或未见验证。

## 实现与兼容边界

SMP U-Net／ResNet18、2类单分割头，无分类辅助头；编码器BN20处统计固定但affine微调，解码器10处BN替换GN8。通用编码器原header102个F32描述；显式去fc.bias／fc.weight，20个缺少的旧版BN计数键仅在未来运行时建立int64标量0，其他键／shape／dtype必须严格匹配。此规则与本地Torch2.8 BN兼容源码身份已冻结，尚未执行实际加载，未重初始化已加载编码器。

CE＋Dice包含空GT并排除补边。Adam encoder／decoder学习率1e-5／1e-4，更新1—1600不衰减，1601—1800用2e-6／2e-5，1801—2000用4e-7／4e-6；正式8000／8001、9000／9001边界只作标量核验，未准备正式训练。seed2026102201，2000次更新，奇R偶N、各1000曝光，R各200、N各62或63；曝光不是独立图数。

CPU准备首次准入在读取图像、建立输入目录之前，因Windows平台查询使用NUL设备而被写入审计误拦；已仅允许该系统空设备，未放宽科学目录或图像访问范围。随后输入准备完成。独立静态审阅发现未知空目录与模型目录根未知文件可能绕过缓存边界，已在未封存的包装／核验层补齐严格清单与恢复检查；输入封存源码未被改写。核验仅限CPU源码／输入／标量，实际模型加载、显存占用与GPU完整恢复尚未实测。

最终2000步验收：全21窗有效域全局精确AP≥0.95、各R窗AP≥0.85，损失／梯度有限、5异常概率图非常量、评分参数／buffer不变，训练BN buffer固定。正常分数和像素0.5占比只描述，无SSN分类0.5门槛；未计算任何AP或得出通过结论。未来CPU指标绑定旧exact_ap并以sklearn／实际ties交叉核对。

## 人工启动、资源与恢复

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.cli train-probe
```

只允许用户手动执行该参数更新入口。启动至少RAM4.5GiB／空闲显存6GiB／磁盘20GiB，实际时间与峰值待首次运行实测；每100步资源采样，可用RAM<2GiB在明确更新边界保存自有停止点并退出。模型／5个预定完整检查点在{c.MODEL.as_posix()}，训练结果在{c.RUN.as_posix()}。每400步保存自有model safetensors／optimizer受限读取文件／JSON随机状态、序列游标／LR历史与身份。Ctrl+C不额外保存，只能恢复最近完整自有点；部分或未知保存事务保留查因，不自动恢复／重跑。GPU恢复尚未实测，标量日程核对不等于恢复可靠性证明。

本次CPU准备时资源快照：可用RAM {p['resource_snapshot']['free_RAM_GiB']:.3f}GiB、磁盘 {p['resource_snapshot']['free_disk_GiB']:.3f}GiB；它不是未来训练启动时的准入结论，运行入口会重新核对。

完整训练缓存只核验，不再次更新；未完成的已知运行须由用户显式 --resume-from 指定本组最新完整点。小试结束不自动评分、计算AP或进入正式训练；用户报告后先只读核验，再另行指定评分与CPU验收。probe权重不用于正式初始化；不承诺真实、稳定增益或屏幕效果。
"""
