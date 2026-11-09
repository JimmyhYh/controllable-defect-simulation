"""CPU-only pilot protocol, identity/cache and scalar schedule preparation."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys

from . import resources as r
from . import data
from . import pilot_config as c
from .contracts import MODEL_CONFIG, INPUT_OUTPUT, OPTIMIZER

OUT = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/pilot_preparation_v1"
RUNS = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/pilot_v1"
MODELS = r.ROOT / "models/ssn_baseline/ksdd2_r5_v1/pilot_v1"
SOURCE_NAMES = ["ssn_baseline/pilot_preparation.py", "ssn_baseline/pilot_config.py", "ssn_baseline/pilot.py",
                "ssn_baseline/pilot_metrics.py", "ssn_baseline/README_stage3.md"]


def source_identity():
    for name in SOURCE_NAMES:
        if name.endswith(".py"):
            ast.parse((r.ROOT / name).read_text(encoding="utf-8"), filename=name)
    return {name: r.digest(r.ROOT / name) for name in SOURCE_NAMES}


def sequence():
    import numpy as np
    selected = data.scope()
    real = [row["id"] for role, row in selected if role == "R5"]
    normal = [row["id"] for role, row in selected if role == "N16"]
    rows = []
    for epoch0 in range(c.EPOCHS):
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.ORDER_SEED, epoch0]))).permutation(16)
        for j, k in enumerate(order):
            for role, sample_id in [("R5", real[(epoch0 * 16 + j) % 5]), ("N16", normal[int(k)])]:
                rows.append({"step": len(rows) + 1, "epoch": epoch0 + 1, "role": role, "id": sample_id})
    return rows


def expected_lrs(step):
    completed_epochs = (step - 1) // c.UPDATES_PER_EPOCH
    factor = MODEL_CONFIG["gamma"] ** sum(completed_epochs >= m for m in c.MILESTONES)
    return [MODEL_CONFIG[k] * factor for k in ["adapt_lr", "seg_lr", "dec_lr"]]


def cache_check():
    import numpy as np
    manifest = data.verify_numeric()
    counts = {"feature_pixels": 0, "feature_positive": 0, "native_pixels": 0, "native_positive": 0}
    for row in manifest["samples"]:
        with np.load(data.OUT / row["cache"], allow_pickle=False) as z:
            r.require(set(z.files) == {"image", "mask_input", "mask_feature", "mask_native"}, "Unexpected NPZ keys")
            r.require(z["image"].shape == (3, 640, 232) and z["image"].dtype == np.float32 and np.isfinite(z["image"]).all(), "Invalid cached normalized input")
            for name, shape in [("mask_input", (640, 232)), ("mask_feature", (160, 58)), ("mask_native", tuple(row["coordinates"]["native_hw"]))]:
                r.require(z[name].shape == shape and z[name].dtype == np.uint8 and set(np.unique(z[name])) <= {0, 1}, "Invalid cached GT")
            soft, ambiguous = data.feature_resize(z["mask_input"])
            r.require(not ambiguous and np.array_equal(soft >= 0.5, z["mask_feature"]), "Cached feature mapping mismatch")
            r.require(int(z["mask_native"].sum()) == row["native_positive"] and int(z["mask_feature"].sum()) == row["feature_positive"], "Cache label counts differ")
            counts["feature_pixels"] += z["mask_feature"].size
            counts["feature_positive"] += int(z["mask_feature"].sum())
            counts["native_pixels"] += z["mask_native"].size
            counts["native_positive"] += int(z["mask_native"].sum())
    return counts


def binding():
    result = data.verify()
    r.require(result["input_ready"], "CPU input gate failed; no pilot admission")
    return {"input_complete": r.digest(data.OUT / "complete.json"), "input_protocol": r.digest(data.OUT / "protocol.json"),
            "stage2_complete": r.digest(data.stage2.OUT / "complete.json"), "sources": source_identity()}


def scalar_protocol(rows):
    from collections import Counter
    r.require(len(rows) == c.STEPS == c.EPOCHS * c.UPDATES_PER_EPOCH, "Pilot update budget mismatch")
    exposures = Counter(row["id"] for row in rows)
    for role, row in data.scope():
        r.require(exposures[row["id"]] == (160 if role == "R5" else 50), "Unbalanced fixed-input exposure")
    return {"updates": c.STEPS, "epochs": c.EPOCHS, "updates_per_epoch": c.UPDATES_PER_EPOCH,
            "real_input_exposures": 800, "normal_input_exposures": 800, "per_id_exposures": dict(exposures),
            "author_internal_feature_views": 3200, "independent_images": 21,
            "scheduler_unit": "epoch; call scheduler.step once AFTER all32 updates of epoch",
            "milestone_epochs": c.MILESTONES, "gamma": 0.4,
            "LR_boundary_updates": {str(k): expected_lrs(k) for k in [1, 1280, 1281, 1440, 1441, 1600]},
            "checkpoint_epochs": c.CHECKPOINT_EPOCHS, "checkpoint_updates": [x * 32 for x in c.CHECKPOINT_EPOCHS],
            "no_flips_or_extra_RGB_augmentation": True,
            "sampling_purpose": "balanced finite-pool fitting pilot, not author datamodule or a formal training budget",
            "resume": "manual only: own complete checkpoint -> NEW run name/output, restore model+BN+optimizer+scheduler+Python/NumPy/Torch CPU/CUDA RNG and same sequence cursor; original partial run preserved",
            "GPU_bitwise_replay_and_full_recovery": "not yet empirically verified; seeded run is not a bitwise guarantee"}


def verify():
    r.verify_seal(OUT)
    p = r.load(OUT / "protocol.json")
    r.require(p["bindings"] == binding() and p["gate"] == c.GATE and p["resources"] == c.RESOURCES, "Pilot identity/policy changed")
    rows = r.load(OUT / "sequence.json")
    r.require(rows == sequence() and p["schedule"] == scalar_protocol(rows), "Fixed pilot sequence/schedule changed")
    r.require(p["label_counts"] == cache_check(), "Actual input denominator changed")
    return {"pilot_preparation_ready": True, "training_started": False, "model_operations_in_verification": 0,
            "steps": c.STEPS, "epochs": c.EPOCHS, "seed": c.INIT_SEED,
            "protocol_sha256": r.digest(OUT / "protocol.json"), "manual_training_entry_implemented": True}


def prepare():
    if OUT.exists():
        return {"reused": True, **verify()}
    identities = binding()
    rows = sequence()
    schedule = scalar_protocol(rows)
    counts = cache_check()
    r.require(not RUNS.exists() and not MODELS.exists(), "Unexpected pilot runtime destination before preparation")
    OUT.mkdir(parents=True, exist_ok=False)
    r.write(OUT / "sequence.json", rows)
    r.write(OUT / "protocol.json", {"schema": "ssn-21-input-technical-pilot-v1", "bindings": identities,
        "model_config": {**MODEL_CONFIG, "epochs": c.EPOCHS}, "input_output": INPUT_OUTPUT, "optimizer": OPTIMIZER,
        "init_seed": c.INIT_SEED, "order_seed": c.ORDER_SEED, "schedule": schedule,
        "gate": c.GATE, "resources": c.RESOURCES, "label_counts": counts,
        "input_ids": [{"role": role, "id": row["id"]} for role, row in data.scope()],
        "checkpoint_path": str(r.BACKBONE), "runs_root": str(RUNS), "models_root": str(MODELS),
        "restore_validation": "complete identity+file hashes, full checkpoint from this same pilot only; fresh run label, no overwrite",
        "inference": "not automatic after train; separately invoked score caches each of21 once using final1600 step, segmentation hook + same eval call; no BN/parameter updates",
        "manual_runtime_preflight": "old CPU interpreter verifies immutable preparation; new model env checks actual Torch interpolation against cache for all21 BEFORE model initialization",
        "runtime_status": "Torch import, parameter match, model fit, resource peaks and full recovery untested",
        "resources_snapshot": r.snapshot(), "no_heldout_Q_E_U_read": True})
    r.write(OUT / "第3阶段小试协议与人工运行.md", report(counts))
    r.seal(OUT, "ssn-technical-pilot-preparation-v1")
    return {"reused": False, **verify()}


def report(counts):
    return f"""# SuperSimpleNet第3阶段：小试协议与人工运行

固定输入21张（原R5全部5张＋旧N16全部16张）的CPU来源、原始对比度、坐标及全部样例审阅通过。R5特征网格阳性为379／95／56／224／810，未替换／膨胀／借U；独立中间NPZ供新环境读取，不向封存.venv-ssn加OpenCV或注入旧site-packages。此处只有CPU准备，未导入Torch、构造模型／优化器、加载参数、运行前向／生成／训练或读取Q／E／U图像／GT。

## 预先固定的小试预算

初始化seed={c.INIT_SEED}，顺序PCG64 SeedSequence([{c.ORDER_SEED},epoch0])；50epochs，每epoch32次更新，batch1／FP32／workers0，共1600更新。每epoch交替16个R5循环项与N16的固定无放回排列；R5每图总160次、N16每图50次，实际输入曝光各800。作者B→2B保留，共3200特征视图，不是3200独立图。无额外RGB合成／翻转增强。这是均衡有限池技术拟合预算，不是作者原数据采样器或正式预算，也不是从旧1000步／0.80结果选出的条件。

保持作者适配器、双头、BN、Perlin／Gaussian特征合成、损失和AdamW三组。LR起点为1e-4／2e-4／2e-4；调度仍按作者epoch单位，在完成40／45epoch后乘0.4，即1—1280更新用原LR，1281—1440用0.4倍，1441—1600用0.16倍。逐更新边界见协议。不照搬旧800／1000或8000／9000日程；不改旧冻结配置。

损失沿用已封存objective.py：分割为正常像素截断L1均值＋异常像素截断L1均值＋focal均值，分类为返回标签的focal；截断阈值0.5，正常与异常分别取均值。空子集贡献标量0，不把空GT改为异常或用Dice／CE替换。输入正常GT为空，但作者特征合成返回的mask／label可含人工异常，损失使用其返回标签，不能把800次正常输入等同于800个全正常特征视图。全部样本有像素监督，无U弱标签。

## 最终验收的工程约定

只验最终1600步，不挑中间权重。先核对预算／检查点／有限损失及资源，再在相同21张训练输入上检查：原始分割头160×58 sigmoid全局像素AP≥0.95，且每张R5各自AP≥0.85；分类头sigmoid以固定严格>0.5作技术参照，TP=5/5、FP≤1/16。全部图有限且特征概率图非常量，评分前后参数／BN缓冲身份不变。该约定是训练前设定的技术准入，不是文献保证、真实验证或业务指标；全局加逐例防止大掩码主导，分类条件要求双头都具备基本拟合能力。门槛可能失败，失败查因，不降门槛或挑检查点。

特征域共有{counts['feature_pixels']}像素／{counts['feature_positive']}阳性；原生域共有{counts['native_pixels']}像素／{counts['native_positive']}阳性。仅有21独立图，三种尺度的像素不能混为相同分母。原生平滑／逆映射概率AP仅作同池描述，不设门槛，以免把输出平滑的几何影响当训练未学会。当前没有实际AP、分类计数或模型效果。

## 资源、停止与恢复

启动要求可用RAM≥6GiB、空闲显存≥6GiB、磁盘≥80GiB；运行采样可用RAM≥3GiB。门槛是工程准入，不保证不会OOM。耗时／峰值未知，不引用旧模型速度；手动运行后每100更新显示当次elapsed与进度。OOM／非有限／身份或真实算子标签差异即保留停止，不改分辨率、骨干、数据或预算。

用户手动启动后，先由旧CPU解释器核验协议与缓存；新环境对21个缓存执行实际Torch GT插值二值核对，差异在模型初始化前停止。随后才加载已固定V1并触发作者构造器尺寸前向。两步都没有在本阶段执行。

资源准入通过后、首次运行期Torch导入／实际插值／模型构造之前先保留独立事务，初始化／搬运／恢复异常也写入failure，不把它们当作从未尝试而自动重复。未来独立评分同样先保留事务，并逐输入检查运行RAM≥3GiB，保存最低采样值；部分或前向不明确时保留停止。

完成epoch10／20／30／40／50（step320／640／960／1280／1600）各存完整自有检查点：模型与BN、optimizer／scheduler、Python／NumPy／Torch CPU／CUDA RNG、已完成epoch／step／序列游标、全部来源身份和资源。最终另存权重。Ctrl+C不额外保存未完成更新；完整恢复只允许用户明确选本小试完整检查点并指定新run-name，旧事务不覆盖，仍以总1600步为终点。GPU位级重放及完整恢复尚未实测，不称已保证。没有自动续训、评分或指标计算。

## 实际已实现的第一条人工命令

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv-ssn\\Scripts\\python.exe' -B -X utf8 -m ssn_baseline.pilot train
```

只训练初次run initial，不自动评分／计算gate或进入Q／E。完整结果再执行只读身份核验；存在partial／unknown则保留并停止，不能反复敲该命令救结果。训练由用户手动启动，当前未启动。

只读准备核验：

```powershell
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m ssn_baseline.pilot_preparation verify
```

静态入口源码已实现／AST与CPU标量已核对，不等于运行兼容通过。实际score及CPU metrics入口另行显式调用；训练完成不自动执行它们。旧实验、冻结环境、源码／权重／阈值和全部历史失败保留，本小试也不能支持未见真实泛化或屏幕收益。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    args = parser.parse_args()
    def guard(event, arguments):
        if event == "import" and (arguments[0].split(".")[0] in {"torch", "torchvision", "diffusers", "transformers", "sklearn"} or arguments[0] == "ssn_baseline.model" or arguments[0].startswith("ssn_baseline.core_v1")):
            raise RuntimeError("CPU preparation cannot import runtime models")
        if event.startswith("pickle.") or event == "socket.connect":
            raise RuntimeError("No pickle or network")
    sys.addaudithook(guard)
    print(json.dumps({"prepare": prepare, "verify": verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
