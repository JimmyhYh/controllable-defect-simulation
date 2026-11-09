"""CPU-only identity preparation for the official-V1 storage-dtype fix.

No Torch import, pickle execution, tensor loading or model construction.
The original frozen preparation and failed initial run remain unchanged.
"""
from __future__ import annotations
import argparse
import ast
import json
import pickletools
import sys

from . import resources as r
from . import pilot_preparation as original
from . import pilot_config as c
from . import data
from .pilot_preparation import expected_lrs

OUT = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_preparation_v1"
RUNS = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_v1"
MODELS = r.ROOT / "models/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_v1"
FAILED = original.RUNS / "initial/train"
OLD_MODELS = original.MODELS / "initial"
CORE = r.ROOT / "ssn_baseline/core_v1_dtype_compat"
SOURCE_NAMES = ["ssn_baseline/pilot_dtype_preparation.py", "ssn_baseline/pilot_dtype_compat.py",
                "ssn_baseline/pilot_dtype_metrics.py", "ssn_baseline/model_dtype_compat.py"]
SOURCE_NAMES += ["ssn_baseline/core_v1_dtype_compat/" + x.name for x in sorted(CORE.glob("*.py"))]


def storage_metadata():
    r.require(r.BACKBONE.stat().st_size == r.BACKBONE_BYTES and r.digest(r.BACKBONE) == r.BACKBONE_HASH, "Original V1 identity changed")
    # Interpret only opcode descriptions in five legacy metadata streams.
    # genops does not execute GLOBAL/REDUCE or deserialize tensor storages.
    result = []
    with r.BACKBONE.open("rb") as f:
        for index in range(5):
            start = f.tell()
            ops = list(pickletools.genops(f))
            result.append({"index": index, "start": start, "end": f.tell(),
                           "globals": [arg for op, arg, _ in ops if op.name == "GLOBAL"]})
            if index == 3:
                prefix = [(op.name, arg) for op, arg, _ in ops[:20]]
                r.require(("BINUNICODE", "conv1.weight") in prefix and ("GLOBAL", "torch HalfStorage") in prefix,
                          "Expected original conv1 HalfStorage metadata")
    r.require(set(result[3]["globals"]) == {"torch._utils _rebuild_tensor_v2", "collections OrderedDict",
              "torch HalfStorage", "torch FloatStorage", "torch LongStorage"}, "Unexpected V1 serialization globals")
    return {"bytes": r.BACKBONE_BYTES, "sha256": r.BACKBONE_HASH,
            "legacy_metadata_streams": result, "conv1_storage": "HalfStorage / FP16",
            "declared_storage_types": ["HalfStorage", "FloatStorage", "LongStorage"],
            "inspection": "pickletools opcode descriptions only; no pickle execution or tensor payload decoding"}


def binding():
    original.verify()
    r.require(r.files_under(FAILED) == {"transaction.json", "failure.json"}, "Original failure changed or unknown files")
    failure = r.load(FAILED / "failure.json")
    r.require(failure["error"] == "RuntimeError('Expected original generic FP32 V1 state dict')" and
              failure["last_completed_update"] == 0 and failure["own_complete_checkpoint_epochs_this_run"] == [],
              "Not the reviewed zero-update initialization failure")
    r.require(OLD_MODELS.is_dir() and not r.files_under(OLD_MODELS), "Unexpected old checkpoint or weights")
    r.require(r.load(FAILED / "transaction.json")["binding"] == {
        "protocol_sha256": r.digest(original.OUT / "protocol.json"),
        "preparation_complete": r.digest(original.OUT / "complete.json")}, "Failed run protocol binding mismatch")
    for name in SOURCE_NAMES:
        ast.parse((r.ROOT / name).read_text(encoding="utf-8"), filename=name)
    # Copy-only author core except the narrowly scoped local load guard.
    for path in CORE.glob("*.py"):
        if path.name != "feature_extractor.py":
            r.require(path.read_bytes() == (r.ROOT / "ssn_baseline/core_v1" / path.name).read_bytes(), "Unrelated core change")
    return {"original_preparation_complete": r.digest(original.OUT / "complete.json"),
            "original_protocol": r.digest(original.OUT / "protocol.json"),
            "failed_attempt": {x: r.digest(FAILED / x) for x in sorted(r.files_under(FAILED))},
            "implementation": {x: r.digest(r.ROOT / x) for x in SOURCE_NAMES},
            "local_torch_load_source": r.digest(r.ROOT / ".venv-ssn/Lib/site-packages/torch/nn/modules/module.py")}


def verify():
    r.verify_seal(OUT)
    protocol = r.load(OUT / "protocol.json")
    r.require(protocol["bindings"] == binding() and r.load(OUT / "storage_metadata.json") == storage_metadata(), "Compatibility identity changed")
    old = r.load(original.OUT / "protocol.json")
    for key in ["model_config", "input_output", "optimizer", "init_seed", "order_seed", "schedule",
                "gate", "resources", "label_counts", "input_ids", "checkpoint_path"]:
        r.require(protocol[key] == old[key], "Original scientific policy changed: " + key)
    r.require(protocol["runs_root"] == str(RUNS) and protocol["models_root"] == str(MODELS), "Compatibility destination changed")
    r.require(r.load(OUT / "sequence.json") == r.load(original.OUT / "sequence.json"), "Input order changed")
    return {"pilot_preparation_ready": True, "dtype_compatibility_preparation": True,
            "original_failure_preserved": True, "manual_training_only": True,
            "steps": c.STEPS, "seed": c.INIT_SEED, "model_operations_in_verification": 0,
            "protocol_sha256": r.digest(OUT / "protocol.json"), "runtime_compatibility": "not yet run"}


def prepare():
    if OUT.exists():
        return {"reused": True, **verify()}
    identities, metadata = binding(), storage_metadata()
    r.require(not RUNS.exists() and not MODELS.exists(), "Unknown compatibility runtime destination")
    protocol = r.load(original.OUT / "protocol.json")
    protocol.update(schema="ssn-21-input-pilot-official-v1-dtype-compatibility-v1", bindings=identities,
                    runs_root=str(RUNS), models_root=str(MODELS), resources_snapshot=r.snapshot(),
                    correction="Same exact official file; allow its FP16/FP32/int64 storage, strict assign=False copy into FP32 model; no scientific-policy change")
    OUT.mkdir(parents=True, exist_ok=False)
    r.write(OUT / "protocol.json", protocol)
    r.write(OUT / "sequence.json", r.load(original.OUT / "sequence.json"))
    r.write(OUT / "storage_metadata.json", metadata)
    r.write(OUT / "V1存储类型诊断与独立兼容准备.md", report())
    r.seal(OUT, "ssn-official-v1-storage-compatibility-preparation-v1")
    return {"reused": False, **verify()}


def report():
    return r"""# 官方V1存储类型诊断与独立兼容准备

## 原因与实际状态

原initial事务的failure记录Expected original generic FP32 V1 state dict，完成更新0、检查点0，模型目录为空。失败在构造器内部的骨干加载检查处，尚未完成模型构造／作者尺寸前向或创建优化器；不是训练损失、显存不足或数据标签消失。用户运行已导入Torch并越过实际GT算子核对，不能继续称原尝试从未执行过任何运行期操作。原失败与所有冻结代码／协议／结果完整保留，不续训该目录。

本地V1为138223492字节，SHA256为95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8，与原协议一致。它是legacy PyTorch格式而非ZIP；只读pickletools元信息opcode可见conv1.weight引用HalfStorage，文件声明HalfStorage／FloatStorage／LongStorage。没有执行pickle、读取张量载荷或导入Torch。这是原始官方序列化本身含FP16权重；原加载守卫误把“运行模型为FP32”写成“文件全部浮点存储必须为FP32”，因此误拒绝正确文件。

## 独立修正与不变条件

新增core_v1_dtype_compat，除feature_extractor的加载守卫外其余文件逐字节复用冻结核心；model_dtype_compat、pilot_dtype_compat与pilot_dtype_metrics使用独立绑定／输出。保留固定路径、大小、全文件hash与weights_only=True，只接受Tensor字典及FP16／FP32／int64存储；strict=True、assign=False按PyTorch标准行为拷入默认FP32骨干，并核对目标浮点参数／buffer全部FP32。依据本机Torch2.8的nn/modules/module.py，assign=False走param.copy_，不保留文件的半精度目标属性。原文件不转换、不重存、不换权重或重新下载；也不进行半精度训练。

固定21输入缓存、R5／N16、顺序、seed2026102101、50epoch／1600更新、LR／作者损失／BN／特征合成、最终训练侧门槛和资源门槛全部与原协议逐项一致。修复版本从同seed从头初始化；不接旧失败、旧检查点或probe。Q／E／U不读、不评分或选型。不因本次加载错误降低验收或改变训练因素。

结果另存results/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_v1/initial；模型另存models/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_v1/initial。自有检查点只能在本兼容版本完整来源内恢复到新run-name；部分／未知事务保留停止，不删除原失败或自动重跑。

## 当前核验与人工命令

本次仅CPU元信息、hash、AST、源码差异与标量核验；未导入Torch、实例化模型／优化器、加载张量、前向或更新参数，没有训练或真实指标。实际完整加载／结构兼容与GPU峰值仍待人工运行，不能把静态修正称为已经训练通过。

启动要求可用RAM≥6GiB、空闲显存≥6GiB、磁盘≥80GiB，运行RAM≥3GiB；耗时仍未知。首个独立人工命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv-ssn\Scripts\python.exe' -B -X utf8 -m ssn_baseline.pilot_dtype_compat train
```

训练只保存，不自动评分或验收。完成／中断／报错后先保留现场核验，不反复敲命令救结果。旧pilot train指向已失败现场，不作为当前入口。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    args = parser.parse_args()
    def guard(event, arguments):
        if event == "import" and (arguments[0].split(".")[0] in {"torch", "torchvision", "diffusers", "transformers", "sklearn"}
                                  or arguments[0].startswith("ssn_baseline.core_v1") or arguments[0].startswith("ssn_baseline.model")):
            raise RuntimeError("CPU compatibility preparation cannot import models")
        if event == "socket.connect" or event.startswith("pickle."):
            raise RuntimeError("No network or pickle execution")
    sys.addaudithook(guard)
    print(json.dumps({"prepare": prepare, "verify": verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
