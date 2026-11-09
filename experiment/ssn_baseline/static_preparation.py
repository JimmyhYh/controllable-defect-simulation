"""Stage 2 CPU/AST preparation and verification; no model import or forward."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

from . import resources as r
from . import environment as e
from . import environment_module_bootstrap as installed
from .contracts import MODEL_CONFIG, INPUT_OUTPUT, OPTIMIZER

OUT = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/core_preparation_v1"
CORE = r.ROOT / "ssn_baseline/core_v1"
SOURCES = ["ssn_baseline/environment.py", "ssn_baseline/environment_continuation.py", "ssn_baseline/environment_module_bootstrap.py", "ssn_baseline/static_preparation.py", "ssn_baseline/contracts.py",
           "ssn_baseline/model.py", "ssn_baseline/objective.py", "ssn_baseline/README_stage2.md"]


def derived_sources():
    model = (r.VENDOR / "model/supersimplenet.py").read_text(encoding="utf-8")
    old_import = "from common.perlin_noise import rand_perlin_2d"
    r.require(model.count(old_import) == 1, "Author Perlin import changed")
    model = model.replace(old_import, "from .perlin_noise import rand_perlin_2d")
    old_call = "            image_size=image_size,\n        )"
    r.require(model.count(old_call) == 1, "Author feature constructor changed")
    model = model.replace(old_call, "            image_size=image_size,\n            checkpoint_path=config[\"local_backbone_path\"],\n        )")
    feat = (r.VENDOR / "model/feature_extractor.py").read_text(encoding="utf-8")
    feat = feat.replace("from importlib import import_module", "import hashlib\nfrom pathlib import Path")
    old_arg = "        image_size: tuple[int, int],\n    ):"
    r.require(feat.count(old_arg) == 1, "Author feature argument changed")
    feat = feat.replace(old_arg, "        image_size: tuple[int, int],\n        checkpoint_path: str,\n    ):")
    start = feat.index("        # tro to get model from torchvision")
    end = feat.index("        self.feature_extractor =", start)
    load_local = '''        # Independent local-only V1 adaptation; original constructor used a
        # torchvision weight enum (which can download). All feature math below
        # and the author's constructor shape forward remain unchanged.
        from torchvision.models import wide_resnet50_2
        if backbone != "wide_resnet50_2" or layers != ["layer2", "layer3"] or patch_size != 3:
            raise RuntimeError("Only the frozen strict-R5 backbone contract is supported")
        path = Path(checkpoint_path).resolve()
        expected_path = Path(__file__).resolve().parents[2] / "models/checkpoints/wide_resnet50_2-95faca4d.pth"
        if path != expected_path.resolve() or not path.is_file() or path.stat().st_size != 138223492:
            raise RuntimeError("Explicit existing generic V1 path/size mismatch; no download")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8":
            raise RuntimeError("Local V1 identity changed; preserve and stop")
        model = wide_resnet50_2(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
            raise RuntimeError("Expected original generic FP32 V1 state dict")
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)

'''
    feat = feat[:start] + load_local + feat[end:]
    header = f"# Derived from blaz-r/SuperSimpleNet@{r.COMMIT}; original license in vendor/SuperSimpleNet_98ab4d5/LICENSE.\n"
    return {"__init__.py": b'"""Versioned author-core adaptation; no eager imports."""\n',
            "supersimplenet.py": (header + model).encode("utf-8"),
            "feature_extractor.py": (header + feat).encode("utf-8"),
            "perlin_noise.py": (r.VENDOR / "common/perlin_noise.py").read_bytes(),
            "loss.py": (r.VENDOR / "common/loss.py").read_bytes()}


def source_check():
    expected = derived_sources()
    r.require(r.files_under(CORE) == set(expected), "Derived core strict directory mismatch")
    for name, payload in expected.items():
        r.require((CORE / name).read_bytes() == payload, f"Unapproved author-core change {name}")
    # Parse only: do not import any prepared runtime source.
    for path in [*(CORE / n for n in expected), *(r.ROOT / n for n in SOURCES if n.endswith(".py"))]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {**{f"ssn_baseline/core_v1/{n}": r.digest(CORE / n) for n in expected},
            **{n: r.digest(r.ROOT / n) for n in SOURCES}}


def scalar_checks():
    h, w = INPUT_OUTPUT["input_hw"]
    layer2_hw = [(h + 7) // 8, (w + 7) // 8]
    shape = [512 + 1024, layer2_hw[0] * 2, layer2_hw[1] * 2]
    r.require(shape == INPUT_OUTPUT["feature_shape_static"] == [1536, 160, 58], "Static feature grid mismatch")
    perlin_hw = [1 << (x - 1).bit_length() for x in shape[1:]]
    scales = [2 ** x for x in range(0, 6)]
    r.require(all(dim % scale == 0 for dim in perlin_hw for scale in scales), "Perlin scalar divisibility mismatch")
    r.require(MODEL_CONFIG["adapt_cls_feat"] is True and MODEL_CONFIG["dt"] is None and MODEL_CONFIG["dilate"] is None, "Wrong candidate")
    threshold_cases = [[x, 0 if x < 0.5 else 1] for x in [0, 0.499999, 0.5, 1]]
    return {"feature_grid": shape, "Perlin_grid": perlin_hw, "Perlin_scales": scales,
            "binary_threshold_scalar_examples": threshold_cases,
            "B1_feature_views": 2, "independent_input_images": 1,
            "pilot_and_formal_epochs": None, "scheduler_not_instantiated": True,
            "native_inverse_mapping": "(y+0.5)*640/Hnative-0.5, (x+0.5)*232/Wnative-0.5 for align_corners=False; boundary clamp, then sigmoid",
            "source_feature_dim_forward_executed": False, "interpretation": "shape/coordinate arithmetic only, no actual labels or tensors verified"}


def bindings():
    r.verify()
    installed.verify()
    return {"stage1_complete_sha256": r.digest(r.OUT / "complete.json"),
            "environment_complete_sha256": r.digest(installed.INSTALLATION / "complete.json"),
            "original_environment_failure": installed.original_failure(),
            "continuation_preparation_sha256": r.digest(installed.PREP / "complete.json"),
            "inventory_complete_sha256": r.digest(e.INVENTORY / "complete.json"),
            "vendor_complete_sha256": r.digest(r.VENDOR / "complete.json"),
            "sources": source_check()}


def verify():
    r.verify_seal(OUT)
    p = r.load(OUT / "protocol.json")
    r.require(p["bindings"] == bindings() and p["model_config"] == MODEL_CONFIG and p["input_output"] == INPUT_OUTPUT and p["optimizer"] == OPTIMIZER, "Static core protocol identity changed")
    r.require(p["scalar_checks"] == scalar_checks(), "Scalar contract changed")
    return {"stage2_static_ready": True, "environment_and_core_metadata_verified": True,
            "model_imports_or_operations": 0, "runtime_compatibility": "not checked",
            "CPU_content_and_pilot": "not started", "protocol_sha256": r.digest(OUT / "protocol.json")}


def prepare():
    if OUT.exists():
        return {"reused": True, **verify()}
    r.verify()
    installed.verify()
    if CORE.exists():
        # Pre-existing directory without the stage-2 seal is never overwritten.
        raise RuntimeError("Unknown/partial derived core directory; retain and diagnose")
    CORE.mkdir(parents=True, exist_ok=False)
    for name, payload in derived_sources().items():
        with (CORE / name).open("xb") as f:
            f.write(payload)
    binding = bindings()
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 1, "Static CPU resource gate")
    OUT.mkdir(parents=True, exist_ok=False)
    r.write(OUT / "protocol.json", {"schema": "ssn-stage2-static-core-v1", "bindings": binding,
        "model_config": MODEL_CONFIG, "input_output": INPUT_OUTPUT, "optimizer": OPTIMIZER,
        "scalar_checks": scalar_checks(), "resources": snap, "roles": r.ROLES, "R5": r.R5,
        "train_roles": ["R_real5", "N_train1885"], "Q": "calibration only, no training or configuration selection",
        "U": "not used, no weak supervision", "E": "not used for selection or periodic evaluation; old benchmark already observed",
        "core_diff": {"supersimplenet.py": "only local Perlin import and explicit checkpoint constructor argument",
                      "feature_extractor.py": "local-only weights=None + verified generic V1 weights_only load; freezes backbone; author feature forward/get_feature_dim preserved",
                      "perlin_noise.py": "exact author bytes", "loss.py": "exact author bytes",
                      "objective.py": "fully segmented R5/normal subset of author loss, no distance/loss-mask branch"},
        "remaining": ["OpenCV decode/resize dependency and CPU R5/N16 content/coordinates", "feature-grid R5 positive survival", "pilot sampling/budget/gate/manual entry", "actual model imports/parameter match/forward/learning/resource peaks", "complete checkpoint recovery"]})
    r.write(OUT / "verification.json", {"source_AST_and_scalar_only": True, "old_sources_metadata_weights_unchanged": True,
        "images_GT_probability_payload_reads": 0, "Torch_model_imports": 0, "models_optimizers_instantiated": 0,
        "forward_generation_training_calibration_metrics": 0, "network_bytes_needed_for_stage2": 0,
        "derivative_sources_reproducible_from_fixed_vendor": True, "environment_runtime_not_validated": True})
    p = r.load(e.INVENTORY / "inventory.json")
    r.write(OUT / "第2阶段隔离环境与核心静态准备.md", report(p, snap))
    r.seal(OUT, "ssn-static-core-preparation-v1")
    return {"reused": False, **verify()}


def report(inventory, snap):
    return f"""# SuperSimpleNet第2阶段：隔离环境与核心静态准备

完成范围：离线新环境安装、作者核心独立包装、AST／依赖元信息／标量协议。**不代表模型已导入、参数匹配、可训练或检测有效**。第3阶段CPU内容与小试协议尚未实施，训练仍由用户手动启动。

## 来源与环境

第1阶段协议、固定vendor及旧元信息／核心源码／ImageNet V1字节hash只读复核通过；封存不改。作者commit `{r.COMMIT}`。本地14份固定wheel合计{inventory['compressed_bytes']/2**30:.6f}GiB，解压条目约{inventory['uncompressed_bytes']/2**30:.6f}GiB，均与既有手动资源CRC／RECORD审计的文件hash一致；这不是官方独立签名认证。

从本地wheel离线安装至`.venv-ssn`，网络流量0；旧`.venv`未改。固定核心Torch2.8.0+cu128／torchvision0.23.0+cu128／NumPy1.26.4／Pillow12.3.0；其余实际pin及完整依赖闭包、解释器、METADATA／RECORD身份见[独立续接环境协议]({(installed.INSTALLATION/'environment.json').as_posix()})与[wheel清单]({(e.INVENTORY/'inventory.json').as_posix()})。传递依赖采用已备本地版本，明确不同于旧环境部分版本，不声称整套依赖相同。无anomalib／Lightning／albumentations／torchmetrics安装。14份原wheel之外，仅使用Python本机自带pip24.2引导包，路径和hash另绑，无联网获取。

安装前元信息检查一度因setuptools包内附带多份`_vendor/*.dist-info/METADATA`而停止；核对发现它们是嵌套依赖许可／说明，顶层wheel身份仍唯一。仅修正本轮尚未封存的选择逻辑为唯一顶层dist-info，保留所有原wheel／hash／pin，未创建或修补失败环境；随后实际完整检查通过。

首次venv创建的ensurepip随后因沙箱默认TEMP拒绝写入而失败。独立只读元信息核对：新解释器正常、隔离配置正确、site-packages为空；仅依赖dry-run复现PermissionError，未安装任何包。原`environment_v1/installation`的transaction／failure保持原hash及未完成状态，原environment.py／inventory不改，不调用其失败入口重跑。独立environment_continuation改用项目内TEMP，但其runpy调用遗漏alter_sys=True，在真正安装前被Windows pip自修改保护拦截；元信息再次确认包数0。该失败与源码／协议同样保留，不在原目录修复。

最终另立`environment_module_bootstrap.py`，将调用改为Python本机ensurepip源码采用的标准`runpy.run_module('pip',run_name='__main__',alter_sys=True)`语义，绑定两份失败、已知空环境、固定14份wheel及自带pip，在独立environment_module_bootstrap_v1事务中完成本阶段获准的离线安装。保留两次失败的原hash及未完成状态，不删标记、不换pin／环境路径、不改旧源码。这是引导权限／调用封装排障，不是模型恢复／运行兼容证据；完整绑定见protocol。独立审阅覆盖原阶段2包装与资源模块；新增引导修正由助手依据ensurepip／pip本机源码、实际依赖解析及元信息核验，不声称执行过模型兼容检查。

离线安装禁用index／缓存与字节码编译，显式require-hashes；新进程读取包元信息，不导入Torch／torchvision／NumPy／PIL或模型。pip／依赖解析不是模型运行验证。完成缓存严格只读复用；异常／未知事务保留停止，不自动修复或换pin。

## 保留核心与必要适配

`ssn_baseline/core_v1/`保存独立派生代码；vendor原字节不改。SSN主文件只调整本地Perlin导入及显式权重路径传入；FeatureExtractor改为`weights=None`构造、核对既有V1字节hash后`weights_only=True`本地加载／严格state匹配，骨干冻结，保留作者所有特征运算与构造器尺寸探测。**以上加载代码均未执行，构造器会前向，当前禁止调用**。Perlin及focal helper与作者字节完全一致，来源许可继续引用vendor。

2024式adapt_cls_feat=True、dt=None、dilate=None；其余特征合成／噪声／双头／BN／损失／AdamW三组参数取固定作者监督入口，完整数值见[protocol.json](protocol.json)。不换旧GN、CE＋Dice、stateless BN或旧10000步日程。作者调度按epoch在0.8／0.9处乘0.4，具体epochs／更新预算／采样／gate留待第3阶段独立冻结，不能现在实例化优化器。R5／N1885以外材料不训练；Q仅未来正常校准，U不弱监督，E不选型／训练期test。无作者pickle或默认N246数据模块接入。

`objective.py`准备严格全分割子集：分割正常／异常截断L1分别取均值（空子集贡献0），再加全像素focal与分类focal；margin0.5、gamma4、alpha−1。无distance loss-mask分支，保留作者特征B→2B复制。复制数不等于独立真实图数。没有训练循环／反向／更新入口。

## 坐标、标签与评分口径

未来RGB保留原始对比度，按作者A.Resize的OpenCV uint8 INTER_LINEAR到640×232，再/255与ImageNet均值／标准差归一化。GT用INTER_NEAREST到输入域，再按作者bilinear／align_corners=True到160×58，以>=0.5置1。原生GT不改；R5任何标签变空就停止，不膨胀／换样救结果。静态推算特征1536×160×58、Perlin网格256×64；本阶段没有实际标签存活证据。

新环境目前只准备模型核心依赖，没有OpenCV；原图解码／resize模块和依赖属于第3阶段先决项。既有CPU环境有OpenCV4.11.0.86可作内容核验，禁止把旧site-packages加入新环境或冒充端到端依赖已齐。后续若新运行入口需要OpenCV，须先核对对应安装材料和流量再独立准入；当前没有下载／安装这一额外包。

评价时作者先把logit上采样至输入域，再Gaussian kernel25／sigma4反射平滑。本分支明确补充：平滑后的logit按bilinear／align_corners=False逆映射至原生H×W，再sigmoid保存float32概率，每原生像素一次；不先sigmoid再插值。分类头logit的sigmoid是独立整图主分数；原生定位图最大值另列补充，两者不混称，不用softmax、测试池minmax或旧阈值。本阶段不校准、不读取Q／E／U图像／GT或计算任何指标。

## 资源与实际入口

封存快照可用RAM {snap['free_RAM_GiB']:.6f}GiB、D盘空闲{snap['free_disk_GiB']:.6f}GiB；GPU {json.dumps(snap['GPUs'],ensure_ascii=False)}。只是元信息，非运行峰值。后续模型起点RAM6GiB／空闲显存6GiB／磁盘80GiB及运行RAM采样3GiB仍是工程候选，不是足够性证明；模型时间、峰值和完整恢复未实测。

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m ssn_baseline.static_preparation verify
```

该入口使用旧解释器做CPU静态核验，并用新环境独立进程只读依赖元信息；不会导入模型／Torch或产生前向。当前已实现并可核验复用的入口为environment inventory、environment_module_bootstrap prepare／install／verify及static_preparation prepare／verify；完整缓存只读复用，不重新安装或覆盖。environment install／verify与environment_continuation install保留失败历史，不能用其充当当前整体恢复。说明见[第2阶段入口]({(r.ROOT/'ssn_baseline/README_stage2.md').as_posix()})。不存在当前可用的训练／小试命令，不提供虚构入口。

下一项仅第3阶段训练侧CPU输入核验与小试协议／人工入口准备；未批准模型操作。旧实验结果与历史失败不改变，现有E已观察，未来复用只作探索比较，不能称未见确认性证据。MIT作者许可／helper许可及V1权重授权局限沿用第1阶段，继续限个人非商业公开研究。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    args = parser.parse_args()
    forbidden = {"torch", "torchvision", "numpy", "PIL", "cv2", "anomalib"}
    r.require(not any(n.split(".")[0] in forbidden for n in sys.modules), "Forbidden package already imported")
    def guard(event, arguments):
        if event == "import" and (arguments[0].split(".")[0] in forbidden or arguments[0] == "ssn_baseline.model" or arguments[0].startswith("ssn_baseline.core_v1")):
            raise RuntimeError("Only CPU metadata/AST preparation authorized")
        if event.startswith("pickle.") or event == "socket.connect":
            raise RuntimeError("No pickle or network authorized")
    sys.addaudithook(guard)
    for target in [OUT, CORE]:
        for parent in [target, *target.parents]:
            if parent == r.ROOT:
                break
            r.require(not parent.is_symlink() and not parent.is_junction(), "Redirected static output ancestor")
    print(json.dumps({"prepare": prepare, "verify": verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
