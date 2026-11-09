"""CPU formal training inputs, complete fixed requests and scalar schedules."""
from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as c
from . import formal_common as f
from .data import starts, window, normalized

def schedule_verification():
    updates = [1, 8000, 8001, 9000, 9001, 10000]
    rates = [[1e-5, 1e-4], [1e-5, 1e-4], [2e-6, 2e-5], [2e-6, 2e-5], [4e-7, 4e-6], [4e-7, 4e-6]]
    c.require([c.lr_at(n, f.STEPS) for n in updates] == rates, "Formal LR boundary differs")
    return {"updates": dict(zip(map(str, updates), rates)),
            "resume_next": {str(done): c.lr_at(done + 1, f.STEPS) if done < f.STEPS else None for done in [0, 999, 1000, 7999, 8000, 8999, 9000, 9999, 10000]}}

def requests(items):
    real_rng, normal_rng, flip_rng = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([f.SEED, k]))) for k in range(3)]
    real_order = [int(x) for _ in range(1000) for x in real_rng.permutation(5)]
    result = []
    for index in range(f.STEPS):
        if index % 2 == 0:
            item_index = real_order[index // 2]
            choices = [x for x in items[item_index]["windows"] if x["positive_pixels"] > 0]
            choice = choices[int(real_rng.integers(len(choices)))]
        else:
            item_index = 5 + int(normal_rng.integers(1885))
            choices = items[item_index]["windows"]
            choice = choices[int(normal_rng.integers(len(choices)))]
        result.append({"update": index + 1, "item_index": item_index, "id": items[item_index]["id"], "y": choice["y"],
                       "positive_pixels": choice["positive_pixels"], "flip_h": bool(flip_rng.random() < .5), "flip_v": bool(flip_rng.random() < .5)})
    return result

def load_native(item):
    with np.load(f.PREP / item["cache"], allow_pickle=False) as z:
        c.require(set(z.files) == {"rgb", "mask"}, "Formal native cache keys differ")
        rgb, mask = z["rgb"], z["mask"]
    h, w = item["height"], item["width"]
    c.require(rgb.shape == (h, w, 3) and rgb.dtype == np.uint8 and mask.shape == (h, w) and mask.dtype == np.uint8, "Formal cache geometry/dtype differs")
    c.require(set(np.unique(mask).tolist()) <= {0, 1} and bool(mask.any()) == (item["kind"] == "real"), "Formal cache GT semantics differ")
    return rgb, mask

def request_input(item, request):
    rgb, mask = load_native(item)
    c.require(request["id"] == item["id"] and request["y"] in starts(item["height"]), "Request ID/coordinate differs")
    pack = window(rgb, mask, request["y"])
    if request["flip_h"]:
        for key in ["rgb_uint8", "target", "valid"]: pack[key] = pack[key][:, ::-1].copy()
    if request["flip_v"]:
        for key in ["rgb_uint8", "target", "valid"]: pack[key] = pack[key][::-1].copy()
    pack["image"] = normalized(pack["rgb_uint8"])
    c.require(int(pack["target"][pack["valid"]].sum()) == request["positive_pixels"] and int(pack["valid"].sum()) == 256 * item["width"] and not pack["target"][~pack["valid"]].any(), "Flipped GT/valid domain differs")
    return pack

def verify(decode=False):
    seal = f.strict_seal(f.PREP)
    p = c.load(f.PREP / "protocol.json"); items = c.load(f.PREP / "inputs.json"); seq = c.load(f.PREP / "requests.json")
    c.require(p["implementation"] == f.implementation() and p["pilot_implementation"] == c.code_identity(), "Formal/pilot source identity differs")
    c.verify_upstream(p["upstream"])
    pilot = c.load(c.PREP / "protocol.json")
    c.require(p["compatibility"] == pilot["compatibility"] and p["model_and_loss"] == {k: pilot[k] for k in ["Adam", "loss", "window", "batch", "workers", "precision"]}, "Frozen model/loss/compatibility policy differs")
    split = c.load(c.DATA / "split.json")
    original = split["R_real5"] + split["N_train1885"]
    c.require(len(items) == 1890 and len(split["R_real5"]) == 5 and len(split["N_train1885"]) == 1885, "Training pool count differs")
    c.require([x["id"] for x in items] == [x["id"] for x in original] and [x["id"] for x in items[:5]] == c.R5, "Formal role/order differs")
    expected_files = {"transaction.json", "inputs.json", "requests.json", "protocol.json", "第5阶段正式基线准备与人工运行.md", "complete.json"} | {f"inputs/{x['id']}.npz" for x in original}
    c.require(c.paths(f.PREP) == expected_files and all(x["cache"] == f"inputs/{x['id']}.npz" for x in items), "Unknown formal preparation/cache member")
    c.require([x["kind"] for x in items] == ["real"] * 5 + ["normal"] * 1885, "Formal input roles differ")
    for item, source in zip(items, original):
        c.require(all(item[k] == source[k] for k in ["id", "image", "mask", "height", "width"]), "Formal original coordinate metadata differs")
        c.require([x["y"] for x in item["windows"]] == starts(item["height"]), "Formal tail/stride windows differ")
        c.require(2 <= item["width"] < 256 and item["source_sha256"]["image"] == source["image_sha256"], "Formal source shape/hash differs")
        if decode:
            rgb, mask = load_native(item)
            c.require([int(mask[x["y"]:x["y"]+256].sum()) for x in item["windows"]] == [x["positive_pixels"] for x in item["windows"]], "Formal legal-window positivity differs")
    c.require(p["seed"] == f.SEED and p["steps"] == f.STEPS and p["schedules"] == schedule_verification(), "Formal budget/schedule differs")
    c.require(seq == requests(items) and len(seq) == 10000, "Formal frozen sequence differs")
    counts = Counter(x["id"] for x in seq)
    c.require(all(counts[x] == 1000 for x in c.R5) and sum(x["item_index"] >= 5 for x in seq) == 5000 and dict(counts) == p["exposure_by_id"], "Formal exposure differs")
    c.require(p["synthesis"] is False and p["initialization"] == "same generic ImageNet encoder plus new decoder/head; never probe/task weights", "Formal initialization/synthesis scope differs")
    c.require(seal["identity"]["meaning"] == "CPU formal preparation only; no model or training executed", "Preparation meaning differs")
    return p, items, seq

def prepare():
    if f.PREP.exists():
        p, items, seq = verify(decode=True)
        return {"formal_ready_for_manual_training": True, "reused": True, "training_executed": False, "unique_training_images": len(items), "requests": len(seq)}
    upstream, pilot = f.upstream()
    resource = c.resources(2, 2)
    c.require(not f.RUN.exists() and not f.MODEL.exists(), "Formal output already present before preparation")
    for name in f.CODE:
        if name.endswith(".py"): ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"), filename=name)
    split = c.load(c.DATA / "split.json")
    rows = split["R_real5"] + split["N_train1885"]
    c.require(len(rows) == 1890 and len({x["id"] for x in rows}) == 1890, "Formal pool differs")
    prohibited = {r["id"] for k in ["Q_normal200", "U_anomaly241", "E_official1004"] for r in split[k]}
    c.require(not prohibited & {r["id"] for r in rows}, "Formal role leakage")
    inventory = c.load(c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json")["files"]
    old_r = c.load(c.INPUT / "protocol.json")["items"][:5]
    f.PREP.mkdir(parents=True, exist_ok=False)
    c.save(f.PREP / "transaction.json", {"scope": "R5/N1885 CPU decode only; no visual selection/model", "upstream": upstream})
    items = []
    try:
        for index, row in enumerate(rows):
            identity = {k: c.sha(c.DATA / row[k]) for k in ["image", "mask"]}
            c.require(identity["image"] == row["image_sha256"] and all(identity[k] == inventory[row[k]] for k in identity), "Canonical formal source hash differs")
            c.require(row["height"] >= 256 and 2 <= row["width"] < 256, "Unsupported native geometry")
            with Image.open(c.DATA / row["image"]) as im:
                c.require(im.size == (row["width"], row["height"]) and im.mode in ("RGB", "L"), "Formal source RGB mode/size differs")
                rgb = np.asarray(im.convert("RGB")).copy()
            with Image.open(c.DATA / row["mask"]) as im:
                c.require(im.size == (row["width"], row["height"]) and im.mode in ("L", "1"), "Formal GT mode/size differs")
                raw_gt = np.asarray(im.convert("L")).copy()
            c.require(set(np.unique(raw_gt).tolist()) <= {0, 255}, "Nonbinary formal source GT")
            mask = (raw_gt > 0).astype(np.uint8)
            c.require(bool(mask.any()) == (index < 5), "R5/N1885 source GT semantics differ")
            windows = [{"y": y, "positive_pixels": int(mask[y:y+256].sum())} for y in starts(row["height"])]
            if index < 5:
                c.require(windows == old_r[index]["all_legal_windows"] and row["id"] == old_r[index]["id"], "R5 full legal-window counts differ from sealed pilot input")
            name = f"inputs/{row['id']}.npz"
            (f.PREP / "inputs").mkdir(exist_ok=True)
            np.savez_compressed(f.PREP / name, rgb=rgb, mask=mask)
            items.append({k: row[k] for k in ["id", "image", "mask", "height", "width"]} | {"kind": "real" if index < 5 else "normal", "source_sha256": identity, "windows": windows, "cache": name})
            if (index + 1) % 100 == 0:
                c.resources(2, 1)
                print({"CPU_training_inputs": index + 1, "of": 1890, "model_updates": 0}, flush=True)
        seq = requests(items)
        for req in seq:  # Geometry/label scalar verification on every fixed request.
            item = items[req["item_index"]]
            c.require(req["positive_pixels"] > 0 if req["item_index"] < 5 else req["positive_pixels"] == 0, "Invalid request GT role")
            c.require(req["y"] in starts(item["height"]), "Invalid request y")
        checked = 0
        for item in items[:5] + [r for r in items[5:] if r["id"] in c.N16]:
            for option in item["windows"]:
                if item["kind"] == "real" and option["positive_pixels"] == 0: continue
                for h, v in [(False, False), (True, False), (False, True), (True, True)]:
                    request_input(item, {"id": item["id"], "y": option["y"], "positive_pixels": option["positive_pixels"], "flip_h": h, "flip_v": v})
                    checked += 1
        counts = Counter(x["id"] for x in seq)
        p = {"schema": "pretrained-segmentation-formal-preparation-v1", "seed": f.SEED, "steps": f.STEPS,
             "implementation": f.implementation(), "pilot_implementation": c.code_identity(), "upstream": upstream,
             "compatibility": pilot["compatibility"], "model_and_loss": {k: pilot[k] for k in ["Adam", "loss", "window", "batch", "workers", "precision"]},
             "initialization": "same generic ImageNet encoder plus new decoder/head; never probe/task weights", "synthesis": False,
             "schedules": schedule_verification(), "exposure_by_id": dict(counts), "real_exposures": 5000, "normal_exposures": 5000,
             "unique_pool_images": 1890, "normal_pool_images": 1885, "unique_sampled_normal_images": len({x["id"] for x in seq if x["item_index"] >= 5}),
             "source_RGB_decodes": 1890, "source_GT_decodes": 1890, "Q_E_U_decodes": 0, "additional_visual_review": False,
             "request_randomness": "PCG64 SeedSequence([2026102202,0]): pre-generate1000 R permutations then uniform positive window; namespace1 uniform N image/window; namespace2 independent p.5 horizontal/vertical flips; no model RNG consumption",
             "CPU_flip_cases_verified": checked, "horizontal_flips": sum(x["flip_h"] for x in seq), "vertical_flips": sum(x["flip_v"] for x in seq),
             "checkpoints": list(range(1000, 10001, 1000)), "resource_snapshot": resource,
             "manual_interpreter": str(c.ENV / "Scripts/python.exe"), "GPU_complete_resume_verified": False,
             "meaning": "formal CPU preparation only; no fitting gate/real evaluation computed"}
        c.save(f.PREP / "inputs.json", items); c.save(f.PREP / "requests.json", seq); c.save(f.PREP / "protocol.json", p)
        c.save(f.PREP / "第5阶段正式基线准备与人工运行.md", report(p))
        expected_files = {"transaction.json", "inputs.json", "requests.json", "protocol.json", "第5阶段正式基线准备与人工运行.md"} | {x["cache"] for x in items}
        c.require(c.paths(f.PREP) == expected_files and {x.relative_to(f.PREP).as_posix() for x in f.PREP.rglob("*") if x.is_dir()} == {"inputs"}, "Unknown formal preparation member before sealing")
        c.seal(f.PREP, {"meaning": "CPU formal preparation only; no model or training executed"})
    except Exception as exc:
        c.save(f.PREP / "failure.json", {"error": repr(exc), "rule": "preserve, no altered sample/seed or automatic retry"})
        raise
    verify(decode=True)
    return {"formal_ready_for_manual_training": True, "reused": False, "unique_training_images": 1890, "sampled_normal_images": p["unique_sampled_normal_images"], "requests": 10000, "training_executed": False}

def report(p):
    return f'''# 第5阶段：正式纯真实基线准备与人工运行

小试最终2000步固定21训练窗gate通过证据、来源、旧源码／环境和通用编码器身份已核对。本次仅CPU准备，不导入Torch／SMP／timm／模型、创建tensor／优化器、加载参数、前向／反向、更新、训练、校准或评价，不读取Q／E／U图像／GT。封存小试源码、入口、全局路径、输入、权重和成果不改。

独立静态审阅未发现核心采样／模型复用／恢复边界的实质阻碍；末次准备目录精确清单与跨恢复CUDA峰值记录增量由执行方补充源码／AST核验，独立审阅未覆盖该增量。没有以静态审阅冒充GPU运行或完整恢复实测。

## 实际CPU输入与冻结请求

严格R5／N1885原角色原顺序，共1890训练池图像；RGB／GT各解码1890次，逐文件核原包规范导出hash、原尺寸、二值GT及正常零GT。此为程序合法性核对，没有新增视觉筛选或外观统计。原始RGB／GT以独立npz保存，无拉伸／缩放／合成。R5全部合法窗阳性计数与原封存计数一致；N仅合法原生窗。反射补边在右，翻转后GT和有效域同步，补边不参与损失。

固定seed2026102202、10000更新，奇R偶N各5000曝光，R各1000。R流namespace0先固定1000个R5洗牌循环再均匀取其非空窗；N流namespace1均匀有放回选N1885图及合法窗，实际涉及{p['unique_sampled_normal_images']}张不同N；独立namespace2每请求生成水平／垂直p0.5翻转。模型随机流独立。水平翻转{p['horizontal_flips']}次、垂直{p['vertical_flips']}次；这些是更新曝光，不是独立图数。全部10000请求角色／几何已标量核对，固定R5＋旧N16所有适用窗的四种翻转共{p['CPU_flip_cases_verified']}例CPU有效域／阳性计数核对通过。未用旧真实评价分数选采样。

模型与损失只读复用已封存model.py：同一ImageNet ResNet18＋新U-Net decoder／head、无辅助分类头；编码器20 BN统计固定但affine可训练，decoder10处GN8，CE＋Dice含空GT，Adam两组1e-5／1e-4、FP32／batch1／workers0。不能加载probe／旧任务权重初始化。通用header仍按fc两键移除／20旧BN零计数显式清单严格加载，无未解释缺项。

更新1—8000为1e-5／1e-4，8001—9000为2e-6／2e-5，9001—10000为4e-7／4e-6。8000／8001、9000／9001及自有恢复下一更新均仅作CPU标量核验；未运行模型或恢复。

## 唯一当前人工训练命令

```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_segmentation.formal_cli train
```

这是参数更新入口，仅由用户手动启动。启动RAM≥4.5GiB／空闲显存≥6GiB／磁盘≥20GiB，CPU准备时RAM{p['resource_snapshot']['free_RAM_GiB']:.3f}GiB、磁盘{p['resource_snapshot']['free_disk_GiB']:.3f}GiB仅为当时快照。每100步显示与采样资源，可用RAM<2GiB在明确更新边界保存完整停止点并退出；不自动降低规格。小试2000步实测8.88分钟，按更新数线性参考约44分钟，正式输入IO／增强／检查点不同且尚未实测，不能保证该耗时或沿用小试峰值。

模型目录：{f.MODEL.as_posix()}；训练结果：{f.RUN.as_posix()}；输入／配置：{f.PREP.as_posix()}。每1000步保存完整自有检查点共10份，含模型safetensors、受限读取的本组optimizer、Python／NumPy／Torch CPU／CUDA RNG、序列游标、实际LR／损失历史、来源与实现身份。Ctrl+C不额外保存。仅用户显式--resume-from指定本组最新完整点可恢复；未知文件／锁／不完整保存／failure保留停止，不自动恢复、重训或删标记。完整缓存只核验不更新；GPU完整恢复尚未实测，标量检查不等于恢复成功。

用户跑完后先核最终10000步、10保存点、实际曝光／LR、参数与BN身份及实测资源，不自动评分、Q校准或E评价。E为旧已见探索集，不作为选型或未见验证。本次只准备，不生成正式权重或检测成绩。
'''
