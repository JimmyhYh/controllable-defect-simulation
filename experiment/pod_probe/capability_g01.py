"""G0/G1 能力卡：用 AnoLCD 真实背景刺激测量可学习生成模型的响应。

这是**跨域**测量：G0/G1 在自建渲染器背景上训练，此处测的是它们对
AnoLCD 真实显示屏背景 + 参数化刺激的响应。故结果只能读作
"该模型对真实背景刺激有无响应"，不与同域三行直接比 a50。

口径与 holdout 冻结协议一致：
- 预处理：PIL RGB whole-image bilinear 256，float32/255，无其他归一化；
- 分数：raw anomaly-channel softmax 在 256×256 上的最大值；
- 阈值：各自冻结 q0.95，不重新校准。
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "DRAEM"))
from model_unet import DiscriminativeSubNetwork, ReconstructiveSubNetwork  # noqa: E402

STAGE2 = ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"
OUT = ROOT / "results" / "pod_probe" / "capability_v1"
CALIB = ROOT / "results" / "controlled_generation" / "calibration" / "calibration_verification.json"
MODELS = {"G0": "G0_20261030", "G1": "G1_20261030"}


def frozen_tau(cond: str) -> float:
    data = json.loads(CALIB.read_text(encoding="utf-8"))
    return float(data["results"][cond]["threshold"])


def _disable_running_stats(net) -> None:
    """协议规定 track_running_stats=False、running_buffers=None：
    BN 用当前单张输入的 batch/spatial 统计，不读 running buffer。
    权重里因此没有 running_* 键，加载时必须按此构造。"""
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None


def load_g(cond: str):
    folder = ROOT / "models" / "controlled_generation" / "formal_draem" / MODELS[cond]
    rec = ReconstructiveSubNetwork(3, 3).cuda()
    seg = DiscriminativeSubNetwork(6, 2).cuda()
    _disable_running_stats(rec)
    _disable_running_stats(seg)
    rec_state = torch.load(folder / "reconstruction.pth", map_location="cpu", weights_only=True)
    seg_state = torch.load(folder / "segmentation.pth", map_location="cpu", weights_only=True)
    missing_rec = rec.load_state_dict(rec_state, strict=False).missing_keys
    missing_seg = seg.load_state_dict(seg_state, strict=False).missing_keys
    # 只允许缺失 BN 的运行统计缓冲；出现其他缺失说明权重与结构不匹配
    for name in list(missing_rec) + list(missing_seg):
        if "running_" not in name and "num_batches_tracked" not in name:
            raise RuntimeError(f"unexpected missing key: {name}")
    rec.eval()
    seg.eval()
    return rec, seg


def score_one(rec, seg, rgb: np.ndarray) -> float:
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).cuda()
    with torch.inference_mode():
        probability = torch.softmax(seg(torch.cat((rec(x), x), 1)), dim=1)[0, 1]
        return float(probability.max().cpu())


def run(cond: str, overwrite: bool = False) -> dict:
    if cond not in MODELS:
        raise ValueError(f"condition {cond} not in {list(MODELS)}")
    target = OUT / f"g01_{cond}"
    record = target / "scores.json"
    if record.exists() and not overwrite:
        raise RuntimeError(f"{cond} already sealed; use --overwrite or a new branch")

    tau = frozen_tau(cond)
    idx = json.loads((STAGE2 / "stimuli" / "index.json").read_text(encoding="utf-8"))
    records = idx["records"]
    rec, seg = load_g(cond)

    rows, skipped = [], 0
    for i, item in enumerate(records, 1):
        npz = STAGE2 / "stimuli" / f"{item['id']}.npz"
        if not npz.exists():
            skipped += 1
            continue
        with np.load(npz) as handle:
            image = np.array(handle["image"], dtype=np.float32)
        value = score_one(rec, seg, image)
        del image
        if i % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append({"id": item["id"], "kind": item["kind"],
                     "shape": item.get("shape"),
                     "target_SEMU": item.get("target_SEMU"),
                     "SEMU_actual": item.get("SEMU_actual"),
                     "background_index": item.get("background_index"),
                     "score": value, "hit": bool(value > tau)})

    baseline = [r for r in rows if r["kind"] == "baseline"]
    stim = [r for r in rows if r["kind"] == "stimulus"]
    base_fp = sum(1 for r in baseline if r["hit"])
    base_rate = base_fp / len(baseline) if baseline else None

    cells = {}
    for shape in sorted({r["shape"] for r in stim}):
        for level in sorted({r["target_SEMU"] for r in stim if r["shape"] == shape}):
            sel = [r for r in stim if r["shape"] == shape and r["target_SEMU"] == level]
            if not sel:
                continue
            hits = sum(1 for r in sel if r["hit"])
            cells["%s@%g" % (shape, level)] = {
                "n": len(sel), "hits": hits, "detection_rate": hits / len(sel)}

    by_shape = {}
    for shape in sorted({r["shape"] for r in stim}):
        sel = [r for r in stim if r["shape"] == shape]
        rates = [c["detection_rate"] for k, c in cells.items() if k.startswith(shape + "@")]
        lo, hi = (min(rates), max(rates)) if rates else (None, None)
        by_shape[shape] = {
            "n": len(sel),
            "detection_rate_overall": sum(1 for r in sel if r["hit"]) / len(sel),
            "lowest_cell_rate": lo,
            "highest_cell_rate": hi,
            "no_response": bool(hi is not None and hi < 0.50),
            "response_minus_baseline": (None if hi is None else hi - base_rate),
            "a50_identifiable": bool(lo is not None and lo <= 0.5 < hi),
            "a90_identifiable": bool(lo is not None and lo < 0.9 <= hi),
        }

    result = {
        "condition": cond,
        "model_dir": MODELS[cond],
        "tau": tau,
        "tau_source": "controlled_generation/calibration/calibration_verification.json",
        "tau_domain": "softmax probability, max over pixels",
        "domain_note": ("跨域测量：训练域为自建渲染器背景，刺激背景为 AnoLCD 真实正常图。"
                        "结果只能读作“对真实背景刺激有无响应”，不与同域三行比 a50。"),
        "n_images": len(rows), "n_skipped_missing": skipped,
        "n_baseline": len(baseline),
        "baseline_false_positive": base_fp,
        "baseline_false_positive_rate": base_rate,
        "n_stimulus": len(stim),
        "stimulus_detection_rate_overall": (
            sum(1 for r in stim if r["hit"]) / len(stim) if stim else None),
        "by_shape": by_shape,
        "cells": cells,
    }

    target.mkdir(parents=True, exist_ok=True)
    (target / "scores.json").write_text(
        json.dumps({**result, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    codes = sys.argv[1:] or ["G0", "G1"]
    out = {}
    for c in codes:
        r = run(c)
        out[c] = {"tau": r["tau"],
                  "baseline_false_positive": r["baseline_false_positive"],
                  "n_baseline": r["n_baseline"],
                  "stimulus_detection_rate_overall": r["stimulus_detection_rate_overall"],
                  "by_shape": {s: {"overall": round(v["detection_rate_overall"], 3),
                                   "low": v["lowest_cell_rate"],
                                   "high": v["highest_cell_rate"],
                                   "no_response": v["no_response"]}
                               for s, v in r["by_shape"].items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
