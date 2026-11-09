"""替代生成器实验的测量与配对对比。

对 G1 在新版刺激上推理，并与原版逐条配对比较。
因为两版 S 与 Cx 相同、背景与位置相同、τ 相同，
检出率的差异只能归因于边缘剖面（软边 → 硬边）。
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from .capability_g01 import frozen_tau, load_g, score_one

ROOT = Path(__file__).resolve().parent.parent
ALT = ROOT / "results" / "pod_probe" / "alt_generator_v1"
CAP = ROOT / "results" / "pod_probe" / "capability_v1"
OUT = ALT / "measure"

SHAPES = ("mura_p25", "mura_med", "mura_p75", "line_med")
DROP_STRONG = 0.30
DROP_WEAK = 0.10


def score_alt(cond: str = "G1", overwrite: bool = False) -> dict:
    target = OUT / f"{cond}_scores.json"
    if target.exists() and not overwrite:
        raise RuntimeError(f"{cond} alt scores already sealed")

    index = json.loads((ALT / "stimuli" / "index.json").read_text(encoding="utf-8"))
    tau = frozen_tau(cond)
    rec, seg = load_g(cond)

    rows = []
    for i, item in enumerate(index["records"], 1):
        npz = ALT / "stimuli" / f"{item['id']}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as handle:
            image = np.array(handle["image"], dtype=np.float32)
        value = score_one(rec, seg, image)
        del image
        if i % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        rows.append({**{k: item.get(k) for k in
                        ("id", "kind", "shape", "target_SEMU", "background_index",
                         "paired_original", "S_mm2_original", "S_mm2_measured",
                         "Cx_percent_original", "Cx_percent_measured")},
                     "score": value, "hit": bool(value > tau)})

    result = {"condition": cond, "tau": tau, "n_images": len(rows),
              "generator": "alt_generator_v1: hard edge, same S and Cx"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({**result, "rows": rows}, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return result


def compare(cond: str = "G1") -> dict:
    alt = json.loads((OUT / f"{cond}_scores.json").read_text(encoding="utf-8"))["rows"]
    base_path = CAP / f"g01_{cond}" / "scores.json"
    base = {r["id"]: r for r in
            json.loads(base_path.read_text(encoding="utf-8"))["rows"]}
    tau = frozen_tau(cond)

    paired = []
    for r in alt:
        if r["kind"] != "stimulus":
            continue
        src = base.get(r["paired_original"])
        if src is None:
            continue
        paired.append({"shape": r["shape"], "level": r["target_SEMU"],
                       "orig_hit": bool(src["hit"]), "alt_hit": bool(r["hit"]),
                       "orig_score": float(src["score"]), "alt_score": float(r["score"]),
                       "S_orig": r["S_mm2_original"], "S_alt": r["S_mm2_measured"]})

    alt_base = [r for r in alt if r["kind"] == "baseline"]
    orig_base = [v for k, v in base.items() if k.startswith("baseline")]

    by_shape = {}
    for shape in SHAPES:
        sel = [p for p in paired if p["shape"] == shape]
        if not sel:
            continue
        o = np.mean([p["orig_hit"] for p in sel])
        a = np.mean([p["alt_hit"] for p in sel])
        # 逐档
        levels = {}
        for lv in sorted({p["level"] for p in sel}):
            g = [p for p in sel if p["level"] == lv]
            levels["%g" % lv] = {"n": len(g),
                                 "orig": float(np.mean([p["orig_hit"] for p in g])),
                                 "alt": float(np.mean([p["alt_hit"] for p in g]))}
        hi_orig = max(v["orig"] for v in levels.values())
        hi_alt = max(v["alt"] for v in levels.values())
        lo_orig = min(v["orig"] for v in levels.values())
        lo_alt = min(v["alt"] for v in levels.values())
        drop = hi_orig - hi_alt
        by_shape[shape] = {
            "n_pairs": len(sel),
            "orig_rate": float(o), "alt_rate": float(a), "delta": float(a - o),
            "orig_highest_cell": hi_orig, "alt_highest_cell": hi_alt,
            "highest_drop": float(drop),
            "orig_lowest_cell": lo_orig, "alt_lowest_cell": lo_alt,
            "consistent_drop": bool(drop > 0 and lo_alt <= lo_orig),
            "levels": levels,
            "verdict": ("strong_dependence" if drop >= DROP_STRONG
                        else "partial_dependence" if drop >= DROP_WEAK
                        else "no_evidence"),
        }

    flips_hit_to_miss = sum(1 for p in paired if p["orig_hit"] and not p["alt_hit"])
    flips_miss_to_hit = sum(1 for p in paired if not p["orig_hit"] and p["alt_hit"])

    overall_drop = (max(v["orig_highest_cell"] for v in by_shape.values())
                    - max(v["alt_highest_cell"] for v in by_shape.values()))

    result = {
        "condition": cond, "tau": tau,
        "design": ("paired: same background, position, S, Cx and tau; "
                   "only edge profile differs (soft -> hard)"),
        "n_pairs": len(paired),
        "orig_rate_overall": float(np.mean([p["orig_hit"] for p in paired])),
        "alt_rate_overall": float(np.mean([p["alt_hit"] for p in paired])),
        "delta_overall": float(np.mean([p["alt_hit"] for p in paired])
                               - np.mean([p["orig_hit"] for p in paired])),
        "highest_cell_drop_max": float(overall_drop),
        "hit_to_miss": flips_hit_to_miss,
        "miss_to_hit": flips_miss_to_hit,
        "baseline_fp_orig": (sum(1 for r in orig_base if r["hit"]),
                             len(orig_base)),
        "baseline_fp_alt": (sum(1 for r in alt_base if r["hit"]), len(alt_base)),
        "by_shape": by_shape,
        "verdict": ("strong_dependence" if overall_drop >= DROP_STRONG
                    else "partial_dependence" if overall_drop >= DROP_WEAK
                    else "no_evidence"),
        "note": ("strong_dependence 表示该检测器的响应主要来自原生成器的边缘剖面，"
                 "即它记住的是生成器特征而非异常本身。no_evidence 表示仅改边缘几乎不影响响应。"),
    }

    (OUT / f"{cond}_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    cond = sys.argv[1] if len(sys.argv) > 1 else "G1"
    score_alt(cond)
    r = compare(cond)
    print(json.dumps({k: r[k] for k in
                      ("condition", "n_pairs", "orig_rate_overall", "alt_rate_overall",
                       "delta_overall", "highest_cell_drop_max", "hit_to_miss",
                       "miss_to_hit", "baseline_fp_orig", "baseline_fp_alt", "verdict")},
                     ensure_ascii=False, indent=1))
    for s, v in r["by_shape"].items():
        print("%-10s 原%.3f → 新%.3f  Δ%+.3f  最高档 %.3f→%.3f  %s" % (
            s, v["orig_rate"], v["alt_rate"], v["delta"],
            v["orig_highest_cell"], v["alt_highest_cell"], v["verdict"]))
