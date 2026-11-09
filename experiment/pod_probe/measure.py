"""推理与三读数：整图判定、定位命中、相对基线的信号变化。

只推理一次并缓存；部分或未知事务保留停止，不补推、不重复推理。
"""

from __future__ import annotations

import json

import numpy as np

from . import bank256
from . import protocol as P
from . import stimuli as S


def run(device: str = "cuda", overwrite: bool = False) -> dict:
    score_dir = P.out_dir("scores")
    if (score_dir / "complete.json").exists() and not overwrite:
        raise RuntimeError("scores already sealed; use verify-scores or a new branch")

    thr_path = P.out_dir("calibration") / "threshold.json"
    if not thr_path.exists():
        raise RuntimeError("calibration missing; run calibration first")
    tau = json.loads(thr_path.read_text(encoding="utf-8"))["tau"]

    index = json.loads((P.out_dir("stimuli") / "index.json").read_text(encoding="utf-8"))
    built = bank256.build(device)
    if abs(built["tau"] - tau) > 1e-9:
        raise RuntimeError("rebuilt threshold does not match sealed calibration; stop")

    stim_dir = P.out_dir("stimuli")
    rows = []
    heats = {}
    for rec in index["records"]:
        rgb = S.load_image(stim_dir / f"{rec['id']}.npz")
        score, heat = bank256.predict_score(built["model"], rgb)
        heats[rec["id"]] = heat.astype(np.float32)
        row = {"id": rec["id"], "kind": rec["kind"],
               "background_index": rec["background_index"], "score": score,
               "peak_xy": [int(np.unravel_index(int(np.argmax(heat)), heat.shape)[1]),
                           int(np.unravel_index(int(np.argmax(heat)), heat.shape)[0])]}
        if rec["kind"] == "stimulus":
            eff = S.load_effective(stim_dir / f"{rec['id']}.npz")
            row.update({"shape": rec["shape"], "target_SEMU": rec["target_SEMU"],
                        "SEMU_actual": rec["SEMU_actual"], "amplitude": rec["amplitude"],
                        "peak_in_mask": bool(eff[tuple(row["peak_xy"])[::-1]])})
        rows.append(row)

    base = {r["background_index"]: r["score"] for r in rows if r["kind"] == "baseline"}
    for r in rows:
        r["baseline_score"] = base[r["background_index"]]
        r["delta_score"] = r["score"] - base[r["background_index"]]
        r["hit"] = bool(r["score"] > tau)

    score_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_dir / "heatmaps.npz", **heats)
    P.write_json(score_dir / "scores.json", {
        "version": P.VERSION, "tau": tau, "device": device, "rows": rows,
        "baseline_false_positive": int(sum(1 for r in rows
                                          if r["kind"] == "baseline" and r["hit"])),
        "baseline_count": int(sum(1 for r in rows if r["kind"] == "baseline")),
    })
    P.write_json(score_dir / "complete.json", {"stage": "scores", "rows": len(rows)})
    return {"rows": len(rows), "tau": tau,
            "baseline_fp": int(sum(1 for r in rows if r["kind"] == "baseline" and r["hit"]))}


def verify() -> dict:
    score_dir = P.out_dir("scores")
    path = score_dir / "scores.json"
    if not path.exists():
        raise RuntimeError("no sealed scores")
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for r in data["rows"]:
        if not np.isfinite(r["score"]):
            problems.append({"id": r["id"], "reason": "non-finite score"})
        if abs(r["score"] - r["baseline_score"] - r["delta_score"]) > 1e-9:
            problems.append({"id": r["id"], "reason": "delta inconsistent"})
        if r["hit"] != (r["score"] > data["tau"]):
            problems.append({"id": r["id"], "reason": "hit inconsistent"})
    stim = [r for r in data["rows"] if r["kind"] == "stimulus"]
    shapes = {(r["shape"], r["target_SEMU"]) for r in stim}
    expected = len(P.SHAPES) * len(P.SEMU_LEVELS)
    if len(shapes) != expected:
        problems.append({"reason": f"missing (shape, level) cells: {len(shapes)}/{expected}"})
    return {"rows": len(data["rows"]), "tau": data["tau"], "problems": problems,
            "passed": not problems}
