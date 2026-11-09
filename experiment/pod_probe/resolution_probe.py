"""受控探针阶段 3：分辨率验证。

只做建库、校准、推理与聚合，不训练、不修改冻结管线。
每个分辨率是独立身份，落在独立子目录。
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import wide_resnet50_2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "patchcore-inspection" / "src"))

import patchcore.patchcore
import patchcore.sampler
from detector import UnicodePathFaissNN

OUT = ROOT / "results" / "pod_probe" / "resolution_probe_v1"
SPLIT = ROOT / "data" / "weak_localization" / "split.json"
WEIGHTS = ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"

SIZES = (256, 512)
BUILD_IMAGES = 80
# 与冻结配置相同的覆盖密度：核心数随特征池同步增长。
# 固定绝对核心数会使 512 档覆盖密度降到 1/4，等于同时改变两个因素（见 attempt1 结果文档）。
SAMPLER_RATIO = 0.01
SEED = 42
Q = 0.95

# A 档（256）必须复现的区间；不满足则停止，不解释 B 档
REPRO_SPOT = (0.50, 0.62)
REPRO_OVERALL = (0.80, 0.88)
REPRO_MURA_MIN = 0.90

KINDS = ("mura", "line", "spot", "scrathes", "cylinder")

# 建库最低可用内存门槛（GiB）；不足则拒绝运行，不自动降档
REQUIRED_RAM = {256: 2.5, 512: 6.5}


def estimated_peak_gib(size: int) -> dict:
    """按实测结构估算建库峰值：特征池 + 128 维投影 + anchor 距离矩阵 + 运行时基线。"""
    positions = BUILD_IMAGES * _positions_per_image(size)
    vectors = expected_vectors(size)
    return {
        "feature_pool": positions * 1024 * 4 / 2 ** 30,
        "projection_128d": positions * 128 * 4 / 2 ** 30,
        "anchor_matrix": positions * (vectors + 10) * 4 / 2 ** 30,
        "runtime_baseline": 1.0,
    }


def available_ram_gib() -> float:
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    m = Memory()
    m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return float("nan")
    return m.available / 2 ** 30


def _split() -> dict:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def _positions_per_image(size: int) -> int:
    return (size // 8) ** 2


def sampler_pct(size: int) -> float:
    """覆盖密度固定为 1%，与冻结配置一致。"""
    return SAMPLER_RATIO


def expected_vectors(size: int) -> int:
    return int(BUILD_IMAGES * _positions_per_image(size) * SAMPLER_RATIO)


def load_rgb(root, rel: str, size: int) -> np.ndarray:
    image = Image.open(Path(root) / rel)
    return np.asarray(
        image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / np.float32(255)


def _digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(size: int):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    backbone.name = "wideresnet50"
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, size, size),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(sampler_pct(size), device),
        nn_method=UnicodePathFaissNN(on_gpu=False, num_workers=4),
    )
    return model, device


def _tensor(rgb: np.ndarray):
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return ((t - mean) / std).unsqueeze(0)


def build(size: int, overwrite: bool = False) -> dict:
    if size not in SIZES:
        raise ValueError(f"size {size} not in {SIZES}")
    split = _split()
    folder = OUT / f"bank_{size}"
    record = OUT / f"bank_{size}.json"
    if record.exists() and not overwrite:
        raise RuntimeError(f"bank {size} already sealed; use a new branch or --overwrite")

    folder.mkdir(parents=True, exist_ok=True)
    parts = estimated_peak_gib(size)
    estimate = sum(parts.values())
    ram = available_ram_gib()
    need = REQUIRED_RAM.get(size, estimate)
    if not (ram >= need):
        detail = ", ".join(f"{k} {v:.2f}" for k, v in parts.items())
        raise RuntimeError(
            f"available RAM {ram:.2f} GiB below {need:.2f} GiB required for input {size} "
            f"(estimate {estimate:.2f} GiB = {detail}); "
            "close other programs and retry, do not silently downsize")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, _ = make_model(size)
    root = split["root"]

    def batches():
        for rel in split["N"]:
            yield _tensor(load_rgb(root, rel, size))

    started = time.perf_counter()
    model.fit(batches())
    torch.cuda.synchronize()
    model.save_to_path(str(folder))
    index = model.anomaly_scorer.nn_method.search_index
    actual = int(index.ntotal)
    expected = expected_vectors(size)
    if actual != expected:
        raise RuntimeError(
            f"coreset vectors {actual} != expected {expected}; sampling ratio was rounded "
            "differently, abort rather than compare a different coverage density")

    info = {
        "stage": "build",
        "input_size": size,
        "sampler_pct": sampler_pct(size),
        "coreset_vectors_target": expected_vectors(size),
        "sampler_ratio_fixed": SAMPLER_RATIO,
        "coverage_density": actual / (BUILD_IMAGES * _positions_per_image(size)),
        "coreset_vectors_actual": int(index.ntotal),
        "dimensions": int(index.d),
        "positions_per_image": _positions_per_image(size),
        "build_images": len(split["N"]),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "ram_available_before_fit_GiB": ram,
        "seed": SEED,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "complete_build.json").write_text(
        json.dumps({"size": size, "vectors": info["coreset_vectors_actual"]}), encoding="utf-8")
    return info


def _load_bank(size: int):
    record = OUT / f"bank_{size}.json"
    if not record.exists():
        raise RuntimeError(f"bank {size} not built")
    model, _ = make_model(size)
    model.anomaly_scorer.load(str(OUT / f"bank_{size}"))
    model.forward_modules.eval()
    return model


def _infer(model, paths, root, size):
    scores = np.zeros(len(paths), dtype=np.float64)
    for i, rel in enumerate(paths):
        rgb = load_rgb(root, rel, size)
        with torch.inference_mode():
            s, _maps = model.predict(_tensor(rgb))
        scores[i] = float(s[0])
    return scores


def calibrate(size: int, overwrite: bool = False) -> dict:
    split = _split()
    target = OUT / f"calibration_{size}.json"
    if target.exists() and not overwrite:
        raise RuntimeError(f"calibration {size} already sealed")
    model = _load_bank(size)
    scores = _infer(model, split["Vnew"], split["root"], size)
    threshold = float(np.quantile(scores, Q, method="linear"))
    info = {
        "stage": "calibrate",
        "input_size": size,
        "n_calibration": len(split["Vnew"]),
        "q": Q,
        "method": "linear",
        "alarm": "score>threshold",
        "threshold": threshold,
        "calibration_min": float(scores.min()),
        "calibration_max": float(scores.max()),
        "calibration_median": float(np.median(scores)),
    }
    np.save(OUT / f"calibration_scores_{size}.npy", scores)
    target.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def score(size: int, overwrite: bool = False) -> dict:
    split = _split()
    npz = OUT / f"scores_{size}.npy"
    meta = OUT / f"scores_{size}.json"
    if npz.exists() and not overwrite:
        raise RuntimeError(f"scores {size} already sealed")
    cal = json.loads((OUT / f"calibration_{size}.json").read_text(encoding="utf-8"))
    model = _load_bank(size)
    started = time.perf_counter()
    values = _infer(model, split["Tfinal"], split["root"], size)
    np.save(npz, values)
    info = {
        "stage": "score",
        "input_size": size,
        "n_images": int(values.size),
        "threshold": cal["threshold"],
        "elapsed_seconds": time.perf_counter() - started,
        "scores_sha256": _digest(npz),
    }
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def _kind_of(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "test" else "good"


def _auroc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # 处理并列：取平均秩
    sorted_vals = allv[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


REAL_SEMU = (ROOT / "results" / "pod_probe" / "anolcd_stage2_v1" / "real_semu"
             / "per_image.json")


def _stage2a_applicable() -> set:
    """2a 中 SEMU 可计算的图，用于与 2b 补充拆解的召回口径对齐。"""
    if not REAL_SEMU.exists():
        return set()
    rows = json.loads(REAL_SEMU.read_text(encoding="utf-8"))["rows"]
    return {(r["kind"], r["image"]) for r in rows}


def analyze() -> dict:
    split = _split()
    labels = [_kind_of(r) for r in split["Tfinal"]]
    applicable = _stage2a_applicable()
    per_size = {}
    for size in SIZES:
        npz = OUT / f"scores_{size}.npy"
        calp = OUT / f"calibration_{size}.json"
        if not npz.exists() or not calp.exists():
            continue
        values = np.load(npz).ravel()
        threshold = json.loads(calp.read_text(encoding="utf-8"))["threshold"]
        good = [v for v, k in zip(values, labels) if k == "good"]
        cells = {}
        for kind in KINDS:
            pairs = [(v, r) for v, k, r in zip(values, labels, split["Tfinal"]) if k == kind]
            if not pairs:
                continue
            sel = [v for v, _ in pairs]
            aligned = [v for v, r in pairs
                       if (kind, r.replace("\\", "/").split("/")[-1]) in applicable]
            cells[kind] = {
                "n": len(sel),
                "recall": float(np.mean([v > threshold for v in sel])),
                "auroc": _auroc(sel, good),
                "median_score_over_tau": float(np.median(sel) / threshold),
                "n_stage2a_aligned": len(aligned),
                "recall_stage2a_aligned": (float(np.mean([v > threshold for v in aligned]))
                                           if aligned else None),
                "auroc_stage2a_aligned": _auroc(aligned, good) if aligned else None,
            }
        abnormal = [v for v, k in zip(values, labels) if k != "good"]
        per_size[size] = {
            "threshold": threshold,
            "fpr": float(np.mean([v > threshold for v in good])),
            "n_normal": len(good),
            "recall_overall": float(np.mean([v > threshold for v in abnormal])),
            "auroc_overall": _auroc(abnormal, good),
            "by_kind": cells,
        }

    report = {"stage": "analyze", "sizes": list(per_size), "per_size": per_size}
    if 256 in per_size and 512 in per_size:
        a, b = per_size[256], per_size[512]
        spot_a = a["by_kind"].get("spot", {})
        spot_b = b["by_kind"].get("spot", {})
        # 主判读用 2a 适用子集口径（与 2b 的 0.562 可直接对照）
        ra = spot_a.get("recall_stage2a_aligned")
        rb = spot_b.get("recall_stage2a_aligned")
        ua = spot_a.get("auroc_stage2a_aligned")
        ub = spot_b.get("auroc_stage2a_aligned")
        if ra is None or rb is None:
            ra, rb = spot_a.get("recall", 0.0), spot_b.get("recall", 0.0)
        if ua is None or ub is None:
            ua, ub = spot_a.get("auroc"), spot_b.get("auroc")
        d_recall = (rb or 0.0) - (ra or 0.0)
        d_auroc = (ub or 0.0) - (ua or 0.0)

        repro = {
            "spot_recall_in_range": REPRO_SPOT[0] <= (ra or -1) <= REPRO_SPOT[1],
            "overall_recall_in_range":
                REPRO_OVERALL[0] <= a["recall_overall"] <= REPRO_OVERALL[1],
            "mura_recall_min": (a["by_kind"].get("mura", {}).get("recall", 0.0)
                                >= REPRO_MURA_MIN),
        }
        repro["passed"] = all(repro.values())

        if not repro["passed"]:
            verdict = "前置条件未通过：A 档未复现旧结果，停止解释 B 档"
        elif d_recall >= 0.20 and d_auroc >= 0.05:
            verdict = "分辨率是主因，512 档已显著缓解 → 补样本不对症，生成不必需"
        elif d_recall >= 0.05:
            verdict = "部分归因：分辨率有贡献但不是全部，报连续值"
        else:
            verdict = "分辨率不是主因（至少 512 档不是）→ spot 失效另有原因"

        report["comparison"] = {
            "basis": "stage2a aligned subset (same as 2b 0.562)",
            "spot_n_aligned_256": spot_a.get("n_stage2a_aligned"),
            "spot_n_aligned_512": spot_b.get("n_stage2a_aligned"),
            "spot_recall_256": ra,
            "spot_recall_512": rb,
            "spot_recall_delta": d_recall,
            "spot_auroc_256": ua,
            "spot_auroc_512": ub,
            "spot_auroc_delta": d_auroc,
            "spot_recall_all_256": spot_a.get("recall"),
            "spot_recall_all_512": spot_b.get("recall"),
            "fpr_256": a["fpr"],
            "fpr_512": b["fpr"],
            "fpr_delta": b["fpr"] - a["fpr"],
        }
        report["reproduction_check"] = repro
        report["verdict"] = verdict

    out = OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return report
