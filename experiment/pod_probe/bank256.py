"""256×256 leather PatchCore 实例：建库与 V49 阈值校准。

与旧 224 基线并存但不混用、不比较；本实例只服务阶段 1 的零缩放需求。
构造参数除 input_shape 外与 detector.make_detector 保持一致。
"""

from __future__ import annotations

import random
import sys

import numpy as np
import torch
from PIL import Image

from . import protocol as P

ROOT = P.ROOT
_VENDOR = ROOT / "vendor" / "patchcore-inspection" / "src"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import patchcore.backbones  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402
import patchcore.sampler  # noqa: E402


class _UnicodeFaissNN(patchcore.common.FaissNN):
    """Windows 中文路径下的 FAISS 存取（与 detector.py 同法）。"""

    def save(self, filename):
        import faiss
        from pathlib import Path
        data = faiss.serialize_index(self._index_to_cpu(self.search_index))
        Path(filename).write_bytes(data.tobytes())

    def load(self, filename):
        import faiss
        from pathlib import Path
        data = np.frombuffer(Path(filename).read_bytes(), dtype=np.uint8)
        self.search_index = self._index_to_gpu(faiss.deserialize_index(data))


def make(device: str = "cuda", input_size: int = P.INPUT_SIZE):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / "models"))
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if target.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    weight = ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
    if not weight.is_file():
        raise FileNotFoundError("missing official ImageNet weight: " + str(weight))

    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    model = patchcore.patchcore.PatchCore(target)
    model.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
               device=target, input_shape=(3, input_size, input_size),
               pretrain_embed_dimension=1024, target_embed_dimension=1024, patchsize=3,
               anomaly_score_num_nn=1,
               featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(0.01, target),
               nn_method=_UnicodeFaissNN(on_gpu=False, num_workers=4))
    return model


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    mean = np.asarray(P.IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(P.IMAGENET_STD, dtype=np.float32)
    arr = (rgb.astype(np.float32) - mean) / std
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).float()


def read_rgb(relative: str) -> np.ndarray:
    path = P.DATA / relative
    raw = np.asarray(Image.open(path).convert("RGB").resize(
        (P.INPUT_SIZE, P.INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
    return raw.astype(np.float32) / np.float32(255.0)


def predict_score(model, rgb: np.ndarray) -> tuple[float, np.ndarray]:
    scores, masks = model.predict(to_tensor(rgb))
    score = float(np.max(np.asarray(scores)))
    heat = np.asarray(masks[0], dtype=np.float32)
    return score, heat


def build(device: str = "cuda") -> dict:
    """建库 + 校准，返回身份记录。可重复调用，结果确定。"""
    model = make(device)
    fit_paths = P.bank_paths()
    model.fit([to_tensor(read_rgb(p)) for p in fit_paths])

    cal_scores = []
    for rel in P.calib_paths():
        s, _ = predict_score(model, read_rgb(rel))
        cal_scores.append(s)
    cal = np.asarray(cal_scores, dtype=np.float64)
    tau = float(np.quantile(cal, P.CALIB_QUANTILE, method="linear"))

    return {"model": model, "tau": tau, "calibration_scores": cal,
            "fit_paths": fit_paths, "calib_paths": P.calib_paths()}


def run(device: str = "cuda", overwrite: bool = False) -> dict:
    bank_dir = P.out_dir("bank")
    cal_dir = P.out_dir("calibration")
    if (bank_dir / "complete.json").exists() and not overwrite:
        raise RuntimeError("bank already sealed; use verify-bank or a new branch")
    bank_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    built = build(device)
    cal = built["calibration_scores"]
    P.write_json(bank_dir / "identity.json", {
        "version": P.VERSION, "input_size": P.INPUT_SIZE, "backbone": "wideresnet50",
        "layers": ["layer2", "layer3"], "patchsize": 3, "n_neighbors": 1,
        "sampler": "ApproximateGreedyCoresetSampler(0.01)", "device": device,
        "bank_images": built["fit_paths"], "bank_count": len(built["fit_paths"]),
        "weight_sha256": _sha256(ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"),
    })
    np.save(cal_dir / "calibration_scores.npy", cal)
    P.write_json(cal_dir / "threshold.json", {
        "quantile": P.CALIB_QUANTILE, "method": "linear", "tau": built["tau"],
        "n": int(cal.shape[0]), "min": float(cal.min()), "median": float(np.median(cal)),
        "max": float(cal.max()),
        "same_pool_false_positives": int((cal > built["tau"]).sum()),
    })
    P.write_json(bank_dir / "complete.json", {"stage": "bank", "bank_count": len(built["fit_paths"])})
    P.write_json(cal_dir / "complete.json", {"stage": "calibration", "tau": built["tau"]})
    return {"tau": built["tau"], "bank_count": len(built["fit_paths"]),
            "same_pool_fp": int((cal > built["tau"]).sum())}


def verify(device: str = "cuda") -> dict:
    """重建并按身份核验 τ 复现性（不写任何结果）。"""
    import json
    bank_dir, cal_dir = P.out_dir("bank"), P.out_dir("calibration")
    for required in (bank_dir / "identity.json", bank_dir / "complete.json",
                     cal_dir / "threshold.json", cal_dir / "calibration_scores.npy"):
        if not required.exists():
            raise RuntimeError(f"bank not sealed: missing {required.name}; run build-bank first")
    ident = json.loads((bank_dir / "identity.json").read_text(encoding="utf-8"))
    thr = json.loads((cal_dir / "threshold.json").read_text(encoding="utf-8"))
    problems = []
    if ident["input_size"] != P.INPUT_SIZE:
        problems.append("input_size mismatch")
    if ident["bank_count"] != P.BANK_IMAGES:
        problems.append("bank_count mismatch")

    built = build(device)
    delta = abs(built["tau"] - thr["tau"])
    if delta > 1e-9:
        problems.append(f"tau not reproducible (delta={delta:.3e})")
    return {"tau": thr["tau"], "tau_reproduced": built["tau"], "delta": delta,
            "problems": problems, "passed": not problems}


def _sha256(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
