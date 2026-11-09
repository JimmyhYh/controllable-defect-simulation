"""Stage-3 CPU input preparation; raw byte access is limited to fixed R5/N16.

Uses only the existing CPU codec environment. No Torch/model imports, model
construction, synthesis, scoring, training or package installation occurs.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

from . import resources as r
from . import static_preparation as stage2
from .contracts import INPUT_OUTPUT

DATA = r.ROOT / "data/fewshot_synthetic/ksdd2_v1"
OUT = r.ROOT / "results/ssn_baseline/ksdd2_r5_v1/input_preparation_v1"
AUDIT = r.ROOT / "results/fewshot_synthetic/ksdd2_input_audit_v1/archive_verification.json"


def scope():
    split = r.load(DATA / "split.json")
    normal = {x["id"]: x for x in split["N_train1885"]}
    return [("R5", row) for row in split["R_real5"]] + [("N16", normal[n]) for n in split["visual_N16_ids"]]


def bindings():
    stage2.verify()
    old = r.load(r.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/input_protocol.json")
    r.require(r.digest(AUDIT) == old["audit_sha256"], "Original canonical audit changed")
    return {"stage2_complete": r.digest(stage2.OUT / "complete.json"), "stage2_protocol": r.digest(stage2.OUT / "protocol.json"),
            "split": r.digest(DATA / "split.json"), "audit": r.digest(AUDIT),
            "data_implementation": r.digest(Path(__file__)),
            "packages_metadata": {n: md.version(n) for n in ["numpy", "Pillow", "opencv-python"]}}


def feature_resize(mask):
    """Float32 bilinear align_corners=True formula; not cv2 INTER_LINEAR.

CPU preparation implements the author Tensor interpolation rule explicitly.
The future manual runtime must compare its Torch result before model creation;
current stage does not claim a Torch numerical equivalence run.
"""
    import numpy as np
    h, w = mask.shape
    oh, ow = 160, 58
    y = np.arange(oh, dtype=np.float32) * np.float32((h - 1) / (oh - 1))
    x = np.arange(ow, dtype=np.float32) * np.float32((w - 1) / (ow - 1))
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    ay, ax = y - y0.astype(np.float32), x - x0.astype(np.float32)
    arr = mask.astype(np.float32)
    top = (np.float32(1) - ax)[None, :] * arr[y0[:, None], x0] + ax[None, :] * arr[y0[:, None], x1]
    bottom = (np.float32(1) - ax)[None, :] * arr[y1[:, None], x0] + ax[None, :] * arr[y1[:, None], x1]
    result = (np.float32(1) - ay)[:, None] * top + ay[:, None] * bottom
    # A higher precision calculation detects a possible preparation ambiguity.
    yd, xd = np.linspace(0, h - 1, oh), np.linspace(0, w - 1, ow)
    ya, xa = np.floor(yd).astype(int), np.floor(xd).astype(int)
    yb, xb = np.minimum(ya + 1, h - 1), np.minimum(xa + 1, w - 1)
    vy, vx = yd - ya, xd - xa
    dbl = ((1 - vy)[:, None] * ((1 - vx)[None, :] * arr[ya[:, None], xa] + vx[None, :] * arr[ya[:, None], xb])
           + vy[:, None] * ((1 - vx)[None, :] * arr[yb[:, None], xa] + vx[None, :] * arr[yb[:, None], xb]))
    return result, int(np.count_nonzero((result >= 0.5) != (dbl >= 0.5)))


def coordinates(h, w):
    # Center-coordinate round trip is an algebra check, not a pixelwise inverse
    # of lossy resizing. Every native/input pixel remains valid; no padding/crop.
    points = [(0.0, 0.0), ((h - 1) / 2, (w - 1) / 2), (float(h - 1), float(w - 1))]
    errors = []
    for y, x in points:
        yy, xx = (y + 0.5) * 640 / h - 0.5, (x + 0.5) * 232 / w - 0.5
        errors.append(max(abs((yy + 0.5) * h / 640 - 0.5 - y), abs((xx + 0.5) * w / 232 - 0.5 - x)))
    return {"native_hw": [h, w], "input_hw": [640, 232], "feature_hw": [160, 58],
            "padding": 0, "crop": None, "input_valid_pixels": 640 * 232,
            "native_valid_pixels": h * w, "center_roundtrip_max_abs_error": max(errors),
            "RGB_native_to_input_center": "(coord+0.5)*output_size/input_size-0.5; OpenCV INTER_LINEAR on uint8",
            "GT_input_to_native_lookup": "INTER_NEAREST: min(floor(output_index*native_size/input_size),native_size-1)",
            "feature_to_input_center": "feature_index*(input_size-1)/(feature_size-1); align_corners=True"}


def preview(index, role, row, rgb, gt, resized, mask, feature):
    import numpy as np
    from PIL import Image, ImageDraw
    from PIL.Image import Resampling
    if role == "R5":
        panels = [Image.fromarray(rgb), Image.fromarray((gt * 255).astype(np.uint8)).convert("RGB"),
                  Image.fromarray(resized), Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"),
                  Image.fromarray((feature * 255).astype(np.uint8)).resize((232, 640), Resampling.NEAREST).convert("RGB")]
        labels = ["native RGB", "native GT", "input RGB 640x232", "input GT nearest", "feature GT 160x58 (NN display)"]
        canvas = Image.new("RGB", (1390, 720), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"R5 {row['id']} | CPU inputs, labels only; NO predictions | original contrast", fill="black")
        for i, (im, label) in enumerate(zip(panels, labels)):
            draw.text((10 + i * 277, 33), label, fill="black")
            canvas.paste(im, (10 + i * 277, 65))
        canvas.save(OUT / f"previews/R_{row['id']}.png")
    else:
        # Four original, unscaled RGBs per page; no normalization/contrast
        # stretch used for inspection. Ground-truth emptiness is numeric evidence.
        page, cell = (index - 5) // 4, (index - 5) % 4
        path = OUT / f"previews/N_page{page + 1}.png"
        if cell == 0:
            canvas = Image.new("RGB", (1120, 740), "white")
        else:
            with Image.open(path) as im:
                canvas = im.copy()
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), "Fixed N16 native RGB; original contrast, no crop; all GT checked separately", fill="black")
        draw.text((10 + cell * 278, 35), f"N {row['id']} | native GT=0", fill="black")
        canvas.paste(Image.fromarray(rgb), (10 + cell * 278, 65))
        canvas.save(path)


def verify_numeric():
    seal = r.load(OUT / "preparation_complete.json")
    expected = set(seal["files"]) | {"preparation_complete.json"}
    if (OUT / "complete.json").exists():
        expected |= {"complete.json", "visual_review.json", "第3阶段CPU输入与坐标核验.md"}
    r.require(r.files_under(OUT) == expected, "Input cache incomplete/unknown files")
    for name, sha in seal["files"].items():
        r.require(r.digest(OUT / name) == sha, f"Input cache changed: {name}")
    protocol = r.load(OUT / "protocol.json")
    r.require(protocol["bindings"] == bindings() and protocol["scope"] == [{"role": a, **b} for a, b in scope()], "CPU source/role/implementation identity changed")
    # Hash source bytes only within the permitted 21 selected pairs.
    for row in r.load(OUT / "manifest.json")["samples"]:
        for key in ["image", "mask"]:
            r.require(r.digest(DATA / row[key]) == row[key + "_sha256"], "Selected raw source changed")
    return r.load(OUT / "manifest.json")


def verify():
    manifest = verify_numeric()
    r.verify_seal(OUT)
    visual = r.load(OUT / "visual_review.json")
    return {"CPU_inputs_verified": True, "input_ready": manifest["numeric_ready"] and visual["status"] == "pass",
            "samples": len(manifest["samples"]), "R5_feature_positive": {x["id"]: x["feature_positive"] for x in manifest["samples"] if x["role"] == "R5"},
            "Torch_or_model_operations": 0, "seal_sha256": r.digest(OUT / "complete.json")}


def prepare():
    if OUT.exists():
        if (OUT / "complete.json").exists():
            return {"reused": True, **verify()}
        return {"reused": True, "numeric_only": True, "manifest": verify_numeric()}
    binding = bindings()
    r.require(binding["packages_metadata"]["opencv-python"] == "4.11.0.86", "Use existing frozen CPU codec, not new SSN environment")
    snap = r.snapshot()
    r.require(snap["free_RAM_GiB"] >= 2 and snap["free_disk_GiB"] >= 1, "CPU preparation resource gate")
    import cv2
    import numpy as np
    selected = scope()
    r.require(len(selected) == 21 and [b["id"] for a, b in selected if a == "R5"] == r.R5, "Fixed scope changed")
    audit = r.load(AUDIT)
    members = {x["path"]: x for x in audit["members"]}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "arrays").mkdir()
    (OUT / "previews").mkdir()
    r.write(OUT / "transaction.json", {"action": "CPU fixed R5/N16 only", "raw_image_pairs_allowed": 21, "bindings": binding})
    r.write(OUT / "protocol.json", {"schema": "ssn-fixed-cpu-inputs-v1", "bindings": binding,
          "scope": [{"role": role, **row} for role, row in selected], "contract": INPUT_OUTPUT,
          "cache": "numeric NPZ with image CHW float32, binary input/feature/native GT uint8; allow_pickle=False",
          "feature_mapping": "NumPy float32 align_corners=True formula; float64 threshold consistency; future manual Torch comparison before model init",
          "resources_start": snap, "roles_read": ["R5", "fixed_N16"], "new_environment_modified": False})
    rows = []
    try:
        for index, (role, row) in enumerate(selected):
            image_path, mask_path = DATA / row["image"], DATA / row["mask"]
            image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
            import hashlib
            image_hash, mask_hash = hashlib.sha256(image_bytes).hexdigest(), hashlib.sha256(mask_bytes).hexdigest()
            r.require(image_hash == row["image_sha256"] == members[row["image"]]["sha256"] and mask_hash == members[row["mask"]]["sha256"], "Selected canonical source hash mismatch")
            bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            native_gt = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            r.require(bgr is not None and native_gt is not None, "CPU decode failed")
            r.require(bgr.shape == (row["height"], row["width"], 3) and bgr.dtype == np.uint8, "RGB shape/dtype differs")
            r.require(native_gt.shape == bgr.shape[:2] and native_gt.dtype == np.uint8 and set(np.unique(native_gt).tolist()) <= {0, 255}, "Native GT must be binary 8-bit and native-sized")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            native_gt = (native_gt > 0).astype(np.uint8)
            r.require(bool(native_gt.any()) == (role == "R5"), "Selected GT role mismatch")
            resized = cv2.resize(rgb, (232, 640), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(native_gt, (232, 640), interpolation=cv2.INTER_NEAREST)
            soft, ambiguous = feature_resize(mask)
            feature = (soft >= np.float32(0.5)).astype(np.uint8)
            image = ((resized.astype(np.float32) / np.float32(255) - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy()
            r.require(image.dtype == np.float32 and np.isfinite(image).all(), "Normalized input not finite FP32")
            out = OUT / f"arrays/{row['id']}.npz"
            np.savez_compressed(out, image=image, mask_input=mask.astype(np.uint8), mask_feature=feature, mask_native=native_gt)
            preview(index, role, row, rgb, native_gt, resized, mask, feature)
            rows.append({"id": row["id"], "role": role, "image": row["image"], "mask": row["mask"],
                         "image_sha256": image_hash, "mask_sha256": mask_hash, "cache": out.relative_to(OUT).as_posix(),
                         "native_positive": int(native_gt.sum()), "input_positive": int(mask.sum()), "feature_positive": int(feature.sum()),
                         "float32_vs_float64_binary_disagreements": ambiguous, "coordinates": coordinates(row["height"], row["width"])})
        ready = all(x["input_positive"] > 0 and x["feature_positive"] > 0 for x in rows if x["role"] == "R5") and not any(x["float32_vs_float64_binary_disagreements"] for x in rows)
        r.write(OUT / "manifest.json", {"samples": rows, "numeric_ready": ready, "visual_review": "pending", "raw_pairs_decoded": 21,
            "Q_E_U_content_read": False, "input_pixels_per_image": 148480, "feature_pixels_per_image": 9280,
            "numeric_gate": "all R5 positive labels survive input/feature mapping; N16 empty; float32/64 binary agreement"})
        r.write(OUT / "preparation_complete.json", {"files": {n: r.digest(OUT / n) for n in sorted(r.files_under(OUT))}, "numeric_ready": ready})
    except Exception as exc:
        r.write(OUT / "failure.json", {"error": repr(exc), "completed_selected_pairs": [x["id"] for x in rows], "rule": "preserve partial state, no retries/resampling/label rescue"})
        raise
    manifest = verify_numeric()
    return {"CPU_numeric_preparation_complete": True, "numeric_ready": manifest["numeric_ready"], "visual_review": "pending",
            "samples": len(rows), "R5_counts": [{k: x[k] for k in ["id", "native_positive", "input_positive", "feature_positive"]} for x in rows if x["role"] == "R5"]}


def accept_visual(status):
    if (OUT / "complete.json").exists():
        return verify()
    manifest = verify_numeric()
    r.require(status in {"pass", "fail"}, "Explicit visual decision required")
    ready = manifest["numeric_ready"] and status == "pass"
    r.write(OUT / "visual_review.json", {"status": status, "all_fixed_R5_N16_reviewed": True,
        "scope": [row["id"] for _, row in scope()], "source_contrast_preserved": True,
        "limitation": "visual input/GT alignment check only, not predicted localization or performance"})
    lines = ["# SuperSimpleNet第3阶段CPU输入与坐标核验", "", f"固定21对输入的CPU准备与审阅完成；input_ready={ready}。无模型操作。", "",
             "R5原生／输入／特征网格阳性像素：", ""]
    for row in manifest["samples"]:
        if row["role"] == "R5":
            lines.append(f"- {row['id']}：{row['native_positive']}／{row['input_positive']}／{row['feature_positive']}；均为不同网格的标签计数，不能作检测指标。")
    lines += ["", "全部旧固定N16的原生、输入及特征GT为空。RGB无对比度拉伸，无裁剪／补边；整图到640×232，原生GT独立保留。所有21例的坐标及来源hash见manifest，全部CPU样例见previews。",
              "", "特征标签按作者bilinear／align_corners=True／>=0.5的NumPy float32公式计算，与float64公式二值复核；未导入Torch或作Torch数值对照。未来人工运行须先用实际Torch算子对缓存逐例核验二值标签，相异则在模型初始化前停止。坐标往返是中心位置代数核验，不代表有损图像缩放可逐像素逆还原。",
              "", "既有CPU环境OpenCV4.11.0.86／NumPy／Pillow完成解码和缓存，未向封存.venv-ssn安装任何包；未来模型环境只读标准NPZ数值输入，不借用旧site-packages。固定R5与N16仅是训练侧技术小试材料，未读取Q／E／U图像／GT、生成缺陷、评分、前向、训练或校准。",
              "", "数值／视觉失败不得替换样本、膨胀标签、更换插值或借U救结果；保持本次证据并停止。小试预算与人工入口另行准备，当前输入完成不等于允许自动启动模型。"]
    r.write(OUT / "第3阶段CPU输入与坐标核验.md", "\n".join(lines) + "\n")
    r.seal(OUT, "ssn-stage3-cpu-inputs-v1")
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "accept-visual"])
    parser.add_argument("--status", choices=["pass", "fail"])
    args = parser.parse_args()
    allowed = {(DATA / row[k]).resolve() for _, row in scope() for k in ["image", "mask"]}
    forbidden = {"torch", "torchvision", "diffusers", "transformers", "sklearn"}
    r.require(not any(n.split(".")[0] in forbidden for n in sys.modules), "Unexpected ML package imported")
    def guard(event, arguments):
        if event == "import" and (arguments[0].split(".")[0] in forbidden or arguments[0].startswith("ssn_baseline.core_v1") or arguments[0] == "ssn_baseline.model"):
            raise RuntimeError("CPU input preparation forbids model imports")
        if event.startswith("pickle.") or event == "socket.connect":
            raise RuntimeError("No pickle/network in CPU input preparation")
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            # Windows platform/subprocess metadata probes open the character
            # device in read/write mode; it is not a project filesystem write.
            if os.fsdecode(arguments[0]).lower() == os.devnull.lower():
                return
            path = Path(os.fsdecode(arguments[0])).resolve()
            mode, flags = arguments[1] or "", arguments[2] or 0
            writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                r.require(path.is_relative_to(OUT), "CPU writes outside independent input destination")
            elif path.is_relative_to(r.ROOT / "data"):
                r.require(path in allowed or path == (DATA / "split.json").resolve(), "Raw access outside fixed R5/N16")
            elif path.is_relative_to(r.ROOT / "results") and path.suffix.lower() in {".png", ".npy", ".npz"}:
                r.require(path.is_relative_to(OUT), "Old image/probability payload access forbidden")
    sys.addaudithook(guard)
    if args.command == "accept-visual":
        result = accept_visual(args.status)
    else:
        result = {"prepare": prepare, "verify": verify}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
