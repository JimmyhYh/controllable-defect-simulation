"""DRAEM-derived paired probes on held-out P; no training or E inputs."""
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import cv2
import imgaug as ia
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from data import ROOT, MEAN, STD
from demo import memory, synchronize
from evaluation import CONFIG_PATH, digest, load_formal, process_memory, timestamp, write_json

DRAEM_COMMIT = "2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef"
DRAEM_DIR = ROOT / "vendor" / "DRAEM"
DTD_DIR = ROOT / "data" / "dtd"
OUT = ROOT / "results" / "synthetic_probes"
LAMBDA = (0.5, 0.1)
sys.path.insert(0, str(DRAEM_DIR))
from data_loader import MVTecDRAEMTrainDataset
from perlin import rand_perlin_2d_np


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                          dtype=np.float32) / np.float32(255)


def texture_pool():
    names = sorted((DTD_DIR / "labels" / "train1.txt").read_text().splitlines())
    if len(names) != 1880 or len(set(names)) != 1880:
        raise ValueError("DTD split 1 training must have 1880 distinct paths")
    order = np.random.default_rng(42).permutation(len(names))[:20]
    pool = [names[i] for i in order]
    for name in pool:
        if len(Path(name).parts) != 2 or ".." in Path(name).parts:
            raise ValueError("Unexpected DTD relative path")
        if not (DTD_DIR / "images" / name).is_file():
            raise FileNotFoundError(name)
    return pool


def generate_components(generator, texture_path, seed):
    # Seed all random sources used by the author's torch / numpy / imgaug paths.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ia.seed(seed)
    # cv2.imread cannot open the Chinese workspace; decode bytes, retaining BGR.
    source = cv2.imdecode(np.frombuffer(texture_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Invalid DTD texture: " + str(texture_path))
    source = cv2.resize(source, (224, 224), interpolation=cv2.INTER_LINEAR)
    aug = generator.randAugmenter()
    augmented = aug(image=source)
    texture = np.ascontiguousarray(augmented[:, :, ::-1], dtype=np.float32) / np.float32(255)
    # Reuse original Perlin function and original mask rotation; B is never rotated.
    for attempt in range(1, 11):
        scales = tuple(int(2 ** torch.randint(0, 6, (1,)).item()) for _ in range(2))
        noise = rand_perlin_2d_np((224, 224), scales)
        noise = generator.rot(image=noise)
        mask = noise > 0.5
        if mask.any():
            return texture, mask, {"seed": seed, "mask_attempts": attempt,
                                    "perlin_resolution": list(scales),
                                    "texture_augmenters": [type(a).__name__ for a in aug]}
    return texture, None, {"seed": seed, "mask_attempts": 10, "reason": "empty mask after 10 draws"}


def residual_statistics(image, background, mask):
    delta = image - background
    inside = np.abs(delta[mask]).astype(np.float64)
    outside = np.abs(delta[~mask]).astype(np.float64)
    # Explicit RGB luma proxy on stored gamma-encoded values, not linear luminance.
    luma = np.einsum("hwc,c->hw", delta.astype(np.float64), [0.2126, 0.7152, 0.0722])
    return {"inside_mean_abs_rgb": float(inside.mean()), "inside_max_abs_rgb": float(inside.max()),
            "inside_mean_abs_luma_proxy": float(np.abs(luma[mask]).mean()),
            "inside_max_abs_luma_proxy": float(np.abs(luma[mask]).max()),
            "energy_sum_squared_rgb": float(np.square(delta.astype(np.float64)).sum()),
            "outside_mean_abs_rgb": float(outside.mean()) if outside.size else None,
            "outside_max_abs_rgb": float(outside.max()) if outside.size else None,
            "mask_pixels": int(mask.sum()), "outside_pixels": int((~mask).sum()),
            "changed_pixels": int(np.any(delta != 0, axis=2).sum()),
            "range_rgb": [float(image.min()), float(image.max())]}


def infer_rgb(model, rgb):
    # Only this RGB array enters PatchCore; B, M, source paths and labels do not.
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.float32 or not np.isfinite(rgb).all():
        raise ValueError("Invalid float32 RGB input")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB input outside [0,1]")
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    tensor = ((tensor - MEAN) / STD).unsqueeze(0)
    synchronize(model.device.type)
    start = time.perf_counter()
    with torch.no_grad():
        scores, maps = model.predict(tensor)
    synchronize(model.device.type)
    score = float(scores[0])
    heatmap = np.asarray(maps[0], dtype=np.float32)
    if not np.isfinite(score) or heatmap.shape != (224, 224) or not np.isfinite(heatmap).all():
        raise ValueError("Invalid detector output")
    return score, heatmap, time.perf_counter() - start


def first_preview(pair, maps, scores, threshold, upper):
    """Convenience preview only; day 4 scientific figures remain a separate task."""
    canvas = Image.new("RGB", (896, 538), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    draw = ImageDraw.Draw(canvas)
    for j, (key, title) in enumerate(zip(("background", "strong", "weak"), ("P正常背景", "λ=0.5", "λ=0.1"))):
        rgb = pair[key]
        canvas.paste(Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), (j*224, 0))
        z = np.clip(maps[j]/upper, 0, 1)
        colored = np.stack((z, np.zeros_like(z), 1-z), axis=-1)
        overlay = np.rint(255*(0.6*rgb + 0.4*colored)).astype(np.uint8)
        canvas.paste(Image.fromarray(overlay), (j*224, 272))
        draw.text((j*224+4, 227), title, font=font, fill="black")
        draw.text((j*224+4, 250), f"分数 {scores[j]:.4f}", font=font, fill="black")
    mask = pair["mask"]
    canvas.paste(Image.fromarray(mask.astype(np.uint8)*255).convert("RGB"), (672, 0))
    residual = np.abs(pair["strong"]-pair["background"]).mean(axis=2)
    # Fixed theoretical 0.5 RGB absolute-difference bound, not per-image rescaling.
    image = np.rint(255*np.clip(residual/0.5, 0, 1)).astype(np.uint8)
    canvas.paste(Image.fromarray(image).convert("RGB"), (672, 272))
    draw.text((676, 227), "操作掩码（非可见GT）", font=font, fill="black")
    draw.text((676, 499), "强组RGB平均绝对残差", font=font, fill="black")
    draw.text((4, 514), f"固定V阈值 {threshold:.5f}；热图沿用V色阶，可能饱和；PNG仅供查看。", font=font, fill="black")
    canvas.save(OUT / "first_pair.png")


def run():
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    split_path = ROOT / "data" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    category = Path(split["category_root"])
    backgrounds = split["splits"]["P"]
    if len(backgrounds) != 20 or len(set(backgrounds)) != 20:
        raise ValueError("Expected the already fixed P20")
    if set(backgrounds) & (set(split["splits"]["R"]) | set(split["splits"]["V"])):
        raise ValueError("P overlaps normal bank / validation")
    for p in backgrounds:
        if not p.startswith("train/good/") or digest(category/p) != split["train_sha256"][p]:
            raise ValueError("P source path or content changed")
    frozen_hash = digest(CONFIG_PATH)
    if frozen_hash != "30fde76a4638c24f85785a8ba166c9bfeb21b9f2ee70dcde2a318fe7d6219c67":
        raise ValueError("Day 2 frozen configuration changed")
    pool = texture_pool()
    source_hashes = {n: digest(DRAEM_DIR/n) for n in ("data_loader.py", "perlin.py", "LICENSE")}
    pool_hashes = {n: digest(DTD_DIR/"images"/n) for n in pool}
    if (OUT / "evaluation.json").exists():
        saved = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
        if (saved["configuration_sha256"] != frozen_hash or saved["texture_pool_sha256"] != pool_hashes
                or saved["draem_source_sha256"] != source_hashes
                or saved["synthesis_sha256"] != digest(Path(__file__))):
            raise ValueError("Completed probes differ from current sources; do not silently overwrite")
        for name, expected in saved["output_sha256"].items():
            if digest(OUT/name) != expected:
                raise ValueError("Saved probe output changed: " + name)
        # Validates the saved index / split / weights, without recalibrating or E inference.
        from evaluation import check_saved_config
        check_saved_config(config)
        print("合成观察已完成；展示固定结果，不重新生成或评价。", flush=True)
        print(json.dumps(saved["summary"], ensure_ascii=False, indent=2))
        return
    OUT.mkdir(exist_ok=True)
    (OUT / "pairs").mkdir(exist_ok=True)
    generator = MVTecDRAEMTrainDataset(str(category/"train"/"good"), str(DTD_DIR/"images"), [224,224])
    # Never use __getitem__/transform_image: those randomize background, source and beta.
    pairs, records, exclusions = [], [], []
    generation_start = time.perf_counter()
    for i, p in enumerate(backgrounds):
        background = read_rgb(category / p)
        texture, mask, settings = generate_components(generator, DTD_DIR/"images"/pool[i%20], 42000+i)
        if mask is None:
            exclusions.append({"P_index": i, "background": p, **settings})
            continue
        images = [np.asarray(background + np.float32(lam)*mask[...,None]*(texture-background),
                             dtype=np.float32) for lam in LAMBDA]
        stats = [residual_statistics(image, background, mask) for image in images]
        for image, stat in zip(images, stats):
            if (not np.isfinite(image).all() or stat["range_rgb"][0] < 0 or stat["range_rgb"][1] > 1
                    or stat["outside_max_abs_rgb"] not in (0.0, None)
                    or stat["energy_sum_squared_rgb"] <= 0):
                raise ValueError("Nonlocal, zero, or invalid synthesis residual")
        # Quantify float-rounding departure from exact 1/5 residual and 1/25 energy.
        linear_error = float(np.max(np.abs((images[1]-background)-np.float32(0.2)*(images[0]-background))))
        pair = {"background": background, "texture": texture, "mask": mask,
                "strong": images[0], "weak": images[1]}
        name = f"pairs/pair_{i:02d}.npz"
        np.savez_compressed(OUT/name, **pair)
        pairs.append(pair)
        records.append({"P_index": i, "background": p, "texture": pool[i%20], **settings,
                        "array_file": name, "strong_residual": stats[0], "weak_residual": stats[1],
                        "weak_vs_0p2_strong_residual_max_abs_error": linear_error,
                        "energy_ratio_weak_over_strong": stats[1]["energy_sum_squared_rgb"]/stats[0]["energy_sum_squared_rgb"]})
    generation_seconds = time.perf_counter()-generation_start
    if not pairs:
        raise ValueError("No valid pairs; use the plan's real baseline fallback")
    before = memory()
    model = load_formal(config)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    score_rows, map_rows = [], []
    inference_seconds = 0.0
    threshold = config["threshold"]
    for pair, record in zip(pairs, records):
        scores, maps, decisions, peaks = [], [], [], []
        for key in ("background", "strong", "weak"):
            score, heatmap, elapsed = infer_rgb(model, pair[key])
            y,x = (int(v) for v in np.unravel_index(np.argmax(heatmap), heatmap.shape))
            scores.append(score); maps.append(heatmap); decisions.append(bool(score > threshold))
            peaks.append({"yx": [y,x], "in_operation_mask": bool(pair["mask"][y,x])})
            inference_seconds += elapsed
        score_rows.append(scores); map_rows.append(maps)
        record.update(scores_P_A_B=scores, decisions_P_A_B=decisions, heatmap_peaks_P_A_B=peaks,
                      score_changes={"A_minus_P": scores[1]-scores[0], "B_minus_P": scores[2]-scores[0],
                                     "B_minus_A": scores[2]-scores[1]})
        print(f"P{record['P_index']:02d}: scores P/A/B={scores}; flags={decisions}", flush=True)
    scores = np.asarray(score_rows, dtype=np.float64)
    maps = np.asarray(map_rows, dtype=np.float32)
    decisions = scores > threshold
    counts = decisions.sum(axis=0).astype(int).tolist()
    summary = {"valid_pairs": len(pairs), "excluded_pairs": len(exclusions),
               "P_false_alarms": counts[0], "A_detected": counts[1], "B_detected": counts[2],
               "P_false_alarm_rate": counts[0]/len(pairs), "A_detection_rate": counts[1]/len(pairs),
               "B_detection_rate": counts[2]/len(pairs),
               "A_peak_in_mask": sum(r["heatmap_peaks_P_A_B"][1]["in_operation_mask"] for r in records),
               "B_peak_in_mask": sum(r["heatmap_peaks_P_A_B"][2]["in_operation_mask"] for r in records),
               "mean_score_change_A_minus_P": float(np.mean(scores[:,1]-scores[:,0])),
               "mean_score_change_B_minus_P": float(np.mean(scores[:,2]-scores[:,0])),
               "mean_score_change_B_minus_A": float(np.mean(scores[:,2]-scores[:,1])),
               "B_score_lower_than_A": int((scores[:,2] < scores[:,1]).sum()),
               "A_only_detected": int((decisions[:,1] & ~decisions[:,2]).sum()),
               "B_only_detected": int((decisions[:,2] & ~decisions[:,1]).sum()),
               "P_normal_count": int((~decisions[:,0]).sum()),
               "A_detected_when_P_normal": int((decisions[:,1] & ~decisions[:,0]).sum()),
               "B_detected_when_P_normal": int((decisions[:,2] & ~decisions[:,0]).sum()),
               "max_outside_residual": max((r[g]["outside_max_abs_rgb"] or 0) for r in records
                                            for g in ("strong_residual", "weak_residual")),
               "mean_inside_abs_rgb_A": float(np.mean([r["strong_residual"]["inside_mean_abs_rgb"] for r in records])),
               "mean_inside_abs_rgb_B": float(np.mean([r["weak_residual"]["inside_mean_abs_rgb"] for r in records])),
               "mean_energy_ratio_B_over_A": float(np.mean([r["energy_ratio_weak_over_strong"] for r in records])),
               "max_residual_ratio_roundoff": max(r["weak_vs_0p2_strong_residual_max_abs_error"] for r in records)}
    np.savez_compressed(OUT/"scores_heatmaps.npz", scores_P_A_B=scores, maps_P_A_B=maps)
    first_preview(pairs[0], maps[0], scores[0], threshold, config["display_scale"][1])
    from evaluation import check_saved_config
    check_saved_config(config)
    if digest(CONFIG_PATH) != frozen_hash:
        raise ValueError("Frozen model configuration modified during probes")
    result = {"scope": "held-out P synthetic sensitivity only; no training, no E inputs",
              "created": timestamp(), "configuration_sha256": frozen_hash,
              "threshold": threshold, "bank_vectors": config["bank_vectors"],
              "split_sha256": digest(split_path), "draem_commit": DRAEM_COMMIT,
              "draem_source_sha256": source_hashes, "synthesis_sha256": digest(Path(__file__)),
              "dtd_archive_sha256": digest(ROOT/"data"/"dtd-r1.0.1.tar.gz"),
              "dtd_train1_sha256": digest(DTD_DIR/"labels"/"train1.txt"),
              "texture_pool": pool, "texture_pool_sha256": pool_hashes,
              "lambdas_A_B": list(LAMBDA), "summary": summary, "pairs": records, "exclusions": exclusions,
              "generation_seconds_including_pair_save": generation_seconds,
              "inference_seconds_60_arrays_no_io": inference_seconds,
              "memory_before_model": before, "memory_after": memory(), "process_memory_after": process_memory(),
              "versions": {n: importlib.metadata.version(n) for n in
                           ("numpy", "torch", "torchvision", "imgaug", "opencv-python", "Pillow", "scikit-image")},
              "device": model.device.type, "display_scale": config["display_scale"],
              "output_sha256": {r["array_file"]: digest(OUT/r["array_file"]) for r in records}}
    for name in ("scores_heatmaps.npz", "first_pair.png"):
        result["output_sha256"][name] = digest(OUT/name)
    if model.device.type == "cuda":
        result.update(gpu_name=torch.cuda.get_device_name(), peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    result["total_seconds_excluding_imports_and_final_json_write"] = time.perf_counter()-started
    write_json(OUT/"evaluation.json", result)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
