"""Public PatchCore demo: fixed split, normal pilot, real evaluation and single image."""
import argparse
import ctypes
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from data import ROOT, NormalImages, prepare_split


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]


def memory():
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("Cannot read Windows RAM status")
    return {"ram_total_gib": value.total / 2**30, "ram_available_gib": value.available / 2**30}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def probe(device):
    from detector import make_detector
    result_path = ROOT / "results" / "day1.json"
    # Re-running the normal pilot must not erase the already fixed study configuration.
    retained = {}
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        retained = {key: previous[key] for key in
                    ("formal_configuration", "data_archive", "timing_scope") if key in previous}
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    paths = split["splits"]["R"]
    fit_paths, predict_paths = paths[:8], paths[8:12]
    before = memory()
    model = make_detector(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        model.fit(DataLoader(NormalImages(split["category_root"], fit_paths), batch_size=1, num_workers=0))
    synchronize(device)
    fit_seconds = time.perf_counter() - start
    scores, maps, durations = [], [], []
    dataset = NormalImages(split["category_root"], predict_paths)
    for i in range(len(dataset)):
        synchronize(device)
        start = time.perf_counter()
        score, heatmap = model.predict(dataset[i].unsqueeze(0))
        synchronize(device)
        durations.append(time.perf_counter() - start)
        scores.append(float(score[0]))
        maps.append(np.asarray(heatmap[0], dtype=np.float32))
    maps = np.stack(maps)
    if maps.shape != (4, 224, 224) or not np.isfinite(maps).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite or incorrectly shaped author inference output")
    if np.any(maps < 0) or np.any(np.asarray(scores) < 0):
        raise ValueError("Unexpected negative nearest-neighbour distances")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    np.savez_compressed(results / "day1_normal_probe.npz", scores=np.asarray(scores), maps=maps)
    # Shared display-only scale for these four normal pilot maps; raw scores remain untouched.
    upper = max(float(maps.max()), 1e-12)
    canvas = Image.new("RGB", (672, 248 * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for i, relative in enumerate(predict_paths):
        with Image.open(Path(split["category_root"]) / relative) as image:
            rgb = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        z = np.clip(maps[i] / upper, 0, 1)
        color = np.stack([z, np.zeros_like(z), 1-z], axis=-1)
        colored = (255 * color).astype(np.uint8)
        overlay = (0.6*rgb + 0.4*colored).astype(np.uint8)
        for j, value in enumerate((rgb, colored, overlay)):
            canvas.paste(Image.fromarray(value), (224*j, 248*i))
        draw.text((3, 248*i+225), f"R normal {relative} | score={scores[i]:.5f}", fill="black")
    canvas.save(results / "day1_normal_probe.png")
    bank_size = int(model.anomaly_scorer.nn_method.search_index.ntotal)
    after = memory()
    result = {"scope": "Day 1 runtime probe only; 8 R fit + 4 other R normals; no V/P/E inference",
              "device": device, "faiss_device": "cpu", "source_commit": "fcaa92f124fb1ad74a7acf56726decd4b27cbcad",
              "fit_paths": fit_paths, "predict_paths": predict_paths,
              "fit_seconds": fit_seconds, "inference_seconds": durations,
              "image_scores_raw": scores, "heatmap_shape": list(maps.shape),
              "heatmap_min_max_raw": [float(maps.min()), float(maps.max())],
              "pilot_display_scale": [0, upper], "pilot_bank_vectors": bank_size,
              "bank_dimension": 1024, "memory_before": before, "memory_after": after,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "torchvision", "numpy", "faiss-cpu", "timm", "scipy")}}
    if device == "cuda":
        result.update(gpu_name=torch.cuda.get_device_name(),
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    result.update(retained)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--category-root", required=True)
    run = commands.add_parser("probe")
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    commands.add_parser("evaluate", help="Build the fixed formal bank, calibrate V, then evaluate all E")
    commands.add_parser("synthesize", help="DRAEM-derived paired probes on fixed P; reuse the frozen bank and V threshold")
    single = commands.add_parser("predict", help="Use the saved formal bank and V threshold for one public image")
    single.add_argument("--image", required=True)
    single.add_argument("--output-dir", default=str(ROOT / "results" / "single_demo"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_split(args.category_root)
        print(json.dumps({"train_count": result["train_count"],
                          "split_counts": {k: len(v) for k, v in result["splits"].items()}}, indent=2))
    elif args.command == "probe":
        probe(args.device)
    elif args.command == "evaluate":
        from evaluation import evaluate
        evaluate()
    elif args.command == "synthesize":
        from synthesis import run
        run()
    else:
        from evaluation import predict_single
        predict_single(args.image, args.output_dir)
