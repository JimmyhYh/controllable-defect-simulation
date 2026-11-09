"""Fixed R5/N16 native CPU inputs; no model/optimizer imports or execution."""
from __future__ import annotations
import importlib.metadata as md
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from . import common as c

MEAN = np.asarray([.485, .456, .406], np.float32)
STD = np.asarray([.229, .224, .225], np.float32)

def starts(height):
    c.require(height >= 256, "Native height too small")
    return sorted(set(range(0, height - 256 + 1, 128)) | {height - 256})

def normalized(rgb):
    return np.ascontiguousarray(((rgb.astype(np.float32) / np.float32(255) - MEAN) / STD).transpose(2, 0, 1))

def window(rgb, mask, y):
    h, w = mask.shape
    c.require(rgb.shape == (h, w, 3) and rgb.dtype == np.uint8 and mask.dtype == np.uint8 and 2 <= w < 256 and y in starts(h), "Invalid native input/coordinate")
    tile = np.pad(rgb[y:y + 256], ((0, 0), (0, 256 - w), (0, 0)), mode="reflect")
    target = np.pad(mask[y:y + 256], ((0, 0), (0, 256 - w)), mode="constant")
    valid = np.zeros((256, 256), bool); valid[:, :w] = True
    return {"rgb_uint8": tile, "image": normalized(tile), "target": target, "valid": valid}

def read_cache(item):
    with np.load(c.INPUT / item["cache"], allow_pickle=False) as z:
        c.require(set(z.files) == {"rgb_uint8", "image", "target", "valid"}, "Input cache keys differ")
        p = {k: z[k] for k in z.files}
    c.require(p["rgb_uint8"].shape == (256, 256, 3) and p["rgb_uint8"].dtype == np.uint8, "RGB cache differs")
    c.require(p["image"].shape == (3, 256, 256) and p["image"].dtype == np.float32 and np.isfinite(p["image"]).all(), "Normalized input invalid")
    c.require(np.array_equal(p["image"], normalized(p["rgb_uint8"])), "ImageNet normalization changed")
    c.require(p["target"].shape == (256, 256) and p["target"].dtype == np.uint8 and set(np.unique(p["target"]).tolist()) <= {0, 1}, "Invalid binary cache GT")
    c.require(p["valid"].shape == (256, 256) and p["valid"].dtype == bool and p["valid"].sum() == item["valid_pixels"], "Valid domain changed")
    expected = np.zeros((256, 256), bool); expected[:, :item["width"]] = True
    c.require(np.array_equal(expected, p["valid"]) and not p["target"][~p["valid"]].any(), "Padding entered GT/valid domain")
    c.require(int(p["target"][p["valid"]].sum()) == item["positive_pixels"], "GT denominator changed")
    return p

def verify_inputs(decode=True, source_bytes=True):
    seal = c.verify_seal(c.INPUT)
    p = c.load(c.INPUT / "protocol.json")
    c.require(p["implementation"] == c.code_identity(["common.py", "data.py"]), "Frozen input code changed")
    c.verify_upstream(p["upstream"])
    c.require(p["R5"] == c.R5 and p["N16"] == c.N16, "Fixed scope changed")
    items = p["items"]
    c.require([x["id"] for x in items] == c.R5 + c.N16 and len(items) == 21, "Fixed21 original order changed")
    for item in items:
        if source_bytes:
            for field in ["image", "mask"]:
                c.require(c.sha(c.DATA / item[field]) == item["source_sha256"][field], "Fixed training source bytes changed")
        if decode:
            read_cache(item)
    c.require(sum(x["valid_pixels"] for x in items) == p["valid_pixels"] and sum(x["positive_pixels"] for x in items) == p["positive_pixels"], "Pool denominator changed")
    c.require(seal["identity"]["scope"] == "fixed21 windows CPU inputs; not a model result", "Input meaning changed")
    return p

def prepare_inputs():
    if c.INPUT.exists():
        p = verify_inputs()
        return {"cpu_inputs_ready": True, "reused": True, "windows": 21, "valid_pixels": p["valid_pixels"], "positive_pixels": p["positive_pixels"]}
    from . import environment_resources as e
    e.verify()
    c.resources(2, 1)
    c.require(md.version("numpy") == "1.26.4" and md.version("Pillow") == "12.3.0", "CPU image dependencies changed")
    upstream = c.upstream_files()
    split = c.load(c.DATA / "split.json")
    normals = {r["id"]: r for r in split["N_train1885"]}
    rows = split["R_real5"] + [normals[i] for i in c.N16]
    inventory = c.load(c.ROOT / "results/fewshot_synthetic/ksdd2_v1/input/export_inventory.json")["files"]
    c.INPUT.mkdir(parents=True, exist_ok=False)
    c.save(c.INPUT / "transaction.json", {"scope": c.R5 + c.N16, "rule": "first positive R window / first N window; no substitutions", "upstream": upstream})
    items, normal_previews = [], []
    try:
        for i, row in enumerate(rows):
            real = i < 5
            identity = {key: c.sha(c.DATA / row[key]) for key in ["image", "mask"]}
            c.require(identity["image"] == row["image_sha256"] and all(identity[k] == inventory[row[k]] for k in identity), "Canonical training input identity mismatch")
            with Image.open(c.DATA / row["image"]) as im:
                c.require(im.mode in ("RGB", "L") and im.size == (row["width"], row["height"]), "Unexpected source RGB geometry/mode")
                rgb = np.asarray(im.convert("RGB")).copy()
            with Image.open(c.DATA / row["mask"]) as im:
                c.require(im.mode in ("L", "1") and im.size == (row["width"], row["height"]), "Unexpected GT geometry/mode")
                raw_gt = np.asarray(im.convert("L")).copy()
            c.require(set(np.unique(raw_gt).tolist()) <= {0, 255}, "Source GT is not binary0/255")
            mask = (raw_gt > 0).astype(np.uint8)
            options = [{"y": y, "positive_pixels": int(mask[y:y + 256].sum())} for y in starts(row["height"])]
            c.require(bool(mask.any()) == real, "R5/N16 original GT semantics differ")
            y = next(x["y"] for x in options if x["positive_pixels"] > 0) if real else options[0]["y"]
            pack = window(rgb, mask, y)
            cache = f"windows/{i:02d}_{row['id']}.npz"
            (c.INPUT / "windows").mkdir(exist_ok=True)
            np.savez_compressed(c.INPUT / cache, **pack)
            item = {"id": row["id"], "kind": "real" if real else "normal", "image": row["image"], "mask": row["mask"],
                    "source_sha256": identity, "width": row["width"], "height": row["height"], "y": y, "all_legal_windows": options,
                    "cache": cache, "valid_pixels": int(pack["valid"].sum()), "positive_pixels": int(pack["target"][pack["valid"]].sum())}
            c.require(not real or item["positive_pixels"] > 0, "R window label vanished")
            items.append(item)
            if real:
                w, h = row["width"], row["height"]
                canvas = Image.new("RGB", (2 * w + 2 * 256 + 60, h + 40), "#333333")
                canvas.paste(Image.fromarray(rgb), (0, 30)); canvas.paste(Image.fromarray(raw_gt).convert("RGB"), (w + 10, 30))
                canvas.paste(Image.fromarray(pack["rgb_uint8"]), (2 * w + 20, 30))
                canvas.paste(Image.fromarray(pack["target"] * 255).convert("RGB"), (2 * w + 286, 30))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, y + 30, w - 1, y + 285), outline="lime", width=2)
                draw.text((0, 5), f"{row['id']} raw RGB | full GT | fixed crop y={y} | crop GT; valid width={w}", fill="white")
                (c.INPUT / "previews").mkdir(exist_ok=True)
                canvas.save(c.INPUT / f"previews/R_{row['id']}.png")
            else:
                normal_previews.append((row["id"], Image.fromarray(rgb), row["width"]))
        max_h = max(im.height for _, im, _ in normal_previews)
        canvas = Image.new("RGB", (4 * 256, 4 * (max_h + 30)), "#333333")
        draw = ImageDraw.Draw(canvas)
        for i, (sample_id, im, width) in enumerate(normal_previews):
            x, yy = (i % 4) * 256, (i // 4) * (max_h + 30)
            canvas.paste(im, (x, yy + 25)); draw.text((x, yy + 3), sample_id, fill="white")
            draw.rectangle((x, yy + 25, x + width - 1, yy + 280), outline="lime", width=2)
        canvas.save(c.INPUT / "previews/N16_original.png")
        p = {"schema": "pretrained-r5-fixed21-native-inputs-v1", "implementation": c.code_identity(["common.py", "data.py"]), "upstream": upstream,
             "R5": c.R5, "N16": c.N16, "items": items, "valid_pixels": sum(x["valid_pixels"] for x in items),
             "positive_pixels": sum(x["positive_pixels"] for x in items), "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
             "source_RGB_decodes": 21, "source_GT_decodes": 21, "other_raw_inputs_decoded": 0,
             "selection": "R first nonempty legal window; N first window; no augmentation/synthesis/resize/contrast stretch",
             "visual_review": "required separately in pilot preparation; numerical input completion is not pilot readiness"}
        c.save(c.INPUT / "protocol.json", p)
        c.save(c.INPUT / "CPU输入说明.md", f"# 固定21窗CPU输入\n\n实际21张原图各解码一次，R5各首个GT非空窗、N16各首窗。有效像素{p['valid_pixels']}、阳性{p['positive_pixels']}；排除右补边。原始对比度／二值GT不变，未缩放／膨胀／合成。预览原RGB与GT分列，绿色框仅指工作窗。数值准备完成不等于模型或视觉验收，视觉判读写入独立pilot协议。\n")
        c.seal(c.INPUT, {"scope": "fixed21 windows CPU inputs; not a model result"})
    except Exception as exc:
        c.save(c.INPUT / "failure.json", {"error": repr(exc), "rule": "preserve and stop, no sample substitution or retry"})
        raise
    return {"cpu_inputs_ready": True, "reused": False, "windows": 21, "valid_pixels": p["valid_pixels"], "positive_pixels": p["positive_pixels"]}
