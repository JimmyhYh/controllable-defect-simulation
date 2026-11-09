"""Public MVTec data interface; no company inputs or test-based model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def prepare_split(category):
    category = Path(category).resolve()
    train = sorted(category.glob("train/good/*.png"))
    if len(train) // 5 < 20:
        raise ValueError("At least 100 train/good images required by this protocol")
    order = np.random.default_rng(42).permutation(len(train))
    names = [train[i].relative_to(category).as_posix() for i in order]
    n_v = len(train) // 5
    n_p = min(20, len(train) - n_v)
    splits = {"V": names[:n_v], "P": names[n_v:n_v+n_p], "R": names[n_v+n_p:]}
    if len(splits["R"]) < 12:
        raise ValueError("Pilot requires 12 distinct R images")
    evaluation = []
    for image in sorted(category.glob("test/*/*.png")):
        kind = image.parent.name
        mask = None if kind == "good" else category / "ground_truth" / kind / (image.stem + "_mask.png")
        if mask is not None and not mask.is_file():
            raise FileNotFoundError(mask)
        # Only inspect image/mask geometry here; no E inference or performance metrics.
        if mask is not None:
            with Image.open(image) as im, Image.open(mask) as gt:
                if im.size != gt.size:
                    raise ValueError(f"Image/mask size mismatch: {image}")
        evaluation.append({"image": image.relative_to(category).as_posix(),
                           "mask": mask.relative_to(category).as_posix() if mask else None,
                           "kind": kind})
    if not evaluation:
        raise ValueError("Missing official test images")
    splits["E"] = evaluation
    result = {"category": "leather", "category_root": str(category), "seed": 42,
              "rng": "numpy.random.default_rng (PCG64)", "train_count": len(train),
              "preprocessing": "whole-image RGB resize 224x224 bilinear, /255, ImageNet mean/std; no crop",
              "train_sha256": {p.relative_to(category).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in train},
              "splits": splits}
    path = ROOT / "data" / "split.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise ValueError("Existing split differs; do not silently replace the fixed protocol")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def read_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        value = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(value.transpose(2, 0, 1).copy())
    return (tensor - MEAN) / STD


def read_mask(path):
    if path is None:
        return np.zeros((224, 224), dtype=bool)
    with Image.open(path) as image:
        image = image.convert("L").resize((224, 224), Image.Resampling.NEAREST)
        return np.array(image) > 0


class NormalImages(Dataset):
    """Only normal R paths enter this reader; labels are not detector inputs."""
    def __init__(self, category, paths):
        self.category, self.paths = Path(category), list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return read_image(self.category / self.paths[index])
