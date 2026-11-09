"""Thin wrapper around the pinned author implementation, not a new PatchCore."""
import random
import sys
from pathlib import Path

import numpy as np
import faiss
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "patchcore-inspection" / "src"))
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler


class UnicodePathFaissNN(patchcore.common.FaissNN):
    """Use FAISS byte serialization with Python Unicode-safe file I/O on Windows."""
    def save(self, filename):
        data = faiss.serialize_index(self._index_to_cpu(self.search_index))
        Path(filename).write_bytes(data.tobytes())

    def load(self, filename):
        data = np.frombuffer(Path(filename).read_bytes(), dtype=np.uint8)
        self.search_index = self._index_to_gpu(faiss.deserialize_index(data))


def make_detector(device="cuda"):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / "models"))
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; inspect runtime before choosing CPU")
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    weight = ROOT / "models" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
    if not weight.is_file():
        raise FileNotFoundError("Download the official ImageNet V1 weight first: " + str(weight))
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    model = patchcore.patchcore.PatchCore(target)
    model.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
               device=target, input_shape=(3, 224, 224), pretrain_embed_dimension=1024,
               target_embed_dimension=1024, patchsize=3, anomaly_score_num_nn=1,
               featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(0.01, target),
               nn_method=UnicodePathFaissNN(on_gpu=False, num_workers=4))
    return model
