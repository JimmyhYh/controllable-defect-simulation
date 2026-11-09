# Derived from blaz-r/SuperSimpleNet@98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d; original license in vendor/SuperSimpleNet_98ab4d5/LICENSE.
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

import hashlib
from pathlib import Path


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        patch_size: int,
        image_size: tuple[int, int],
        checkpoint_path: str,
    ):
        super().__init__()
        self.layers = layers

        # Independent local-only V1 adaptation; original constructor used a
        # torchvision weight enum (which can download). All feature math below
        # and the author's constructor shape forward remain unchanged.
        from torchvision.models import wide_resnet50_2
        if backbone != "wide_resnet50_2" or layers != ["layer2", "layer3"] or patch_size != 3:
            raise RuntimeError("Only the frozen strict-R5 backbone contract is supported")
        path = Path(checkpoint_path).resolve()
        expected_path = Path(__file__).resolve().parents[2] / "models/checkpoints/wide_resnet50_2-95faca4d.pth"
        if path != expected_path.resolve() or not path.is_file() or path.stat().st_size != 138223492:
            raise RuntimeError("Explicit existing generic V1 path/size mismatch; no download")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8":
            raise RuntimeError("Local V1 identity changed; preserve and stop")
        model = wide_resnet50_2(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        # The exact hash-pinned official V1 serialization contains FP16
        # convolution weights, FP32 BN tensors and int64 counters. Preserve
        # file bytes; strict default assign=False load copies into the FP32
        # destination exactly as torchvision's normal pretrained load does.
        if not isinstance(state, dict) or not state or any(
            not isinstance(v, torch.Tensor) or v.dtype not in {torch.float16, torch.float32, torch.int64}
            for v in state.values()
        ):
            raise RuntimeError("Expected hash-pinned original V1 tensor dictionary")
        model.load_state_dict(state, strict=True, assign=False)
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in model.state_dict().values()):
            raise RuntimeError("Destination backbone must remain FP32")
        model.requires_grad_(False)

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers)
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size, stride=1, padding=patch_size // 2
        )
        self.feature_dim = self.get_feature_dim(image_size)

    def forward(self, images: Tensor) -> Tensor:
        # extract features
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)

        features = list(features.values())

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer, size=(h * 2, w * 2), mode="bilinear", align_corners=True
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        feature_map = self.pooler(feature_map)

        return feature_map

    def get_feature_dim(self, image_shape: tuple[int, int]) -> tuple[int, int, int]:
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, *image_shape))
        # sum channels
        channels = sum(feature.shape[1] for feature in features.values())
        _, _, h, w = next(iter(features.values())).shape
        return channels, h * 2, w * 2
