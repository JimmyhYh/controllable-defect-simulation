"""Paired DRAEM components. Sampling RNG is separate from model RNG."""
import contextlib
import random
import sys
import cv2
import imgaug as ia
import numpy as np
import torch
from localization_data import ROOT, normal_rgb, digest
sys.path.insert(0, str(ROOT / 'vendor/DRAEM'))
from data_loader import MVTecDRAEMTrainDataset
from perlin import rand_perlin_2d_np

@contextlib.contextmanager
def isolated_random(seed):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed); ia.seed(seed)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])

class PairStream:
    def __init__(self, split, seed, first_eight=False):
        self.split = split
        self.paths = split['N'][:8] if first_eight else split['N']
        self.pool = sorted((ROOT / 'data/dtd/labels/train1.txt').read_text().splitlines())
        if len(self.pool) != 1880 or len(set(self.pool)) != 1880:
            raise ValueError('DTD train1 inventory differs')
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.ordinal = 0
        self.cache = {}
        self.generator = MVTecDRAEMTrainDataset('', '', [256, 256])

    def next(self, force_anomaly=False):
        path = self.paths[int(self.rng.integers(len(self.paths)))]
        texture_path = self.pool[int(self.rng.integers(len(self.pool)))]
        component_seed = int(self.rng.integers(0, 2**31 - 1))
        if path not in self.cache:
            self.cache[path] = normal_rgb(self.split, path)
        base = self.cache[path].copy()
        source_path = ROOT / 'data/dtd/images' / texture_path
        with isolated_random(component_seed):
            # Shared author's 30% training background rotation; not evaluation preprocessing.
            rotated = bool(torch.rand(1).item() > 0.7)
            if rotated:
                base = self.generator.rot(image=base)
            source = cv2.imdecode(np.frombuffer(source_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError('Invalid texture')
            source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_LINEAR)
            texture = self.generator.randAugmenter()(image=source)
            texture = np.ascontiguousarray(texture[:, :, ::-1], dtype=np.float32) / np.float32(255)
            scales = tuple(2 ** int(torch.randint(0, 6, (1,)).item()) for _ in range(2))
            noise = self.generator.rot(image=rand_perlin_2d_np((256, 256), scales))
            mask = noise > 0.5
            strength = 1.0 - float(torch.rand(1).item()) * 0.8
            normal = bool(torch.rand(1).item() > 0.5)
        active = bool(mask.any()) and (force_anomaly or not normal)
        if not active:
            mask = np.zeros_like(mask)
        if active:
            self.ordinal += 1
        return {'background': base, 'texture': texture, 'mask': mask, 'lambda_A': strength,
                'replace': active and self.ordinal % 2 == 0, 'active': active,
                'metadata': {'background': path, 'texture': texture_path,
                             'component_seed': component_seed, 'background_rotated': rotated,
                             'effective_anomaly_ordinal': self.ordinal}}

def blend(sample, strength):
    b = sample['background']
    return np.asarray(b + np.float32(strength) * sample['mask'][..., None] * (sample['texture'] - b), dtype=np.float32)

def residual(image, sample):
    mask = sample['mask']; delta = np.abs(image - sample['background'])
    inside = delta[mask]
    outside = delta[~mask]
    return {'mean_abs': float(inside.mean()), 'q05_q50_q95_abs': np.quantile(inside, [.05,.5,.95], method='linear').tolist(),
            'effective_pixel_fraction': float((inside.max(axis=1) > 1/255).mean()),
            'outside_max_abs': float(outside.max()) if outside.size else 0.0,
            'mask_pixels': int(mask.sum()), 'finite': bool(np.isfinite(image).all())}
