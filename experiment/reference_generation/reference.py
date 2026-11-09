"""Original RGB reference preparation; no contrast enhancement."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rgb(path, expected):
    if sha256(path) != expected['sha256']:
        raise ValueError('Source file identity changed: ' + str(path))
    with Image.open(path) as im:
        mode = im.mode
        if mode not in ('RGB', 'RGBA'):
            raise ValueError('Unexpected color mode: ' + mode)
        if mode == 'RGBA' and np.any(np.asarray(im.getchannel('A')) != 255):
            raise ValueError('Nonopaque alpha requires a new explicit policy')
        rgb = np.array(im.convert('RGB'))
    if list(rgb.shape[1::-1]) != expected['size']:
        raise ValueError('Image dimensions changed')
    # The old manifest's RGB group digest convention is not defined in its
    # saved schema. Bind original file bytes above; use its immutable group
    # IDs for exclusion, and store a separate explicitly defined new RGB hash.
    return rgb


def read_mask(path, expected):
    if sha256(path) != expected['sha256']:
        raise ValueError('Mask identity changed')
    with Image.open(path) as im:
        if im.mode != 'L':
            raise ValueError('Mask must be L')
        a = np.array(im)
    if not set(np.unique(a)).issubset({0, 255}) or not np.any(a):
        raise ValueError('Empty or nonbinary mask')
    if int(np.count_nonzero(a)) != expected['positive_pixels']:
        raise ValueError('Positive mask count changed')
    return a > 0


def bbox(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty mask')
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def crop_box(center, side, shape):
    side = int(side)
    left, top = np.floor(np.asarray(center) - (side-1)/2).astype(int)
    h, w = shape[:2]
    if side < 1 or left < 0 or top < 0 or left+side > w or top+side > h:
        raise ValueError('Crop exceeds image; no repositioning, padding or rescue')
    return [int(left), int(top), int(left+side), int(top+side)]


def prepare_reference(image, mask, crop_spec):
    if image.shape[:2] != mask.shape:
        raise ValueError('Mask/image dimensions differ')
    b = bbox(mask)
    center = [(b[0]+b[2]-1)/2, (b[1]+b[3]-1)/2]
    box = crop_box(center, 4*max(b[2]-b[0], b[3]-b[1]), mask.shape)
    if crop_spec != {'context_factor': 4, 'model_side': 512}:
        raise ValueError('Unsupported reference crop protocol')
    rgb = np.array(Image.fromarray(image).crop(box).resize((512, 512), Image.Resampling.BILINEAR))
    local_mask = np.array(Image.fromarray(mask.astype('uint8')*255).crop(box).resize(
        (512, 512), Image.Resampling.NEAREST)) > 0
    if not local_mask.any():
        raise ValueError('Reference mask disappeared at model scale')
    return {'rgb': rgb, 'mask': local_mask, 'bbox_xyxy_exclusive': b, 'crop_xyxy_exclusive': box,
            'source_positive_pixels': int(mask.sum()), 'model_positive_pixels': int(local_mask.sum())}
