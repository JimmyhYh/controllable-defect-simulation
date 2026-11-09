"""Pixel-center affine geometry for original and local512 frames."""
import math

import cv2
import numpy as np

from .reference import bbox, crop_box

VARIANTS = ('base', 'move', 'scale', 'rotate')


def affine(reference_mask, background_shape, variant):
    h, w = background_shape[:2]
    b = bbox(reference_mask)
    center = np.array([(b[0]+b[2]-1)/2, (b[1]+b[3]-1)/2])
    scale = .1*min(w, h)/max(b[2]-b[0], b[3]-b[1])
    target = np.array([(.65 if variant == 'move' else .35)*w, .5*h])
    if variant == 'scale':
        scale *= 1.25
    angle = math.radians(30 if variant == 'rotate' else 0)
    # x right, y down: positive angle appears clockwise.
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])*scale
    matrix = np.eye(3); matrix[:2, :2] = rotation; matrix[:2, 2] = target - rotation@center
    return matrix, b


def build_geometry(reference_mask, background_shape, request):
    variant = request['variant']
    if variant not in VARIANTS or request.get('model_side') != 512:
        raise ValueError('Unsupported geometry request')
    h, w = background_shape[:2]
    transforms = {}; longest = 0
    y, x = np.nonzero(reference_mask)
    points = np.stack((x, y, np.ones(len(x))))
    for name in VARIANTS:
        a, _ = affine(reference_mask, background_shape, name)
        projected = a@points
        # Pixel cells, not only centers: include half-pixel footprint.
        half = .5*np.abs(a[:2, :2]).sum(axis=1)
        if np.any(projected[:2].min(axis=1)-half < -.5) or np.any(
                projected[:2].max(axis=1)+half > np.array([w-.5, h-.5])):
            raise ValueError('Transformed mask leaves original image')
        m = cv2.warpAffine(reference_mask.astype('uint8'), a[:2], (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)>0
        b = bbox(m)
        longest = max(longest, b[2]-b[0], b[3]-b[1])
        transforms[name] = (a, m)
    side = int(math.ceil(4*longest/8)*8)
    a, roi = transforms[variant]
    center = [(.65 if variant == 'move' else .35)*w, .5*h]
    box = crop_box(center, side, (h, w))
    factor = 512/side
    # OpenCV resizing maps destination center u to (u+.5)/factor-.5.
    original_to_local = np.array([[factor, 0, factor*(.5-box[0])-.5],
                                  [0, factor, factor*(.5-box[1])-.5], [0, 0, 1.]])
    inverse = np.linalg.inv(original_to_local)
    samples = np.array([[0, 0, 1], [w-1, h-1, 1], [*center, 1]], dtype=float).T
    error = float(np.max(np.abs(inverse@original_to_local@samples-samples)))
    if error > 1e-9:
        raise ValueError('Coordinate roundtrip fails')
    left, top, right, bottom = box
    if int(roi[top:bottom, left:right].sum()) != int(roi.sum()):
        raise ValueError('Crop clips target support')
    local_roi = cv2.resize(roi[top:bottom,left:right].astype('uint8'), (512,512),
                           interpolation=cv2.INTER_NEAREST_EXACT)>0
    if not local_roi.any():
        raise ValueError('Target mask disappears')
    return {'roi': roi, 'local_roi': local_roi, 'metadata': {
        'variant': variant, 'requested_center_xy': center, 'crop_xyxy_exclusive': box,
        'model_side': 512, 'crop_side_original_px': side,
        'reference_to_original': a.tolist(), 'original_to_local512': original_to_local.tolist(),
        'local512_to_original': inverse.tolist(), 'reference_to_local512': (original_to_local@a).tolist(),
        'roundtrip_max_error_px': error, 'original_support_pixels': int(roi.sum()),
        'local_support_pixels': int(local_roi.sum()), 'original_bbox_xyxy_exclusive': bbox(roi),
        'local_bbox_xyxy_exclusive': bbox(local_roi),
        'positive_angle': 'clockwise on image x-right/y-down grid',
        'support_note': 'nearest rasterization; request support is not defect semantic GT'}}
