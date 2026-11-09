"""Specified geometry; AnoStyler blob outlines are reused with an explicit adapter.

Line construction follows the center/rotation/segment idea in its MIT msp_line,
but uses pixel-center distance for explicit width, rather than random thickness.
"""
import importlib.util
import math
import random

import cv2
import numpy as np

from .data import ROOT


def point(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('A point must contain finite [x,y]')
    return a


def positive(value, name, minimum=0.0):
    x = float(value)
    if not math.isfinite(x) or x <= minimum:
        raise ValueError(name + ' is nonfinite or below the allowed working scale')
    return x


def bounds(lo, hi, width, height):
    if np.any(lo < 0) or np.any(hi > [width - 1, height - 1]):
        raise ValueError('Geometry exceeds the working frame; no silent clipping')


def describe(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return {'pixels': 0, 'centroid_xy': None, 'bbox_xyxy': None}
    cov = np.cov(np.stack((x, y)), bias=True)
    _, vectors = np.linalg.eigh(cov)
    direction = vectors[:, -1]
    return {'pixels': int(len(x)), 'centroid_xy': [float(x.mean()), float(y.mean())],
            'bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            'principal_angle_deg': float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)}


def polygon_vertices(value, width, height):
    vertices = np.asarray(value, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError('Polygon requires at least three finite [x,y] vertices')
    rounded = np.rint(vertices).astype(np.int32)
    if len(np.unique(rounded, axis=0)) != len(rounded):
        raise ValueError('Polygon vertices collapse at the working scale')
    bounds(vertices.min(0), vertices.max(0), width, height)
    if abs(cv2.contourArea(rounded)) < 4:
        raise ValueError('Degenerate polygon')
    def orient(a, b, c):
        return float(np.cross(b - a, c - a))
    def meets(a, b, c, d):
        o = [orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)]
        if o[0] * o[1] < 0 and o[2] * o[3] < 0:
            return True
        for v, p, u, z in ((o[0],c,a,b),(o[1],d,a,b),(o[2],a,c,d),(o[3],b,c,d)):
            if abs(v) < 1e-9 and np.all(p >= np.minimum(u,z)) and np.all(p <= np.maximum(u,z)):
                return True
        return False
    n = len(rounded)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if meets(rounded[i], rounded[(i+1)%n], rounded[j], rounded[(j+1)%n]):
                raise ValueError('Polygon must be simple, without self intersections')
    return rounded


def author_blob(width, height, seed):
    """Call pinned msp_dot, preserving all caller random state."""
    module_spec = importlib.util.spec_from_file_location('anostyler_mask_source', ROOT / 'vendor/AnoStyler/meta_shape_priors.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed); random.seed(seed)
        raw = module.msp_dot(W=256, H=256)
    finally:
        np.random.set_state(numpy_state); random.setstate(python_state)
    y, x = np.nonzero(raw)
    if not len(x):
        raise ValueError('Upstream random outline is empty')
    crop = raw[y.min():y.max()+1, x.min():x.max()+1]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def rasterize(spec, height=256, width=256, seed=20261014):
    if (height, width) != (256, 256) or spec.get('units') != 'px256':
        raise ValueError('Only explicit px256 coordinates and 256 working frame are supported')
    shape = spec.get('shape')
    y, x = np.mgrid[:height, :width]
    meta = {'shape': shape, 'units': 'px256', 'coordinate_order': 'x,y; pixel-center grid', 'upstream_reuse': False}
    if shape in ('circle', 'ellipse'):
        center = point(spec['center'])
        axes = np.repeat(positive(spec['radius'], 'radius', 1), 2) if shape == 'circle' else point(spec['axes'])
        if np.any(axes <= 1):
            raise ValueError('Ellipse semiaxes must exceed one working pixel')
        angle = float(spec.get('angle_deg', 0))
        if not np.isfinite(angle):
            raise ValueError('Nonfinite ellipse angle')
        theta = np.deg2rad(angle); c, s = np.cos(theta), np.sin(theta)
        extent = np.sqrt([(axes[0]*c)**2+(axes[1]*s)**2, (axes[0]*s)**2+(axes[1]*c)**2])
        bounds(center - extent, center + extent, width, height)
        u = (x-center[0])*c + (y-center[1])*s
        v = -(x-center[0])*s + (y-center[1])*c
        mask = (u/axes[0])**2 + (v/axes[1])**2 <= 1
        meta.update(requested_center_xy=center.tolist(), requested_semiaxes=axes.tolist(), requested_angle_deg=angle)
    elif shape == 'polygon':
        vertices = polygon_vertices(spec['vertices'], width, height)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, [vertices], 1)
        mask = canvas > 0
        meta['raster_vertices_xy'] = vertices.tolist()
    elif shape in ('line', 'polyline'):
        thickness = positive(spec['width'], 'line width', 1)
        if shape == 'line':
            center = point(spec['center']); length = positive(spec['length'], 'line length', 2)
            angle = float(spec['angle_deg'])
            if not np.isfinite(angle):
                raise ValueError('Nonfinite line angle')
            direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            points = np.stack((center - direction*length/2, center + direction*length/2))
            meta.update(requested_center_xy=center.tolist(), requested_angle_deg=angle % 180, requested_length=length)
        else:
            points = np.asarray(spec['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError('Polyline requires at least two finite points')
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths < 3):
            raise ValueError('Degenerate line segment at 256 working scale')
        bounds(points.min(0) - thickness/2, points.max(0) + thickness/2, width, height)
        distance = np.full((height, width), np.inf)
        for a, b in zip(points[:-1], points[1:]):
            d = b-a
            t = np.clip(((x-a[0])*d[0] + (y-a[1])*d[1]) / float(d @ d), 0, 1)
            distance = np.minimum(distance, np.hypot(x-a[0]-t*d[0], y-a[1]-t*d[1]))
        mask = distance <= thickness/2
        meta.update(points_xy=points.tolist(), requested_width=thickness, path_length=float(lengths.sum()),
                    pixel_center_width_rule='distance to finite centerline <= width/2, round caps')
    elif shape == 'anostyler_blob':
        center = point(spec['center']); size = point(spec['size'])
        if np.any(size < 4) or np.any(size != np.rint(size)):
            raise ValueError('Blob width/height must be integer and at least 4 pixels')
        w, h = size.astype(int)
        left, top = np.rint(center - (size-1)/2).astype(int)
        bounds(np.array([left, top]), np.array([left+w-1, top+h-1]), width, height)
        mask = np.zeros((height, width), dtype=bool)
        mask[top:top+h, left:left+w] = author_blob(w, h, seed)
        meta.update(upstream_reuse=True, upstream_function='AnoStyler.msp_dot -> cropped and nearest-resized bbox',
                    requested_bbox_xyxy=[int(left), int(top), int(left+w-1), int(top+h-1)])
    else:
        raise ValueError('Unsupported geometry: ' + str(shape))
    if int(mask.sum()) < 4:
        raise ValueError('Empty or degenerate raster region')
    return mask, {**meta, 'raster': describe(mask)}
