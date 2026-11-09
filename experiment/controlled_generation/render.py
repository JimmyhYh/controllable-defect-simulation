"""Interpretable Mura-like/line appearance, not a manufacturing physics model."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import describe, rasterize


def render(rgb, spec, seed=20261014):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('Input must be finite RGB float data in [0,1], shape256x256x3')
    amplitude = float(spec['amplitude'])
    if not np.isfinite(amplitude) or amplitude == 0 or abs(amplitude) > .25:
        raise ValueError('Signed amplitude must be nonzero, finite and at most0.25')
    kind = spec.get('kind')
    if kind not in ('mura_like', 'line_like'):
        raise ValueError('Use explicitly named mura_like or line_like appearance')
    roi, geometry = rasterize(spec, seed=seed)
    inside = distance_transform_edt(roi).astype(np.float32)
    if kind == 'mura_like':
        # Finite-support, smooth interior; the last pixel has a nonzero step.
        # This makes appearance controllable, but is not a measured physical edge.
        profile = np.float32(.5) + np.float32(.5) * (inside / inside.max())
        alpha = np.clip(inside / np.float32(2), 0, 1) * profile
    else:
        alpha = np.clip(inside / np.float32(1.25), 0, 1)
    alpha = np.where(roi, alpha, np.float32(0)).astype(np.float32)
    unclipped = rgb + np.float32(amplitude) * alpha[..., None]
    generated = np.clip(unclipped, 0, 1).astype(np.float32)
    residual = np.max(np.abs(generated-rgb), axis=-1)
    effective = residual > np.float32(1/255)
    before8 = np.rint(rgb*255).astype(np.uint8)
    after8 = np.rint(generated*255).astype(np.uint8)
    residual8 = np.max(np.abs(after8.astype(np.int16)-before8.astype(np.int16)), axis=-1)
    effective8 = residual8 > 1
    coverage = float((effective & roi).sum() / roi.sum())
    outside = float(residual[~roi].max(initial=0))
    issues = []
    if coverage < .9:
        issues.append('Effective float coverage below0.90')
    if outside != 0 or np.any(effective & ~roi):
        issues.append('Unexpected change outside request region')
    stats = {'valid': not issues, 'invalid_reasons': issues, 'kind': kind, 'signed_amplitude': amplitude,
             'geometry': geometry, 'effective': describe(effective), 'png_effective': describe(effective8),
             'effective_coverage': coverage, 'png_effective_coverage': float((effective8 & roi).sum() / roi.sum()),
             'float_png_mask_disagreement_pixels': int(np.count_nonzero(effective != effective8)),
             'outside_ROI_max_float_residual': outside, 'outside_ROI_max_png_codes': int(residual8[~roi].max(initial=0)),
             'roi_residual_min_median_max': [float(residual[roi].min()), float(np.median(residual[roi])), float(residual[roi].max())],
             'roi_clipped_channel_fraction': float(((unclipped < 0) | (unclipped > 1))[roi].mean()),
             'residual_metric': 'max abs RGB channel difference in [0,1]',
             'float_label_threshold': float(np.float32(1/255)), 'png_label_threshold_codes': 1,
             'mask_note': 'M_roi is requested region; effective float label differs from PNG label after quantization',
             'physical_simulation': False}
    if 'requested_center_xy' in geometry:
        stats['effective_center_error_px'] = float(np.linalg.norm(np.asarray(stats['effective']['centroid_xy'])-
                                                   geometry['requested_center_xy'])) if effective.any() else None
    if spec['shape'] == 'line' and effective.any():
        difference = abs(stats['effective']['principal_angle_deg'] - geometry['requested_angle_deg']) % 180
        stats['effective_angle_error_deg'] = min(difference, 180-difference)
        stats['effective_area_over_centerline_length_px'] = float(effective.sum() / geometry['path_length'])
        stats['width_note'] = 'area/length includes round end caps; not an exact line-width measurement'
    return {'background': rgb, 'image': generated, 'M_roi': roi, 'alpha': alpha,
            'M_eff': effective, 'M_eff_png': effective8, 'residual': residual,
            'background_png': before8, 'image_png': after8, 'stats': stats}
