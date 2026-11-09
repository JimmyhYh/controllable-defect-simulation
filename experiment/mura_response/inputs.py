"""Pure CPU metadata, fixed geometry and paired digital measurements.

Only artificial arrays are used by numerical_audit. No image/file decoder,
detector package, historical runtime import, or real-data execution entry exists.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


SIZE = 256
AMPLITUDES = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
K_VALUES = (0.4, 0.2, 0.8)
LB_FLOOR = 1e-6
LABEL_FLOOR = np.float32(1 / 255)


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no replacement or retry')


def array_identity(value):
    a = np.ascontiguousarray(value)
    header = f'{a.dtype.str}|{",".join(map(str, a.shape))}|C\n'.encode('ascii')
    return hashlib.sha256(header + a.tobytes(order='C')).hexdigest()


def working_rgb_identity(rgb):
    validate_rgb(rgb)
    return array_identity(rgb)


def validate_rgb(rgb):
    require(isinstance(rgb, np.ndarray), 'RGB must be an explicit NumPy array')
    require(rgb.dtype == np.float32 and rgb.shape == (SIZE, SIZE, 3),
            'RGB must be float32, HWC256; no implicit conversion or scaling')
    require(bool(np.isfinite(rgb).all()) and float(rgb.min()) >= 0
            and float(rgb.max()) <= 1, 'RGB must be finite and within [0,1]')


def geometry_specs():
    result = []
    for radius in (4, 8, 12):
        for family, angle in (('circle', 0), ('ellipse_h', 0), ('ellipse_d', 45)):
            axes = ([float(radius)] * 2 if family == 'circle'
                    else [radius * math.sqrt(3), radius / math.sqrt(3)])
            result.append({'id': f'r{radius:02d}_{family}', 'family': family,
                           'radius_scale': radius, 'center_xy': [128., 128.],
                           'axes_xy': axes, 'angle_deg': float(angle),
                           'units': 'px256', 'frame_hw': [SIZE, SIZE]})
    return result


def make_geometry(spec):
    """Same reviewed pixel-centre ellipse predicate; no old-module import."""
    require(spec in geometry_specs(), 'Geometry is outside the frozen nine specifications')
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cx, cy = spec['center_xy']; ax, ay = spec['axes_xy']
    theta = np.deg2rad(spec['angle_deg']); c, s = np.cos(theta), np.sin(theta)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    mask = ((u / ax) ** 2 + (v / ay) ** 2 <= 1)
    require(int(mask.sum()) >= 4, 'Degenerate geometry')
    inside = distance_transform_edt(mask).astype(np.float32)
    profile = np.float32(.5) + np.float32(.5) * (inside / inside.max())
    alpha = np.where(mask, np.clip(inside / np.float32(2), 0, 1) * profile,
                     np.float32(0)).astype(np.float32)
    require(not np.any(alpha[~mask]) and bool((alpha[mask] > 0).all()),
            'Alpha must have exactly the fixed geometry support')
    y, x = np.nonzero(mask)
    metadata = {'spec': spec, 'area_pixels': int(mask.sum()),
                'bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                'centroid_xy': [float(x.mean()), float(y.mean())],
                'mask_dtype': mask.dtype.str, 'alpha_dtype': alpha.dtype.str,
                'M_geo_sha256': array_identity(mask), 'alpha_sha256': array_identity(alpha),
                'alpha_min_mean_max': [float(alpha[mask].min()),
                                       float(alpha[mask].mean(dtype=np.float64)),
                                       float(alpha[mask].max())]}
    return {'M_geo': mask, 'alpha': alpha, 'metadata': metadata}


def digital_y(rgb):
    # Measurement accumulation is float64; model/generation inputs remain float32.
    return (rgb[..., 0].astype(np.float64) * .2126
            + rgb[..., 1].astype(np.float64) * .7152
            + rgb[..., 2].astype(np.float64) * .0722)


def background_measurement(rgb, geometry):
    validate_rgb(rgb)
    mask = geometry['M_geo']; values = digital_y(rgb)[mask]
    lb = float(values.mean()); defined = lb > LB_FLOOR
    return {'Lb': lb, 'defined': bool(defined), 'area_pixels': int(mask.sum()),
            'background_std_Y': float(values.std(ddof=0)),
            'background_relative_std': float(values.std(ddof=0) / lb) if defined else None}


def measure_pair(background, image, geometry):
    validate_rgb(background); validate_rgb(image)
    mask = geometry['M_geo']; base = background_measurement(background, geometry)
    dy = digital_y(image) - digital_y(background)
    delta = float(dy[mask].mean())
    c = 100 * abs(delta) / base['Lb'] if base['defined'] else None
    area = base['area_pixels']
    semu = {str(k): (c / (1.97 / (area * k * k) ** .33 + .72)
                     if c is not None else None) for k in K_VALUES}
    residual = np.max(np.abs(image - background), axis=-1)
    require(float(residual[~mask].max(initial=0)) == 0,
            'Unexpected change outside fixed geometry')
    return {**base, 'DeltaY': delta, 'C_pair_pct': c, 'SEMU_proxy': semu,
            'nonzero_pixels': int((residual > 0).sum()),
            'M_eff_1_over_255_pixels': int((residual > LABEL_FLOOR).sum()),
            'zero_actual_change': bool(not np.any(residual)),
            'outside_max': float(residual[~mask].max(initial=0)),
            'residual_max': float(residual.max()),
            'energy_RGB': float(np.square(image.astype(np.float64)
                                          - background.astype(np.float64)).sum()),
            'measurement_domain': 'encoded RGB weighted Y; not radiometric luminance',
            'admission_uses_effective_mask': False}


def render_measurement(background, geometry, sign, amplitude):
    validate_rgb(background)
    require(sign in (-1, 1) and amplitude in AMPLITUDES,
            'Sign/amplitude is outside the frozen grid')
    raw = background + np.float32(sign * amplitude) * geometry['alpha'][..., None]
    image = np.clip(raw, 0, 1).astype(np.float32)
    measurement = measure_pair(background, image, geometry)
    measurement['clipped_channel_fraction'] = float(((raw < 0) | (raw > 1))[
        geometry['M_geo']].mean())
    measurement.update(sign=sign, amplitude=amplitude)
    return {'image': image, 'measurement': measurement}


def work_to_original(points):
    points = np.asarray(points, dtype=np.float64)
    require(points.shape[-1] == 2 and bool(np.isfinite(points).all()), 'Invalid x,y points')
    return (points + .5) * 4 - .5


def original_to_work(points):
    points = np.asarray(points, dtype=np.float64)
    require(points.shape[-1] == 2 and bool(np.isfinite(points).all()), 'Invalid x,y points')
    return (points + .5) / 4 - .5


def request_dictionary(roles):
    rows = []
    for role, prefix in (('D_fit60', 'fit'), ('D_hold60', 'hold')):
        for bg in roles[role]:
            for spec in geometry_specs():
                for polarity, sign in (('bright', 1), ('dark', -1)):
                    for i, amp in enumerate(AMPLITUDES):
                        rows.append({'id': f'{prefix}_b{bg["ordinal"]:03d}_{spec["id"]}_{polarity}_a{i}',
                                     'role': role, 'background_id': bg['id'],
                                     'geometry_id': spec['id'], 'polarity': polarity,
                                     'sign': sign, 'amplitude_index': i, 'amplitude': amp})
    require(len(rows) == 15120 and len({row['id'] for row in rows}) == 15120,
            'Nominal request dictionary does not close')
    return rows


def numerical_audit():
    """Scientific CPU input/measurement checks, only artificial arrays."""
    geometries = [make_geometry(spec) for spec in geometry_specs()]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    bump = .2 + .5 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 25. ** 2))
    gradient = .15 + .7 * xx / 255
    artificial = {'uniform_mid': np.full((SIZE, SIZE, 3), .5, np.float32),
                  'spatial_bump': np.repeat(bump[..., None], 3, axis=2).astype(np.float32),
                  'gradient': np.repeat(gradient[..., None], 3, axis=2).astype(np.float32),
                  'all_zero': np.zeros((SIZE, SIZE, 3), np.float32),
                  'all_one': np.ones((SIZE, SIZE, 3), np.float32),
                  'near_one': np.full((SIZE, SIZE, 3), .999, np.float32)}
    rows = []; weak_retained = 0; zero_achieved = 0; undefined = 0
    energy_errors = []; independent_formula_errors = []; zero_injection = []
    for name, background in artificial.items():
        for geometry in geometries:
            local = []; area = geometry['metadata']['area_pixels']
            for sign in (1, -1):
                for amp in AMPLITUDES:
                    result = render_measurement(background, geometry, sign, amp)
                    row = result['measurement']
                    require(row['area_pixels'] == area and row['outside_max'] == 0,
                            'Artificial-array fixed-support property failed')
                    if row['defined']:
                        c_direct = 100 * abs(row['DeltaY']) / row['Lb']
                        for k in K_VALUES:
                            direct = c_direct / (1.97 / ((area * k * k) ** .33) + .72)
                            independent_formula_errors.append(abs(row['SEMU_proxy'][str(k)] - direct))
                    else:
                        undefined += 1
                        require(row['C_pair_pct'] is None
                                and all(v is None for v in row['SEMU_proxy'].values()),
                                'Undefined Lb was silently regularized')
                    weak_retained += int(row['nonzero_pixels'] > 0
                                         and row['M_eff_1_over_255_pixels'] == 0)
                    zero_achieved += int(row['zero_actual_change'])
                    saved = {'artificial_background': name, 'geometry_id': geometry['metadata']['spec']['id'],
                             'image_sha256': working_rgb_identity(result['image']), **row}
                    rows.append(saved); local.append(saved)
            for sign in (1, -1):
                by_amp = {row['amplitude']: row for row in local if row['sign'] == sign}
                for low, high in zip(AMPLITUDES[:-1], AMPLITUDES[1:]):
                    a, b = by_amp[low], by_amp[high]
                    if a['clipped_channel_fraction'] == b['clipped_channel_fraction'] == 0 and a['energy_RGB'] > 0:
                        relative = abs((b['energy_RGB'] / a['energy_RGB']) / (high / low) ** 2 - 1)
                        energy_errors.append(relative)
            if name == 'spatial_bump':
                m = measure_pair(background, background.copy(), geometry)
                legacy_global_offset = 100 * abs(digital_y(background)[geometry['M_geo']].mean()
                                                  - digital_y(background).mean()) / digital_y(background).mean()
                require(m['C_pair_pct'] == 0 and legacy_global_offset > 1,
                        'Zero injection did not isolate spatial background offset')
                zero_injection.append({'geometry_id': geometry['metadata']['spec']['id'],
                                       'paired_contrast_pct': m['C_pair_pct'],
                                       'unpaired_global_offset_pct': float(legacy_global_offset)})
    require(len(rows) == 756 and weak_retained > 0 and zero_achieved > 0 and undefined > 0,
            'Artificial audit lacks required edge conditions')
    require(max(energy_errors, default=0) <= 5e-4, 'Unclipped energy scaling exceeds float32 allowance')
    require(max(independent_formula_errors, default=0) <= 1e-12, 'Proxy scalar formula mismatch')
    points = np.asarray([[0., 0.], [128., 128.], [255., 255.], [3.5, 7.25]])
    roundtrip = float(np.abs(original_to_work(work_to_original(points)) - points).max())
    require(roundtrip <= 1e-12, 'Coordinate inverse mismatch')
    cut = []
    for value in (0., 1e-6, 1.1e-6):
        m = background_measurement(np.full((SIZE, SIZE, 3), value, np.float32), geometries[0])
        require(m['defined'] == (m['Lb'] > LB_FLOOR), 'Lb boundary mismatch')
        cut.append({'requested_scalar': value, **m})
    return {'passed': True, 'real_RGB_decoded': 0, 'real_GT_read': 0,
            'formal_stimuli_generated': 0, 'artificial_request_count': len(rows),
            'artificial_background_count': len(artificial), 'fixed_geometries': len(geometries),
            'weak_nonzero_without_M_eff_retained': weak_retained,
            'zero_actual_change_requests_retained': zero_achieved,
            'undefined_proxy_requests_retained': undefined,
            'unclipped_energy_pair_count': len(energy_errors),
            'unclipped_energy_max_relative_error': max(energy_errors, default=0),
            'unclipped_energy_relative_tolerance': 5e-4,
            'scalar_formula_max_absolute_error': max(independent_formula_errors, default=0),
            'zero_injection_spatial_offset_checks': zero_injection,
            'coordinate_roundtrip_max_error': roundtrip,
            'working_center_xy': [128., 128.],
            'original_pixel_center_xy': work_to_original([[128., 128.]]).tolist()[0],
            'coordinate_note': '(x+0.5)*4-0.5; pixel centres, not x*4',
            'Lb_boundary_checks': cut, 'rows': rows,
            'interpretation': 'artificial numerical evidence only; no model/real-background validity claim'}
