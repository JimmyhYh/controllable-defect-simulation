"""G1 train-side sampling and CPU feasibility only; never loads a detector."""
import argparse
import copy
from collections import Counter
import time

import cv2
import numpy as np

from .data import OUT, ROOT, SPLIT, digest, load_split, normal_rgb, read_json, save_json
from .render import render
from .geometry import rasterize

FOLDER = OUT / 'training_preparation'
POLICY_PATH = FOLDER / 'g1_sampling.json'
AUDIT_PATH = FOLDER / 'N_gen_sampling_audit.json'


def generator_identity():
    return {name: digest(ROOT / 'controlled_generation' / name)
            for name in ('data.py', 'geometry.py', 'render.py', 'sampling.py')}


def policy():
    return {'schema': 'g1-controlled-sampling-v2-brightness-feasible-placement', 'input_size': 256, 'units': 'px256',
            'rng': 'PCG64 via SeedSequence([root_seed, request_ordinal, attempt_index])',
            'audit_seed': 20261028, 'formal_sampling_seed': 20261030,
            'signed_amplitudes': [-.04, .04], 'max_attempts_per_request': 5,
            'Mura_shapes_uniform': ['circle', 'ellipse', 'anostyler_blob'],
            'line_shapes_uniform': ['line', 'polyline'], 'circle_radius': [12, 36],
            'ellipse_semiaxes': [[16, 40], [10, 28]], 'blob_size_integer_inclusive': [[24, 72], [16, 48]],
            'line_length': [45, 140], 'line_width_integer_inclusive': [3, 7], 'angle_deg': [0, 180],
            'polyline_segments_integer_inclusive': [2, 3], 'polyline_bend_px': [-12, 12],
            'geometry_distribution': 'continuous uniform unless integer inclusive explicitly named',
            'placement': 'uniform over shape-fitting integer translations with >=90% normal-only conservative brightness capacity; 1px margin',
            'capacity': 'max RGB residual from signed0.01 > float32(1/255); lower bound since every ROI alpha >=0.25',
            'placement_revision_reason': 'Unconstrained positive line request42 on0386 failed five attempts due to white saturation; same background/polarity/amplitude retained',
            'unconstrained_evidence': 'unconstrained_g1_sampling.json and unconstrained_sampling_audit.json retained separately',
            'retry': 'same background, defect kind and polarity; resample geometry and outline seed only',
            'invalid_after_5': 'stop; no background replacement or conversion to normal',
            'float_label': 'max RGB abs residual > float32(1/255)', 'coverage_minimum': .90,
            'audit_cycle': [['mura_like', 1], ['mura_like', -1], ['line_like', 1], ['line_like', -1]],
            'formal_effective_anomaly_cycle': [['mura_like', 1], ['line_like', 1], ['mura_like', -1], ['line_like', -1]],
            'split_sha256': digest(SPLIT), 'generator_identity': generator_identity(),
            'trial_protocol_sha256': digest(OUT / 'generation_protocol.json'),
            'screen_physics_claim': False, 'model_score_selection': False}


def sample_spec(kind, polarity, rng, cfg):
    if kind not in ('mura_like', 'line_like') or polarity not in (-1, 1):
        raise ValueError('Explicit kind and polarity required')
    spec = {'kind': kind, 'units': cfg['units'], 'amplitude': .04 * polarity}
    if kind == 'mura_like':
        shape = str(rng.choice(cfg['Mura_shapes_uniform'])); spec['shape'] = shape
        if shape == 'circle':
            radius = float(rng.uniform(*cfg['circle_radius'])); spec['radius'] = radius
            lo, hi = np.repeat(-radius, 2), np.repeat(radius, 2)
        elif shape == 'ellipse':
            axes = np.array([rng.uniform(*axis) for axis in cfg['ellipse_semiaxes']])
            angle = float(rng.uniform(*cfg['angle_deg'])); theta = np.deg2rad(angle)
            extent = np.sqrt([(axes[0]*np.cos(theta))**2+(axes[1]*np.sin(theta))**2,
                              (axes[0]*np.sin(theta))**2+(axes[1]*np.cos(theta))**2])
            spec.update(axes=axes.tolist(), angle_deg=angle); lo, hi = -extent, extent
        else:
            size = np.array([rng.integers(a, b+1) for a, b in cfg['blob_size_integer_inclusive']])
            spec['size'] = size.tolist(); lo, hi = -(size-1)/2, (size-1)/2
    else:
        shape = str(rng.choice(cfg['line_shapes_uniform'])); spec['shape'] = shape
        length = float(rng.uniform(*cfg['line_length']))
        a, b = cfg['line_width_integer_inclusive']; width = int(rng.integers(a, b+1)); spec['width'] = width
        angle = float(rng.uniform(*cfg['angle_deg'])); theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        if shape == 'line':
            points = np.array([[-length/2, 0], [length/2, 0]]) @ rotation.T
            spec.update(length=length, angle_deg=angle)
        else:
            a, b = cfg['polyline_segments_integer_inclusive']; segments = int(rng.integers(a, b+1))
            # Monotone local x prevents crossing/degenerate segments, without changing a failed request's polarity.
            u = np.linspace(-length/2, length/2, segments+1)
            v = np.zeros_like(u); v[1:-1] = rng.uniform(*cfg['polyline_bend_px'], size=segments-1)
            points = np.stack((u, v), axis=1) @ rotation.T
        lo, hi = points.min(0)-width/2, points.max(0)+width/2
    minimum, maximum = 1-lo, 254-hi
    if np.any(minimum > maximum):
        raise ValueError('Proposed shape cannot fit working frame')
    center = rng.uniform(minimum, maximum)
    if shape == 'polyline': spec['points'] = (points+center).tolist()
    else: spec['center'] = center.tolist()
    return spec


def feasible_placement(rgb, spec, outline_seed, rng):
    """Condition on normal brightness capacity, never on model scores or labels."""
    roi, meta = rasterize(spec, seed=outline_seed)
    x0,y0,x1,y1 = meta['raster']['bbox_xyxy']
    kernel = roi[y0:y1+1,x0:x1+1].astype(np.float32)
    signed_minimum = np.float32(.01 if spec['amplitude'] > 0 else -.01)
    capacity = (np.max(np.abs(np.clip(rgb+signed_minimum,0,1)-rgb),axis=-1)>np.float32(1/255)).astype(np.float32)
    count = np.rint(cv2.matchTemplate(capacity,kernel,cv2.TM_CCORR)).astype(np.int32)
    y,x = np.mgrid[:count.shape[0],:count.shape[1]]
    dx,dy = x-x0,y-y0
    shape=spec['shape']
    if shape=='circle':lo=np.asarray(spec['center'])-spec['radius'];hi=np.asarray(spec['center'])+spec['radius']
    elif shape=='ellipse':
        a,b=spec['axes'];t=np.deg2rad(spec['angle_deg'])
        extent=np.sqrt([(a*np.cos(t))**2+(b*np.sin(t))**2,(a*np.sin(t))**2+(b*np.cos(t))**2])
        lo=np.asarray(spec['center'])-extent;hi=np.asarray(spec['center'])+extent
    elif shape=='anostyler_blob':
        box=meta['requested_bbox_xyxy'];lo=np.asarray(box[:2]);hi=np.asarray(box[2:])
    else:
        points=np.asarray(meta['points_xy']);lo=points.min(0)-spec['width']/2;hi=points.max(0)+spec['width']/2
    fitting=(lo[0]+dx>=1)&(lo[1]+dy>=1)&(hi[0]+dx<=254)&(hi[1]+dy<=254)
    eligible=fitting&(count>=int(np.ceil(.9*roi.sum())))
    ys,xs=np.nonzero(eligible)
    if not len(xs):raise ValueError('No shape-fitting location meets normal brightness capacity for this proposal')
    selected=int(rng.integers(len(xs))); shift=np.array([xs[selected]-x0,ys[selected]-y0])
    placed=copy.deepcopy(spec)
    if 'center' in placed:placed['center']=(np.asarray(placed['center'])+shift).tolist()
    else:placed['points']=(np.asarray(placed['points'])+shift).tolist()
    return placed, {'eligible_integer_translations':int(len(xs)), 'all_shape_fitting_integer_translations':int(fitting.sum()),
                    'minimum_capacity_fraction':.9,'uses_normal_brightness_only':True}


def generate_request(rgb, kind, polarity, root_seed, ordinal, cfg):
    attempts = []
    for attempt in range(cfg['max_attempts_per_request']):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([root_seed, ordinal, attempt])))
        spec = sample_spec(kind, polarity, rng, cfg)
        outline_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        try:
            spec, placement = feasible_placement(rgb,spec,outline_seed,rng)
            result = render(rgb, spec, outline_seed)
            valid = result['stats']['valid']
            info = {'attempt': attempt+1, 'spec': spec, 'outline_seed': outline_seed,
                    'valid': valid, 'stats': result['stats'],'placement':placement}
        except ValueError as error:
            result = None; valid = False
            info = {'attempt': attempt+1, 'spec': spec, 'outline_seed': outline_seed,
                    'valid': False, 'geometry_error': str(error)}
        attempts.append(info)
        if valid: return result, attempts
    return None, attempts


def validate_policy(cfg):
    if cfg != policy():
        raise ValueError('Sampling configuration or pinned generator changed; preserve existing frozen preparation')


def prepare_sampling():
    split = load_split(); cfg = policy(); FOLDER.mkdir(parents=True, exist_ok=True)
    if POLICY_PATH.exists() and read_json(POLICY_PATH) != cfg:
        raise ValueError('Sampling policy exists with another identity; no overwrite')
    save_json(POLICY_PATH, cfg)
    if AUDIT_PATH.exists():
        result = read_json(AUDIT_PATH)
        if result['policy_sha256'] != digest(POLICY_PATH):
            raise ValueError('Saved sampling audit identity changed')
        if not result['ready_for_probe_preparation']:
            raise ValueError('Previous sampling feasibility failed; diagnose instead of silently regenerating')
        print('SAMPLING_CACHED', result['accepted_requests'], result['attempts'], flush=True)
        return result
    start = time.perf_counter(); records = []; stop = False
    for background_index, relative in enumerate(split['N_gen']):
        rgb = normal_rgb(split, relative)  # Generation access: N_gen only, no held-out pool opened.
        for cycle_index, (kind, polarity) in enumerate(cfg['audit_cycle']):
            ordinal = background_index*4 + cycle_index
            generated, attempts = generate_request(rgb, kind, polarity, cfg['audit_seed'], ordinal, cfg)
            records.append({'request_ordinal': ordinal, 'background': relative, 'kind': kind, 'polarity': polarity,
                            'accepted': generated is not None, 'attempts': attempts})
            if generated is None:
                stop = True; break
        if stop: break
        if (background_index+1)%20 == 0:
            print('SAMPLING_NORMAL_BACKGROUNDS', background_index+1, '/80', flush=True)
    count = sum(row['accepted'] for row in records)
    stats = [row['attempts'][-1]['stats'] for row in records if row['accepted']]
    ready = not stop and count == 320
    result = {'policy_sha256': digest(POLICY_PATH), 'planned_requests': 320, 'completed_requests': len(records),
              'normal_backgrounds_completed': len({row['background'] for row in records}),
              'accepted_requests': count, 'failed_requests': len(records)-count,
              'unexecuted_requests': 320-len(records), 'attempts': sum(len(row['attempts']) for row in records),
              'rejected_attempts': sum(sum(not x['valid'] for x in row['attempts']) for row in records),
              'accepted_by_kind_polarity': dict(Counter(f"{row['kind']}_{row['polarity']:+d}" for row in records if row['accepted'])),
              'accepted_by_shape': dict(Counter(x['geometry']['shape'] for x in stats)),
              'minimum_effective_coverage': min(x['effective_coverage'] for x in stats) if stats else None,
              'max_outside_ROI_float_residual': max(x['outside_ROI_max_float_residual'] for x in stats) if stats else None,
              'max_clipped_channel_fraction': max(x['roi_clipped_channel_fraction'] for x in stats) if stats else None,
              'records': records, 'CPU_generate_and_audit_seconds': time.perf_counter()-start,
              'ready_for_probe_preparation': ready, 'training_started': False, 'model_predictions': 0,
              'pool_opened': 'N_gen only', 'independent_real_data_status': 'pending'}
    save_json(AUDIT_PATH, result)
    print('SAMPLING_AUDIT', {key: result[key] for key in ('accepted_requests', 'planned_requests', 'attempts',
          'rejected_attempts', 'minimum_effective_coverage', 'ready_for_probe_preparation')}, flush=True)
    if not ready:
        raise ValueError('Feasibility condition failed; see exact background/polarity and attempts, do not train')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=['prepare'])
    parser.parse_args(); prepare_sampling()
