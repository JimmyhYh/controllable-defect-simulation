"""Explicit read-only source/input/statistics audit. Never runs old entry points."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / 'results/pod_probe'
OUT = OLD / 'audit_20261028_v1'
LEVELS = [1., 1.6, 2.5, 4., 6.3, 10., 16., 25.]
SHAPES = ['mura_p25', 'mura_med', 'mura_p75', 'line_med']
FORBIDDEN = ('torch', 'torchvision', 'timm', 'segmentation_models_pytorch',
             'patchcore', 'controlled_generation', 'pod_probe', 'localization_calibration')


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in FORBIDDEN):
            raise RuntimeError('Prohibited import during CPU audit: ' + fullname)
        return None


sys.meta_path.insert(0, BlockModels())


def require(ok, why):
    if not ok:
        raise RuntimeError(why + '; preserve all state; no automatic retry')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, data):
    p = Path(p)
    require(not p.exists(), 'Refuse overwrite: ' + str(p))
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def resource():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ['total', 'available', 'page', 'page_available',
                                            'virtual', 'virtual_available', 'extended']]
    m = Memory(); m.length = ctypes.sizeof(m)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    f = kernel.GlobalMemoryStatusEx
    f.argtypes = [ctypes.POINTER(Memory)]; f.restype = ctypes.c_int
    require(f(ctypes.byref(m)), 'RAM query failed')
    state = {'available_RAM_GiB': m.available / 2 ** 30,
             'free_disk_GiB': shutil.disk_usage(ROOT).free / 2 ** 30}
    require(state['available_RAM_GiB'] >= 2 and state['free_disk_GiB'] >= 2, 'CPU resource admission failed')
    return state


def sources():
    paths = list((ROOT / 'pod_probe').glob('*.py'))
    for d in ['anolcd_stage2_v1', 'spot_probe_v1', 'capability_v1', 'alt_generator_v1',
              'resolution_probe_v1']:
        paths += [p for p in (OLD / d).rglob('*') if p.is_file() and p.suffix in ('.json', '.md', '.npz', '.npy')]
    paths += [ROOT / p for p in ['controlled_generation/render.py', 'controlled_generation/geometry.py',
        'controlled_generation/formal_sampling.py', 'controlled_generation/formal_preparation.py',
        'localization_synthesis.py', 'localization_training.py', 'localization_calibration.py',
        'results/controlled_generation/formal_preparation/common_config.json',
        'data/weak_localization/split.json']]
    paths += [ROOT.parent.parent / '报告/受控探针检测器能力标定专题报告.md']
    paths += list((ROOT.parent.parent / '计划').glob('受控探针阶段*方案.md'))
    paths += [ROOT.parent.parent / '计划/受控探针主线研究问题.md']
    return sorted(set(paths))


def current_identity():
    return {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('__init__.py')]}


def check_sources():
    s = read(OUT / 'sources.json')
    require(s['implementation'] == current_identity(), 'Audit implementation changed')
    for p, h in s['files'].items():
        require(Path(p).is_file() and digest(p) == h, 'Frozen source changed: ' + p)
    return len(s['files'])


def cached(stage):
    path = OUT / (stage + '_complete.json')
    if path.exists():
        c = read(path)
        for p, h in c['outputs'].items():
            require(digest(p) == h, 'Audit output changed: ' + p)
        check_sources()
        return c
    require(not (OUT / (stage + '_started.json')).exists(), 'Partial/unknown transaction: ' + stage)
    return None


def begin(stage):
    resource(); check_sources()
    save(OUT / (stage + '_started.json'), {'stage': stage, 'time': time.time(), 'pid': os.getpid()})


def seal(stage, files, summary):
    check_sources()
    c = {'stage': stage, 'outputs': {str(p): digest(p) for p in files}, 'summary': summary,
         'forbidden_imports_present': [n for n in sys.modules if n.split('.')[0] in FORBIDDEN]}
    require(not c['forbidden_imports_present'], 'Forbidden runtime imported')
    save(OUT / (stage + '_complete.json'), c)
    return summary


def checked_rows(path, index=None):
    data = read(path); rows = data['rows']; tau = float(data['tau'])
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate score IDs: ' + str(path))
    for r in rows:
        require(math.isfinite(r['score']), 'Nonfinite saved score')
        require(r['hit'] == (r['score'] > tau), 'Saved hit differs from frozen tau')
    if index is not None:
        require([r['id'] for r in rows] == [r['id'] for r in index['records']], 'Score/request order differs')
    return data, rows


def prepare():
    if OUT.exists():
        require((OUT / 'prepare_complete.json').exists(), 'Unknown/partial audit output directory')
        return cached('prepare')['summary']
    snap = resource()
    inputs = sources()
    require(all(p.is_file() for p in inputs), 'Missing source')
    frozen = {str(p): digest(p) for p in inputs}
    idx = read(OLD / 'anolcd_stage2_v1/stimuli/index.json')
    sets = [('PatchCore', OLD / 'anolcd_stage2_v1/scores/scores.json', idx),
            ('Spot', OLD / 'spot_probe_v1/scores/scores.json', read(OLD / 'spot_probe_v1/stimuli/index.json'))]
    sets += [(m, OLD / ('capability_v1/' + d + '/scores.json'), idx)
             for m, d in [('G0', 'g01_G0'), ('G1', 'g01_G1'), ('DRAEM-A', 'draem_A'), ('DRAEM-B', 'draem_B')]]
    counts = {}
    for name, p, i in sets:
        data, rows = checked_rows(p, i)
        b = [r for r in rows if r['kind'] == 'baseline']; a = [r for r in rows if r['kind'] == 'stimulus']
        require(len(b) == 60 and len({r['background_index'] for r in b}) == 60, 'Baseline scope differs')
        counts[name] = {'baseline_n': len(b), 'baseline_FP': sum(r['hit'] for r in b),
                        'stimulus_n': len(a), 'stimulus_hits': sum(r['hit'] for r in a), 'tau': data['tau']}
    code = (ROOT / 'pod_probe/pod.py').read_text(encoding='utf-8')
    require('np.percentile(b90, 5)' in code, 'Legacy confidence-bound expression differs')
    cfg = read(ROOT / 'results/controlled_generation/formal_preparation/common_config.json')
    provenance = {'G0': 'DRAEM author-style Perlin/DTD synthetic anomalies',
        'G1': 'procedural controlled Mura/line renderer; not a learned image generator',
        'G0_G1_normal_backgrounds': cfg['N_gen'], 'G0_G1_real_anomaly_training': False,
        'DRAEM_A_B': 'normal N backgrounds plus DTD/Perlin synthetic anomalies; B changes blend strength; not real anomalous training',
        'training_source_evidence': ['common_config.json', 'formal_sampling.py', 'localization_synthesis.py', 'localization_training.py']}
    OUT.mkdir()
    save(OUT / 'sources.json', {'files': frozen, 'implementation': current_identity(),
        'identity_limit': 'Observed current-byte hash; not proof of historical execution identity'})
    save(OUT / 'protocol.json', {'version': 'pod_audit_20261028_v1', 'bootstrap_n': 2000,
        'bootstrap_seed': 20261028, 'fit_basis': 'saved hits and original measured SEMU; biased proxy retained for sensitivity only',
        'confidence_bound': 'exploratory clustered percentile P95 of a90, not standard-certified a90/95',
        'sample_unit': 'background; acquisition groups unknown', 'inputs_read': 'saved NPZ and score arrays only; no original RGB/GT',
        'python': sys.version, 'executable': sys.executable,
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy']}, 'resources': snap,
        'source_url': 'https://ntrs.nasa.gov/api/citations/20140005337/downloads/20140005337.pdf',
        'source_location': 'printed p56: lower one-sided 95% POD curve crossing 0.90; upper bound on required size',
        'stages': ['prepare', 'inputs', 'statistics', 'verify'], 'training': False, 'model_forward': False})
    save(OUT / 'source_audit.json', {'counts': counts, 'legacy_bound_expression': 'percentile(b90,5)',
        'definition_issue_confirmed': True, 'training_provenance': provenance,
        'consequences': ['3.80/8.43 are lower-tail a90 quantiles, not conservative required-stimulus upper bounds',
          'synthetic SEMU uses global background mean; real uses local ring median; direct bridge invalid',
          'G0/G1 labels and real-anomaly source labels in old capability card are wrong',
          'original paired training effects positive on average but inconsistent; not proof of zero effect']})
    return seal('prepare', [OUT / f for f in ['sources.json', 'protocol.json', 'source_audit.json']],
                {'sources': len(frozen), 'counts': counts, 'resources': snap})


def science():
    import numpy as np
    from scipy import ndimage, optimize, special
    return np, ndimage, optimize, special


def quantiles(values):
    np, *_ = science()
    a = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(a): return None
    return {'n': len(a), 'min': float(a.min()), 'p05': float(np.quantile(a, .05)),
            'median': float(np.median(a)), 'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def load_image(path):
    np, *_ = science()
    with np.load(path, allow_pickle=False) as z:
        a = z['image'].copy()
    require(a.shape == (256, 256, 3) and a.dtype == np.float32 and np.isfinite(a).all()
            and a.min() >= 0 and a.max() <= 1, 'Invalid cached RGB array: ' + str(path))
    return a


def roi_for(shape):
    np, *_ = science()
    yy, xx = np.mgrid[:256, :256]
    if shape == 'line_med':
        # Original finite horizontal centerline, round end caps; no call to old geometry.
        dx = np.maximum(np.abs(xx - 128.) - 56., 0.)
        return np.hypot(dx, yy - 128.) <= 2.
    radius = {'mura_p25': 4., 'mura_med': 7., 'mura_p75': 12., 'spot_xs': 1.4}[shape]
    return ((xx - 128.) / radius) ** 2 + ((yy - 128.) / radius) ** 2 <= 1


def measure(base, image, shape):
    np, ndimage, *_ = science()
    d = np.max(np.abs(image - base), axis=2); eff = d > np.float32(1 / 255)
    roi = roi_for(shape)
    yb = (base * np.array([.2126, .7152, .0722], np.float32)).sum(2)
    yi = (image * np.array([.2126, .7152, .0722], np.float32)).sum(2)
    n = int(eff.sum()); require(n > 0, 'Empty original effective support')
    outside = float(d[~roi].max(initial=0))
    ring = ndimage.binary_dilation(eff, iterations=8) & ~ndimage.binary_dilation(eff, iterations=2)
    fallback = int(ring.sum()) < 50
    if fallback: ring = ~eff
    area = n * .16; jnd = 1.97 / area ** .33 + .72
    bg_global = float(yb.mean()); bg_local = float(yb[eff].mean())
    delta = float((yi[eff] - yb[eff]).mean())
    ring_bg = float(np.median(yi[ring])); ring_lo = float(np.median(yi[eff]))
    counterfactual_contrast = delta / bg_local * 100 if bg_local > 0 else None
    same_ring_contrast = (ring_lo - ring_bg) / ring_bg * 100 if ring_bg > 0 else None
    return {'effective_pixels': n, 'area_mm2_assumed': area, 'outside_requested_ROI_max': outside,
        'residual_max': float(d.max()), 'residual_sum': float(d.sum()), 'residual_energy': float((d.astype(float) ** 2).sum()),
        'saturated_at_one_changed_channels': int(((image == 1) & (image != base)).sum()),
        'changed_pixels_below_label_floor': int(((d > 0) & ~eff).sum()),
        'global_SEMU_recomputed': abs((float(yi[eff].mean()) - bg_global) / bg_global * 100) / jnd,
        'zero_injection_global_SEMU_on_same_support': abs((bg_local - bg_global) / bg_global * 100) / jnd,
        'paired_local_delta_mean': delta, 'paired_local_contrast_percent': counterfactual_contrast,
        'paired_local_SEMU_proxy': abs(counterfactual_contrast) / jnd if counterfactual_contrast is not None else None,
        'same_ring_SEMU_proxy': abs(same_ring_contrast) / jnd if same_ring_contrast is not None else None,
        'ring_fallback': fallback}, eff, d


def cells(rows):
    np, *_ = science()
    bases = {r['background_index']: r for r in rows if r['kind'] == 'baseline'}
    stim = [r for r in rows if r['kind'] == 'stimulus']
    result = {}
    for shape in sorted({r['shape'] for r in stim}):
        subset = [r for r in stim if r['shape'] == shape]
        sets = [set(r['background_index'] for r in subset if r['target_SEMU'] == t) for t in LEVELS]
        common = set.intersection(*sets)
        by = []
        for t in LEVELS:
            g = [r for r in subset if r['target_SEMU'] == t]
            clean = [r for r in g if not bases[r['background_index']]['hit']]
            com = [r for r in g if r['background_index'] in common]
            by.append({'target': t, 'n': len(g), 'backgrounds': [r['background_index'] for r in g],
                'hit': sum(r['hit'] for r in g), 'baseline_positive_n': len(g) - len(clean),
                'baseline_negative_n': len(clean), 'baseline_negative_hit': sum(r['hit'] for r in clean),
                'common_n': len(com), 'common_hit': sum(r['hit'] for r in com),
                'joint_hit_and_peak_in_effective': sum(r['hit'] and r.get('peak_in_mask', False) for r in g)
                   if all('peak_in_mask' in r for r in g) else None,
                'actual_SEMU': quantiles([r['SEMU_actual'] for r in g]),
                'delta_score': quantiles([r['score'] - bases[r['background_index']]['score'] for r in g])})
        result[shape] = {'common_backgrounds': sorted(common), 'cells': by}
    return result


def wilson(k, n):
    z = 1.959963984540054; p = k / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - rad, mid + rad]


def inputs():
    c = cached('inputs')
    if c: return c['summary']
    require(cached('prepare') is not None, 'Preparation incomplete'); begin('inputs')
    np, *_ = science(); t0 = time.perf_counter()
    idx = read(OLD / 'anolcd_stage2_v1/stimuli/index.json')
    pc, rows = checked_rows(OLD / 'anolcd_stage2_v1/scores/scores.json', idx)
    spotidx = read(OLD / 'spot_probe_v1/stimuli/index.json')
    sp, sprows = checked_rows(OLD / 'spot_probe_v1/scores/scores.json', spotidx)
    base_records = {r['background_index']: r for r in idx['records'] if r['kind'] == 'baseline'}
    # 60 backgrounds occupy ~45MiB; no original PNG is opened.
    bases = {i: load_image(OLD / 'anolcd_stage2_v1/stimuli' / (r['id'] + '.npz')) for i, r in base_records.items()}
    for i, b in bases.items():
        require(np.array_equal(b, load_image(OLD / 'spot_probe_v1/stimuli' / (base_records[i]['id'] + '.npz'))),
                'Spot baseline differs')
    measures = []; originals = {}
    for folder, index in [('anolcd_stage2_v1', idx), ('spot_probe_v1', spotidx)]:
        for r in index['records']:
            if r['kind'] != 'stimulus': continue
            f = OLD / folder / 'stimuli' / (r['id'] + '.npz')
            im = load_image(f); m, eff, d = measure(bases[r['background_index']], im, r['shape'])
            require(abs(m['area_mm2_assumed'] - r['S_mm2']) < 1e-6, 'Saved area differs: ' + r['id'])
            require(m['outside_requested_ROI_max'] == 0, 'Original changes outside requested ROI')
            require(abs(m['global_SEMU_recomputed'] - r['SEMU_actual']) < 2e-3, 'Original SEMU replay differs')
            if folder == 'anolcd_stage2_v1':
                with np.load(f, allow_pickle=False) as z:
                    mask = np.unpackbits(z['M_eff'], count=256 * 256).reshape(256, 256).astype(bool)
                    require(np.array_equal(mask, eff), 'Saved M_eff differs')
                originals[r['id']] = r
            measures.append({'id': r['id'], 'folder': folder, 'shape': r['shape'], 'background': r['background_index'],
                'target': r['target_SEMU'], 'saved_SEMU': r['SEMU_actual'], **m})
    print('INPUT_RESIDUALS_VERIFIED', len(measures), flush=True)
    mapping = {r['id']: r for r in measures}
    _, altrows = checked_rows(OLD / 'alt_generator_v1/measure/G1_scores.json')
    altidx = read(OLD / 'alt_generator_v1/stimuli/index.json')
    require([r['id'] for r in altrows] == [r['id'] for r in altidx['records']], 'Alt request/score order differs')
    _, grows = checked_rows(OLD / 'capability_v1/g01_G1/scores.json', idx)
    gm = {r['id']: r for r in grows}
    altmeasures = []; paired = []
    for r in altidx['records']:
        if r['kind'] == 'baseline':
            require(np.array_equal(bases[r['background_index']], load_image(OLD / 'alt_generator_v1/stimuli' / (r['id'] + '.npz'))), 'Alt baseline differs')
            continue
        orig = originals[r['paired_original']]
        base = bases[r['background_index']]
        soft = load_image(OLD / 'anolcd_stage2_v1/stimuli' / (orig['id'] + '.npz'))
        hard = load_image(OLD / 'alt_generator_v1/stimuli' / (r['id'] + '.npz'))
        sd = np.max(np.abs(soft - base), axis=2); hd = np.max(np.abs(hard - base), axis=2)
        se = sd > np.float32(1 / 255); he = hd > np.float32(1 / 255)
        both = se & he; union = se | he
        ax = np.nonzero(he); sx = np.nonzero(se)
        sa = next(a for a in altrows if a['id'] == r['id'])
        so = gm[orig['id']]
        paired.append({'background': r['background_index'], 'shape': r['shape'], 'target': r['target_SEMU'],
                       'old_hit': int(so['hit']), 'new_hit': int(sa['hit'])})
        local = mapping[orig['id']]
        weights = np.array([.2126, .7152, .0722], np.float32)
        hard_delta = float(((hard - base) * weights).sum(2)[he].mean()) if he.any() else None
        altmeasures.append({'id': r['id'], 'paired': orig['id'], 'shape': r['shape'],
            'soft_effective_n': int(se.sum()), 'hard_effective_n_common_floor': int(he.sum()),
            'hard_effective_n_old_half_floor': int((hd > np.float32(.5 / 255)).sum()),
            'support_IoU': float(both.sum() / union.sum()), 'identical_support': bool(np.array_equal(se, he)),
            'soft_bbox_width_height': [int(sx[1].max()-sx[1].min()+1), int(sx[0].max()-sx[0].min()+1)],
            'hard_bbox_width_height': [int(ax[1].max()-ax[1].min()+1), int(ax[0].max()-ax[0].min()+1)] if he.any() else None,
            'soft_actual_delta_mean': local['paired_local_delta_mean'], 'hard_actual_delta_mean': hard_delta,
            'delta_ratio_hard_over_soft': hard_delta / local['paired_local_delta_mean'] if hard_delta is not None and local['paired_local_delta_mean'] != 0 else None,
            'soft_requested_effective_n': orig['effective_pixels'], 'saved_hard_area': r['S_mm2_measured']})
    print('ALT_PAIRING_AUDITED', len(altmeasures), flush=True)
    spot_high = []
    for t1, t2 in [(10., 16.), (16., 25.)]:
        a = {r['background_index']: r for r in sprows if r['kind'] == 'stimulus' and r['target_SEMU'] == t1}
        b = {r['background_index']: r for r in sprows if r['kind'] == 'stimulus' and r['target_SEMU'] == t2}
        require(set(a) == set(b), 'High spot background set differs')
        changes = []
        for bg in sorted(a):
            ma, mb = mapping[a[bg]['id']], mapping[b[bg]['id']]
            changes.append({'background': bg, 'old_hit': a[bg]['hit'], 'new_hit': b[bg]['hit'],
                'score_difference': b[bg]['score'] - a[bg]['score'],
                'residual_max_difference': mb['residual_max'] - ma['residual_max'],
                'energy_ratio': mb['residual_energy'] / ma['residual_energy'],
                'SEMU_old': a[bg]['SEMU_actual'], 'SEMU_new': b[bg]['SEMU_actual']})
        spot_high.append({'levels': [t1, t2], 'n': len(a), 'hit_to_miss': sum(x['old_hit'] and not x['new_hit'] for x in changes),
            'miss_to_hit': sum(not x['old_hit'] and x['new_hit'] for x in changes), 'rows': changes})
    # Resolution: only native saved scores and path labels; no RGB/GT, no AUROC recomputation.
    split = read(ROOT / 'data/weak_localization/split.json'); paths = split['Tfinal']
    values = {}; taus = {}
    for sz in [256, 512]:
        f = OLD / 'resolution_probe_v1' / f'scores_{sz}.npy'
        values[sz] = np.load(f, allow_pickle=False)
        require(values[sz].shape == (len(paths),) and np.isfinite(values[sz]).all(), 'Resolution score scope differs')
        meta = read(f.with_suffix('.json')); require(digest(f) == meta['scores_sha256'], 'Resolution stored source hash differs')
        taus[sz] = read(OLD / 'resolution_probe_v1' / f'calibration_{sz}.json')['threshold']
    resolution = {}
    for kind in ['good', 'spot', 'mura', 'line', 'scrathes', 'cylinder']:
        loc = [i for i, p in enumerate(paths) if p.replace('\\','/').split('/')[2] == kind]
        a = values[256][loc] > taus[256]; b = values[512][loc] > taus[512]
        resolution[kind] = {'n': len(loc), 'positive_256': int(a.sum()), 'positive_512': int(b.sum()),
            'positive_to_negative': int((a & ~b).sum()), 'negative_to_positive': int((~a & b).sum())}
    np.savez_compressed(OUT / 'paired_edge_rows.npz', background=np.array([r['background'] for r in paired]),
        difference=np.array([r['new_hit'] - r['old_hit'] for r in paired]), shape=np.array([r['shape'] for r in paired]))
    # Full counts are returned; no statistical draws here.
    inp = {'n_original_measurements': len(measures), 'stage2_stimulus_n': len(originals),
        'spot_stimulus_n': len([r for r in measures if r['folder'] == 'spot_probe_v1']),
        'baseline_n': 60, 'all_baselines_equal_across_three_input_sets': True,
        'response_cells': {'PatchCore': cells(rows), 'Spot': cells(sprows), 'G1': cells(grows)},
        'spot_high_pairs': spot_high, 'resolution_fixed_tau_transitions': resolution,
        'legacy_SEMU_bias_zero_injection': quantiles([r['zero_injection_global_SEMU_on_same_support'] for r in measures]),
        'legacy_vs_paired_SEMU_ratio': quantiles([r['global_SEMU_recomputed'] / r['paired_local_SEMU_proxy'] for r in measures if r['paired_local_SEMU_proxy']]),
        'alt_summary': {s: {'n': len([r for r in altmeasures if r['shape']==s]),
            'identical_support_n': sum(r['identical_support'] for r in altmeasures if r['shape']==s),
            'IoU': quantiles([r['support_IoU'] for r in altmeasures if r['shape']==s]),
            'actual_delta_ratio': quantiles([r['delta_ratio_hard_over_soft'] for r in altmeasures if r['shape']==s])} for s in SHAPES},
        'notes': ['global-background contrast contains preexisting spatial brightness, not only injected residual',
           'soft and hard effective-area thresholds differ in old code',
           'line hard generator uses vertical rectangle whereas original angle0 line is horizontal with round caps',
           'float32 stimulus input is not quantized before inference; 1/255 is effective-label/admission rule',
           'no full probability maps saved in spot/G1 caches; local response there cannot be reconstructed'],
        'elapsed_seconds': time.perf_counter()-t0}
    save(OUT / 'stimulus_measurements.json', {'rows': measures, 'alt_pairs': altmeasures})
    save(OUT / 'input_audit.json', inp)
    return seal('inputs', [OUT / f for f in ['stimulus_measurements.json', 'input_audit.json', 'paired_edge_rows.npz']],
                {'original_stimuli': len(measures), 'alt_pairs': len(altmeasures), 'scientific_bridge_valid': False})


def fit_probit(x, y, w, start=None):
    np, _, optimize, special = science()
    def fun(beta):
        eta = beta[0] + beta[1] * x
        lp = special.log_ndtr(eta); lq = special.log_ndtr(-eta)
        logpdf = -.5 * eta ** 2 - .5 * math.log(2 * math.pi)
        score = y * np.exp(logpdf - lp) - (1-y) * np.exp(logpdf - lq)
        return -float(np.sum(w*(y*lp+(1-y)*lq))), -np.array([np.sum(w*score), np.sum(w*score*x)])
    z = optimize.minimize(fun, np.array(start if start is not None else [-2.,1.]), jac=True,
                          method='BFGS', options={'gtol':1e-6, 'maxiter':200})
    grad = float(np.max(np.abs(z.jac)))
    ok = bool(np.isfinite(z.x).all() and math.isfinite(z.fun) and (z.success or grad < 1e-5) and z.x[1] > 0)
    logs = [(p-z.x[0])/z.x[1] for p in [0., 1.2815515655446004]] if z.x[1]>0 else [math.inf,math.inf]
    levels = [math.exp(l) if l<700 else math.inf for l in logs]
    ok = ok and all(math.isfinite(v) for v in levels)
    return z.x, levels, ok, {'optimizer_success':bool(z.success), 'status':int(z.status), 'gradient_max':grad}


def statistics():
    c=cached('statistics')
    if c: return c['summary']
    require(cached('inputs') is not None,'Input audit incomplete'); begin('statistics')
    np, *_=science(); t0=time.perf_counter()
    nboot=2000; seed=20261028; result={}; arrays={}
    idx=read(OLD/'anolcd_stage2_v1/stimuli/index.json')
    legacy=read(OLD/'capability_v1/pod_patchcore_256.json')
    for name, path in [('PatchCore',OLD/'anolcd_stage2_v1/scores/scores.json'),
                       ('G1',OLD/'capability_v1/g01_G1/scores.json')]:
        _,rows=checked_rows(path,idx)
        for shape in SHAPES:
            rs=[r for r in rows if r['kind']=='stimulus' and r['shape']==shape]
            x=np.log([r['SEMU_actual'] for r in rs]); y=np.array([float(r['hit']) for r in rs])
            bg=np.array([r['background_index'] for r in rs]); groups=sorted(set(bg))
            beta,lv,ok,diag=fit_probit(x,y,np.ones(len(y)))
            altbeta,altlv,altok,altdiag=fit_probit(x,y,np.ones(len(y)),start=[0.,.5])
            require(ok and altok and np.max(np.abs(beta-altbeta))<1e-4,'Full-data numerical fit disagreement')
            draws=np.random.Generator(np.random.PCG64(seed)).integers(0,len(groups),size=(nboot,len(groups)))
            b=np.full((nboot,2),np.nan); slopes=np.full(nboot,np.nan); statuses=np.zeros(nboot,np.int8)
            rawstatus=np.zeros(nboot,np.int16); grad=np.full(nboot,np.nan)
            groupidx=np.searchsorted(np.array(groups),bg)
            for k,pick in enumerate(draws):
                w=np.bincount(pick,minlength=len(groups))[groupidx]
                be,v,valid,d=fit_probit(x,y,w,start=beta)
                slopes[k]=be[1]; rawstatus[k]=d['status']; grad[k]=d['gradient_max']
                statuses[k]=int(valid)
                if valid:b[k]=v
            kept=int(statuses.sum()); failures=nboot-kept
            rates=[sum(r['hit'] for r in rs if r['target_SEMU']==t)/sum(r['target_SEMU']==t for r in rs) for t in LEVELS]
            validb=b[statuses.astype(bool)]
            report={'n':len(rs),'background_n':len(groups),'beta':beta.tolist(),'fit':diag,
                    'second_start_beta':altbeta.tolist(),'a50_point':lv[0],'a90_point':lv[1],
                    'observed_actual_SEMU_range':[float(np.exp(x.min())),float(np.exp(x.max()))],
                    'cell_hit_rates':rates,'observed_a50_crossing':min(rates)<=.5<max(rates),
                    'observed_a90_crossing':min(rates)<.9<=max(rates),'bootstrap_requested':nboot,
                    'bootstrap_valid':kept,'bootstrap_failures':failures,'seed':seed,
                    'a90_percentiles_conditional_on_valid': {str(q):float(np.percentile(validb[:,1],q)) for q in [2.5,5,50,95,97.5]} if kept else None,
                    'a50_ci95_conditional_on_valid': np.percentile(validb[:,0],[2.5,97.5]).tolist() if kept else None,
                    'exploratory_a90_upper95':float(np.percentile(validb[:,1],95)) if kept>=.95*nboot and min(rates)<.9<=max(rates) else None,
                    'scope':'original proxy axis; percentile bootstrap only, no certified coverage or real-defect bridge'}
            if name=='PatchCore':
                old=legacy['shapes'][shape]
                report['legacy_a90_95_p5']=old['a90_95'];report['legacy_a90_point']=old['a90']
                report['point_difference_from_legacy']=lv[1]-old['a90']
            key=name+'__'+shape;result[key]=report
            arrays[key+'__levels']=b;arrays[key+'__valid']=statuses;arrays[key+'__slope']=slopes
            arrays[key+'__optimizer_status']=rawstatus; arrays[key+'__gradient']=grad
            print('BOOTSTRAP_DONE',key,{'valid':kept,'upper95':report['exploratory_a90_upper95']},flush=True)
    sp=read(OUT/'input_audit.json')['response_cells']['Spot']['spot_xs']['cells']
    spot=[{'target':r['target'],'n':r['n'],'hits':r['hit'],'wilson95':wilson(r['hit'],r['n'])} for r in sp]
    with np.load(OUT/'paired_edge_rows.npz',allow_pickle=False) as z:
        bgs=z['background'];diff=z['difference']; shapes=z['shape']
    edge={}
    for s in ['all']+SHAPES:
        select=np.ones(len(diff),bool) if s=='all' else shapes==s
        d=diff[select]; bg=bgs[select]; gs=sorted(set(bg))
        sums=np.array([d[bg==g].sum() for g in gs]);counts=np.array([(bg==g).sum() for g in gs])
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,37])))
        draws=rng.integers(0,len(gs),size=(nboot,len(gs)))
        vals=sums[draws].sum(1)/counts[draws].sum(1)
        edge[s]={'n_pairs':len(d),'background_n':len(gs),'difference':float(d.mean()),
                 'cluster_bootstrap_ci95':np.percentile(vals,[2.5,97.5]).tolist(),
                 'interpretation':'paired descriptive sensitivity; support/contrast mismatch prevents edge-only causal interpretation'}
        arrays['edge__'+s]=vals
    np.savez_compressed(OUT/'bootstrap_replicates.npz',**arrays)
    stat={'fits':result,'spot_cells':spot,'spot_a50':'not identified; no numerical lower bound supported',
          'edge':edge,'elapsed_seconds':time.perf_counter()-t0,
          'conclusion':'confidence direction corrected as exploratory sensitivity; stimulus-axis bridge remains invalid'}
    save(OUT/'statistics.json',stat)
    return seal('statistics',[OUT/'statistics.json',OUT/'bootstrap_replicates.npz'],
                {'fits':len(result),'bootstrap_per_fit':nboot,'elapsed_seconds':stat['elapsed_seconds']})


def verify():
    n=check_sources(); stages={}
    for s in ['prepare','inputs','statistics']:
        c=cached(s);require(c is not None,'Stage incomplete: '+s);stages[s]=c['summary']
    result={'source_files_unchanged':n,'implementation_unchanged':True,'stages':stages,
            'no_model_imports':not any(n.split('.')[0] in FORBIDDEN for n in sys.modules),
            'new_model_forward':0,'new_training_updates':0,'new_render_calls':0,
            'old_thresholds_modified':False,'original_RGB_GT_decoded':0,
            'scientific_bridge_accepted':False}
    if (OUT/'verification.json').exists():
        require(read(OUT/'verification.json')==result,'Verification identity differs')
        old=read(OUT/'complete.json')
        for p,h in old['outputs'].items():require(digest(p)==h,'Complete output changed')
    else:
        save(OUT/'verification.json',result)
        files=sorted(p for p in OUT.iterdir() if p.is_file())
        save(OUT/'complete.json',{'outputs':{str(p):digest(p) for p in files},'scientific_bridge_accepted':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','inputs','statistics','verify'])
    a=p.parse_args();r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
