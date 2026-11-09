"""Frozen PatchCore D_hold scoring: stdlib-only preparation; explicit GPU run.

This fixed input set has ZERO content matches with prior normal/D_fit scores.
Only an intact own complete cache can be reused; partial transactions never
resume. Prediction consumes only the saved float RGB, never masks or outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

from . import patchcore_hold_inputs as h

ROOT, OLD, WEEK, TAU = h.ROOT, h.OLD, h.WEEK, h.TAU
INPUTS = WEEK / 'patchcore_hold_inputs_v1/run'
OUT = WEEK / 'patchcore_hold_scoring_v1'
PREP, RUN = OUT / 'preparation', OUT / 'run'
EVIDENCE = h.PROJECT / '报告/PatchCore_D_hold60_CPU输入取得与独立核验结果.md'
INPUT_COMPLETE = '5cbc1351afe334cefd23486aa35efd2143a9e38def506f385876c3e48d31e355'
HOLD_CODE = '81e191feda4268cab439c9ba86df7ea40c0afdec9e4cea923fa8de2667e35952'
COUNTERS = {'model_loads': 0, 'tensor_loads': 0, 'author_placeholder_forwards': 0,
            'model_input_forwards': 0}
EVENTS = {'source_RGB_reads': [], 'heatmap_reads': [], 'blocked': [],
          'NUL_opens': 0, 'temporary_write_opens': [], 'system_queries': []}
MODE = None
require, read, digest, write = h.require, h.read, h.digest, h.write


def source_context():
    require(digest(h.__file__) == HOLD_CODE, 'Frozen CPU input implementation differs')
    original = h.context()  # JSON/source/asset byte hashes only, no array API.
    files = dict(original['files'])
    def bind(p, expected=None):
        actual = digest(p); require(expected is None or actual == expected, 'Source hash differs: ' + str(p))
        files[str(Path(p))] = actual
    def metadata_cache(folder, expected=None):
        bind(folder / 'complete.json', expected)
        outputs = read(folder / 'complete.json')['outputs']; h.inventory(folder, outputs)
        for n, value in outputs.items():
            require(not Path(n).is_absolute() and '..' not in Path(n).parts, 'Nonlocal cache member')
            if n.endswith('.json'): bind(folder / n, value)
        return outputs
    require(not OUT.exists() or (OUT.is_dir() and not OUT.is_symlink() and not OUT.is_junction()
            and OUT.absolute() == OUT.resolve() and {p.name for p in OUT.iterdir()} <= {'preparation', 'run'}),
            'Unknown/redirected independent output root')
    metadata_cache(INPUTS, INPUT_COMPLETE)
    metadata_cache(INPUTS.parent / 'preparation', '4056569613a57ffb2308c545cdc45dd3193cecf6eff0eb0e78a0838a8e0c6d8c')
    bind(EVIDENCE); bind(Path(__file__).resolve())
    match = re.search(r'```json\s*\n(.*?)\n```', EVIDENCE.read_text(encoding='utf-8'), re.S)
    require(match is not None, 'Independent input verification evidence missing')
    proof = json.loads(match.group(1))
    require(proof['status'] == 'HOLD_INPUTS_COMPLETE_CACHE_VERIFIED'
            and proof['complete_sha256'] == INPUT_COMPLETE
            and proof['boundary']['counters']['original_decodes'] == 0
            and proof['boundary']['counters']['formal_render_calls'] == 0,
            'Independent full input verification identity differs')
    audit = read(INPUTS / 'source_audit.json'); manifest = read(INPUTS / 'manifest.json')
    rows = manifest['rows']; unique = read(INPUTS / 'unique_inputs.json')['rows']
    require(len(audit['backgrounds']) == 60 and len(rows) == 7560 and len(unique) == 7116
            and [r['id'] for r in audit['backgrounds']] == [r['id'] for r in original['backgrounds']],
            'Holdout denominators differ')
    refs = [{'id': r['id'], 'kind': 'baseline', 'background_id': r['id'], **r['image']}
            for r in audit['backgrounds']]
    for row, frozen in zip(rows, original['requests']):
        require(all(row[k] == v for k, v in frozen.items()) and row['retained'], 'Request identity/order differs')
        refs.append({'id': row['id'], 'kind': 'request', 'background_id': row['background_id'],
                     'geometry_id': row['geometry_id'], 'geometry_file': row['geometry_file'],
                     'zero_actual_change': row['measurement']['zero_actual_change'],
                     'measurement_defined': row['measurement']['defined'], **row['image']})
    ref_ids = defaultdict(list)
    for row in refs: ref_ids[row['working_RGB_sha256']].append(row['id'])
    input_outputs = read(INPUTS / 'complete.json')['outputs']
    planned = []
    for i, row in enumerate(unique):
        rgb_hash = row['working_RGB_sha256']
        require(row['file'] == f'images/{rgb_hash}.npy' and row['references'] == ref_ids[rgb_hash],
                'Unique input path/reference map differs')
        h.safe(INPUTS / row['file'])  # stat only, no RGB bytes.
        planned.append({**row, 'ordinal': i, 'file_sha256': input_outputs[row['file']]})
    require(len(refs) == len({r['id'] for r in refs}) == 7620
            and set(ref_ids) == {r['working_RGB_sha256'] for r in unique}, 'Full reference closure differs')
    cv = read(INPUTS / 'coverage.json'); verified = read(INPUTS / 'verification.json')
    require(verified['passed'] and len(verified['semantics']['by_geometry']) == 9
            and all(verified['semantics'].get(k) == v for k, v in proof['summary'].items())
            and manifest['summary'] == {k: v for k, v in verified['semantics'].items()
                if k not in ('geometry_arrays_verified', 'request_residuals_verified',
                             'unclipped_energy_max_relative_error', 'no_original_redecode_or_render_during_verification')}
            and cv['defined_backgrounds_at_least1'] == 60 and not cv['working_collisions']
            and cv['formal_render_calls_before_gate'] == 0, 'Input saved verification/coverage differs')
    fit = WEEK / 'patchcore_fit_scoring_v1'; normal = OLD / 'normal_scoring_v1/PatchCore'
    metadata_cache(fit / 'preparation'); metadata_cache(fit / 'run', h.FIXED[fit / 'run/complete.json'])
    metadata_cache(normal, h.FIXED[normal / 'complete.json'])
    fit_state = read(fit / 'run/verification.json'); normal_state = read(normal / 'verification.json')
    state = normal_state['state_before']
    require(fit_state['passed'] and normal_state['passed']
            and state == normal_state['state_after'] == fit_state['state_before'] == fit_state['state_after']
            and not state['PatchCore']['training_modules'] and not state['PatchCore']['gradients_present'],
            'Parameters/BN/eval identity differs between reference scores')
    ncfg_path = OLD / 'normal_preparation_v1/protocol.json'
    bind(ncfg_path, read(normal / 'q_result.json')['identity']['protocol_sha256'])
    cfg = read(ncfg_path)
    require(cfg['precision'] == 'fp32' and cfg['batch_size'] == 1 and cfg['workers'] == 0
            and cfg['eval'] and not cfg['TF32'], 'Frozen model runtime precision differs')
    for p, value in cfg['implementation'].items(): bind(p, value)
    for p, value in cfg['sources']['author_and_evidence'].items():
        if 'vendor\\patchcore-inspection\\' in p: bind(p, value)
    for p, value in cfg['sources']['assets'].items():
        if 'formal_draem' not in p: bind(p, value)
    runtime = cfg['sources']['runtime']
    require(sys.version.split()[0] == runtime['python'] and sys.executable == runtime['executable']
            and digest(sys.executable) == runtime['executable_sha256']
            and {p: importlib.metadata.version(p) for p in runtime['packages_metadata_only']}
                == runtime['packages_metadata_only'], 'GPU package/interpreter metadata differs')
    prior_fit = read(fit / 'preparation/unique_inputs.json')['rows']
    prior_normal = read(OLD / 'normal_inputs_v1/records.json')['rows']
    require(len(prior_fit) == 7305 and len(prior_normal) == 300, 'Prior input comparison budget differs')
    fit_hashes = {r['working_RGB_sha256'] for r in prior_fit}
    normal_hashes = {r['working_RGB_sha256'] for r in prior_normal}
    current_hashes = set(ref_ids)
    overlaps = {'D_fit': sorted(current_hashes & fit_hashes), 'normal': sorted(current_hashes & normal_hashes)}
    # Fixed scientific inputs are known disjoint; a changed pool is a source error,
    # not permission to invent a cross-experiment reuse/load path.
    require(not any(overlaps.values()), 'Unexpected content overlap; preserve and review provenance')
    require(read(fit / 'run/result.json')['tau'] == original['tau'] == TAU, 'Original tau differs')
    return {'files': files, 'normal_state': state, 'runtime_metadata': runtime,
            'frozen_model_protocol': cfg['models']['PatchCore'], 'tau': TAU,
            'hold_complete_sha256': INPUT_COMPLETE, 'fit_complete_sha256': original['fit_complete_sha256'],
            'frozen_p0': original['p0'], 'models_and_transforms': original['models'],
            'overlaps': overlaps, 'previous_input_comparison': {'normal': 300, 'D_fit': 7305},
            'denominators': {'backgrounds': 60, 'baseline_references': 60, 'request_references': 7560,
                'all_references': 7620, 'unique_float_RGB': 7116, 'reused_prior_outputs': 0,
                'new_model_input_forwards': 7116, 'new_author_ones_probes': 1,
                'zero_actual': 504, 'nonzero_actual': 7056, 'weak_nonzero_empty_M_eff': 2016,
                'undefined': 0}}, planned, refs


def eligible_reuse(a, b):
    keys = ('working_RGB_sha256', 'input_file_sha256', 'model_state', 'preprocessing', 'score_definition', 'tau')
    return all(k in a and k in b and a[k] == b[k] for k in keys)


def scalar_audit():
    a = dict(working_RGB_sha256='artificial', input_file_sha256='bytes', model_state='BN+parameters',
             preprocessing='frozen', score_definition='max_patch/native_heat', tau=TAU)
    checks = {'same_identity_eligible': eligible_reuse(a, dict(a))}
    for k in a:
        b = dict(a); b[k] = 'different'
        checks['reject_' + k] = not eligible_reuse(a, b)
    checks['strict_equal_not_alarm'] = not (TAU > TAU)
    checks['strict_above_alarm'] = math.nextafter(TAU, math.inf) > TAU
    checks['references_not_forwards'] = 60 + 7560 == 7620 and 7620 - 504 == 7116
    require(all(checks.values()), 'Artificial identity/threshold/count scalar audit failed')
    return {'passed': True, 'checks': checks, 'actual_array_reads': 0, 'model_forwards': 0}


def protocol(sources):
    return {'model': 'PatchCore', 'tau': TAU, 'rule': 'strictly greater',
            'denominators': sources['denominators'], 'eval': True, 'precision': 'FP32',
            'batch_size': 1, 'workers': 0, 'seed_for_author_construction': 42,
            'input': 'saved HWC float32 RGB[0,1]; no PNG/no rescaling/no extra division',
            'runtime': 'frozen inference_runtime.build_model/predict_rgb; original explicit WRN/V1/N80',
            'output': 'native max patch score and original float32 author heatmap, not heatmap max score',
            'state_check': 'full parameter/BN/buffer/training/gradient signature after every input',
            'reuse_prior': 'same content bytes, parameters+BN, preprocessing, score and tau required; fixed matches0',
            'complete_cache': 'strict full identity/hash/semantic read-only reuse; unknown/partial stop, no retry/resume',
            'prediction_consumes': 'RGB only; masks and frozen fits never passed to predictor',
            'startup_RAM_GiB': 4.5, 'runtime_RAM_GiB': 2, 'initial_free_GPU_GiB': 4, 'free_D_GiB': 40,
            'ones_probe_is_not_a_real_background': True, 'CPU_response_statistics': False}


PREP_FILES = {'sources.json', 'protocol.json', 'unique_inputs.json', 'references.json', 'scalar_verification.json'}


def preparation():
    sources, unique, refs = source_context(); resources = h.resources('hold_scoring_preparation')
    if PREP.exists(): return check_preparation()
    require(not OUT.exists(), 'Unknown output directory; do not adopt it')
    audit = scalar_audit(); compile(Path(__file__).read_text(encoding='utf-8'), __file__, 'exec')
    OUT.mkdir(exist_ok=False); PREP.mkdir(exist_ok=False)
    write(PREP / 'sources.json', sources); write(PREP / 'protocol.json', protocol(sources))
    write(PREP / 'unique_inputs.json', {'rows': unique}); write(PREP / 'references.json', {'rows': refs})
    write(PREP / 'scalar_verification.json', {'passed': True, 'audit': audit, 'resources': resources,
                                             'boundary': boundary()})
    h.seal(PREP)
    return {'status': 'HOLD_SCORING_PREPARATION_COMPLETE', 'denominators': sources['denominators'],
            'complete_sha256': digest(PREP / 'complete.json'), 'resources': resources, 'boundary': boundary()}


def check_preparation(return_data=False):
    sources, unique, refs = source_context()
    c = read(PREP / 'complete.json'); require(set(c['outputs']) == PREP_FILES, 'Preparation whitelist differs')
    h.inventory(PREP, c['outputs'])
    require(all(digest(PREP / n) == value for n, value in c['outputs'].items()), 'Preparation hash differs')
    require(read(PREP / 'sources.json') == sources and read(PREP / 'protocol.json') == protocol(sources)
            and read(PREP / 'unique_inputs.json')['rows'] == unique
            and read(PREP / 'references.json')['rows'] == refs, 'Preparation binding/config differs')
    v = read(PREP / 'scalar_verification.json')
    require(v['passed'] and v['audit'] == scalar_audit() and not any(v['boundary']['counters'].values())
            and not v['boundary']['events']['blocked'], 'Scalar/static boundary evidence differs')
    if return_data: return sources, unique, refs
    return {'status': 'HOLD_SCORING_PREPARATION_CACHE_VERIFIED', 'denominators': sources['denominators'],
            'complete_sha256': digest(PREP / 'complete.json'), 'boundary': boundary()}


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        banned = ('torch', 'torchvision', 'timm', 'faiss', 'patchcore', 'mura_response.inference_runtime')
        if MODE != 'run' and any(fullname == p or fullname.startswith(p + '.') for p in banned):
            EVENTS['blocked'].append('import:' + fullname); require(False, 'Model import outside run')
        if MODE in ('prepare', 'verify-preparation') and fullname.split('.')[0] in ('numpy', 'scipy', 'PIL'):
            EVENTS['blocked'].append('import:' + fullname); require(False, 'Array/decoder package outside scalar preparation')


def boundary():
    require(not EVENTS['blocked'], 'Forbidden operation attempted')
    if MODE != 'run':
        require('torch' not in sys.modules and not any(COUNTERS.values()), 'CPU command invoked model')
    return {'counters': dict(COUNTERS), 'events': dict(EVENTS), 'GT_original_reads': 0,
            'generation_training_calibration_fitting': 0}


def install_boundary():
    allowed_inputs = {str((INPUTS / r['file']).resolve()) for r in read(PREP / 'unique_inputs.json')['rows']} \
        if MODE == 'run' and PREP.is_dir() else set()
    assets = {str(p.resolve()) for p in (ROOT / 'models/checkpoints/wide_resnet50_2-95faca4d.pth',
        ROOT / 'models/weak_localization/patchcore/nnscorer_search_index.faiss',
        ROOT / 'models/weak_localization/patchcore/patchcore_params.pkl')}
    temp_roots = {Path(v).resolve() for k in ('TEMP', 'TMP') if (v := os.environ.get(k))}
    def writable(p): return p.is_relative_to(OUT) or any(p.is_relative_to(t) for t in temp_roots)
    def deny(reason): EVENTS['blocked'].append(reason); require(False, reason)
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'pickle.find_class', 'os.system'): deny(event)
        if event == 'subprocess.Popen':
            f = sys._getframe(1); okay = False
            while f:
                if f.f_code.co_name == '_syscmd_ver' and Path(f.f_code.co_filename).resolve() == (
                        Path(sys.base_prefix) / 'Lib/platform.py').resolve():
                    okay = f.f_locals.get('cmd') in ('ver', 'command /c ver', 'cmd /c ver'); break
                f = f.f_back
            if not okay: deny('Unlisted subprocess:' + str(args[:2]))
            EVENTS['system_queries'].append(str(args[:2]))
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'):
            for v in (args[:2] if event == 'os.rename' else args[:1]):
                if isinstance(v, (str, bytes)) and not writable(Path(os.fsdecode(v)).resolve()): deny('mutation:' + str(v))
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)): return
        raw = os.fsdecode(args[0])
        if raw.upper() in ('NUL', '\\\\.\\NUL', '\\\\?\\NUL'): EVENTS['NUL_opens'] += 1; return
        p = Path(raw).resolve(); ext = p.suffix.lower()
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if not writable(p): deny('write:' + str(p))
            if any(p.is_relative_to(t) for t in temp_roots): EVENTS['temporary_write_opens'].append(str(p))
            return
        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'): deny('original/GT:' + str(p))
        if ext in ('.npy', '.npz'):
            if MODE == 'run' and str(p) in allowed_inputs: EVENTS['source_RGB_reads'].append(str(p))
            elif MODE in ('run', 'verify') and p.is_relative_to(RUN): EVENTS['heatmap_reads'].append(str(p))
            else: deny('array/heatmap:' + str(p))
        if MODE == 'run' and ext in ('.pth', '.pt', '.pkl', '.pickle', '.faiss', '.safetensors') and str(p) not in assets:
            deny('unlisted asset:' + str(p))
    sys.meta_path.insert(0, ImportGuard()); sys.addaudithook(audit)


def rgb_identity(rgb):
    header = f'{rgb.dtype.str}|{",".join(map(str, rgb.shape))}|C\n'.encode('ascii')
    return hashlib.sha256(header + rgb.tobytes(order='C')).hexdigest()


def valid_heat(score, heat, np):
    require(math.isfinite(score) and heat.dtype == np.float32 and heat.shape == (256, 256)
            and np.isfinite(heat).all(), 'Native score/heatmap finite/shape check failed')


def run_names(unique):
    return {'started.json', 'references.json', 'result.json', 'verification.json'} | {
        f'records/{r["working_RGB_sha256"]}.json' for r in unique} | {
        f'heatmaps/{r["working_RGB_sha256"]}.npy' for r in unique}


def run():
    sources, unique, refs = check_preparation(return_data=True)
    if RUN.exists(): return verify()
    snapshot = h.resources('before_hold_model'); require(snapshot['available_RAM_GiB'] >= 4.5, 'Need4.5GiB RAM')
    require(not any(COUNTERS.values()), 'Fresh model process required')
    RUN.mkdir(exist_ok=False); (RUN / 'records').mkdir(); (RUN / 'heatmaps').mkdir()
    write(RUN / 'started.json', {'preparation_sha256': digest(PREP / 'complete.json'), 'pid': os.getpid(),
        'resources': snapshot, 'temporary_environment': {k: os.environ.get(k) for k in ('TEMP', 'TMP', 'TMPDIR')}})
    write(RUN / 'references.json', {'rows': refs}); started = time.perf_counter(); completed = 0
    current = 'model_construction'; timings = []; samples = [snapshot]
    try:
        write(RUN / 'inflight.json', {'operation': current})
        import numpy as np
        from mura_response import inference_runtime as native
        torch, modules, handle = native.build_model(SimpleNamespace(ROOT=ROOT, COUNTERS=COUNTERS, require=require), 'PatchCore', {})
        def state(): return json.loads(json.dumps(native.state_signature(torch, modules), allow_nan=False))
        initial = state(); require(initial == sources['normal_state'], 'Loaded parameter/BN state differs')
        (RUN / 'inflight.json').unlink(); constructed = time.perf_counter()
        for row in unique:
            current = row['working_RGB_sha256']; samples.append(h.resources('hold_' + str(row['ordinal'])))
            payload = (INPUTS / row['file']).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == row['file_sha256'], 'Frozen input bytes differ')
            rgb = np.load(io.BytesIO(payload), allow_pickle=False)
            require(rgb.dtype == np.float32 and rgb.shape == (256, 256, 3) and rgb.flags.c_contiguous
                    and np.isfinite(rgb).all() and 0 <= float(rgb.min()) <= float(rgb.max()) <= 1
                    and rgb_identity(rgb) == current, 'Float RGB content/definition differs')
            write(RUN / 'inflight.json', {'operation': 'unique_RGB_forward', 'ordinal': row['ordinal'], 'working_RGB_sha256': current})
            t = time.perf_counter(); COUNTERS['model_input_forwards'] += 1
            score, heat = native.predict_rgb(torch, 'PatchCore', handle, rgb); torch.cuda.synchronize()
            valid_heat(score, heat, np); require(state() == initial, 'Parameter/BN/training/gradient state changed')
            target = RUN / f'heatmaps/{current}.npy'
            with target.open('xb') as stream: np.save(stream, heat, allow_pickle=False)
            write(RUN / f'records/{current}.json', {'ordinal': row['ordinal'], 'working_RGB_sha256': current,
                'input_file_sha256': row['file_sha256'], 'score': score, 'tau': TAU, 'state_unchanged': True,
                'heatmap_path': str(target), 'heatmap_sha256': digest(target),
                'heat_min': float(heat.min()), 'heat_max': float(heat.max()), 'source': 'new_forward'})
            (RUN / 'inflight.json').unlink(); completed += 1; timings.append(time.perf_counter() - t)
            if completed == 1 or completed % 25 == 0 or completed == len(unique):
                print(json.dumps({'unique_completed': completed, 'unique_total': len(unique),
                    'new_input_forwards': COUNTERS['model_input_forwards'], 'elapsed_seconds': time.perf_counter() - started}), flush=True)
        require(COUNTERS == {'model_loads': 1, 'tensor_loads': 1, 'author_placeholder_forwards': 1,
                             'model_input_forwards': 7116} and completed == 7116, 'Actual call counts differ')
        require(EVENTS['source_RGB_reads'] == [str((INPUTS / r['file']).resolve()) for r in unique]
                and not EVENTS['blocked'], 'Input read order/boundary differs')
        write(RUN / 'result.json', {'status': 'HOLD_SCORING_COMPLETE', 'denominators': sources['denominators'],
            'counters': COUNTERS, 'tau': TAU, 'elapsed_seconds': time.perf_counter() - started,
            'construction_seconds': constructed - started, 'input_scoring_state_IO_seconds': sum(timings),
            'CPU_response_statistics_performed': False})
        write(RUN / 'verification.json', {'passed': True, 'state_before': initial, 'state_after': state(),
            'events': EVENTS, 'resources': samples, 'GPU': torch.cuda.get_device_name(),
            'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
            'GPU_peak_scope': 'after CUDA admission, before construction through all inputs',
            'original_RGB_GT_reads': 0})
        require(source_context()[0] == sources, 'Sources changed during model transaction')
        require({p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()} == run_names(unique),
                'Pre-seal run whitelist differs')
        h.seal(RUN)
    except BaseException as exc:
        if not (RUN / 'failure.json').exists():
            write(RUN / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                'current': current, 'completed_unique': completed, 'counters': dict(COUNTERS),
                'events': dict(EVENTS), 'no_automatic_retry_resume': True})
        raise
    return {'status': 'HOLD_SCORING_COMPLETE', 'output': str(RUN), 'counters': COUNTERS}


def verify():
    sources, unique, refs = check_preparation(return_data=True)
    require((RUN / 'complete.json').is_file(), 'Missing complete; preserve partial transaction')
    c = read(RUN / 'complete.json'); require(set(c['outputs']) == run_names(unique), 'Run output whitelist differs')
    h.inventory(RUN, c['outputs'])
    require(all(digest(RUN / n) == value for n, value in c['outputs'].items()), 'Run output hash differs')
    import numpy as np
    result = read(RUN / 'result.json'); v = read(RUN / 'verification.json')
    require(result['status'] == 'HOLD_SCORING_COMPLETE' and result['denominators'] == sources['denominators']
            and result['tau'] == TAU and result['counters'] == {'model_loads': 1, 'tensor_loads': 1,
                'author_placeholder_forwards': 1, 'model_input_forwards': 7116}
            and not result['CPU_response_statistics_performed'] and v['passed']
            and v['state_before'] == v['state_after'] == sources['normal_state']
            and not v['events']['blocked'] and v['original_RGB_GT_reads'] == 0,
            'Saved result/state/counters differ')
    require(v['events']['source_RGB_reads'] == [str((INPUTS / r['file']).resolve()) for r in unique]
            and read(RUN / 'references.json')['rows'] == refs
            and read(RUN / 'started.json')['preparation_sha256'] == digest(PREP / 'complete.json'),
            'Saved source/ref/preparation binding differs')
    for row in unique:
        rgb_hash = row['working_RGB_sha256']; rec = read(RUN / f'records/{rgb_hash}.json')
        target = RUN / f'heatmaps/{rgb_hash}.npy'
        require(rec['ordinal'] == row['ordinal'] and rec['working_RGB_sha256'] == rgb_hash
                and rec['input_file_sha256'] == row['file_sha256'] and rec['tau'] == TAU
                and rec['state_unchanged'] and rec['source'] == 'new_forward'
                and rec['heatmap_path'] == str(target) and digest(target) == rec['heatmap_sha256'],
                'Saved native output identity differs')
        heat = np.load(target, allow_pickle=False); valid_heat(rec['score'], heat, np)
        require(rec['heat_min'] == float(heat.min()) and rec['heat_max'] == float(heat.max()), 'Heat range differs')
    boundary()
    return {'status': 'HOLD_SCORING_COMPLETE_CACHE_VERIFIED', 'complete_sha256': digest(RUN / 'complete.json'),
            'denominators': sources['denominators'], 'new_model_input_forwards': 0,
            'new_source_RGB_reads': 0, 'tau_recomputed': False, 'CPU_response_statistics_performed': False}


def main():
    global MODE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify-preparation', 'run', 'verify'))
    args = p.parse_args(); MODE = args.command; install_boundary()
    result = {'prepare': preparation, 'verify-preparation': check_preparation, 'run': run, 'verify': verify}[MODE]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
