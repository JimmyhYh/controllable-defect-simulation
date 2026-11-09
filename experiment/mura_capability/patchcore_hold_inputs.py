"""Independent stage-6 CPU holdout preparation, acquisition and verification.

prepare/verify-preparation use metadata and artificial arrays only. acquire is
an explicit, separate authorization boundary; it never invokes a detector or
fits statistics. Complete same-identity caches are read-only, never repaired.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent.parent
OLD = ROOT / 'results/mura_response/semu_applicability_v1'
WEEK = ROOT / 'results/mura_capability/week1_v1'
FIT = WEEK / 'patchcore_fit_analysis_v1'
OUT = WEEK / 'patchcore_hold_inputs_v1'
PREP, RUN = OUT / 'preparation', OUT / 'run'
DESIGN = PROJECT / '计划/Mura检测器响应与SEMU代理刻度研究设计.md'
FIT_REPORT = PROJECT / '报告/PatchCore_D_fit统计拟合准备与人工运行说明.md'
FIT_COMPLETE = '983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649'
FIXED = {
    OLD / 'preparation_v1/complete.json': '0fd936ea58f8c8a7d16c209c96d116daa0f34c6044d1b809781bd4b01f7c7ba5',
    OLD / 'fit_inputs_identity_v2/complete.json': '3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593',
    OLD / 'normal_inputs_v1/complete.json': 'bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da',
    OLD / 'normal_scoring_v1/PatchCore/complete.json': '2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec',
    FIT / 'preparation/complete.json': '00a2997392b9c995781b5c0160596644d67f975b71281644cf7abb0ba92b02c2',
    FIT / 'run/complete.json': FIT_COMPLETE,
    WEEK / 'patchcore_fit_scoring_v1/run/complete.json': '501b547812f6cb941e8f4f8ed7f92046c8451e330d5698302fb77d7578428f7e',
    WEEK / 'patchcore_fit_response_v1/run/complete.json': '69f7c691b241a4175f4a8c1f2442fddc1ea6de978cc44572ee278f73d652adae',
}
TAU = 1.852951288223265
MODEL_IDS = ['C-only', 'C+area', 'SEMU-only_k0.4', 'C+area+context',
             'SEMU-only_k0.2', 'SEMU-only_k0.8']
FORBIDDEN = ('torch', 'torchvision', 'timm', 'faiss', 'patchcore',
    'controlled_generation', 'mura_response.resources', 'mura_response.generation',
    'mura_response.generation_v2', 'mura_response.inference', 'mura_response.fit_inference',
    'mura_capability.patchcore_fit_analysis', 'scipy.optimize')
MODE = None
COUNTERS = Counter(original_file_opens=0, original_decodes=0, formal_render_calls=0,
                   saved_array_reads=0, model_forwards=0, statistical_fits=0)
ALLOWED_ORIGINALS = set()
OPENS = Counter()
BLOCKED = []
SAMPLES = []


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no retry, replacement, or resume')


def safe(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and not p.is_junction()
            and p.absolute() == p.resolve(), 'Missing or redirected file: ' + str(p))
    return p


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def digest(path):
    with safe(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + '\n')


def seal(folder):
    require(not any((folder / n).exists() for n in ('complete.json', 'failure.json', 'inflight.json')),
            'Cannot seal failed or unknown transaction')
    write(folder / 'complete.json', {'outputs': {
        p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}})


def inventory(folder, outputs):
    require(folder.is_dir() and not folder.is_symlink() and not folder.is_junction()
            and folder.absolute() == folder.resolve(), 'Redirected cache root')
    members = list(folder.rglob('*'))
    expected_dirs = {str(Path(n).parent).replace('\\', '/') for n in outputs if '/' in n}
    expected_dirs.discard('.')
    require(all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(folder)
                for p in members), 'Redirected cache member')
    require({p.relative_to(folder).as_posix() for p in members if p.is_file()}
            == set(outputs) | {'complete.json'}, 'Unknown/partial cache files')
    require({p.relative_to(folder).as_posix() for p in members if p.is_dir()} == expected_dirs,
            'Unknown cache directories')


def resources(stage):
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'page_available',
                                            'virtual', 'virtual_available', 'extended')]
    state = Memory(); state.length = ctypes.sizeof(state)
    q = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
    q.argtypes = [ctypes.POINTER(Memory)]; q.restype = ctypes.c_int
    require(q(ctypes.byref(state)), 'RAM query failed')
    row = {'stage': stage, 'available_RAM_GiB': state.available / 2**30,
           'D_free_GiB': shutil.disk_usage(ROOT).free / 2**30}
    SAMPLES.append(row)
    require(row['available_RAM_GiB'] >= 2 and row['D_free_GiB'] >= 40, 'CPU RAM2/disk40GiB admission')
    return row


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == x or fullname.startswith(x + '.') for x in FORBIDDEN):
            BLOCKED.append({'import': fullname}); require(False, 'Forbidden model/optimization/runtime import')


def install_boundary():
    temps = {Path(v).resolve() for k in ('TEMP', 'TMP') if (v := os.environ.get(k))}
    def mutation_allowed(p):
        return p.is_relative_to(OUT) or any(p.is_relative_to(t) for t in temps)
    def deny(data):
        BLOCKED.append(data); require(False, str(data))
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'pickle.find_class', 'os.system'):
            deny({'event': event})
        if event == 'subprocess.Popen':
            # Only standard-library platform version discovery, no task subprocess.
            f = sys._getframe(1); permitted = False
            while f:
                if f.f_code.co_name == '_syscmd_ver' and Path(f.f_code.co_filename).resolve() == (
                        Path(sys.base_prefix) / 'Lib/platform.py').resolve():
                    permitted = f.f_locals.get('cmd') in ('ver', 'command /c ver', 'cmd /c ver'); break
                f = f.f_back
            if not permitted: deny({'subprocess': str(args[:2])})
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'):
            for value in (args[:2] if event == 'os.rename' else args[:1]):
                if isinstance(value, (str, bytes)) and not mutation_allowed(Path(os.fsdecode(value)).resolve()):
                    deny({'mutation': str(value)})
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)): return
        raw = os.fsdecode(args[0])
        if raw.upper() in ('NUL', '\\\\.\\NUL', '\\\\?\\NUL'): return
        p = Path(raw).resolve(); ext = p.suffix.lower()
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if not mutation_allowed(p): deny({'write': str(p)})
            return
        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            if MODE != 'acquire' or p not in ALLOWED_ORIGINALS: deny({'original_read': str(p)})
            OPENS[str(p)] += 1; COUNTERS['original_file_opens'] += 1
            if OPENS[str(p)] != 1: deny({'repeat_original_open': str(p)})
        if ext in ('.npy', '.npz'):
            if MODE not in ('acquire', 'verify-inputs') or not p.is_relative_to(RUN):
                deny({'array_read': str(p)})
            COUNTERS['saved_array_reads'] += 1
    sys.meta_path.insert(0, ImportGuard()); sys.addaudithook(audit)


def boundary():
    loaded = [n for n in sys.modules if any(n == p or n.startswith(p + '.') for p in FORBIDDEN)]
    require(not loaded and not BLOCKED, 'CPU boundary crossed')
    return {'counters': dict(COUNTERS), 'blocked': list(BLOCKED), 'forbidden_modules_loaded': loaded,
            'old_training_Q_F_D_fit_original_reads': 0, 'GT_probability_reads': 0,
            'calibrations': 0, 'detector_parameter_updates': 0}


def context():
    """All old arrays/images are opaque: only JSON/source/asset byte hashes."""
    hashes = {}
    def bind(p, h=None):
        actual = digest(p); require(h is None or actual == h, 'Source hash differs: ' + str(p))
        hashes[str(Path(p))] = actual
    for complete, expected in FIXED.items():
        bind(complete, expected)
        outputs = read(complete)['outputs']; inventory(complete.parent, outputs)
        for name, h in outputs.items():
            require(Path(name).is_relative_to(Path('.')) and '..' not in Path(name).parts,
                    'Nonlocal source cache member')
            if name.endswith('.json'): bind(complete.parent / name, h)
    orig = OLD / 'preparation_v1'
    protocol = read(orig / 'protocol.json'); roles_doc = read(orig / 'roles.json')
    roles = roles_doc['roles']; requests = [r for r in read(orig / 'requests.json')['rows']
                                          if r['role'] == 'D_hold60']
    geo = read(orig / 'geometry.json')['rows']
    bind(DESIGN, protocol['design_sha256'])
    for p, h in protocol['implementation'].items(): bind(p, h)
    for p, h in read(FIT / 'preparation/sources.json')['files'].items(): bind(p, h)
    bind(Path(__file__).resolve()); bind(FIT_REPORT)
    proof_match = re.search(r'```json\s*\n(.*?)\n```', FIT_REPORT.read_text(encoding='utf-8'), re.S)
    require(proof_match is not None, 'Missing independent fit verification evidence')
    proof = json.loads(proof_match.group(1))
    require(proof['status'] == 'STATISTICAL_FIT_COMPLETE_CACHE_VERIFIED'
            and proof['complete_sha256'] == FIT_COMPLETE and proof['new_optimizer_calls'] == 0,
            'Independent fitting proof mismatch')
    models = read(FIT / 'run/models.json')['models']; p0 = read(FIT / 'run/constant_reference.json')['p0']
    require([m['spec']['id'] for m in models] == MODEL_IDS
            and all(m['optimizer_success'] for m in models) and 0 < p0 < 1,
            'Frozen six-model/p0 admission not met')
    require(read(FIT / 'run/verification.json')['status'] == 'ALL_DECLARED_D_FIT_MODELS_FROZEN',
            'Final statistical seal missing')
    q = read(OLD / 'normal_scoring_v1/PatchCore/q_result.json')
    require(q['threshold'] == TAU and q['q'] == .95 and q['method'] == 'linear'
            and q['rule'] == 'strictly greater', 'Frozen workpoint differs')
    scoring_src = read(WEEK / 'patchcore_fit_scoring_v1/preparation/sources.json')
    require(scoring_src['tau'] == TAU, 'D_fit score tau differs')
    for p, h in scoring_src['minimal_sources']['files'].items():
        if Path(p).suffix.lower() in ('.py', '.pth', '.pkl', '.faiss', '.exe'): bind(p, h)
    require(len(roles['D_hold60']) == 60 and len(requests) == 7560 and len(geo) == 9,
            'Holdout nominal budget differs')
    require([r['ordinal'] for r in roles['D_hold60']] == list(range(60))
            and len({r['id'] for r in requests}) == 7560, 'Holdout order or IDs differ')
    audit = read(OLD / 'fit_inputs_identity_v2/source_audit.json')
    normal = read(OLD / 'normal_inputs_v1/records.json')['rows']
    prior = ([{'role': 'old_training', **r} for r in audit['old_training']]
             + [{'role': 'D_fit60', **r} for r in audit['backgrounds']]
             + normal)
    require(len(audit['old_training']) == 160 and len(audit['backgrounds']) == 60
            and len(normal) == 300 and Counter(r['role'] for r in normal) == {'Q200': 200, 'F100': 100},
            'Working hash comparison pool budget differs')
    for role in ('D_fit60', 'Q200', 'F100'):
        actual = [r for r in prior if r['role'] == role]
        require([r['id'] for r in actual] == [r['id'] for r in roles[role]]
                and all(r['path'] == f['path'] and r['file_sha256'] == f['file_sha256']
                        for r, f in zip(actual, roles[role])), 'Prior role provenance differs')
    manifest_path = ROOT / 'results/weak_localization/dataset_manifest.json'
    old_source = read(orig / 'sources.json')['files']; bind(manifest_path, old_source[str(manifest_path)])
    manifest = {r['path']: r for r in read(manifest_path)['files'] if 'rgb_content_sha256' in r}
    splits = {}
    for name in ('weak_localization', 'controlled_generation'):
        p = ROOT / 'data' / name / 'split.json'; bind(p, old_source[str(p)]); splits[name] = read(p)
    training = sorted(set(splits['weak_localization']['N']) | set(splits['controlled_generation']['N_gen']))
    require([r['path'] for r in audit['old_training']] == training and len(training) == 160,
            'Old training comparison pool differs')
    for r in prior:
        source = manifest[r['path']]
        require(r['file_sha256'] == source['sha256'] and r['legacy_rgb_group_id'] == source['rgb_content_sha256']
                and len(r['working_RGB_sha256']) == 64, 'Prior work hash metadata differs')
    hold_paths = [r['path'] for r in roles['D_hold60']]
    require(len(set(hold_paths)) == 60 and not set(hold_paths) & {r['path'] for r in prior},
            'Holdout source path overlaps prior roles')
    for r in roles['D_hold60']:
        source = manifest[r['path']]
        require(r['path'].startswith('display/train/good/') and r['file_sha256'] == source['sha256']
                and r['rgb_sha256'] == source['rgb_content_sha256'] and r['bytes'] == source['bytes']
                and r['original_mode'] == source['mode'] and r['original_size_hw'] == [1024, 1024],
                'Frozen holdout manifest/role identity differs')
    runtime = {'python': sys.version.split()[0], 'executable': sys.executable,
               'executable_sha256': digest(sys.executable),
               'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'Pillow')}}
    frozen = protocol['runtime']
    require(runtime['python'] == frozen['python'] and runtime['executable'] == frozen['executable']
            and runtime['executable_sha256'] == frozen['executable_sha256']
            and all(runtime['packages'][p] == frozen['packages_metadata_only'][p] for p in runtime['packages']),
            'Frozen CPU interpreter/package metadata differs')
    return {'files': hashes, 'runtime': runtime, 'tau': TAU, 'fit_complete_sha256': FIT_COMPLETE,
            'p0': p0, 'models': models, 'dataset_root': roles_doc['dataset_root'],
            'backgrounds': roles['D_hold60'], 'requests': requests, 'geometry': geo,
            'prior_work_hashes': [{'role': r['role'], 'path': r['path'],
                                  'working_RGB_sha256': r['working_RGB_sha256']} for r in prior],
            'preprocessing': protocol['preprocessing_future']}


def arrays_api():
    import numpy as np
    from mura_response import inputs
    return np, inputs


def coverage(backgrounds, prior, geometry_ids):
    old = defaultdict(list); current = defaultdict(list); collisions = []
    for row in prior: old[row['working_RGB_sha256']].append({'role': row['role'], 'path': row['path']})
    for row in backgrounds:
        h = row['working_RGB_sha256']; current[h].append(row['id'])
        if h in old: collisions.append({'type': 'hold_vs_prior', 'id': row['id'], 'prior': old[h], 'hash': h})
    for h, ids in current.items():
        if len(ids) > 1: collisions.append({'type': 'within_hold', 'ids': ids, 'hash': h})
    counts = [sum(r['measurements'][g]['defined'] for g in geometry_ids) for r in backgrounds]
    return {'backgrounds': len(backgrounds), 'geometry_conditions': len(backgrounds) * len(geometry_ids),
            'defined_backgrounds_at_least1': sum(c > 0 for c in counts),
            'defined_all9_backgrounds': sum(c == 9 for c in counts),
            'defined_partial_backgrounds': sum(0 < c < 9 for c in counts),
            'defined_none_backgrounds': sum(c == 0 for c in counts),
            'per_geometry_defined_backgrounds': {g: sum(r['measurements'][g]['defined'] for r in backgrounds)
                                                 for g in geometry_ids},
            'working_collisions': collisions, 'prior_comparison_paths': len(prior),
            'unique_hold_backgrounds': len(current), 'minimum_defined_backgrounds': 40,
            'physical_capture_groups_known': False}


def admitted(c):
    return c['backgrounds'] == 60 and c['geometry_conditions'] == 540 and not c['working_collisions'] \
        and c['defined_backgrounds_at_least1'] >= 40


def artificial_audit(ctx):
    np, api = arrays_api()
    geometries = [api.make_geometry(s) for s in api.geometry_specs()]
    require([g['metadata'] for g in geometries] == ctx['geometry'], 'Frozen geometry numerical identity differs')
    roles = {'D_fit60': [{'id': 'dummy', 'ordinal': i} for i in range(60)],
             'D_hold60': ctx['backgrounds']}
    requests = [r for r in api.request_dictionary(roles) if r['role'] == 'D_hold60']
    require(requests == ctx['requests'], 'Holdout original request ordering differs')
    gids = [g['metadata']['spec']['id'] for g in geometries]
    def fake(n, duplicate=False, prior_collision=False):
        return [{'id': f'b{i}', 'working_RGB_sha256': ('prior' if prior_collision and i == 0 else
                 'dup' if duplicate and i < 2 else f'h{i}'),
                 'measurements': {g: {'defined': i < n} for g in gids}} for i in range(60)]
    prior = [{'role': 'old', 'path': 'artificial', 'working_RGB_sha256': 'prior'}]
    cases = {'39_stops': not admitted(coverage(fake(39), prior, gids)),
             '40_passes': admitted(coverage(fake(40), prior, gids)),
             'within_collision_stops': not admitted(coverage(fake(60, duplicate=True), prior, gids)),
             'prior_collision_stops': not admitted(coverage(fake(60, prior_collision=True), prior, gids))}
    require(all(cases.values()), 'Artificial coverage/collision gate failed')
    audit = api.numerical_audit()
    require(audit['passed'] and audit['real_RGB_decoded'] == audit['formal_stimuli_generated'] == 0,
            'Artificial-only numerical audit scope differs')
    audit.pop('rows')
    return {'coverage_gate_cases': cases, 'numerical': audit,
            'fixed_request_dictionary_matches': True, 'hold_images_or_arrays_read': 0,
            'model_compatibility_or_runtime_verified': False}


def protocol(ctx):
    return {'schema': 'patchcore-hold-inputs-v1', 'role': 'D_hold60',
            'fit_complete_sha256': FIT_COMPLETE, 'tau': TAU, 'frozen_p0': ctx['p0'],
            'frozen_model_ids': MODEL_IDS, 'preprocessing': ctx['preprocessing'],
            'work_identity': 'SHA256(dtype.str|comma-shape|C newline + HWC float32 C bytes)',
            'raw_identity': 'SHA256(RGB1024x1024x3 uint8 HWC C bytes; no header)',
            'legacy_RGB_group': 'opaque role/exclusion ID; not equated to new content digests',
            'float_input': 'RGB float32 PIL bilinear256 / float32(255); no later rescale or PNG roundtrip',
            'request_count': 7560, 'backgrounds': 60, 'baseline_plus_request_references': 7620,
            'geometry_conditions_before_render': 540, 'minimum_defined_backgrounds': 40,
            'coverage_definition': 'at least one of nine fixed geometries has Lb>1e-6',
            'comparison_pool': {'old_training': 160, 'Q_F': 300, 'D_fit': 60},
            'pre_generation_gate': 'all60 decoded/hash checked, all540 Lb, no working collision, >=40 defined backgrounds',
            'render': 'I=clip(B+float32(sign*amplitude)*alpha,0,1), exactly once per fixed request',
            'D': 'max(abs(I-B),axis=RGB) float32', 'M_eff': 'D>float32(1/255), diagnostic only',
            'retention': 'all7560 including zero/weak/undefined; never replace or rescue',
            'generation_gate_never_uses_fit_predictions': True,
            'output': str(RUN), 'CPU_RAM_GiB': 2, 'D_free_GiB': 40,
            'cache': 'closed full hash/semantic verify; partial/unknown/failure stop; no resume/retry',
            'no_model_scoring_calibration_or_fitting': True}


PREP_FILES = {'sources.json', 'protocol.json', 'backgrounds.json', 'requests.json',
              'geometry.json', 'scalar_verification.json'}


def prepare():
    ctx = context(); snapshot = resources('preparation')
    if PREP.exists(): return verify_preparation(ctx)
    require(not OUT.exists(), 'Unknown independent output root already exists')
    scalar = artificial_audit(ctx)
    compile(Path(__file__).read_text(encoding='utf-8'), str(Path(__file__)), 'exec')
    require(not any(COUNTERS.values()), 'Preparation touched real inputs')
    OUT.mkdir(exist_ok=False); PREP.mkdir(exist_ok=False)
    write(PREP / 'sources.json', ctx)
    write(PREP / 'protocol.json', protocol(ctx))
    write(PREP / 'backgrounds.json', {'rows': ctx['backgrounds']})
    write(PREP / 'requests.json', {'rows': ctx['requests']})
    write(PREP / 'geometry.json', {'rows': ctx['geometry']})
    write(PREP / 'scalar_verification.json', {'passed': True, 'artificial': scalar,
                                            'resources': snapshot, 'boundary': boundary()})
    seal(PREP)
    return {'status': 'HOLD_INPUT_PREPARATION_COMPLETE', 'output': str(PREP),
            'complete_sha256': digest(PREP / 'complete.json'), 'resources': snapshot, 'boundary': boundary()}


def verify_preparation(ctx=None):
    ctx = context() if ctx is None else ctx
    c = read(PREP / 'complete.json'); require(set(c['outputs']) == PREP_FILES, 'Preparation inventory differs')
    inventory(PREP, c['outputs'])
    require(all(digest(PREP / n) == h for n, h in c['outputs'].items()), 'Preparation file hash differs')
    require(read(PREP / 'sources.json') == ctx and read(PREP / 'protocol.json') == protocol(ctx)
            and read(PREP / 'backgrounds.json')['rows'] == ctx['backgrounds']
            and read(PREP / 'requests.json')['rows'] == ctx['requests']
            and read(PREP / 'geometry.json')['rows'] == ctx['geometry'], 'Preparation source/config differs')
    v = read(PREP / 'scalar_verification.json')
    require(v['passed'] and v['artificial'] == artificial_audit(ctx)
            and not any(v['boundary']['counters'].values()) and not v['boundary']['blocked'],
            'Preparation scalar/scope evidence differs')
    return {'status': 'HOLD_INPUT_PREPARATION_CACHE_VERIFIED',
            'complete_sha256': digest(PREP / 'complete.json'), 'boundary': boundary()}


def save_array(path, **arrays):
    np, _ = arrays_api()
    with path.open('xb') as f:
        if path.suffix == '.npy': np.save(f, arrays['rgb'], allow_pickle=False)
        else: np.savez_compressed(f, **arrays)


def store_rgb(rgb, unique, reference):
    _, api = arrays_api(); h = api.working_rgb_identity(rgb)
    if h not in unique:
        unique[h] = {'working_RGB_sha256': h, 'file': f'images/{h}.npy', 'references': []}
        save_array(RUN / unique[h]['file'], rgb=rgb)
    unique[h]['references'].append(reference)
    return {'working_RGB_sha256': h, 'file': unique[h]['file']}


def load_array(relative, keys=None):
    np, _ = arrays_api(); p = RUN / relative
    require(p.resolve().is_relative_to(RUN) and p.suffix in ('.npy', '.npz'), 'Nonlocal array path')
    safe(p)
    if keys is None: return np.load(p, allow_pickle=False)
    with np.load(p, allow_pickle=False) as z:
        require(set(z.files) == set(keys), 'Unexpected NPZ keys')
        return {k: z[k] for k in keys}


def check_roi(b, image, geo, row):
    np, api = arrays_api(); mask = geo['M_geo']
    pre = b[mask] + np.float32(row['sign'] * row['amplitude']) * geo['alpha'][mask, None]
    lo, hi = pre < 0, pre > 1; free = ~(lo | hi); obs = image[mask]
    require(np.array_equal(obs[free], pre[free]) and (obs[lo] == 0).all() and (obs[hi] == 1).all(),
            'ROI preclip/clipping relation differs')
    require(float((lo | hi).mean()) == row['measurement']['clipped_channel_fraction'], 'Clipping fraction differs')
    measured = api.measure_pair(b, image, geo)
    require(all(row['measurement'][k] == v for k, v in measured.items()), 'Actual pair measurements differ')


def summarize(rows, unique):
    def counts(part):
        return {'requests': len(part), 'defined': sum(r['measurement']['defined'] for r in part),
                'undefined': sum(not r['measurement']['defined'] for r in part),
                'zero_actual': sum(r['measurement']['zero_actual_change'] for r in part),
                'nonzero_actual': sum(not r['measurement']['zero_actual_change'] for r in part),
                'weak_nonzero_empty_M_eff': sum(r['measurement']['nonzero_pixels'] > 0
                    and r['measurement']['M_eff_1_over_255_pixels'] == 0 for r in part),
                'clipped': sum(r['measurement']['clipped_channel_fraction'] > 0 for r in part)}
    _, api = arrays_api()
    return {**counts(rows), 'backgrounds': 60, 'references': len(rows) + 60, 'unique_RGB': len(unique),
            'by_amplitude': {str(a): counts([r for r in rows if r['amplitude'] == a]) for a in api.AMPLITUDES},
            'by_geometry': {g['id']: counts([r for r in rows if r['geometry_id'] == g['id']])
                            for g in api.geometry_specs()},
            'by_polarity': {p: counts([r for r in rows if r['polarity'] == p]) for p in ('bright', 'dark')}}


def acquire():
    ctx = context(); verify_preparation(ctx); resources('acquisition_start')
    if RUN.exists(): return verify_inputs(ctx)
    require(not any(COUNTERS.values()) and not OPENS, 'Acquisition requires fresh process')
    np, api = arrays_api()
    from PIL import Image
    root = Path(ctx['dataset_root'])
    for frozen in ctx['backgrounds']:
        p = root / frozen['path']; safe(p)
        require(p.resolve().is_relative_to(root), 'Original outside dataset')
        ALLOWED_ORIGINALS.add(p.resolve())
    RUN.mkdir(exist_ok=False)
    for n in ('images', 'geometry', 'residuals'): (RUN / n).mkdir(exist_ok=False)
    start = time.perf_counter()
    write(RUN / 'started.json', {'preparation_complete_sha256': digest(PREP / 'complete.json'),
                                 'source_files': ctx['files'], 'no_automatic_retry': True})
    try:
        geometries = {}
        for expected in ctx['geometry']:
            g = api.make_geometry(expected['spec']); require(g['metadata'] == expected, 'Geometry mismatch')
            gid = expected['spec']['id']; geometries[gid] = g
            save_array(RUN / f'geometry/{gid}.npz', M_geo=g['M_geo'], alpha=g['alpha'])
        unique, backgrounds = {}, []
        for frozen in ctx['backgrounds']:
            p = root / frozen['path']; require(p.stat().st_size == frozen['bytes'], 'Source size differs')
            payload = p.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == frozen['file_sha256'], 'Original file hash differs')
            with Image.open(io.BytesIO(payload)) as im:
                COUNTERS['original_decodes'] += 1
                require(list(reversed(im.size)) == frozen['original_size_hw'] and im.mode == frozen['original_mode'],
                        'Original size/mode differs')
                rgb = im.convert('RGB'); raw = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
                work = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255)
            api.validate_rgb(work)
            backgrounds.append({**frozen, 'legacy_rgb_group_id': frozen['rgb_sha256'],
                'decoded_RGB_u8_sha256': hashlib.sha256(raw.tobytes(order='C')).hexdigest(),
                'decode_count': 1, 'working_RGB_sha256': api.working_rgb_identity(work),
                'image': store_rgb(work, unique, frozen['id']),
                'measurements': {gid: api.background_measurement(work, g) for gid, g in geometries.items()}})
            resources('decoded_' + frozen['id'])
        c = coverage(backgrounds, ctx['prior_work_hashes'], list(geometries))
        write(RUN / 'source_audit.json', {'backgrounds': backgrounds, 'comparison_pool_metadata_only': ctx['prior_work_hashes'],
            'original_opens': dict(OPENS), 'boundary_before_generation': boundary()})
        write(RUN / 'coverage.json', {**c, 'formal_render_calls_before_gate': COUNTERS['formal_render_calls']})
        require(COUNTERS['original_decodes'] == COUNTERS['original_file_opens'] == 60
                and COUNTERS['formal_render_calls'] == 0 and admitted(c), 'Pre-generation identity/collision/coverage gate')
        bybg = {r['id']: r for r in backgrounds}; rows = []; current = None
        for request in ctx['requests']:
            bg = bybg[request['background_id']]
            if current != bg['id']:
                b = load_array(bg['image']['file']); current = bg['id']; resources('render_' + current)
            g = geometries[request['geometry_id']]
            COUNTERS['formal_render_calls'] += 1
            rendered = api.render_measurement(b, g, request['sign'], request['amplitude'])
            image, m = rendered['image'], rendered['measurement']
            require(all(m[k] == v for k, v in bg['measurements'][request['geometry_id']].items()),
                    'Generated measurement differs from pre-render coverage')
            d = np.max(np.abs(image - b), axis=-1); eff = d > api.LABEL_FLOOR
            residual_file = f'residuals/{request["id"]}.npz'
            save_array(RUN / residual_file, D=d, M_eff=eff)
            row = {**request, 'image': store_rgb(image, unique, request['id']), 'background_image': bg['image'],
                   'geometry_file': f'geometry/{request["geometry_id"]}.npz', 'residual_file': residual_file,
                   'D_sha256': api.array_identity(d), 'M_eff_sha256': api.array_identity(eff),
                   'measurement': m, 'retained': True}
            check_roi(b, image, g, row); rows.append(row)
            if len(rows) % 126 == 0:
                print(json.dumps({'background': len(rows) // 126, 'of': 60,
                                  'render_calls': COUNTERS['formal_render_calls']}), flush=True)
        write(RUN / 'manifest.json', {'rows': rows, 'summary': summarize(rows, unique)})
        write(RUN / 'unique_inputs.json', {'rows': list(unique.values()), 'references': 7620})
        semantics = verify_arrays(ctx)
        require(context() == ctx, 'Sources changed before seal')
        write(RUN / 'verification.json', {'passed': True, 'semantics': semantics,
            'elapsed_seconds': time.perf_counter() - start, 'resources': SAMPLES, 'boundary': boundary()})
        require(set(p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file())
                == expected_run_files(ctx), 'Pre-seal inventory differs')
        seal(RUN)
        return {'status': 'HOLD_INPUTS_COMPLETE', 'complete_sha256': digest(RUN / 'complete.json'),
                'summary': semantics, 'boundary': boundary()}
    except BaseException as exc:
        if not (RUN / 'failure.json').exists():
            write(RUN / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                        'boundary': boundary(), 'preserve_no_resume': True})
        raise


def expected_run_files(ctx):
    unique = read(RUN / 'unique_inputs.json')['rows']
    return {'started.json', 'source_audit.json', 'coverage.json', 'manifest.json', 'unique_inputs.json',
            'verification.json'} | {f'geometry/{g["spec"]["id"]}.npz' for g in ctx['geometry']} | {
            f'residuals/{r["id"]}.npz' for r in ctx['requests']} | {r['file'] for r in unique}


def verify_arrays(ctx):
    np, api = arrays_api()
    audit = read(RUN / 'source_audit.json'); backgrounds = audit['backgrounds']
    require(len(backgrounds) == 60 and audit['comparison_pool_metadata_only'] == ctx['prior_work_hashes'],
            'Background/comparison metadata differs')
    geos = {}
    for expected in ctx['geometry']:
        gid = expected['spec']['id']; a = load_array(f'geometry/{gid}.npz', ('M_geo', 'alpha'))
        require(a['M_geo'].dtype == np.bool_ and a['alpha'].dtype == np.float32
                and a['M_geo'].shape == a['alpha'].shape == (256, 256)
                and api.array_identity(a['M_geo']) == expected['M_geo_sha256']
                and api.array_identity(a['alpha']) == expected['alpha_sha256'], 'Geometry content differs')
        geos[gid] = {**a, 'metadata': expected}
    unique_doc = read(RUN / 'unique_inputs.json'); items = unique_doc['rows']
    index = {r['working_RGB_sha256']: r for r in items}; require(len(index) == len(items), 'Duplicate unique index')
    for h, r in index.items():
        require(r['file'] == f'images/{h}.npy' and api.working_rgb_identity(load_array(r['file'])) == h,
                'Saved float RGB identity differs')
    refs = defaultdict(list)
    for frozen, bg in zip(ctx['backgrounds'], backgrounds):
        require(all(bg[k] == v for k, v in frozen.items()) and bg['decode_count'] == 1
                and bg['legacy_rgb_group_id'] == frozen['rgb_sha256']
                and re.fullmatch('[0-9a-f]{64}', bg['decoded_RGB_u8_sha256']), 'Background source record differs')
        b = load_array(bg['image']['file'])
        require(bg['working_RGB_sha256'] == bg['image']['working_RGB_sha256'] == api.working_rgb_identity(b),
                'Background work identity differs')
        require(bg['measurements'] == {gid: api.background_measurement(b, g) for gid, g in geos.items()},
                'All540 saved Lb/coverage fields differ')
        refs[bg['working_RGB_sha256']].append(bg['id'])
    c = coverage(backgrounds, ctx['prior_work_hashes'], list(geos))
    require(read(RUN / 'coverage.json') == {**c, 'formal_render_calls_before_gate': 0} and admitted(c),
            'Saved coverage/collision gate differs')
    saved_boundary = audit['boundary_before_generation']['counters']
    require(saved_boundary['original_decodes'] == saved_boundary['original_file_opens'] == 60
            and saved_boundary['formal_render_calls'] == 0
            and audit['original_opens'] == {str((Path(ctx['dataset_root']) / r['path']).resolve()): 1
                                           for r in ctx['backgrounds']}, 'Pre-render once-only source counts differ')
    rows = read(RUN / 'manifest.json')['rows']; require(len(rows) == 7560, 'Missing nominal request rows')
    bybg = {b['id']: b for b in backgrounds}; current = None; energy_groups = defaultdict(list)
    for frozen, row in zip(ctx['requests'], rows):
        require(all(row[k] == v for k, v in frozen.items()) and row['retained'], 'Request content/order differs')
        bg = bybg[row['background_id']]
        if current != bg['id']: b = load_array(bg['image']['file']); current = bg['id']
        h = row['image']['working_RGB_sha256']
        require(h in index and row['image']['file'] == index[h]['file'] and row['background_image'] == bg['image']
                and row['geometry_file'] == f'geometry/{row["geometry_id"]}.npz'
                and row['residual_file'] == f'residuals/{row["id"]}.npz', 'Unbound request cache path')
        image = load_array(row['image']['file']); d = np.max(np.abs(image - b), axis=-1)
        arrays = load_array(row['residual_file'], ('D', 'M_eff'))
        require(arrays['D'].dtype == np.float32 and arrays['M_eff'].dtype == np.bool_
                and arrays['D'].shape == arrays['M_eff'].shape == (256, 256)
                and np.array_equal(arrays['D'], d) and np.array_equal(arrays['M_eff'], d > api.LABEL_FLOOR)
                and api.array_identity(d) == row['D_sha256']
                and api.array_identity(arrays['M_eff']) == row['M_eff_sha256'], 'Saved residual/diagnostic mask differs')
        check_roi(b, image, geos[row['geometry_id']], row)
        require(row['measurement']['sign'] == row['sign'] and row['measurement']['amplitude'] == row['amplitude'],
                'Measurement amplitude/sign differs')
        refs[h].append(row['id']); energy_groups[(bg['id'], row['geometry_id'], row['sign'])].append(row)
    require(set(refs) == set(index) and all(index[h]['references'] == v for h, v in refs.items())
            and unique_doc['references'] == sum(map(len, refs.values())) == 7620
            and len({ref for values in refs.values() for ref in values}) == 7620, 'Dedup/reference closure differs')
    energy_errors = []
    for group in energy_groups.values():
        for a, b_row in zip(group[:-1], group[1:]):
            ma, mb = a['measurement'], b_row['measurement']
            if ma['clipped_channel_fraction'] == mb['clipped_channel_fraction'] == 0 and ma['energy_RGB'] > 0:
                energy_errors.append(abs(mb['energy_RGB'] / ma['energy_RGB'] /
                                         (b_row['amplitude'] / a['amplitude'])**2 - 1))
    require(max(energy_errors, default=0) <= 5e-4, 'Unclipped energy scaling differs')
    summary = summarize(rows, index)
    require(summary == read(RUN / 'manifest.json')['summary'], 'Complete nominal/actual denominators differ')
    return {**summary, 'geometry_arrays_verified': 9, 'request_residuals_verified': 7560,
            'unclipped_energy_max_relative_error': max(energy_errors, default=None),
            'no_original_redecode_or_render_during_verification': True}


def verify_inputs(ctx=None):
    ctx = context() if ctx is None else ctx; verify_preparation(ctx); resources('verify_inputs')
    require((RUN / 'complete.json').is_file(), 'Missing complete; preserve partial transaction')
    c = read(RUN / 'complete.json'); require(set(c['outputs']) == expected_run_files(ctx), 'Run whitelist differs')
    inventory(RUN, c['outputs'])
    require(all(digest(RUN / n) == h for n, h in c['outputs'].items()), 'Frozen run file hash differs')
    started = read(RUN / 'started.json')
    require(started['preparation_complete_sha256'] == digest(PREP / 'complete.json')
            and started['source_files'] == ctx['files'], 'Run/preparation source binding differs')
    semantic = verify_arrays(ctx); v = read(RUN / 'verification.json')
    require(v['passed'] and semantic == v['semantics'] and v['boundary']['counters']['formal_render_calls'] == 7560
            and v['boundary']['counters']['original_decodes'] == 60 and not v['boundary']['blocked'],
            'Saved generation/verification evidence differs')
    require(COUNTERS['formal_render_calls'] == COUNTERS['original_decodes'] == COUNTERS['original_file_opens'] == 0,
            'Read-only reuse attempted acquisition/render')
    return {'status': 'HOLD_INPUTS_COMPLETE_CACHE_VERIFIED', 'complete_sha256': digest(RUN / 'complete.json'),
            'summary': semantic, 'boundary': boundary()}


def main():
    global MODE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify-preparation', 'acquire', 'verify-inputs'))
    args = p.parse_args(); MODE = args.command; install_boundary()
    result = {'prepare': prepare, 'verify-preparation': verify_preparation,
              'acquire': acquire, 'verify-inputs': verify_inputs}[MODE]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
