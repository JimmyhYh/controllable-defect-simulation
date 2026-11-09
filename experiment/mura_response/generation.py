"""Independent stage-3 CPU generation; only the frozen D_fit60 is writable input.

Metadata verification runs in a separate interpreter. Never import resources in
this process. Complete caches are read-only; incomplete transactions are never
continued, regenerated, repaired, or silently replaced by these entry points.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results/mura_response/semu_applicability_v1'
PREP = BASE / 'preparation_v1'
OUT = BASE / 'fit_inputs_v1'
FORBIDDEN = ('torch', 'torchvision', 'timm', 'segmentation_models_pytorch',
             'faiss', 'patchcore', 'controlled_generation', 'pod_probe',
             'mura_response.resources', 'localization_training')
ALLOWED_ORIGINALS = set()
ACCESS = collections.Counter()
DECODED = set()
COUNTERS = {'original_decodes': 0, 'old_training_decodes': 0,
            'fit_decodes': 0, 'render_calls': 0}
RESOURCE_SAMPLES = []


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in FORBIDDEN):
            raise RuntimeError('Forbidden model/metadata-process import: ' + fullname)
        return None


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no replacement, retry, or resume')


def payload_guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    suffix = p.suffix.lower()
    if suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        require(p in ALLOWED_ORIGINALS, 'Original image outside D_fit/old-training allowlist: ' + str(p))
        require(len(args) < 2 or args[1] is None or not any(c in args[1] for c in 'wax+'),
                'Original image write is forbidden')
        ACCESS[str(p)] += 1
    if suffix in ('.npy', '.npz'):
        require(p.is_relative_to(OUT), 'Array/probability outside this independent output: ' + str(p))
    require(suffix not in ('.pth', '.pt', '.pkl', '.pickle', '.faiss', '.safetensors'),
            'Model/tensor/pickle payload access is forbidden in generation process')


sys.meta_path.insert(0, BlockModels())
sys.addaudithook(payload_guard)

import numpy as np
from PIL import Image
from . import inputs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')


def save_npy(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def runtime():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'executable_sha256': digest(sys.executable),
            'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'Pillow')}}


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'page_available',
                                        'virtual', 'virtual_available', 'extended')]


def resource(stage):
    state = MemoryStatus(); state.length = ctypes.sizeof(state)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(MemoryStatus)]; query.restype = ctypes.c_int
    require(query(ctypes.byref(state)) != 0, 'RAM query failed')
    snapshot = {'stage': stage, 'available_RAM_GiB': state.available / 2**30,
                'free_D_GiB': shutil.disk_usage(ROOT).free / 2**30}
    RESOURCE_SAMPLES.append(snapshot)
    require(snapshot['available_RAM_GiB'] >= 2 and snapshot['free_D_GiB'] >= 40,
            'CPU RAM2GiB/full-round free disk40GiB admission failed')
    return snapshot


def preparation():
    """Read metadata only; the child alone imports the permanent resources guards."""
    check = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m',
                            'mura_response.resources', 'verify'], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8', timeout=120)
    require(check.returncode == 0, 'Isolated preparation verification failed: ' + check.stderr[-2000:])
    verified = json.loads(check.stdout)
    require(verified['status'] == 'PREPARATION_CACHE_VERIFIED', 'Preparation status differs')
    complete = read(PREP / 'complete.json')
    hashes = {name: digest(PREP / name) for name in complete['outputs']}
    require(hashes == complete['outputs'], 'Preparation changed after isolated verification')
    hashes['complete.json'] = digest(PREP / 'complete.json')
    source = read(PREP / 'sources.json')
    protocol = read(PREP / 'protocol.json')
    roles = read(PREP / 'roles.json')
    requests = read(PREP / 'requests.json')['rows']
    selected = [row for row in requests if row['role'] == 'D_fit60']
    require(len(selected) == 7560 and len(roles['roles']['D_fit60']) == 60,
            'Frozen fit budget differs')
    require(protocol['preprocessing_future'] == 'PIL RGB bilinear256 then float32/255; unchanged contrast',
            'Frozen preprocessing differs')
    manifest_path = ROOT / 'results/weak_localization/dataset_manifest.json'
    weak_path = ROOT / 'data/weak_localization/split.json'
    cg_path = ROOT / 'data/controlled_generation/split.json'
    for p in (manifest_path, weak_path, cg_path):
        require(digest(p) == source['files'][str(p)], 'Old metadata identity differs')
    manifest = read(manifest_path)
    bypath = {row['path']: row for row in manifest['files'] if 'rgb_content_sha256' in row}
    weak = read(weak_path); cg = read(cg_path)
    training = sorted(set(weak['N']) | set(cg['N_gen']))
    require(len(weak['N']) == len(cg['N_gen']) == 80 and len(training) == 160,
            'Original N/N_gen path budget differs')
    dataset_root = Path(roles['dataset_root'])
    require(dataset_root == Path(weak['root']) == Path(cg['root'])
            == ROOT / 'data/weak_localization/AnoLCD', 'Dataset source roots differ')
    protected = {row['path'] for role in ('Q200', 'F100', 'D_hold60') for row in roles['roles'][role]}
    protected |= set(cg['S_bg']) | set(cg['Q_cal']) | set(weak['Tfinal'])
    allowed = training + [row['path'] for row in roles['roles']['D_fit60']]
    require(len(allowed) == len(set(allowed)) == 220 and not set(allowed) & protected,
            'Generation source roles overlap protected image roles')
    require(all(p.startswith('display/train/good/') and (dataset_root / p).resolve().is_relative_to(dataset_root)
                for p in allowed), 'Source path outside allowed normal training tree')
    return {'hashes': hashes, 'verified': verified, 'protocol': protocol, 'roles': roles,
            'requests': selected, 'geometry': read(PREP / 'geometry.json')['rows'],
            'bypath': bypath, 'training': training, 'dataset_root': dataset_root,
            'training_roles': {'N': weak['N'], 'N_gen': cg['N_gen']}, 'allowed': allowed}


def binding(prep):
    return {'schema': 'mura-fit-generation-binding-v1', 'stage': 3,
            'preparation': prep['hashes'],
            'implementation': {str(Path(__file__).resolve()): digest(__file__),
                               **prep['protocol']['implementation']},
            'runtime': runtime(), 'fit_role': 'D_fit60', 'nominal_requests': 7560,
            'source_snapshot_basis': 'original file bytes and decoded RGB verified once on acquisition',
            'generation': prep['protocol']['generation'],
            'D': 'max(abs(I-B),axis=RGB), float32, HxW',
            'signed_residual_recoverable': 'I-B from saved float32 I and B',
            'M_eff': 'D>float32(1/255); diagnostic only; no weak filtering',
            'undefined_and_zero_change': 'retain all requests; never replace or relabel as FN',
            'model_input_dedup': 'canonical HWC float32 C array content hash; baseline references included',
            'minimum_defined_backgrounds': 40, 'CPU_min_RAM_GiB': 2,
            'full_round_min_free_disk_GiB': 40,
            'no_models_forwards_training_calibration_or_fitting': True}


def boundary():
    loaded = [name for name in sys.modules if any(name == n or name.startswith(n + '.') for n in FORBIDDEN)]
    require(not loaded, 'Forbidden import detected at boundary')
    return {'forbidden_modules_loaded': loaded, **COUNTERS,
            'original_paths_opened': len(ACCESS), 'original_file_opens': sum(ACCESS.values()),
            'Q200_F100_D_hold60_decodes': 0, 'GT_or_probability_reads': 0,
            'model_forwards': 0, 'parameter_updates': 0, 'calibration': 0,
            'statistical_fits': 0, 'old_training_usage': 'source and canonical working hash only'}


def decode_original(prep, relative, kind):
    require(relative in prep['allowed'] and relative not in DECODED, 'Unlisted/repeated original decode')
    row = prep['bypath'][relative]; path = prep['dataset_root'] / relative
    require(path.is_file() and path.stat().st_size == row['bytes'], 'Source missing/size differs: ' + relative)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'Original file hash differs: ' + relative)
    DECODED.add(relative)
    with Image.open(io.BytesIO(payload)) as original:
        COUNTERS['original_decodes'] += 1; COUNTERS[kind + '_decodes'] += 1
        require(list(original.size) == row['size'] == [1024, 1024]
                and original.mode == row['mode'], 'Original shape/mode differs: ' + relative)
        rgb = original.convert('RGB')
        raw_rgb = np.asarray(rgb, dtype=np.uint8)
        raw_hash = hashlib.sha256(raw_rgb.tobytes(order='C')).hexdigest()
        require(raw_hash == row['rgb_content_sha256'], 'Decoded original RGB hash differs: ' + relative)
        work = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR),
                          dtype=np.float32) / np.float32(255)
    inputs.validate_rgb(work)
    return work, {'path': relative, 'file_sha256': row['sha256'], 'bytes': len(payload),
                  'original_RGB_sha256': raw_hash, 'working_RGB_sha256': inputs.working_rgb_identity(work),
                  'original_mode': row['mode'], 'original_size_hw': [1024, 1024],
                  'usage': kind, 'decode_count': 1}


def store_input(rgb, index, reference):
    h = inputs.working_rgb_identity(rgb)
    if h not in index:
        relative = f'images/{h}.npy'
        save_npy(OUT / relative, rgb)
        index[h] = {'working_RGB_sha256': h, 'file': relative, 'references': []}
    index[h]['references'].append(reference)
    return {'working_RGB_sha256': h, 'file': index[h]['file']}


def coverage_summary(backgrounds, geometry_ids, training):
    collisions = []
    byhash = collections.defaultdict(list)
    training_byhash = collections.defaultdict(list)
    for row in training:
        training_byhash[row['working_RGB_sha256']].append(row['path'])
    for row in backgrounds:
        h = row['working_RGB_sha256']; byhash[h].append(row['id'])
        if h in training_byhash:
            collisions.append({'type': 'fit_vs_old_training', 'fit_id': row['id'],
                               'working_RGB_sha256': h, 'old_paths': training_byhash[h]})
    for h, ids in byhash.items():
        if len(ids) > 1:
            collisions.append({'type': 'within_fit', 'fit_ids': ids, 'working_RGB_sha256': h})
    counts = [sum(m['defined'] for m in row['measurements'].values()) for row in backgrounds]
    return {'backgrounds': len(backgrounds), 'geometry_conditions': len(backgrounds) * 9,
            'defined_backgrounds_at_least1': sum(n > 0 for n in counts),
            'defined_all9_backgrounds': sum(n == 9 for n in counts),
            'defined_partial_backgrounds': sum(0 < n < 9 for n in counts),
            'defined_none_backgrounds': sum(n == 0 for n in counts),
            'defined_geometry_conditions': sum(counts),
            'per_geometry_defined_backgrounds': {g: sum(row['measurements'][g]['defined'] for row in backgrounds)
                                                 for g in geometry_ids},
            'working_collisions': collisions, 'fit_unique_working_RGB': len(byhash),
            'old_training_paths': len(training), 'old_training_unique_working_RGB': len(training_byhash),
            'old_training_duplicate_groups': {h: paths for h, paths in training_byhash.items() if len(paths) > 1},
            'cross_new_roles_not_yet_checked': ['Q200', 'F100', 'D_hold60'],
            'physical_capture_groups_known': False, 'minimum_defined_backgrounds': 40,
            'formal_render_calls_before_coverage_gate': COUNTERS['render_calls']}


def aggregate(rows, unique):
    def counts(part):
        return {'requests': len(part), 'defined': sum(r['measurement']['defined'] for r in part),
                'undefined': sum(not r['measurement']['defined'] for r in part),
                'nonzero_actual': sum(not r['measurement']['zero_actual_change'] for r in part),
                'zero_actual': sum(r['measurement']['zero_actual_change'] for r in part),
                'clipped_requests': sum(r['measurement']['clipped_channel_fraction'] > 0 for r in part),
                'weak_nonzero_empty_old_M_eff': sum(r['measurement']['nonzero_pixels'] > 0
                    and r['measurement']['M_eff_1_over_255_pixels'] == 0 for r in part),
                'diagnostic_M_eff_nonempty': sum(r['measurement']['M_eff_1_over_255_pixels'] > 0 for r in part)}
    out = counts(rows)
    out['unique_model_inputs_including60_baselines'] = len(unique)
    out['model_input_references_including60_baselines'] = len(rows) + 60
    out['request_unique_RGB'] = len({r['image']['working_RGB_sha256'] for r in rows})
    out['by_amplitude'] = {str(a): counts([r for r in rows if r['amplitude'] == a]) for a in inputs.AMPLITUDES}
    out['by_geometry'] = {g['id']: counts([r for r in rows if r['geometry_id'] == g['id']])
                          for g in inputs.geometry_specs()}
    out['by_polarity'] = {p: counts([r for r in rows if r['polarity'] == p]) for p in ('bright', 'dark')}
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row['background_id'], row['geometry_id'], row['sign'])].append(row)
    energy_errors = []
    for group in grouped.values():
        for a, b in zip(group[:-1], group[1:]):
            ma, mb = a['measurement'], b['measurement']
            if ma['clipped_channel_fraction'] == mb['clipped_channel_fraction'] == 0 and ma['energy_RGB'] > 0:
                error = abs((mb['energy_RGB'] / ma['energy_RGB']) / (b['amplitude'] / a['amplitude'])**2 - 1)
                energy_errors.append(error)
    out['unclipped_adjacent_energy_pairs'] = len(energy_errors)
    out['unclipped_energy_max_relative_error'] = max(energy_errors, default=None)
    out['unclipped_energy_relative_tolerance'] = 5e-4
    require(not energy_errors or max(energy_errors) <= 5e-4, 'Unclipped energy scaling exceeds frozen float32 allowance')
    return out


def generate_fit():
    prep = preparation(); source = binding(prep); resource('start')
    if OUT.exists():
        return verify_fit(prep=prep, source=source)
    ALLOWED_ORIGINALS.update((prep['dataset_root'] / p).resolve() for p in prep['allowed'])
    OUT.mkdir(exist_ok=False)
    for folder in ('images', 'geometry', 'residuals'):
        (OUT / folder).mkdir(exist_ok=False)
    started = time.perf_counter()
    write(OUT / 'started.json', {'stage': 3, 'scope': 'D_fit60 only; CPU',
                                'no_automatic_retry': True, 'source_identity': source})
    try:
        write(OUT / 'sources.json', source)
        write(OUT / 'protocol.json', {'frozen_preparation': str(PREP), 'role': 'D_fit60',
            'backgrounds': 60, 'requests': 7560, 'generation_binding': source,
            'protected_roles': ['Q200', 'F100', 'D_hold60', 'S_bg', 'Q_cal', 'old_test', 'GT'],
            'original_decode_once': True, 'old_training_hash_only': True,
            'original_RGB_hash': 'sha256 of decoded original RGB uint8 C bytes',
            'working_RGB_hash': 'inputs.array_identity: dtype/shape/C header plus float32 RGB bytes',
            'preprocessing': prep['protocol']['preprocessing_future'],
            'render_once_per_request': True, 'no_rejection_on_M_eff': True,
            'cache_verification': 'closed full file hashes, shape/dtype/content/measurements; no original opens or render',
            'D_definition': source['D'], 'M_eff_definition': source['M_eff'],
            'statistics_domain': 'encoded RGB; k assumed; no calibrated physical luminance'})
        geometries = {}
        for spec, expected in zip(inputs.geometry_specs(), prep['geometry']):
            geometry = inputs.make_geometry(spec)
            require(geometry['metadata'] == expected, 'Frozen geometry numerical identity differs')
            geometries[spec['id']] = geometry
            save_npz(OUT / f'geometry/{spec["id"]}.npz', M_geo=geometry['M_geo'], alpha=geometry['alpha'])
        training = []
        for relative in prep['training']:
            _, row = decode_original(prep, relative, 'old_training')
            row['roles'] = [role for role, paths in prep['training_roles'].items() if relative in paths]
            training.append(row)
        resource('after_old_training_hashes')
        unique = {}; backgrounds = []
        for frozen in prep['roles']['roles']['D_fit60']:
            rgb, row = decode_original(prep, frozen['path'], 'fit')
            require(row['file_sha256'] == frozen['file_sha256'] and row['original_RGB_sha256'] == frozen['rgb_sha256'],
                    'Fit source differs from frozen role')
            row.update(id=frozen['id'], ordinal=frozen['ordinal'], image=store_input(rgb, unique, frozen['id']),
                       measurements={g: inputs.background_measurement(rgb, geo) for g, geo in geometries.items()})
            backgrounds.append(row)
        coverage = coverage_summary(backgrounds, list(geometries), training)
        write(OUT / 'source_audit.json', {'old_training': training, 'backgrounds': backgrounds,
                                        'original_source_hash_verified': True,
                                        'allowlist': prep['allowed'], 'boundary': boundary()})
        write(OUT / 'coverage.json', coverage)
        resource('coverage_before_formal_generation')
        require(len(backgrounds) == 60 and not coverage['working_collisions'], 'Canonical working RGB background collision')
        require(coverage['defined_backgrounds_at_least1'] >= 40,
                'Fewer than40 fixed backgrounds have any defined geometry; stop before formal stimuli')
        require(COUNTERS['render_calls'] == 0, 'Formal stimulus generated before coverage gate')
        rows = []; bybg = {b['id']: b for b in backgrounds}; current_id = None
        for request in prep['requests']:
            bid = request['background_id']; background_row = bybg[bid]
            if bid != current_id:
                resource('generation_' + bid)
                background = np.load(OUT / background_row['image']['file'], allow_pickle=False)
                current_id = bid
            geometry = geometries[request['geometry_id']]
            COUNTERS['render_calls'] += 1
            rendered = inputs.render_measurement(background, geometry, request['sign'], request['amplitude'])
            image = rendered['image']; measurement = rendered['measurement']
            expected_background = background_row['measurements'][request['geometry_id']]
            require(all(measurement[k] == v for k, v in expected_background.items()), 'Request measurement differs from pre-generation Lb condition')
            residual = np.max(np.abs(image - background), axis=-1)
            residual_file = f'residuals/{request["id"]}.npz'
            save_npz(OUT / residual_file, D=residual, M_eff=residual > inputs.LABEL_FLOOR)
            rows.append({**request, 'image': store_input(image, unique, request['id']),
                         'background_image': background_row['image'],
                         'geometry_file': f'geometry/{request["geometry_id"]}.npz',
                         'residual_file': residual_file, 'D_sha256': inputs.array_identity(residual),
                         'M_eff_sha256': inputs.array_identity(residual > inputs.LABEL_FLOOR),
                         'measurement': measurement, 'retained': True})
            if len(rows) % 126 == 0:
                print(json.dumps({'stage': 3, 'background': len(rows)//126, 'of': 60,
                                  'render_calls': COUNTERS['render_calls']}), flush=True)
        write(OUT / 'manifest.json', {'rows': rows, 'request_order': 'unchanged frozen D_fit60 order',
                                     'summary': aggregate(rows, unique)})
        write(OUT / 'unique_inputs.json', {'rows': list(unique.values()), 'references': 7620,
                                          'request_count_is_not_independent_background_count': True})
        require(COUNTERS['render_calls'] == 7560 and COUNTERS['fit_decodes'] == 60
                and COUNTERS['old_training_decodes'] == 160 and len(DECODED) == 220,
                'Actual source/render counts do not close')
        semantic = verify_arrays(prep)
        require(binding(preparation()) == source, 'Bound source/runtime changed before sealing')
        resource('before_seal')
        write(OUT / 'verification.json', {'passed': True, 'summary': semantic,
            'elapsed_seconds': time.perf_counter() - started, 'boundary': boundary(),
            'original_opens': dict(ACCESS), 'resources': RESOURCE_SAMPLES,
            'minimum_sampled_available_RAM_GiB': min(r['available_RAM_GiB'] for r in RESOURCE_SAMPLES),
            'peak_process_working_set_bytes': peak_working_set(),
            'measurement_not_detector_effectiveness': True})
        expected_files = expected_inventory(prep)
        check_inventory(expected_files)
        outputs = {name: digest(OUT / name) for name in sorted(expected_files)}
        write(OUT / 'complete.json', {'schema': 'mura-fit-inputs-complete-v1', 'outputs': outputs})
        return {'status': 'FIT_INPUTS_COMPLETE', 'output': str(OUT), 'summary': semantic,
                'boundary': boundary(), 'complete_files': len(outputs) + 1}
    except BaseException as exc:
        if not (OUT / 'failure.json').exists():
            write(OUT / 'failure.json', {'error': type(exc).__name__, 'message': str(exc),
                'boundary': boundary(), 'resources': RESOURCE_SAMPLES,
                'elapsed_seconds': time.perf_counter() - started,
                'no_automatic_retry_resume_or_cleanup': True})
        raise


def peak_working_set():
    """Observed OS process peak; no model/CUDA or system process introspection."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'quota_peak_paged',
                'quota_paged', 'quota_peak_nonpaged', 'quota_nonpaged', 'page', 'peak_page')]
    state = Counters(); state.cb = ctypes.sizeof(state)
    kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi'); query = psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    query.restype = ctypes.c_int
    require(query(kernel.GetCurrentProcess(), ctypes.byref(state), state.cb), 'Process memory query failed')
    return int(state.peak)


def load_arrays(relative, names):
    p = OUT / relative
    require(p.resolve().is_relative_to(OUT) and p.is_file(), 'Missing/escaped array file')
    with np.load(p, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), 'Array archive fields differ')
        return {n: archive[n] for n in names}


def expected_inventory(prep):
    """Independent ownership list; never bless whatever happens to be in OUT."""
    metadata = {'started.json', 'sources.json', 'protocol.json', 'source_audit.json',
                'coverage.json', 'manifest.json', 'unique_inputs.json', 'verification.json'}
    geometry = {f'geometry/{g["spec"]["id"]}.npz' for g in prep['geometry']}
    residuals = {f'residuals/{r["id"]}.npz' for r in prep['requests']}
    unique = read(OUT / 'unique_inputs.json')['rows']
    images = {f'images/{r["working_RGB_sha256"]}.npy' for r in unique}
    require(len(geometry) == 9 and len(residuals) == 7560 and len(images) == len(unique),
            'Expected inventory counts differ from fixed metadata/unique inputs')
    return metadata | geometry | residuals | images


def check_inventory(expected, sealed=False):
    expected_files = set(expected) | ({'complete.json'} if sealed else set())
    members = list(OUT.rglob('*'))
    files = {p.relative_to(OUT).as_posix() for p in members if p.is_file()}
    dirs = {p.relative_to(OUT).as_posix() for p in members if p.is_dir()}
    require(files == expected_files and dirs == {'images', 'geometry', 'residuals'}
            and all(not p.is_symlink() and p.resolve().is_relative_to(OUT) for p in members),
            'Independent expected file/directory inventory differs; preserve unknown content')


def verify_roi_relation(background, image, geometry, sign, amplitude, claimed_clipping):
    """ROI algebra audit only: no full stimulus/render call or replacement image.

Unclipped channels must equal the float32 preclip sum; clipped channels must
equal their boundary. Counting the preclip ROI violations checks clipping
independently of render_measurement's returned metadata.
"""
    mask = geometry['M_geo']
    roi_sum = background[mask] + np.float32(sign * amplitude) * geometry['alpha'][mask, None]
    low = roi_sum < 0; high = roi_sum > 1; free = ~(low | high)
    observed = image[mask]
    require(np.array_equal(observed[free], roi_sum[free])
            and bool((observed[low] == 0).all()) and bool((observed[high] == 1).all()),
            'Saved ROI violates frozen sign/amplitude/alpha/clipping algebra')
    actual_clipping = float((low | high).mean())
    require(actual_clipping == claimed_clipping, 'Saved clipping fraction differs from preclip ROI conditions')


def verify_arrays(prep):
    """Complete CPU cache check: never reads originals or calls render."""
    audit = read(OUT / 'source_audit.json'); manifest = read(OUT / 'manifest.json')
    rows = manifest['rows']; unique = read(OUT / 'unique_inputs.json')['rows']
    geometry = {}
    for expected in prep['geometry']:
        gid = expected['spec']['id']
        arrays = load_arrays(f'geometry/{gid}.npz', ('M_geo', 'alpha'))
        require(arrays['M_geo'].dtype == np.bool_ and arrays['M_geo'].shape == (256, 256)
                and arrays['alpha'].dtype == np.float32 and arrays['alpha'].shape == (256, 256), 'Geometry array type/shape differs')
        require(inputs.array_identity(arrays['M_geo']) == expected['M_geo_sha256']
                and inputs.array_identity(arrays['alpha']) == expected['alpha_sha256'], 'Geometry content differs')
        geometry[gid] = {**arrays, 'metadata': expected}
    require(len(audit['backgrounds']) == 60 and len(audit['old_training']) == 160, 'Source denominator differs')
    expected_paths = set(prep['allowed'])
    require({r['path'] for r in audit['backgrounds'] + audit['old_training']} == expected_paths, 'Decoded source set differs')
    for row in audit['backgrounds'] + audit['old_training']:
        expected = prep['bypath'][row['path']]
        require(row['file_sha256'] == expected['sha256'] and row['original_RGB_sha256'] == expected['rgb_content_sha256']
                and row['bytes'] == expected['bytes'] and row['decode_count'] == 1, 'Saved original source identity differs')
    require(coverage_summary(audit['backgrounds'], list(geometry), audit['old_training'])
            == {**read(OUT / 'coverage.json'), 'formal_render_calls_before_coverage_gate': COUNTERS['render_calls']},
            'Coverage/collision metadata differs')
    require(not read(OUT / 'coverage.json')['working_collisions']
            and read(OUT / 'coverage.json')['defined_backgrounds_at_least1'] >= 40
            and read(OUT / 'coverage.json')['formal_render_calls_before_coverage_gate'] == 0,
            'Coverage admission was not satisfied before generation')
    index = {row['working_RGB_sha256']: row for row in unique}
    require(len(index) == len(unique), 'Unique input dictionary contains duplicate content')
    loaded_paths = set()
    for h, item in index.items():
        require(item['file'] == f'images/{h}.npy', 'Canonical input path differs')
        rgb = np.load(OUT / item['file'], allow_pickle=False)
        require(inputs.working_rgb_identity(rgb) == h, 'Saved float32 RGB content identity differs')
        loaded_paths.add(item['file'])
    backgrounds = {row['id']: row for row in audit['backgrounds']}
    actual_refs = collections.defaultdict(list)
    for frozen, row in zip(prep['roles']['roles']['D_fit60'], audit['backgrounds']):
        require(row['id'] == frozen['id'] and row['path'] == frozen['path'] and row['ordinal'] == frozen['ordinal'], 'Background sequence differs')
        b = np.load(OUT / row['image']['file'], allow_pickle=False)
        require(row['working_RGB_sha256'] == inputs.working_rgb_identity(b)
                == row['image']['working_RGB_sha256'], 'Background input reference differs')
        for gid, geo in geometry.items():
            require(inputs.background_measurement(b, geo) == row['measurements'][gid], 'Saved pre-render Lb conditions differ')
        actual_refs[row['image']['working_RGB_sha256']].append(row['id'])
    require(len(rows) == len(prep['requests']) == 7560, 'Not all nominal requests are present')
    current_id = None
    for frozen, row in zip(prep['requests'], rows):
        require(all(row[k] == v for k, v in frozen.items()) and row['retained'], 'Frozen request order/content differs')
        bg = backgrounds[row['background_id']]
        if current_id != row['background_id']:
            background = np.load(OUT / bg['image']['file'], allow_pickle=False)
            current_id = row['background_id']
        require(row['background_image'] == bg['image'], 'Request background reference differs')
        geo = geometry[row['geometry_id']]
        require(row['geometry_file'] == f'geometry/{row["geometry_id"]}.npz'
                and row['residual_file'] == f'residuals/{row["id"]}.npz', 'Request geometry/residual path differs')
        require(row['image'] == {'file': index[row['image']['working_RGB_sha256']]['file'],
                                'working_RGB_sha256': row['image']['working_RGB_sha256']}, 'Request input reference is unbound')
        image = np.load(OUT / row['image']['file'], allow_pickle=False)
        inputs.validate_rgb(image)
        arrays = load_arrays(row['residual_file'], ('D', 'M_eff'))
        residual = np.max(np.abs(image - background), axis=-1)
        require(arrays['D'].dtype == np.float32 and arrays['D'].shape == (256, 256)
                and arrays['M_eff'].dtype == np.bool_ and arrays['M_eff'].shape == (256, 256), 'Residual type/shape differs')
        require(np.array_equal(arrays['D'], residual)
                and np.array_equal(arrays['M_eff'], residual > inputs.LABEL_FLOOR)
                and inputs.array_identity(residual) == row['D_sha256']
                and inputs.array_identity(arrays['M_eff']) == row['M_eff_sha256'], 'Actual residual/diagnostic mask differs')
        measured = inputs.measure_pair(background, image, geo)
        require(all(row['measurement'][k] == v for k, v in measured.items()), 'Saved actual measurement differs')
        require(row['measurement']['sign'] == row['sign'] and row['measurement']['amplitude'] == row['amplitude']
                and 0 <= row['measurement']['clipped_channel_fraction'] <= 1, 'Sign/amplitude/clipping metadata differs')
        verify_roi_relation(background, image, geo, row['sign'], row['amplitude'],
                            row['measurement']['clipped_channel_fraction'])
        actual_refs[row['image']['working_RGB_sha256']].append(row['id'])
    require(set(actual_refs) == set(index) and all(actual_refs[h] == index[h]['references'] for h in index),
            'Input dedup references are incomplete or reordered')
    summary = aggregate(rows, index)
    require(summary == manifest['summary'], 'Saved complete denominators/summary differ')
    require(sum(len(v) for v in actual_refs.values()) == 7620, 'Baseline plus request reference denominator differs')
    return {**summary, 'backgrounds': 60, 'old_training_hash_only_paths': 160,
            'geometry_arrays_verified': 9, 'all_request_arrays_verified': 7560,
            'source_originals_reopened_for_verification': 0,
            'render_calls_in_cache_verification': 0,
            'independence': '60 fixed backgrounds; variants not independent physical panels'}


def verify_fit(prep=None, source=None):
    prep = preparation() if prep is None else prep
    source = binding(prep) if source is None else source
    resource('cache_verification_start')
    require((OUT / 'complete.json').is_file() and not (OUT / 'failure.json').exists(),
            'Missing complete input transaction; do not generate/resume a partial or unknown directory')
    complete = read(OUT / 'complete.json')
    expected_files = expected_inventory(prep)
    require(set(complete['outputs']) == expected_files, 'Complete manifest adopts unknown or misses expected files')
    check_inventory(expected_files, sealed=True)
    require(read(OUT / 'sources.json') == source, 'Generation implementation/preparation/runtime identity differs')
    for relative, expected in complete['outputs'].items():
        p = OUT / relative
        require(p.resolve().is_relative_to(OUT) and digest(p) == expected, 'Frozen generated file hash differs: ' + relative)
    summary = verify_arrays(prep)
    verified = read(OUT / 'verification.json')
    require(verified['passed'] and verified['summary'] == summary
            and verified['boundary']['render_calls'] == 7560
            and verified['boundary']['fit_decodes'] == 60
            and verified['boundary']['old_training_decodes'] == 160, 'Saved generation evidence incomplete')
    require(COUNTERS['render_calls'] == COUNTERS['original_decodes'] == 0 and not ACCESS,
            'Cache verification attempted original decode or new generation')
    return {'status': 'FIT_INPUTS_CACHE_VERIFIED', 'output': str(OUT), 'summary': summary,
            'frozen_files_verified': len(complete['outputs']), 'current_invocation_boundary': boundary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate-fit', 'verify-fit'))
    args = parser.parse_args()
    result = {'generate-fit': generate_fit, 'verify-fit': verify_fit}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
