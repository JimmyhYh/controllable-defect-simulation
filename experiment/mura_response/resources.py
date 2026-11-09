"""Explicit CPU-only metadata preparation. Never opens real image/array payloads.

Process contract: import/audit guards last for this interpreter's lifetime.
Future generation/scoring processes must not import this module. They run
`python -m mura_response.resources verify` in a separate metadata process,
then consume verified JSON and the pure-array inputs API. Their own modules
bind their own implementation identities; the three preparation files freeze.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent.parent
OUT = ROOT / 'results/mura_response/semu_applicability_v1/preparation_v1'
DESIGN = PROJECT / '计划/Mura检测器响应与SEMU代理刻度研究设计.md'
SEED = 2026102801
COUNTS = {'Q200': 200, 'F100': 100, 'D_fit60': 60, 'D_hold60': 60}
PREFIXES = {'Q200': 'q', 'F100': 'f', 'D_fit60': 'fit', 'D_hold60': 'hold'}
FORBIDDEN = ('torch', 'torchvision', 'timm', 'segmentation_models_pytorch',
             'patchcore', 'controlled_generation', 'pod_probe', 'localization_calibration',
             'localization_training', 'PIL', 'cv2')
BLOCKED_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy', '.npz'}
GUARD = {'blocked_import_attempts': [], 'blocked_payload_attempts': []}
STATIC = {
    'data/weak_localization/split.json': 'feb206cd18eeb9594c64b30afead64577f34473bda57bcaef5a58b3294124ddb',
    'data/controlled_generation/split.json': '765aa3f0be86f36f2c5370b0729d18daaa2b63e16d7df8271c89073d27b2750b',
    'results/weak_localization/dataset_manifest.json': '9d0bf06efd17660e2a0ab5e3e3d45337021b7305490f89f1c32512a4f479a270',
    'results/weak_localization/patchcore_bank.json': '582172a55e87de5e4a924f5d90c489dc910acb477ede199257c06e27112b2c42',
    'results/weak_localization/frozen_config.json': '93c29798fdb301363e97059ac02a5e3a389698794a21cd3b991fb1fdccb35ff9',
    'results/controlled_generation/formal_preparation/common_config.json': '51ad58ea35d76e10d10e545e51b10b91ee56b942e6e3630101ddd77680800353',
    'results/controlled_generation/calibration/calibration_verification.json': '5d1e4bd432b0b1baae41114157b82beaa8b81fe7acaaff39740fbbe8f157fb72',
    'results/reference_generation/v2_domain_matched/scope_protocol.json': '5a9b467f718ad71f60da066711d5faebcb623af2f9c3e8802442545a77044b73',
    'results/pod_probe/audit_20261028_v1/verification.json': '628a73188da9c7e1635b1ce4012811207e68c8bcc93757ac5d52466f964df705',
    'models/weak_localization/patchcore/nnscorer_search_index.faiss': '4e452bc6ac0fdab8bbb2a869ab7009ac98b4c8ac63168295af98e96c5a71b7b1',
    'models/weak_localization/patchcore/patchcore_params.pkl': 'f6ccc3e6dc10816392b80c3de4f00f0cb56f66c2d96f0a256eaa917aa2c86667',
    'models/checkpoints/wide_resnet50_2-95faca4d.pth': '95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8',
    'models/controlled_generation/formal_draem/G1_20261030/reconstruction.pth': 'eb5fd05c44317f36f367ab89d734a380361d9a53df4b69a8c14579613f4edb0a',
    'models/controlled_generation/formal_draem/G1_20261030/segmentation.pth': '6e1b991203df330f1687b88dfd565077650372f0f1e8bc3fa50c9725d7b64fba',
    'controlled_generation/render.py': 'c948a8001f562a31c41404deed164f8ae64617989bc76a13239481b7fac37fb1',
    'controlled_generation/geometry.py': '22d206fa7d28ffef92deef6b507da997d4c6023644e26b5bbc378f59d715217b',
    'localization_calibration.py': 'e63e2ba11e5a70f11f758b535a5c078eb273da1a9a224efc90462dbe8a4f76d2',
    'pod_probe/semu.py': '5b2fcb925d2c923c5ece0f9256a5c0017311571930677a9a9a5784a0d97e0f76',
    'pod_probe/stage2.py': '730746e8b39a93fb98473bbc222e95d4df3cf814db49836a49bc46bec3c84c87',
    'vendor/DRAEM/model_unet.py': 'fc8995f7f26599ea323dca5dfdae83ede3e19bf802cd0b748ecd46bea2beec1c',
    'pod_probe_audit/audit.py': 'f4ca19473983e9a720b3c2ef096274431bdf0cb900c08941084b74fcce95fd8b',
}
EXTRA = (
    'results/controlled_generation/formal_training/G1_20261030/day8_verification.json',
    'results/controlled_generation/formal_training/G1_20261030/training_result.json',
    'models/controlled_generation/formal_draem/G1_20261030/latest.json',
)


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in FORBIDDEN):
            GUARD['blocked_import_attempts'].append(fullname)
            raise RuntimeError('Forbidden model/decoder import during CPU preparation: ' + fullname)
        return None


def payload_guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if p.suffix.lower() in BLOCKED_SUFFIXES:
            GUARD['blocked_payload_attempts'].append(str(p))
            raise RuntimeError('Forbidden image/array payload access during metadata preparation: ' + str(p))


sys.meta_path.insert(0, BlockModels())
sys.addaudithook(payload_guard)


def require(ok, why):
    if not ok:
        raise RuntimeError(why + '; preserve all state; no automatic retry or substitution')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def identity():
    names = ('__init__.py', 'resources.py', 'inputs.py')
    return {str(Path(__file__).with_name(name)): digest(Path(__file__).with_name(name))
            for name in names}


def runtime():
    packages = {n: importlib.metadata.version(n) for n in
                ('numpy', 'scipy', 'Pillow', 'opencv-python', 'scikit-learn', 'torch', 'torchvision')}
    return {'python': platform.python_version(), 'executable': sys.executable,
            'executable_sha256': digest(sys.executable), 'platform': platform.platform(),
            'packages_metadata_only': packages}


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'page_available',
                                        'virtual', 'virtual_available', 'extended')]


def resource():
    state = MemoryStatus(); state.length = ctypes.sizeof(state)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    fn = kernel.GlobalMemoryStatusEx
    fn.argtypes = [ctypes.POINTER(MemoryStatus)]; fn.restype = ctypes.c_int
    require(fn(ctypes.byref(state)) != 0, 'RAM query failed')
    result = {'available_RAM_GiB': state.available / 2 ** 30,
              'total_RAM_GiB': state.total / 2 ** 30,
              'free_disk_GiB': shutil.disk_usage(ROOT).free / 2 ** 30,
              'minimum_RAM_GiB': 2, 'full_round_disk_budget_GiB': 40}
    require(result['available_RAM_GiB'] >= 2 and result['free_disk_GiB'] >= 40,
            'CPU/full-round disk admission failed')
    gpu = shutil.which('nvidia-smi')
    if gpu:
        check = subprocess.run([gpu, '--query-gpu=name,memory.total,memory.free,driver_version',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        result['gpu_query'] = {'returncode': check.returncode, 'output': check.stdout.strip(),
                               'model_package_or_CUDA_initialized': False}
    else:
        result['gpu_query'] = {'available': False, 'not_a_CPU_blocker': True}
    return result


def source_snapshot():
    result = {}
    for relative, expected in STATIC.items():
        path = ROOT / relative
        require(path.is_file() and digest(path) == expected, 'Stage1 source identity changed: ' + relative)
        result[str(path)] = expected
    for path in [ROOT / relative for relative in EXTRA] + [DESIGN]:
        require(path.is_file(), 'Required metadata/design missing: ' + str(path))
        result[str(path)] = digest(path)
    bank = read(ROOT / 'results/weak_localization/patchcore_bank.json')
    require(bank['normal_images'] == 80 and bank['input'] == 256
            and bank['N_only'] and bank['vectors'] == 819, 'PatchCore bank scope differs')
    weights = read(ROOT / EXTRA[1]); verified = read(ROOT / EXTRA[0]); latest = read(ROOT / EXTRA[2])
    require(weights['training_complete'] and weights['steps'] == 10000
            and verified['verification_passed'] and verified['verified_steps'] == 10000
            and latest['step'] == 10000, 'G1 final-weight metadata differs')
    for name, expected in weights['weight_files'].items():
        path = ROOT / 'models/controlled_generation/formal_draem/G1_20261030' / name
        require(result[str(path)] == expected, 'G1 weight does not match training record')
    return result


def pool_and_roles():
    """Only existing JSON identity metadata and path existence; no file payload."""
    import numpy as np
    manifest = read(ROOT / 'results/weak_localization/dataset_manifest.json')
    weak = read(ROOT / 'data/weak_localization/split.json')
    cg = read(ROOT / 'data/controlled_generation/split.json')
    scope = read(ROOT / 'results/reference_generation/v2_domain_matched/scope_protocol.json')
    require(Path(weak['root']) == Path(cg['root']) == ROOT / 'data/weak_localization/AnoLCD',
            'Dataset roots differ')
    bypath = {row['path']: row for row in manifest['files'] if 'rgb_content_sha256' in row}
    protected = {key: weak[key] for key in ('N', 'Vnew', 'Tfinal', 'excluded_train')}
    protected.update({key: cg[key] for key in ('N_gen', 'S_bg', 'Q_cal')})
    protected['reference_review64'] = [row['path'] for row in scope['scope']]
    require(len(protected['reference_review64']) == 64, 'Reference review scope differs')
    excluded = {bypath[path]['rgb_content_sha256'] for paths in protected.values() for path in paths}
    groups = collections.defaultdict(list)
    for path, record in bypath.items():
        if path.startswith('display/train/good/') and record['rgb_content_sha256'] not in excluded:
            groups[record['rgb_content_sha256']].append(path)
    ordered = sorted(groups)
    require(len(groups) == 1629 and sum(map(len, groups.values())) == 1634,
            'Candidate pool differs from Stage1')
    require(len(groups) >= sum(COUNTS.values()), 'Insufficient candidate RGB groups')
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(ordered))
    roles = {}; cursor = 0
    for role, n in COUNTS.items():
        rows = []
        for ordinal, selected in enumerate(order[cursor:cursor + n]):
            group = ordered[int(selected)]; path = min(groups[group]); entry = bypath[path]
            require((Path(weak['root']) / path).is_file(), 'Selected original path is absent: ' + path)
            require(entry['size'] == [1024, 1024], 'Original image dimensions differ from coordinate protocol')
            rows.append({'id': f'{PREFIXES[role]}_b{ordinal:03d}', 'ordinal': ordinal,
                         'path': path, 'rgb_sha256': group, 'file_sha256': entry['sha256'],
                         'bytes': entry['bytes'], 'original_size_hw': [1024, 1024],
                         'original_mode': entry['mode']})
        roles[role] = rows; cursor += n
    all_rows = [row for rows in roles.values() for row in rows]
    require(len(all_rows) == len({row['rgb_sha256'] for row in all_rows}) == 420,
            'Cross-role raw RGB uniqueness failed')
    require(not ({row['rgb_sha256'] for row in all_rows} & excluded), 'Protected RGB role leaked')
    summary = {'candidate_RGB_groups': len(groups), 'candidate_paths': sum(map(len, groups.values())),
               'selected_RGB_groups': len(all_rows), 'counts': COUNTS, 'seed': SEED, 'rng': 'PCG64',
               'role_order': list(COUNTS), 'protected_role_path_counts': {k: len(v) for k, v in protected.items()},
               'raw_RGB_groups_disjoint': True, 'physical_capture_groups_known': False,
               'original_file_content_rechecked': False, 'working_RGB_uniqueness_checked': False,
               'working_RGB_collision_rule': 'check upon each role actual decode; any within/cross-role or old-training collision stops; no replacement',
               'original_RGB_decoded': 0, 'H_image_access': 0}
    return {'dataset_root': weak['root'], 'roles': roles, 'summary': summary}


def protocol(implementation, sources):
    from . import inputs
    return {'schema': 'mura-semu-applicability-preparation-v1', 'approved_date': '2026-10-28',
            'design_path': str(DESIGN), 'design_sha256': sources[str(DESIGN)],
            'stage': 2, 'scope': 'metadata and artificial CPU arrays only',
            'roles': COUNTS, 'seed': SEED, 'input_shape_hwc': [256, 256, 3],
            'original_shape_hw': [1024, 1024], 'preprocessing_future': 'PIL RGB bilinear256 then float32/255; unchanged contrast',
            'coordinates': {'order': 'x,y', 'grid': 'pixel centres, indices0..255',
                            'work_to_original': '(p+0.5)*4-0.5', 'original_to_work': '(p+0.5)/4-0.5'},
            'geometry_specs': inputs.geometry_specs(), 'geometry_order': [s['id'] for s in inputs.geometry_specs()],
            'polarity_order': ['bright', 'dark'], 'amplitudes': inputs.AMPLITUDES,
            'request_order': 'D_fit then D_hold; role background ordinal; geometry; polarity; amplitude',
            'requests_per_background': 126, 'fit_requests': 7560, 'hold_requests': 7560,
            'total_nominal_requests': 15120, 'maximum_model_input_references_two_detectors': 31080,
            'alpha': 'min(d/2,1)*(0.5+0.5*d/max(d)); d=float32 Euclidean distance inside fixed mask',
            'generation': 'float32 I=clip(B+float32(sign*amplitude)*alpha,0,1)',
            'measurement': {'Y': 'float64 accumulation of encoded RGB weighted0.2126/0.7152/0.0722',
                            'support': 'fixed M_geo; never M_eff for area or admission',
                            'Lb': 'mean(Y(B)[M_geo])', 'DeltaY': 'mean((Y(I)-Y(B))[M_geo])',
                            'C_pair_pct': '100*abs(DeltaY)/Lb if Lb>1e-6 else undefined',
                            'proxy': 'C_pair_pct/(1.97/(A_geo*k*k)^0.33+0.72)',
                            'k_primary': .4, 'k_sensitivity': [.2, .8], 'k_is_assumed_not_calibrated': True,
                            'M_eff': 'D>float32(1/255), diagnostic only', 'weak_float_retained': True,
                            'zero_achieved': 'retain; reuse background input by content; never call successful defect detection',
                            'Lb_min_exclusive': inputs.LB_FLOOR},
            'coverage': {'background_defined': 'at least1/9 geometries has defined Lb',
                         'minimum_defined_backgrounds': 40,
                         'check_before_formal_generation_and_forward': True,
                         'report_complete_partial_none_and_per_geometry': True,
                         'no_weak_level_selection_or_replacement': True},
            'future_models': ['PatchCore_frozen_N80', 'G1_final10000'],
            'future_calibration': {'q': .95, 'method': 'linear', 'alarm': 'score>new_tau',
                                   'old_tau_unchanged': True, 'F100_interval': 'Wilson two-sided95%',
                                   'z': 1.959963984540054},
            'future_response': {'primary': 'alarm and first-flat-argmax heatmap in M_geo',
                                'name': 'spatial joint alarm, not causal detection or segmentation accuracy',
                                'paired_baseline_9_geometry_readouts': True},
            'future_statistics': {'candidates': ['C-only', 'C+area', 'SEMU-only', 'C+area+context'],
                                  'constant_reference': 'p0 mean across backgrounds of within-background mean H_joint',
                                  'standardization': 'D_fit only, constant continuous columns ->0',
                                  'ridge_lambda': .01, 'optimizer': 'L-BFGS-B analytic gradient, init0',
                                  'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-12,
                                  'fit_manual_user_start': True, 'detector_training': False,
                                  'primary_metric': 'within-background then across-background mean Brier',
                                  'delta': 'SEMU-only error minus reference error; negative is better',
                                  'log_loss_clip': 1e-7, 'reliability_bins': '10 equal-width; last includes1; empty NA',
                                  'bootstrap_replicates': 2000, 'bootstrap_seed': 2026102802,
                                  'bootstrap_unit': 'hold background; no refit; conditional on frozen fit'},
            'figure_selection': {'fit': 'first fit background, all predetermined geometry definitions',
                                 'hold': 'first in frozen request order for bright/dark circle/horizontal/diagonal ellipse; add first baseline alarm; deduplicate and merge reasons; absent NA',
                                 'no_response_based_best_example_selection': True},
            'runtime': runtime(), 'implementation': implementation,
            'preparation_source_names': ['__init__.py', 'resources.py', 'inputs.py'],
            'future_module_binding': 'new generation/inference/analysis/figures modules bind their own source identities and this complete manifest; never modify preparation sources',
            'process_contract': {'metadata_process_only': 'resources import installs permanent import/file guards',
                                 'future_real_IO_process': 'must not import mura_response.resources; execute its verify CLI in a separate metadata interpreter, then read verified JSON and use inputs pure-array API',
                                 'future_generation_entry': 'separate generation.py; never add IO/CLI to frozen inputs.py'},
            'no_original_images_or_probabilities_read': True, 'no_formal_generation': True,
            'no_models_or_tensor_loads': True, 'no_forward_training_or_calibration': True}


def boundary():
    loaded = [name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in FORBIDDEN)]
    require(not loaded and not any(GUARD.values()), 'CPU import/payload boundary was crossed')
    return {'forbidden_modules_loaded': loaded, **GUARD,
            'real_image_array_payloads_opened': 0, 'model_forwards': 0,
            'model_parameter_updates': 0, 'model_tensor_loads': 0, 'pickle_deserializations': 0}


def verify_cache():
    require((OUT / 'complete.json').is_file(), 'Missing complete preparation; partial/unknown transaction')
    complete = read(OUT / 'complete.json')
    expected_names = set(complete['outputs']) | {'complete.json'}
    require({p.name for p in OUT.iterdir()} == expected_names, 'Preparation directory is not strictly closed')
    for name, expected in complete['outputs'].items():
        require((OUT / name).is_file() and digest(OUT / name) == expected, 'Frozen output changed: ' + name)
    source = read(OUT / 'sources.json')
    require(source['implementation'] == identity(), 'Preparation implementation changed')
    require(source['runtime'] == runtime(), 'Preparation runtime identity changed')
    for path, expected in source['files'].items():
        require(Path(path).is_file() and digest(path) == expected, 'Bound source changed: ' + path)
    require(source['files'] == source_snapshot(), 'Bound source set changed')
    roles = read(OUT / 'roles.json')
    require(roles == pool_and_roles(), 'Frozen roles differ from prescribed metadata/RNG')
    from . import inputs
    require(read(OUT / 'requests.json')['rows'] == inputs.request_dictionary(roles['roles']),
            'Frozen request metadata differs')
    numerical = read(OUT / 'numerical_verification.json')
    require(numerical['passed'] and numerical['artificial_request_count'] == 756
            and len(numerical['rows']) == 756 and numerical['real_RGB_decoded'] == 0,
            'Scientific numerical evidence incomplete')
    return {'status': 'PREPARATION_CACHE_VERIFIED', 'roles': COUNTS, 'nominal_requests': 15120,
            'artificial_numerics_rerun': False, 'real_images_read': False,
            'outputs_verified': len(complete['outputs']), 'boundary': boundary(), 'output': str(OUT)}


def inspect():
    state = resource(); sources = source_snapshot(); roles = pool_and_roles()
    return {'status': 'CPU_METADATA_INSPECTED_NOT_SAVED', 'resource': state,
            'sources_checked': len(sources), 'candidate_and_role_scope': roles['summary'],
            'frozen_roles_written': False, 'boundary': boundary()}


def prepare():
    if OUT.exists():
        return verify_cache()
    state = resource(); before = source_snapshot(); implementation = identity()
    roles = pool_and_roles()
    from . import inputs
    requests = inputs.request_dictionary(roles['roles'])
    geometry = [inputs.make_geometry(spec)['metadata'] for spec in inputs.geometry_specs()]
    numerical = inputs.numerical_audit()
    require(before == source_snapshot() and implementation == identity(), 'Source changed during CPU preparation')
    checks = boundary()
    # Nothing is written until all CPU/scalar checks have completed successfully.
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'started.json', {'stage': 2, 'timestamp_unix': time.time(),
                               'scope': 'metadata/artificial CPU; no formal inputs',
                               'implementation': implementation})
    try:
        write(OUT / 'sources.json', {'files': before, 'implementation': implementation,
                                    'runtime': runtime(), 'identity_scope': 'present observed bytes; no claim of original-run sealing'})
        write(OUT / 'protocol.json', protocol(implementation, before))
        write(OUT / 'roles.json', roles)
        write(OUT / 'requests.json', {'nominal_fit': 7560, 'nominal_hold': 7560,
                                     'actual_formal_generated': 0, 'rows': requests})
        write(OUT / 'geometry.json', {'count': len(geometry), 'rows': geometry,
                                     'actual_masks_have_not_been_applied_to_real_images': True})
        write(OUT / 'numerical_verification.json', numerical)
        require(before == source_snapshot() and implementation == identity(), 'Source changed before sealing')
        write(OUT / 'verification.json', {'passed': True, 'source_files_checked': len(before),
                                          'role_counts': COUNTS, 'candidate_RGB_groups': 1629,
                                          'raw_RGB_groups_disjoint': True, 'working_RGB_collision_check': 'future actual decode',
                                          'request_count': len(requests), 'geometry_count': len(geometry),
                                          'numerical_cases': len(numerical['rows']), 'resource': state,
                                          'boundary': checks, 'no_model_compatibility_or_effectiveness_claim': True,
                                          'no_formal_input_generation': True})
        write(OUT / 'complete.json', {'schema': 'mura-preparation-complete-v1',
                                     'outputs': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
    except Exception as exc:
        if not (OUT / 'failure.json').exists():
            write(OUT / 'failure.json', {'error': type(exc).__name__, 'message': str(exc),
                                       'no_automatic_retry': True})
        raise
    result = verify_cache()
    result['status'] = 'CPU_PREPARATION_COMPLETE_AND_VERIFIED'
    result['numerical_summary'] = {key: value for key, value in numerical.items()
                                   if key not in ('rows', 'zero_injection_spatial_offset_checks', 'Lb_boundary_checks')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'prepare', 'verify'))
    args = parser.parse_args()
    result = {'inspect': inspect, 'prepare': prepare, 'verify': verify_cache}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
