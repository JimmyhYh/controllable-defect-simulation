"""Two fixed D_fit inputs, original PatchCore runtime, no training or calibration.

prepare is stdlib-only; run owns one immutable transaction; verify never loads
models or source RGB. Normal dependency cold initialization is explicitly allowed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/mura_response/semu_applicability_v1'
FIT = BASE / 'fit_inputs_identity_v2'
NORMAL = BASE / 'normal_scoring_v1/PatchCore'
OUT = ROOT / 'results/mura_capability/week1_v1/patchcore_minimal_runtime_v1'
PREP = OUT / 'preparation'
RUN = OUT / 'run'
TAU = 1.852951288223265
NORMAL_COMPLETE = '2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec'
FIT_COMPLETE = '3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593'
COUNTERS = {'model_loads': 0, 'tensor_loads': 0,
            'author_placeholder_forwards': 0, 'model_input_forwards': 0}
EVENTS = {'source_RGB_reads': [], 'temporary_write_opens': [],
          'NUL_opens': 0, 'system_queries': [], 'blocked': []}


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no automatic retry')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')


def safe_file(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and not p.is_junction(),
            'Missing or redirected file: ' + str(p))
    require(p.absolute() == p.resolve(), 'Redirected parent path: ' + str(p))
    return p


def closed(folder, names):
    require(folder.is_dir() and not folder.is_symlink() and not folder.is_junction(),
            'Missing or redirected cache')
    files = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(files == set(names) | {'complete.json'}, 'Incomplete or unknown cache members')
    require(not any(p.is_symlink() or p.is_junction() for p in folder.rglob('*')),
            'Cache contains redirected member')
    require(not any(p.is_dir() for p in folder.iterdir()), 'Unexpected cache subdirectory')
    outputs = read(folder / 'complete.json')['outputs']
    require(set(outputs) == set(names), 'Complete manifest coverage differs')
    require(all(digest(safe_file(folder / n)) == h for n, h in outputs.items()),
            'Cache hash differs')


def seal(folder):
    outputs = {p.relative_to(folder).as_posix(): digest(p)
               for p in sorted(folder.rglob('*')) if p.is_file()}
    require('complete.json' not in outputs and not {'failure.json', 'inflight.json'} & set(outputs),
            'Cannot seal failed or ambiguous transaction')
    write(folder / 'complete.json', {'outputs': outputs})


def resources(model=False):
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'page_available',
                                            'virtual', 'virtual_available', 'extended')]
    state = Memory(); state.length = ctypes.sizeof(state)
    query = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(Memory)]; query.restype = ctypes.c_int
    require(query(ctypes.byref(state)), 'RAM query failed')
    result = {'available_RAM_GiB': state.available / 2**30,
              'free_D_GiB': shutil.disk_usage(ROOT).free / 2**30}
    require(result['available_RAM_GiB'] >= (4.5 if model else 2)
            and result['free_D_GiB'] >= 40, 'RAM/disk admission not met')
    return result


def select_pair(roles, requests, rows):
    require(len(roles) == 60 and len(rows) == len(requests) == 7560,
            'Fixed D_fit coverage differs')
    require([r['id'] for r in requests] == [r['id'] for r in rows], 'Request order differs')
    first = roles[0]['id']
    candidates = [r for r in rows if r['background_id'] == first]
    require(candidates, 'First background has no requests')
    chosen = next((r for r in candidates if r['measurement']['zero_actual_change'] is False), None)
    require(chosen is not None, 'No nonzero request for first background')
    for row in candidates:
        require(row['background_image'] == chosen['background_image'], 'Baseline reference differs')
    frozen = requests[[r['id'] for r in requests].index(chosen['id'])]
    require(all(frozen[k] == chosen[k] for k in ('id', 'background_id', 'geometry_id',
                                               'polarity', 'amplitude')), 'Chosen request differs')
    return [dict(id=first, kind='baseline', **chosen['background_image']),
            dict(id=chosen['id'], kind='perturbation', **chosen['image'])], chosen


def binding():
    frozen = read(BASE / 'fit_scoring_preparation_v1/sources.json')
    require(digest(FIT / 'complete.json') == FIT_COMPLETE
            and digest(NORMAL / 'complete.json') == NORMAL_COMPLETE,
            'Frozen input/normal complete differs')
    hashes = {str(Path(__file__).resolve()): digest(__file__)}
    def bind(path, expected=None):
        p = safe_file(path); h = digest(p)
        require(expected is None or expected == h, 'Source identity differs: ' + str(p))
        hashes[str(p)] = h
    for folder in (BASE / 'fit_scoring_preparation_v1', BASE / 'normal_preparation_v1'):
        record = read(folder / 'complete.json')
        bind(folder / 'complete.json')
        for name, h in record['outputs'].items():
            require(name.endswith('.json') and Path(name).name == name, 'Unexpected preparation member')
            bind(folder / name, h)
    cfg = read(BASE / 'normal_preparation_v1/protocol.json')
    for path, h in frozen['frozen_implementation'].items():
        bind(path, h)
    for path, h in cfg['sources']['author_and_evidence'].items():
        if 'vendor\\patchcore-inspection\\' in path:
            bind(path, h)
    for path, h in cfg['sources']['assets'].items():
        if 'formal_draem' not in path:
            bind(path, h)  # Opaque hash only for params.pkl; never deserialize it.
    runtime = cfg['sources']['runtime']
    bind(sys.executable, runtime['executable_sha256'])
    require(Path(sys.executable).resolve() == Path(runtime['executable']).resolve()
            and sys.version.split()[0] == runtime['python'], 'Interpreter differs')
    versions = {name: importlib.metadata.version(name)
                for name in runtime['packages_metadata_only']}
    require(versions == runtime['packages_metadata_only'], 'Installed dependency versions differ')
    fc = read(FIT / 'complete.json')
    for name, h in fc['outputs'].items():
        if name.endswith('.json'):
            bind(FIT / name, h)
    bind(FIT / 'complete.json', FIT_COMPLETE)
    require(read(FIT / 'verification.json')['passed'], 'Fit independent input proof failed')
    for p in (BASE / 'preparation_v1/roles.json', BASE / 'preparation_v1/requests.json',
              ROOT / 'mura_response/inputs.py', ROOT / 'mura_response/fit_inference.py'):
        bind(p)
    roles = read(BASE / 'preparation_v1/roles.json')['roles']['D_fit60']
    requests = [r for r in read(BASE / 'preparation_v1/requests.json')['rows'] if r['role'] == 'D_fit60']
    pair, chosen = select_pair(roles, requests, read(FIT / 'manifest.json')['rows'])
    for row in pair:
        name = row['file']
        require(name == 'images/' + row['working_RGB_sha256'] + '.npy', 'RGB path/domain differs')
        row['file_sha256'] = fc['outputs'][name]  # Do not open RGB during prepare.
        safe_file(FIT / name)
    nc = read(NORMAL / 'complete.json')
    for name in ('q_result.json', 'verification.json', 'result.json'):
        bind(NORMAL / name, nc['outputs'][name])
    bind(NORMAL / 'complete.json', NORMAL_COMPLETE)
    q = read(NORMAL / 'q_result.json'); v = read(NORMAL / 'verification.json')
    require(q['threshold'] == TAU and q['q'] == .95 and q['method'] == 'linear'
            and q['rule'] == 'strictly greater' and v['passed']
            and v['state_before'] == v['state_after'], 'Normal workpoint state/tau differs')
    # Preserve the existing failure and both import diagnostic records byte-for-byte.
    old_roots = [BASE / 'fit_scores_v1/PatchCore'] + [OUT.parent / n for n in (
        'import_diagnosis_v1', 'import_diagnosis_launch_v1',
        'import_diagnosis_result_verification_v1', 'day1_admission_v1')]
    inventories = {}
    for folder in old_roots:
        require(folder.is_dir(), 'Historical evidence directory missing')
        names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file())
        inventories[str(folder)] = names
        for name in names:
            require(Path(name).suffix.lower() in ('.json', '.txt', '.log'),
                    'Do not open historical score arrays')
            bind(folder / name)
    return {'files': hashes, 'old_inventories': inventories, 'versions': versions,
            'pair': pair, 'request': chosen, 'normal_state': v['state_before'],
            'tau': TAU, 'authorization': '2026-11-04 normal cold initialization allowed'}


def prepare():
    sources = binding()
    if PREP.exists():
        return verify_preparation()
    require(not OUT.exists(), 'Unknown output root; do not adopt')
    snapshot = resources()
    PREP.mkdir(parents=True, exist_ok=False)
    write(PREP / 'sources.json', sources)
    write(PREP / 'protocol.json', {'model': 'PatchCore', 'inputs': sources['pair'],
        'input_forwards': 2, 'author_ones_probe': 1, 'tau': TAU, 'score_rule': 'strictly greater',
        'runtime': 'original explicit WRN; eval/FP32/batch1/workers0; no C01 rewrite',
        'rescale_cached_float': False, 'partial_transaction': 'preserve and stop; never retry',
        'training_calibration_generation_fitting': False, 'resources_at_prepare': snapshot})
    seal(PREP)
    return {'status': 'MINIMAL_PREPARATION_COMPLETE', 'inputs': sources['pair'], 'resources': snapshot}


def verify_preparation():
    require({p.name for p in OUT.iterdir()} <= {'preparation', 'run'}, 'Unknown output member')
    closed(PREP, {'sources.json', 'protocol.json'})
    sources = read(PREP / 'sources.json')
    require(binding() == sources, 'Bound sources changed; preserve and stop')
    cfg = read(PREP / 'protocol.json')
    require(cfg['inputs'] == sources['pair'] and cfg['input_forwards'] == 2
            and cfg['author_ones_probe'] == 1 and cfg['tau'] == TAU
            and not cfg['rescale_cached_float'] and not cfg['training_calibration_generation_fitting'],
            'Minimal protocol differs')
    return {'status': 'MINIMAL_PREPARATION_VERIFIED', 'source_arrays_read': 0,
            'complete_sha256': digest(PREP / 'complete.json')}


def install_boundary(pair):
    allowed_arrays = {(FIT / r['file']).resolve() for r in pair}
    assets = {p.resolve() for p in (ROOT / 'models/checkpoints/wide_resnet50_2-95faca4d.pth',
              ROOT / 'models/weak_localization/patchcore/nnscorer_search_index.faiss',
              ROOT / 'models/weak_localization/patchcore/patchcore_params.pkl')}
    temp_roots = {Path(v).resolve() for k in ('TEMP', 'TMP') if (v := os.environ.get(k))}
    platform_source = (Path(sys.base_prefix) / 'Lib/platform.py').resolve()
    def deny(reason):
        EVENTS['blocked'].append(reason); require(False, reason)
    def own(p):
        return p.is_relative_to(RUN.resolve())
    def temporary(p):
        return any(p.is_relative_to(t) for t in temp_roots)
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'pickle.find_class'):
            deny('No network or unrestricted pickle: ' + event)
        if event == 'subprocess.Popen':
            f = sys._getframe(1); approved = False
            while f is not None:
                if (f.f_code.co_name == '_syscmd_ver'
                        and Path(f.f_code.co_filename).resolve() == platform_source
                        and f.f_locals.get('cmd') in ('ver', 'command /c ver', 'cmd /c ver')):
                    approved = True; break
                f = f.f_back
            if not approved:
                deny('Unlisted system subprocess: ' + str(args[:2]))
            EVENTS['system_queries'].append({'executable': str(args[0]), 'command': str(args[1])})
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename'):
            paths = args[:2] if event == 'os.rename' else args[:1]
            for value in paths:
                if isinstance(value, (str, bytes)):
                    p = Path(os.fsdecode(value)).resolve()
                    if not own(p) and not temporary(p):
                        deny('Filesystem mutation outside own result/temp: ' + str(p))
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        raw = os.fsdecode(args[0])
        if raw.upper() in ('NUL', '\\\\.\\NUL', '\\\\?\\NUL'):
            EVENTS['NUL_opens'] += 1; return
        p = Path(raw).resolve(); ext = p.suffix.lower()
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(ch in mode for ch in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if not own(p) and not temporary(p):
                deny('Write outside own result/normal temporary root: ' + str(p))
            if temporary(p):
                EVENTS['temporary_write_opens'].append(str(p))
            return
        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            deny('No original RGB or GT reads: ' + str(p))
        if ext in ('.npy', '.npz') and not own(p):
            if p not in allowed_arrays:
                deny('Unlisted array read: ' + str(p))
            EVENTS['source_RGB_reads'].append(str(p))
        if ext in ('.pth', '.pt', '.pkl', '.pickle', '.faiss', '.safetensors'):
            if p not in assets:
                deny('Unlisted tensor/pickle asset: ' + str(p))
    sys.addaudithook(audit)


def rgb_identity(rgb):
    header = f'{rgb.dtype.str}|{",".join(map(str, rgb.shape))}|C\n'.encode('ascii')
    return hashlib.sha256(header + rgb.tobytes(order='C')).hexdigest()


def run():
    verify_preparation()
    if RUN.exists():
        return verify()
    sources = read(PREP / 'sources.json'); snapshot = resources(model=True)
    require(not any(COUNTERS.values()), 'Fresh process required')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'started.json', {'preparation_sha256': digest(PREP / 'complete.json'),
          'pid': os.getpid(), 'resources': snapshot, 'temporary_environment': {
              k: os.environ.get(k) for k in ('TEMP', 'TMP', 'TMPDIR')}})
    install_boundary(sources['pair'])
    started = time.perf_counter(); rows = []; current = 'model_construction'
    try:
        write(RUN / 'inflight.json', {'operation': current})
        import numpy as np
        from mura_response import inference_runtime as native
        context = SimpleNamespace(ROOT=ROOT, COUNTERS=COUNTERS, require=require)
        torch, modules, handle = native.build_model(context, 'PatchCore', {})
        def state():
            return json.loads(json.dumps(native.state_signature(torch, modules), allow_nan=False))
        initial = state()
        require(initial == sources['normal_state'], 'Constructed state/BN differs from frozen workpoint')
        (RUN / 'inflight.json').unlink()
        model_finished = time.perf_counter()
        for row in sources['pair']:
            current = row['id']; resources(model=True)
            payload = (FIT / row['file']).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == row['file_sha256'], 'RGB file hash differs')
            rgb = np.load(io.BytesIO(payload), allow_pickle=False)
            require(rgb.dtype == np.float32 and rgb.shape == (256, 256, 3)
                    and rgb.flags.c_contiguous and np.isfinite(rgb).all()
                    and 0 <= float(rgb.min()) <= float(rgb.max()) <= 1
                    and rgb_identity(rgb) == row['working_RGB_sha256'], 'RGB domain/hash differs')
            write(RUN / 'inflight.json', {'operation': 'input_forward', 'id': current})
            COUNTERS['model_input_forwards'] += 1
            score, heat = native.predict_rgb(torch, 'PatchCore', handle, rgb)
            torch.cuda.synchronize()
            require(math.isfinite(score) and heat.dtype == np.float32
                    and heat.shape == (256, 256) and np.isfinite(heat).all(), 'Nonfinite/wrong model output')
            require(state() == initial, 'Parameters/BN/training/gradient state changed')
            target = RUN / (current + '.npy')
            with target.open('xb') as stream:
                np.save(stream, heat, allow_pickle=False)
            rows.append({**row, 'score': score, 'tau': TAU, 'above_tau': score > TAU,
                         'heatmap': target.name, 'heatmap_sha256': digest(target),
                         'heat_min': float(heat.min()), 'heat_max': float(heat.max()),
                         'state_unchanged': True})
            (RUN / 'inflight.json').unlink()
            print(json.dumps({'id': current, 'score': score, 'tau': TAU, 'above_tau': score > TAU}), flush=True)
        require(COUNTERS == {'model_loads': 1, 'tensor_loads': 1,
                'author_placeholder_forwards': 1, 'model_input_forwards': 2}, 'Actual call counts differ')
        require(len(EVENTS['source_RGB_reads']) == 2 and not EVENTS['blocked'], 'Input boundary differs')
        write(RUN / 'state.json', {'before': initial, 'after': state()})
        write(RUN / 'result.json', {'status': 'MINIMAL_TWO_INPUTS_COMPLETE', 'rows': rows,
              'counters': COUNTERS, 'events': EVENTS, 'original_RGB_GT_reads': 0,
              'construction_seconds': model_finished-started, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
              'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
              'GPU': torch.cuda.get_device_name(), 'resources_after': resources(),
              'no_training_generation_calibration_fitting': True,
              'not_detection_effectiveness_or_original_temp_caller_proof': True})
        # Recheck opaque frozen sources after inference, without reopening either input.
        require(binding() == sources, 'Frozen provenance changed during run')
        seal(RUN)
    except BaseException as exc:
        if not (RUN / 'failure.json').exists():
            write(RUN / 'failure.json', {'error': type(exc).__name__, 'message': str(exc),
                  'traceback': traceback.format_exc(), 'current': current, 'counters': COUNTERS,
                  'events': EVENTS, 'completed': rows, 'no_automatic_retry': True})
        raise
    return {'status': 'MINIMAL_TWO_INPUTS_COMPLETE', 'output': str(RUN), 'rows': rows, 'counters': COUNTERS}


def verify():
    verify_preparation()
    sources = read(PREP / 'sources.json')
    expected = {'started.json', 'result.json', 'state.json'} | {r['id']+'.npy' for r in sources['pair']}
    closed(RUN, expected)
    import numpy as np
    result = read(RUN / 'result.json'); states = read(RUN / 'state.json')
    require(read(RUN / 'started.json')['preparation_sha256'] == digest(PREP / 'complete.json'),
            'Saved run preparation identity differs')
    require(result['status'] == 'MINIMAL_TWO_INPUTS_COMPLETE'
            and states['before'] == states['after'] == sources['normal_state']
            and result['counters'] == {'model_loads': 1, 'tensor_loads': 1,
                'author_placeholder_forwards': 1, 'model_input_forwards': 2}, 'Saved state/count differs')
    require(len(result['rows']) == 2 and not result['events']['blocked']
            and result['events']['source_RGB_reads'] == [str((FIT/r['file']).resolve()) for r in sources['pair']]
            and result['original_RGB_GT_reads'] == 0
            and result['no_training_generation_calibration_fitting'], 'Saved boundary differs')
    for saved, frozen in zip(result['rows'], sources['pair']):
        require(all(saved[k] == v for k, v in frozen.items()) and saved['state_unchanged']
                and saved['tau'] == TAU and math.isfinite(saved['score'])
                and saved['above_tau'] == (saved['score'] > TAU), 'Saved score/identity differs')
        p = safe_file(RUN / saved['heatmap'])
        require(p.name == frozen['id']+'.npy' and digest(p) == saved['heatmap_sha256'], 'Heatmap hash differs')
        heat = np.load(p, allow_pickle=False)
        require(heat.dtype == np.float32 and heat.shape == (256, 256) and np.isfinite(heat).all()
                and float(heat.min()) == saved['heat_min'] and float(heat.max()) == saved['heat_max'],
                'Saved heatmap domain differs')
    require(not any(COUNTERS.values()) and 'torch' not in sys.modules,
            'Independent verification must not build/score a model')
    return {'status': 'MINIMAL_COMPLETE_CACHE_VERIFIED', 'rows': result['rows'],
            'complete_sha256': digest(RUN / 'complete.json'), 'new_model_forwards': 0,
            'new_source_RGB_reads': 0, 'tau_recomputed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify-preparation', 'run', 'verify'))
    args = parser.parse_args()
    functions = {'prepare': prepare, 'verify-preparation': verify_preparation, 'run': run, 'verify': verify}
    print(json.dumps(functions[args.command](), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
