"""Independent, explicitly invoked single-model continuation; old failure stays intact."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from . import common as c
from . import normal_calibration as old

CONTROL = c.RESULTS / 'normal_calibration_continuation_preparation_v1'
OUTPUT = c.RESULTS / 'normal_calibration_continuation_v1'
ASSEMBLY = c.RESULTS / 'normal_calibration_assembled_v1'
DIAGNOSIS = c.RESULTS / 'normal_calibration_resource_diagnosis_v1'
PENDING = ('A_2026101905', 'B_2026101905')
TOKEN = '92f628776179462a8b558327faa0f999'
ERROR = "RuntimeError('Only 4.441GiB RAM; need 4.5; no generation/update started')"
_RUN_CALLED = False


def install_guard(cpu=True, q_paths=(), writes=()):
    """Process-local audit boundary; not a replacement for an OS sandbox."""
    forbidden = ('torch', 'torchvision', 'transformers', 'diffusers')
    if cpu:
        c.require(not any(n.split('.')[0] in forbidden for n in sys.modules), 'Fresh CPU-only process required')
    allowed = {Path(p).resolve() for p in q_paths}
    destinations = tuple(Path(p).resolve() for p in writes)
    def audit(event, args):
        if cpu and event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('No model imports during CPU continuation preparation/verification')
        if event in ('socket.connect', 'subprocess.Popen') or (cpu and event == 'pickle.find_class'):
            raise RuntimeError('No network, subprocess or CPU pickle execution')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or '', args[2] or 0
            writing = any(x in str(mode) for x in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in destinations), 'Write outside independent destinations: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'data'):
                c.require(path == (c.DATA / 'split.json').resolve() or (not cpu and path in allowed),
                          'Original dataset image/GT read forbidden: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'results'):
                c.require(path == c.d.AUDIT or path.is_relative_to(c.RESULTS), 'Old result read forbidden')
    sys.addaudithook(audit)


def implementations():
    return {name: c.sha(Path(__file__).with_name(name)) for name in
            ('normal_calibration_continuation.py', 'normal_calibration_assembly.py')}


def header(path, request):
    """Read NPY header only; hashing elsewhere never parses float payloads."""
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        c.require(version in ((1, 0), (2, 0)), 'Unexpected probability NPY format')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=10000)
        c.require(shape == (request['height'], request['width']) and dtype == np.dtype('float32')
                  and not dtype.hasobject and not fortran
                  and path.stat().st_size == stream.tell() + request['height'] * request['width'] * 4,
                  'Native float32 probability header/size differs')


def verify_cache(folder, row, requests, expected_binding, *, payload=False, independent=False):
    sealed = c.verify_manifest(folder)
    c.require(sealed['binding'] == expected_binding and sealed['model'] == row, 'Cache source identity differs')
    expected = {'scores.json', 'summary.json'} | {f"probabilities/{r['id']}.npy" for r in requests}
    if independent:
        expected |= {'transaction.json', 'cursor.json'}
        txn, cursor = c.read_json(folder / 'transaction.json'), c.read_json(folder / 'cursor.json')
        c.require(txn['binding'] == expected_binding and txn['model'] == row
                  and cursor == {'token': txn['token'], 'phase': 'complete', 'model': row['label']},
                  'Independent model transaction incomplete')
    c.require(set(sealed['files']) == expected, 'Unknown/missing cache files')
    scores, summary = c.read_json(folder / 'scores.json'), c.read_json(folder / 'summary.json')
    c.require(len(scores) == len(requests) == 200 and summary['model'] == row
              and summary['window_forwards'] == 844
              and summary['parameter_hash_before'] == summary['parameter_hash_after'] == row['final_parameter_hash_recorded']
              and summary['buffer_count'] == 0, 'Model/denominator/parameter record differs')
    cal = summary['calibration']
    c.require(cal['q'] == .95 and cal['method'] == 'linear' and cal['comparison'] == 'strict >'
              and cal['normal_denominator'] == 200 and cal['same_pool_not_independent_validation'] is True
              and type(cal['threshold']) is float and np.isfinite(cal['threshold']) and 0 <= cal['threshold'] <= 1
              and type(cal['FP']) is int and type(cal['TN']) is int and 0 <= cal['FP'] <= 200
              and cal['FP'] + cal['TN'] == 200, 'Saved calibration rule/count differs')
    for request, score in zip(requests, scores):
        name = f"probabilities/{request['id']}.npy"
        c.require(score['request'] == request and score['window_forwards'] == len(request['window_starts'])
                  and score['probability_sha256'] == sealed['files'][name]
                  and type(score['score']) is float and np.isfinite(score['score']) and 0 <= score['score'] <= 1,
                  'Request order/source/max record differs')
        header(folder / name, request)
        if payload:
            probability = np.load(folder / name, allow_pickle=False)
            c.require(np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
                      and float(probability.max()) == score['score'], 'Cached numeric probability/max differs')
    if payload:
        c.require(sum(s['score'] > cal['threshold'] for s in scores) == cal['FP'], 'FP at saved threshold differs')
    # Reference original files: never duplicate a threshold or map in a new editable configuration.
    return {'label': row['label'], 'folder': folder.relative_to(c.ROOT).as_posix(),
            'complete_sha256': c.sha(folder / 'complete.json'), 'files': sealed['files'],
            'checkpoint': row['checkpoint'], 'checkpoint_sha256': row['checkpoint_sha256'],
            'summary': (folder / 'summary.json').relative_to(c.ROOT).as_posix(),
            'scores': (folder / 'scores.json').relative_to(c.ROOT).as_posix()}


def source_context():
    """No permanent legacy guards and no legacy whole-run recovery/verification."""
    identity, requests, inherited = old.load_prepared()
    c.verify_manifest(DIAGNOSIS)
    diagnosis = c.read_json(DIAGNOSIS / 'diagnosis.json')
    c.require(c.sha(old.CONTROL / 'complete.json') == diagnosis['preparation_complete_sha256']
              and c.sha(old.__file__) == diagnosis['frozen_implementation_sha256'], 'Original preparation differs')
    root = old.OUTPUT
    prefix = identity['models'][:4]
    c.require({p.name for p in root.iterdir()} == {r['label'] for r in prefix} |
              {'transaction.json', 'cursor.json', 'failure.json'}, 'Failure boundary namespace differs')
    hashes = {name: c.sha(root / name) for name in ('transaction.json', 'cursor.json', 'failure.json')}
    c.require(hashes == diagnosis['source_control_files'], 'Failure boundary bytes changed')
    txn, cursor, failure = (c.read_json(root / name) for name in ('transaction.json', 'cursor.json', 'failure.json'))
    c.require(txn['binding'] == old.binding(identity) and txn['token'] == TOKEN
              and cursor == {'token': TOKEN, 'phase': 'image_saved', 'model': 'B_2026101904', 'image': '12305', 'y': None}
              and failure['token'] == TOKEN and failure['cursor'] == cursor and failure['error'] == ERROR
              and requests[-1]['id'] == '12305', 'Only the documented pre-A05 resource stop is eligible')
    c.require([r['label'] for r in identity['models'][4:]] == list(PENDING), 'Pending formal models differ')
    refs = []
    for row, recorded in zip(prefix, diagnosis['verified_completed_prefix']):
        ref = verify_cache(root / row['label'], row, requests, old.binding(identity), payload=False)
        c.require(recorded['model'] == row['label'] and recorded['source_complete_sha256'] == ref['complete_sha256'],
                  'Completed prefix differs from diagnosis')
        refs.append(ref)
    c.require(len(refs) == len(diagnosis['verified_completed_prefix']) == 4
              and not any((root / name).exists() for name in PENDING), 'Uncertain/already started original target')
    manifest = {'original_preparation_complete_sha256': c.sha(old.CONTROL / 'complete.json'),
                'original_protocol_sha256': c.sha(old.CONTROL / 'protocol.json'),
                'original_requests_sha256': c.sha(old.CONTROL / 'Q_requests.json'),
                'original_implementation_sha256': c.sha(old.__file__),
                'diagnosis_complete_sha256': c.sha(DIAGNOSIS / 'complete.json'),
                'failure_token': TOKEN, 'failure_files': hashes, 'prefix': refs,
                'inherited_identity_digest': c.digest_json(identity), 'new_implementation': implementations()}
    return identity, requests, inherited, manifest


def protocol(identity, requests, inherited, manifest):
    c.require(sum(len(r['window_starts']) for r in requests) == 844, 'Window denominator differs')
    return {'schema': 'normal-calibration-continuation-v1', 'source_manifest_digest': c.digest_json(manifest),
            'original_identity_digest': c.digest_json(identity), 'implementation': implementations(),
            'inherited_protocol': inherited, 'pending': list(PENDING),
            'original_prefix_reference_only': True, 'real_Q_images_per_model': 200,
            'new_image_evaluations': 400, 'new_window_forwards': 1688, 'unique_normal_images': 200,
            'runtime_root': str(OUTPUT), 'assembly_root': str(ASSEMBLY),
            'one_model_per_process': True, 'partial_transaction': 'preserve; never auto resume/repeat',
            'old_whole_failure_modified': False, 'automatic_training': False}


def prepared_context():
    context = source_context()
    identity, requests, inherited, manifest = context
    sealed = c.verify_manifest(CONTROL)
    c.require(set(sealed['files']) == {'protocol.json', 'Q_requests.json', 'source_manifest.json', 'CPU_validation.json'}
              and c.read_json(CONTROL / 'protocol.json') == protocol(*context)
              and c.read_json(CONTROL / 'Q_requests.json') == requests
              and c.read_json(CONTROL / 'source_manifest.json') == manifest
              and sealed['implementation'] == implementations(), 'Independent preparation identity differs')
    return context


def prepare():
    install_guard(writes=(CONTROL,))
    snapshot = c.resource(2, old.MIN_DISK)
    context = source_context()
    if (CONTROL / 'complete.json').exists():
        return verify_preparation(context)
    c.require(not CONTROL.exists() and not OUTPUT.exists() and not ASSEMBLY.exists(),
              'Unknown/partial independent destination; preserve and diagnose')
    validation = old.scalar_validation(context[1])
    CONTROL.mkdir()
    for name, value in [('protocol.json', protocol(*context)), ('Q_requests.json', context[1]),
                        ('source_manifest.json', context[3]),
                        ('CPU_validation.json', {'resources': snapshot, 'prefix_NPY_headers_verified': 800,
                                                 'stored_score_metadata_records_verified': 800,
                                                 'real_probability_payloads_parsed': 0,
                                                 'real_Q_images_or_GT_read': 0, 'actual_thresholds_computed': 0,
                                                 'new_model_operations': 0, 'Torch_imported': 'torch' in sys.modules,
                                                 'invented_scalar_validation': validation})]:
        c.save(CONTROL / name, value)
    c.seal(CONTROL, {'implementation': implementations()})
    return verify_preparation(context)


def verify_preparation(context=None):
    if context is None:
        install_guard()
    # Rebuild identity, source hashes and headers even for complete preparation cache reuse.
    prepared_context()
    return {'prepared': True, 'prefix_models': 4, 'prefix_NPY_headers': 800, 'pending': list(PENDING),
            'real_probability_payloads_parsed': 0, 'real_Q_images_or_GT_read': 0,
            'actual_thresholds_computed': 0, 'new_model_operations': 0, 'Torch_imported': 'torch' in sys.modules}


def binding():
    return {'preparation_complete_sha256': c.sha(CONTROL / 'complete.json'),
            'protocol_sha256': c.sha(CONTROL / 'protocol.json'), 'implementation': implementations()}


def validate_root():
    if OUTPUT.exists():
        c.require({p.name for p in OUTPUT.iterdir()} <= {'binding.json', *PENDING}
                  and c.read_json(OUTPUT / 'binding.json') == binding(), 'Independent runtime root differs')


def verify_model_internal(model_label, context, payload=True):
    c.require(model_label in PENDING, 'Only never-started A05/B05 targets allowed')
    validate_root()
    row = next(r for r in context[0]['models'] if r['label'] == model_label)
    folder = OUTPUT / model_label
    c.require((folder / 'complete.json').exists(), 'Missing/partial target; no automatic repeat or resume')
    ref = verify_cache(folder, row, context[1], binding(), payload=payload, independent=True)
    if model_label == PENDING[1]:
        verify_model_internal(PENDING[0], context, payload=payload)
    return ref


def verify_model(model_label):
    install_guard()
    ref = verify_model_internal(model_label, prepared_context())
    return {'verified': True, 'model': model_label, 'source_reference': ref, 'new_model_operations': 0,
            'thresholds_recomputed': 0, 'Q_images_or_GT_read': 0}


def cursor_write(folder, value):
    temporary = folder / 'cursor.json.tmp'
    c.require(not temporary.exists(), 'Unknown cursor transaction; preserve')
    c.save(temporary, value)
    os.replace(temporary, folder / 'cursor.json')


def run_model(model_label):
    global _RUN_CALLED
    c.require(model_label in PENDING and not _RUN_CALLED, 'One fixed model per fresh Python process')
    _RUN_CALLED = True
    requests = c.read_json(CONTROL / 'Q_requests.json')
    install_guard(cpu=False, q_paths=[c.DATA / r['image'] for r in requests], writes=(OUTPUT,))
    context = prepared_context()
    identity, requests, _, _ = context
    validate_root()
    folder = OUTPUT / model_label
    if (folder / 'complete.json').exists():
        return {'complete_cache_reused': True, 'source_reference': verify_model_internal(model_label, context),
                'new_model_operations': 0}
    c.require(not folder.exists(), 'Partial/unknown target; preserve, do not repeat any forward')
    if model_label == PENDING[1]:
        verify_model_internal(PENDING[0], context)
    else:
        c.require(not OUTPUT.exists(), 'Unfinished/orphan independent root; preserve and diagnose')
    before_import = c.resource(old.START_RAM, old.MIN_DISK)
    import torch
    from .model import make_model, state_hash
    c.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    startup = c.resource(old.START_RAM, old.MIN_DISK)
    free_gpu = torch.cuda.mem_get_info()[0] / 2**30
    c.require(free_gpu >= old.FREE_VRAM, 'Need6GiB free VRAM before model creation')
    if not OUTPUT.exists():
        OUTPUT.mkdir()
        c.save(OUTPUT / 'binding.json', binding())
    folder.mkdir()
    token = uuid.uuid4().hex
    row = next(r for r in identity['models'] if r['label'] == model_label)
    c.save(folder / 'transaction.json', {'token': token, 'binding': binding(), 'model': row,
                                       'preimport_resources': before_import, 'startup_resources': startup,
                                       'free_VRAM_GiB': free_gpu, 'pid': os.getpid()})
    cursor = {'token': token, 'phase': 'before_model_load', 'model': model_label, 'image': None, 'y': None}
    cursor_write(folder, cursor)
    try:
        (folder / 'probabilities').mkdir()
        checkpoint = c.ROOT / row['checkpoint']
        c.require(c.sha(checkpoint) == row['checkpoint_sha256'], 'Final weight identity differs')
        model = make_model(row['seed'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c.require(payload['identity'] == row['run_identity'] and payload['cursor'] == 10000, 'Only own final10000 allowed')
        model.load_state_dict(payload['model'], strict=True)
        del payload
        c.require(state_hash(model) == row['final_parameter_hash_recorded'], 'Loaded final parameters differ')
        model = model.cuda().float().eval()
        c.require(not list(model.buffers()) and not any(m.training for m in model.modules()), 'Expected eval GN without buffers')
        before = state_hash(model)
        torch.cuda.reset_peak_memory_stats()
        scores, start, minimum = [], time.monotonic(), c.d.available_ram()
        with torch.inference_mode():
            for request in requests:
                resource = c.resource(old.MIN_RAM, old.MIN_DISK)
                minimum = min(minimum, resource['available_RAM_GiB'])
                rgb = old.decode_normal(request)
                padded = np.pad(rgb, ((0, 0), (0, 256 - request['width']), (0, 0)), mode='reflect')
                def windows():
                    for y in request['window_starts']:
                        cursor.update(phase='before_forward', image=request['id'], y=y)
                        cursor_write(folder, cursor)
                        x = np.ascontiguousarray(padded[y:y + 256].transpose(2, 0, 1))
                        tensor = torch.from_numpy(x).unsqueeze(0).cuda().float()
                        probability = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy().astype(np.float32)
                        yield y, probability
                probability = old.native.stitch_native(request['height'], request['width'], windows())
                path = folder / 'probabilities' / f"{request['id']}.npy"
                with path.open('xb') as stream:
                    np.save(stream, probability, allow_pickle=False)
                scores.append({'request': request, 'score': float(probability.max()),
                               'probability_sha256': c.sha(path), 'window_forwards': len(request['window_starts'])})
                cursor.update(phase='image_saved', image=request['id'], y=None)
                cursor_write(folder, cursor)
                print(json.dumps({'model': model_label, 'image': request['ordinal'] + 1, 'of': 200,
                                  'RAM_GiB': resource['available_RAM_GiB']}), flush=True)
        after = state_hash(model)
        c.require(before == after and not list(model.buffers()), 'Inference changed parameters/buffers')
        summary = {'model': row, 'calibration': old.calibration_counts([s['score'] for s in scores]),
                   'parameter_hash_before': before, 'parameter_hash_after': after, 'buffer_count': 0,
                   'window_forwards': 844, 'elapsed_scoring_seconds': time.monotonic() - start,
                   'minimum_image_boundary_RAM_GiB': minimum,
                   'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                   'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20}
        c.save(folder / 'scores.json', scores)
        c.save(folder / 'summary.json', summary)
        cursor_write(folder, {'token': token, 'phase': 'complete', 'model': model_label})
        c.seal(folder, {'binding': binding(), 'model': row})
    except BaseException as error:
        if not (folder / 'failure.json').exists():
            c.save(folder / 'failure.json', {'token': token, 'cursor': cursor, 'error': repr(error),
                                           'action': 'preserve; no automatic repeat or resume'})
        raise
    ref = verify_model_internal(model_label, context)
    return {'finished': model_label, 'source_reference': ref, 'calibration': summary['calibration'],
            'new_window_forwards': 844, 'next': 'stop; manually inspect, never auto start another model'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify-preparation', 'run-model', 'verify-model'))
    parser.add_argument('--model', choices=PENDING)
    args = parser.parse_args()
    if args.command in ('run-model', 'verify-model'):
        if args.model is None:
            parser.error('An explicit fixed --model is required')
        result = run_model(args.model) if args.command == 'run-model' else verify_model(args.model)
    else:
        if args.model is not None:
            parser.error('--model is only valid for single-model run/verify')
        result = prepare() if args.command == 'prepare' else verify_preparation()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
