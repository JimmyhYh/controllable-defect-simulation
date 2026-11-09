"""Independent E1004 preparation and explicit one-model, label-free scoring.

Import/preparation never imports Torch, loads a checkpoint, or opens a dataset image.
Only the explicit run-model command permits the fixed E RGB paths. No training,
calibration, metric calculation, partial recovery, or automatic next-model action.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from . import normal_calibration_assembly as assembly

x, c = assembly.x, assembly.c
CONTROL = c.RESULTS / 'real_evaluation_preparation_v1'
OUTPUT = c.RESULTS / 'real_evaluation_v1'
ORDER = tuple(f'{condition}_{seed}' for seed in c.SEEDS for condition in ('A', 'B'))
IMAGES, PIXELS, WINDOWS = 1004, 146411528, 4270
_RUN_CALLED = False


def install_guard(*, cpu=True, images=(), writes=()):
    """Process-local boundary, not an OS sandbox; metadata-only during preparation."""
    forbidden = ('torch', 'torchvision', 'diffusers', 'transformers')
    if cpu:
        c.require(not any(n.split('.')[0] in forbidden for n in sys.modules), 'Fresh CPU-only process required')
    allowed = {Path(p).resolve() for p in images}
    destinations = tuple(Path(p).resolve() for p in writes)
    def audit(event, args):
        if cpu and event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('No model imports during E CPU preparation/verification')
        if event in ('socket.connect', 'subprocess.Popen') or (cpu and event == 'pickle.find_class'):
            raise RuntimeError('No network/subprocess/CPU pickle execution')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or '', args[2] or 0
            writing = any(v in str(mode) for v in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in destinations), 'Write outside independent E destinations: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'data'):
                c.require(path == (c.DATA / 'split.json').resolve() or (not cpu and path in allowed),
                          'Q/E/U image or GT read forbidden: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'results'):
                c.require(path == c.d.AUDIT or path.is_relative_to(c.RESULTS), 'Old result read forbidden')
    sys.addaudithook(audit)


def implementations():
    return {name: c.sha(Path(__file__).with_name(name)) for name in
            ('real_evaluation.py', 'real_evaluation_cache.py')}


def source_context():
    """Frozen hashes/NPY headers and saved Q scalar metadata; no Q array payloads."""
    inherited = x.prepared_context()
    identity, q_requests, _, _ = inherited
    refs = []
    for index, row in enumerate(identity['models']):
        if index < 4:
            ref = x.verify_cache(x.old.OUTPUT / row['label'], row, q_requests, x.old.binding(identity), payload=False)
        else:
            ref = x.verify_model_internal(row['label'], inherited, payload=False)
        refs.append(ref)
    sealed = c.verify_manifest(x.ASSEMBLY)
    c.require(set(sealed['files']) == {'sources.json', 'verification.json'}
              and sealed['binding'] == x.binding()
              and c.read_json(x.ASSEMBLY / 'sources.json') == assembly.sources_value(inherited, refs)
              and c.read_json(x.ASSEMBLY / 'verification.json') == assembly.verification_value(),
              'Frozen six-source assembly differs')
    c.require(tuple(r['label'] for r in identity['models']) == ORDER, 'Six-model order differs')
    split = c.read_json(c.DATA / 'split.json')
    inventory = c.read_json(c.INPUT / 'export_inventory.json')['files']
    rows = split['E_official1004']
    c.require(len(rows) == IMAGES and len({r['id'] for r in rows}) == IMAGES
              and [r['id'] for r in rows] == sorted(r['id'] for r in rows), 'E original metadata order/count differs')
    others = {r['id'] for role in ('R_real5', 'Q_normal200', 'N_train1885', 'U_anomaly241') for r in split[role]}
    c.require(not others.intersection(r['id'] for r in rows), 'E role overlap')
    requests = []
    for ordinal, row in enumerate(rows):
        h, w = row['height'], row['width']
        c.require(row['part'] == 'test' and row['image'] == f"test/{row['id']}.png"
                  and (c.DATA / row['image']).resolve().is_relative_to(c.DATA.resolve())
                  and inventory[row['image']] == row['image_sha256']
                  and type(h) is int and type(w) is int and h >= 256 and 2 <= w < 256,
                  'E image metadata/source/native size differs')
        # Deliberately strip training_label and mask; scoring never receives either.
        requests.append({'ordinal': ordinal, 'id': row['id'], 'image': row['image'],
                         'image_sha256': row['image_sha256'], 'height': h, 'width': w,
                         'window_starts': c.d.window_starts(h)})
    c.require(sum(r['height'] * r['width'] for r in requests) == PIXELS
              and sum(len(r['window_starts']) for r in requests) == WINDOWS, 'E native denominators differ')
    sources = {'assembly_complete_sha256': c.sha(x.ASSEMBLY / 'complete.json'),
               'assembly_sources_sha256': c.sha(x.ASSEMBLY / 'sources.json'),
               'assembly_verification_sha256': c.sha(x.ASSEMBLY / 'verification.json'),
               'continuation_preparation_sha256': c.sha(x.CONTROL / 'complete.json'),
               'split_sha256': c.sha(c.DATA / 'split.json'),
               'inherited_identity_digest': c.digest_json(identity), 'models': refs,
               'calibration_verification_scope': 'frozen identity/full-file hashes/NPY headers and saved Q scalar metadata; no Q NPY numeric payload reparse'}
    return identity, requests, sources


def protocol(context):
    identity, requests, sources = context
    inherited = c.read_json(x.old.CONTROL / 'protocol.json')
    return {'schema': 'fewshot-real-E1004-scoring-v1', 'implementation': implementations(),
            'source_manifest_digest': c.digest_json(sources), 'model_order': list(ORDER),
            'models': identity['models'], 'E_requests_digest': c.digest_json(requests),
            'images_per_model': IMAGES, 'unique_E_images': IMAGES,
            'native_pixels_per_model': PIXELS, 'windows_per_model': WINDOWS,
            'all_six_image_evaluations': IMAGES * 6, 'all_six_windows': WINDOWS * 6,
            'all_six_float32_payload_bytes': PIXELS * 4 * 6,
            'preprocessing': inherited['preprocessing'], 'architecture': inherited['architecture'],
            'execution': inherited['execution'], 'resource': inherited['resource'],
            'threshold': {'storage': 'only original source summary.json; reference path/hash, no copied threshold',
                          'q': .95, 'method': 'linear', 'comparison': 'strict >', 'recalibration': False},
            'scoring_labels_or_GT': 'not present in E_requests; never read by scorer',
            'order': 'original E metadata order; A03/B03/A04/B04/A05/B05, separate fresh processes',
            'output': str(OUTPUT), 'one_model_per_process': True,
            'cache': 'strict prefix; complete identity reuse; partial/unknown preserve and stop, no repeat/resume',
            'metrics': 'future separately authorized CPU stage; no GT decoding or metrics here',
            'training': False}


def scalar_validation(requests):
    for row in requests:
        starts = row['window_starts']
        c.require(starts == c.d.window_starts(row['height']) and starts[0] == 0
                  and starts[-1] == row['height'] - 256
                  and all(0 < b - a <= 128 for a, b in zip(starts, starts[1:])), 'E coverage/tail metadata differs')
        c.require(set(row) == {'ordinal', 'id', 'image', 'image_sha256', 'height', 'width', 'window_starts'},
                  'Label/GT/extra field leaked into scoring request')
    # Invented scalars only: the frozen stitch implementation is bound by source hash.
    mean = np.float32(np.float32(.25) + np.float32(.75)) / np.float32(2)
    c.require(float(mean) == .5 and sum(v > .5 for v in (.25, .5, .75)) == 1,
              'Invented equal-mean / strict threshold convention differs')
    return {'E_metadata_records': IMAGES, 'windows_per_model': WINDOWS, 'native_pixels_per_model': PIXELS,
            'request_labels_or_GT_paths': 0, 'invented_scalar_mean_and_strict_ties': True,
            'original_images_or_GT_read': 0, 'E_probability_payloads_parsed': 0,
            'Q_probability_numeric_payloads_parsed': 0, 'actual_thresholds_computed': 0,
            'saved_Q_score_summary_scalar_metadata_read': True,
            'model_operations': 0, 'Torch_imported': 'torch' in sys.modules}


def prepared_context():
    context = source_context()
    sealed = c.verify_manifest(CONTROL)
    c.require(set(sealed['files']) == {'protocol.json', 'source_manifest.json', 'E_requests.json', 'CPU_validation.json'}
              and sealed['implementation'] == implementations()
              and c.read_json(CONTROL / 'protocol.json') == protocol(context)
              and c.read_json(CONTROL / 'source_manifest.json') == context[2]
              and c.read_json(CONTROL / 'E_requests.json') == context[1], 'Frozen E preparation differs')
    scalar_validation(context[1])
    return context


def prepare():
    install_guard(writes=(CONTROL,))
    resources = c.resource(2, x.old.MIN_DISK)
    context = source_context()
    if (CONTROL / 'complete.json').exists():
        prepared_context()
        return {'prepared_cache_verified': True, **scalar_validation(context[1])}
    c.require(not CONTROL.exists() and not OUTPUT.exists(), 'Partial/unknown E destination; preserve and diagnose')
    validation = scalar_validation(context[1])
    CONTROL.mkdir()
    for name, value in [('protocol.json', protocol(context)), ('source_manifest.json', context[2]),
                        ('E_requests.json', context[1]), ('CPU_validation.json', {'resources': resources, **validation})]:
        c.save(CONTROL / name, value)
    c.seal(CONTROL, {'implementation': implementations()})
    prepared_context()
    return {'prepared': True, **validation, 'resources': resources}


def verify_preparation():
    install_guard()
    context = prepared_context()
    return {'preparation_verified': True, **scalar_validation(context[1])}


def binding():
    return {'preparation_complete_sha256': c.sha(CONTROL / 'complete.json'),
            'protocol_sha256': c.sha(CONTROL / 'protocol.json'), 'implementation': implementations()}


def threshold_reference(context, label):
    ref = next(r for r in context[2]['models'] if r['label'] == label)
    return {'summary': ref['summary'], 'summary_sha256': ref['files']['summary.json'],
            'source_complete_sha256': ref['complete_sha256'], 'field': 'calibration.threshold',
            'comparison': 'strict >'}


def verify_cached_model(label, context, *, payload=False):
    c.require(label in ORDER, 'Unknown E model')
    row = next(r for r in context[0]['models'] if r['label'] == label)
    folder, requests = OUTPUT / label, context[1]
    sealed = c.verify_manifest(folder)
    expected = {'transaction.json', 'cursor.json', 'scores.json', 'summary.json'} | {
        f"probabilities/{r['id']}.npy" for r in requests}
    c.require(set(sealed['files']) == expected and sealed['binding'] == binding() and sealed['model'] == row,
              'E model seal/source differs')
    txn, cursor = c.read_json(folder / 'transaction.json'), c.read_json(folder / 'cursor.json')
    c.require(txn['binding'] == binding() and txn['model'] == row
              and cursor == {'token': txn['token'], 'phase': 'complete', 'model': label}, 'E transaction incomplete')
    summary, scores = c.read_json(folder / 'summary.json'), c.read_json(folder / 'scores.json')
    c.require(len(scores) == IMAGES and summary['model'] == row and summary['images'] == IMAGES
              and summary['native_pixels'] == PIXELS and summary['window_forwards'] == WINDOWS
              and summary['parameter_hash_before'] == summary['parameter_hash_after'] == row['final_parameter_hash_recorded']
              and summary['buffer_count'] == 0 and summary['threshold_reference'] == threshold_reference(context, label),
              'E denominator/parameter/threshold source differs')
    for request, score in zip(requests, scores):
        name = f"probabilities/{request['id']}.npy"
        c.require(set(score) == {'request', 'score', 'probability_sha256', 'window_forwards'}
                  and score['request'] == request and score['window_forwards'] == len(request['window_starts'])
                  and score['probability_sha256'] == sealed['files'][name]
                  and type(score['score']) is float and np.isfinite(score['score']) and 0 <= score['score'] <= 1,
                  'E source/order/raw max differs')
        x.header(folder / name, request)
        if payload:
            probability = np.load(folder / name, allow_pickle=False)
            c.require(np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
                      and float(probability.max()) == score['score'], 'E probability range/native max differs')
    return {'label': label, 'folder': folder.relative_to(c.ROOT).as_posix(),
            'complete_sha256': c.sha(folder / 'complete.json'), 'files': sealed['files'],
            'checkpoint': row['checkpoint'], 'checkpoint_sha256': row['checkpoint_sha256'],
            'summary': (folder / 'summary.json').relative_to(c.ROOT).as_posix(),
            'scores': (folder / 'scores.json').relative_to(c.ROOT).as_posix(),
            'threshold_reference': threshold_reference(context, label)}


def runtime_prefix(context, *, payload=False, allow_target=None):
    if not OUTPUT.exists():
        return []
    names = {p.name for p in OUTPUT.iterdir()}
    c.require('binding.json' in names and names <= {'binding.json', *ORDER}
              and c.read_json(OUTPUT / 'binding.json') == binding(), 'Unknown E root/binding; preserve')
    present = [label for label in ORDER if label in names]
    c.require(present == list(ORDER[:len(present)]) and present, 'E root must be a nonempty ordered prefix')
    for label in present:
        if label == allow_target and not (OUTPUT / label / 'complete.json').exists():
            raise RuntimeError('Partial E target; preserve, no automatic repeat/resume')
        verify_cached_model(label, context, payload=payload)
    return present


def decode_image(request):
    """Only E RGB path/hash/dimensions; no GT or labels accepted by this function."""
    from PIL import Image
    path = c.DATA / request['image']
    c.require(c.sha(path) == request['image_sha256'], 'E RGB source identity differs')
    with Image.open(path) as im:
        c.require(im.size == (request['width'], request['height']) and im.mode in ('L', 'RGB'),
                  'E native mode/dimensions differ')
        return np.asarray(im.convert('RGB'), dtype=np.float32) / np.float32(255)


def run_model(label):
    global _RUN_CALLED
    c.require(label in ORDER and not _RUN_CALLED, 'One explicit fixed model per fresh process')
    _RUN_CALLED = True
    requests = c.read_json(CONTROL / 'E_requests.json')
    install_guard(cpu=False, images=[c.DATA / r['image'] for r in requests], writes=(OUTPUT,))
    context = prepared_context()
    present = runtime_prefix(context, allow_target=label)
    if label in present:
        ref = verify_cached_model(label, context, payload=True)
        return {'complete_cache_reused': True, 'source_reference': ref, 'new_model_operations': 0}
    c.require(len(present) < len(ORDER) and label == ORDER[len(present)], 'Only the next ordered E model may start')
    folder = OUTPUT / label
    c.require(not folder.exists(), 'Unknown/partial E target; preserve')
    preimport = c.resource(x.old.START_RAM, x.old.MIN_DISK)
    import torch
    from .model import make_model, state_hash
    c.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    startup = c.resource(x.old.START_RAM, x.old.MIN_DISK)
    free_gpu = torch.cuda.mem_get_info()[0] / 2**30
    c.require(free_gpu >= x.old.FREE_VRAM, 'Need6GiB free VRAM before model creation')
    if not OUTPUT.exists():
        OUTPUT.mkdir()
        c.save(OUTPUT / 'binding.json', binding())
    folder.mkdir()
    row = next(r for r in context[0]['models'] if r['label'] == label)
    token = uuid.uuid4().hex
    cursor = {'token': token, 'phase': 'before_model_load', 'model': label, 'image': None, 'y': None}
    c.save(folder / 'transaction.json', {'token': token, 'model': row, 'binding': binding(), 'pid': os.getpid(),
                                       'preimport_resources': preimport, 'startup_resources': startup,
                                       'free_VRAM_GiB': free_gpu})
    x.cursor_write(folder, cursor)
    try:
        (folder / 'probabilities').mkdir()
        checkpoint = c.ROOT / row['checkpoint']
        c.require(c.sha(checkpoint) == row['checkpoint_sha256'], 'Final10000 checkpoint identity differs')
        model = make_model(row['seed'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c.require(saved['identity'] == row['run_identity'] and saved['cursor'] == 10000, 'Only own final10000 allowed')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        c.require(state_hash(model) == row['final_parameter_hash_recorded'], 'Loaded final parameters differ')
        model = model.cuda().float().eval()
        c.require(not list(model.buffers()) and not any(m.training for m in model.modules()), 'Expected eval GN and no buffers')
        before = state_hash(model)
        torch.cuda.reset_peak_memory_stats()
        scores, start, minimum = [], time.monotonic(), c.d.available_ram()
        with torch.inference_mode():
            for request in requests:
                resources = c.resource(x.old.MIN_RAM, x.old.MIN_DISK)
                minimum = min(minimum, resources['available_RAM_GiB'])
                cursor.update(phase='before_image_decode', image=request['id'], y=None)
                x.cursor_write(folder, cursor)
                rgb = decode_image(request)
                padded = np.pad(rgb, ((0, 0), (0, 256 - request['width']), (0, 0)), mode='reflect')
                def windows():
                    for y in request['window_starts']:
                        cursor.update(phase='before_forward', image=request['id'], y=y)
                        x.cursor_write(folder, cursor)
                        array = np.ascontiguousarray(padded[y:y + 256].transpose(2, 0, 1))
                        tensor = torch.from_numpy(array).unsqueeze(0).cuda().float()
                        probability = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy().astype(np.float32)
                        yield y, probability
                probability = x.old.native.stitch_native(request['height'], request['width'], windows())
                path = folder / 'probabilities' / f"{request['id']}.npy"
                with path.open('xb') as stream:
                    np.save(stream, probability, allow_pickle=False)
                scores.append({'request': request, 'score': float(probability.max()),
                               'probability_sha256': c.sha(path), 'window_forwards': len(request['window_starts'])})
                cursor.update(phase='image_saved', image=request['id'], y=None)
                x.cursor_write(folder, cursor)
                if (request['ordinal'] + 1) % 25 == 0 or request['ordinal'] + 1 == IMAGES:
                    print(json.dumps({'model': label, 'image': request['ordinal'] + 1, 'of': IMAGES,
                                      'RAM_GiB': resources['available_RAM_GiB']}), flush=True)
        after = state_hash(model)
        c.require(before == after and not list(model.buffers()), 'E inference changed parameters/buffers')
        summary = {'model': row, 'images': IMAGES, 'native_pixels': PIXELS, 'window_forwards': WINDOWS,
                   'parameter_hash_before': before, 'parameter_hash_after': after, 'buffer_count': 0,
                   'threshold_reference': threshold_reference(context, label),
                   'elapsed_scoring_seconds': time.monotonic() - start,
                   'minimum_image_boundary_RAM_GiB': minimum,
                   'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                   'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
                   'GT_or_labels_read': False, 'metrics_computed': False}
        c.save(folder / 'scores.json', scores)
        c.save(folder / 'summary.json', summary)
        x.cursor_write(folder, {'token': token, 'phase': 'complete', 'model': label})
        c.seal(folder, {'binding': binding(), 'model': row})
    except BaseException as error:
        if not (folder / 'failure.json').exists():
            c.save(folder / 'failure.json', {'token': token, 'cursor': cursor, 'error': repr(error),
                                           'action': 'preserve; no automatic repeat/resume'})
        raise
    ref = verify_cached_model(label, context, payload=True)
    return {'finished': label, 'source_reference': ref, 'new_image_evaluations': IMAGES,
            'new_window_forwards': WINDOWS, 'metrics_computed': False,
            'next': 'stop; inspect saved result before any separately authorized next-model process'}


def verify_model(label):
    install_guard()
    context = prepared_context()
    present = runtime_prefix(context)
    c.require(label in present, 'No complete E result; never run a model implicitly')
    return {'verified': True, 'source_reference': verify_cached_model(label, context, payload=True),
            'new_model_operations': 0, 'metrics_computed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify-preparation', 'run-model', 'verify-model'))
    parser.add_argument('--model', choices=ORDER)
    args = parser.parse_args()
    if args.command in ('run-model', 'verify-model'):
        if args.model is None:
            parser.error('Explicit --model required')
        result = run_model(args.model) if args.command == 'run-model' else verify_model(args.model)
    else:
        if args.model is not None:
            parser.error('--model only valid for run-model/verify-model')
        result = prepare() if args.command == 'prepare' else verify_preparation()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
