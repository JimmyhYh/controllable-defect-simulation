"""Independent Q200 calibration. Preparation/verification never load a model.

The explicit `run` command is for a separately authorized future invocation.
There is no training, optimizer, automatic recovery, E evaluation or GT reader.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from . import common as c
from . import formal_admission as f
from . import inference as native

CONTROL = c.RESULTS / 'normal_calibration_preparation_v1'
OUTPUT = c.RESULTS / 'normal_calibration_v1'
EVIDENCE = c.RESULTS / 'formal_run_verifications_v1/B_2026101905'
ORDER = [(condition, seed) for seed in c.SEEDS for condition in ('A', 'B')]
START_RAM, MIN_RAM, FREE_VRAM, MIN_DISK = 4.5, 3.0, 6.0, 80.0


def install_guard(cpu, q_paths=()):
    """Process-local boundary, not an OS security sandbox. No dataset GT reads."""
    forbidden = ('torch', 'torchvision', 'transformers', 'diffusers')
    if cpu:
        c.require(not any(n.split('.')[0] in forbidden for n in sys.modules),
                  'Fresh CPU-only process required')
    allowed = {Path(p).resolve() for p in q_paths}
    split_path = (c.DATA / 'split.json').resolve()

    def audit(event, args):
        if cpu and event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('Model import forbidden in CPU preparation/cache verification')
        if event in ('socket.connect', 'subprocess.Popen') or (cpu and event == 'pickle.find_class'):
            raise RuntimeError('No network/subprocess/pickle in CPU calibration preparation')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or '', args[2] or 0
            writing = any(x in str(mode) for x in 'wax+') or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(path.is_relative_to(CONTROL) or path.is_relative_to(OUTPUT),
                          'Write outside new calibration destinations: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'data'):
                c.require(path == split_path or (not cpu and path in allowed),
                          'Q/E/U image or dataset label read forbidden: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'results'):
                c.require(path == c.d.AUDIT or path.is_relative_to(c.RESULTS),
                          'Old result read forbidden')
    sys.addaudithook(audit)


def label(condition, seed):
    c.require((condition, seed) in ORDER, 'Unexpected formal model')
    return f'{condition}_{seed}'


def static_admission(source, final):
    """Verify sealed completed admission, never re-check a historical process PID."""
    admission = c.verify_manifest(f.CONTROL)
    cfg = c.read_json(f.CONTROL / 'protocol.json')
    c.require(set(admission['files']) == {'protocol.json', 'CPU_validation.json'}
              and c.sha(f.CONTROL / 'complete.json') == final['formal_admission_complete_sha256']
              and admission['protocol_sha256'] == c.sha(f.CONTROL / 'protocol.json')
              and admission['adapter_sha256'] == cfg['adapter_sha256'] == c.sha(f.__file__)
              and cfg['common_config_sha256'] == c.sha(c.PREP / 'common_config.json')
              and cfg['preparation_complete_sha256'] == c.sha(c.PREP / 'complete.json'),
              'Frozen formal admission identity differs')
    gate = cfg['gate_evidence']
    paths = {'metric_complete_sha256': f.pm.OUTPUT / 'complete.json',
             'metric_preparation_sha256': f.pm.CONTROL / 'complete.json',
             'metric_json_sha256': f.pm.OUTPUT / 'metrics.json',
             'probe_complete_sha256': f.pm.RUN / 'complete.json',
             'retry_protocol_sha256': f.retry.CONTROL / 'protocol.json',
             'retry_binding_sha256': f.retry.NEW_RESULTS / 'binding.json',
             'retry_source_sha256': Path(f.retry.__file__),
             'metric_source_sha256': Path(f.pm.__file__)}
    hashes = {name: c.sha(path) for name, path in paths.items()}
    c.require(all(gate[name] == value for name, value in hashes.items())
              and gate['input_identity'] == source and gate['environment'] == c.environment()
              and gate['core_implementation'] == c.identities() and gate['gate_passed'] is True,
              'Frozen technical gate identity differs')
    metric = c.verify_manifest(f.pm.OUTPUT)  # Hash saved tables, never recompute AP.
    gate_summary = c.read_json(f.pm.OUTPUT / 'metrics.json')
    c.require(metric['preparation_sha256'] == hashes['metric_preparation_sha256']
              and gate_summary['gate_passed'] is True and gate_summary['AP'] == gate['AP'],
              'Sealed gate result differs')
    retry_cfg = c.read_json(f.retry.CONTROL / 'protocol.json')
    old = c.MODELS / f'probe_{c.PROBE_SEED}'
    c.require(all(c.sha(old / name) == expected for name, expected in
                  retry_cfg['original_failure_files'].items()), 'Original failure evidence differs')
    return hashes


def collect_identity():
    """Hash/JSON only. Do not use f.verify_run: its permanent guard blocks Torch."""
    split, source = c.source_metadata()
    evidence = c.verify_manifest(EVIDENCE)
    final = c.read_json(EVIDENCE / 'verification.json')
    c.require(final['formal_runs_verified'] == 6 and final['checkpoints_verified'] == 60,
              'All-six final verification evidence missing')
    gate_hashes = static_admission(source, final)
    f.validate_roots()
    rows = []
    folders = f.t.folders
    f.t.folders = f.routed_folders
    try:
        for condition, seed in ORDER:
            result = f.t.verify_run(condition, seed, split, source)['result']
            model_dir, result_dir = f.routed_folders(condition, seed)
            record = next(r for r in final['runs'] if r['condition'] == condition and r['seed'] == seed)
            run = c.read_json(model_dir / 'run.json')
            checkpoint = model_dir / 'checkpoint_10000.pt'
            c.require(c.sha(result_dir / 'complete.json') == record['source_complete_sha256']
                      and c.sha(checkpoint) == record['final_checkpoint_file_sha256']
                      and result['final_hash'] == record['final_parameter_hash_recorded']
                      and run['identity']['condition'] == condition and run['identity']['seed'] == seed,
                      'Final weight / result binding differs')
            rows.append({'label': label(condition, seed), 'condition': condition, 'seed': seed,
                         'checkpoint': checkpoint.relative_to(c.ROOT).as_posix(),
                         'checkpoint_sha256': c.sha(checkpoint),
                         'run_identity': run['identity'],
                         'final_parameter_hash_recorded': result['final_hash'],
                         'source_complete_sha256': c.sha(result_dir / 'complete.json')})
    finally:
        f.t.folders = folders
    for i in (0, 2, 4):
        a, b = rows[i]['run_identity'], rows[i + 1]['run_identity']
        c.require({k: v for k, v in a.items() if k != 'condition'} ==
                  {k: v for k, v in b.items() if k != 'condition'}, 'Paired identity differs')
    q = split['Q_normal200']
    c.require(len(q) == 200 and len({r['id'] for r in q}) == 200, 'Q200 identities differ')
    inventory = c.read_json(c.INPUT / 'export_inventory.json')['files']
    requests = []
    for ordinal, row in enumerate(q):
        path = c.DATA / row['image']
        c.require(row['part'] == 'train' and row['training_label'] == 'normal'
                  and path.resolve().is_relative_to(c.DATA.resolve())
                  and row['image'] == f"train/{row['id']}.png"
                  and inventory[row['image']] == row['image_sha256'], 'Unexpected normal metadata')
        h, w = row['height'], row['width']
        c.require(type(h) is int and type(w) is int and h >= 256 and 2 <= w < 256,
                  'Native dimensions outside frozen rule')
        requests.append({'ordinal': ordinal, 'id': row['id'], 'image': row['image'],
                         'image_sha256': row['image_sha256'], 'height': h, 'width': w,
                         'window_starts': c.d.window_starts(h)})
    other_roles = ('R_real5', 'N_train1885', 'U_anomaly241', 'E_official1004')
    other_ids = {r['id'] for role in other_roles for r in split[role]}
    c.require(not {r['id'] for r in q} & other_ids, 'Q overlaps another frozen role')
    identity = {'source': source, 'environment': c.environment(), 'frozen_core': c.identities(),
                'additional_implementation': {name: c.sha(Path(__file__).parent / name) for name in
                                              ('data.py', 'verification.py', 'formal_admission.py',
                                               'probe_retry.py', 'probe_retry_metrics.py')},
                'implementation_sha256': c.sha(__file__),
                'static_gate_files': gate_hashes,
                'formal_preparation_sha256': c.sha(f.CONTROL / 'complete.json'),
                'formal_binding_sha256': c.sha(f.RESULTS / 'binding.json'),
                'all_six_evidence_sha256': c.sha(EVIDENCE / 'complete.json'),
                'all_six_files': evidence['files'], 'models': rows,
                'Q_requests_digest': c.digest_json(requests)}
    return identity, requests


def protocol(identity, requests):
    windows = sum(len(r['window_starts']) for r in requests)
    pixels = sum(r['height'] * r['width'] for r in requests)
    return {'schema': 'fewshot-normal-calibration-v1', 'identity': identity,
            'model_order': [r['label'] for r in identity['models']],
            'Q_request_order': 'original frozen split order; no sorting/resampling',
            'images_per_model': 200, 'unique_normal_images': 200,
            'full_image_evaluations': 1200, 'windows_per_model': windows,
            'window_forwards_all_models': windows * 6, 'native_pixels_per_model': pixels,
            'probability_payload_bytes_all_models': pixels * 4 * 6,
            'preprocessing': {'RGB': 'L replicated to RGB, float32 / float32(255)',
                              'contrast': 'unchanged', 'resize': False, 'augmentation': False,
                              'window': 256, 'stride': 128, 'last_window': 'append H-256, sorted unique',
                              'x_start': 0, 'right_padding': 'numpy reflect to width256',
                              'padding_excluded': True, 'overlap': 'equal mean probabilities',
                              'map': 'raw float32 anomaly softmax, native HxW',
                              'smoothing': False, 'score': 'max native stitched map'},
            'architecture': 'frozen author 3/2 base16; GN8; no BN/buffers',
            'execution': {'FP32': True, 'batch': 1, 'workers': 0, 'device': 'cuda',
                          'mode': 'eval + inference_mode', 'AMP': False, 'training': False,
                          'checkpoint': 'own final10000 only; optimizer never instantiated'},
            'threshold': {'q': 0.95, 'method': 'linear', 'comparison': 'strict >',
                          'input': 'float64 array of cached float32 map maxima',
                          'separate_per_model': True, 'normal_denominator': 200,
                          'meaning': 'same-pool FP/200, not independent validation'},
            'resource': {'start_RAM_GiB': START_RAM, 'runtime_RAM_GiB': MIN_RAM,
                         'free_VRAM_GiB': FREE_VRAM, 'disk_GiB': MIN_DISK},
            'cache': {'output': str(OUTPUT), 'scope': 'whole six-model calibration transaction',
                      'complete': 'strict files, hashes, sources, order, maps and scalar checks',
                      'partial': 'preserve and stop; no auto resume/retry or intersection',
                      'forward_unit': 'each original Q image once per model, all frozen windows once',
                      'inference_reads_labels': False, 'E_U_reads': False}}


def scalar_validation(requests):
    """Metadata / invented numbers only; never an actual Q score or image."""
    for row in requests:
        starts = row['window_starts']
        c.require(starts == sorted(set(starts)) and starts[0] == 0
                  and starts[-1] == row['height'] - 256
                  and all(0 < b - a <= 128 for a, b in zip(starts, starts[1:])),
                  'Window coverage/boundary mismatch')
    invented = np.arange(200, dtype=np.float64)
    threshold = float(np.quantile(invented, .95, method='linear'))
    c.require(abs(threshold - 189.05) < 1e-12 and int((invented > threshold).sum()) == 10,
              'Invented scalar quantile convention mismatch')
    ties = np.ones(200, dtype=np.float64)
    c.require(int((ties > np.quantile(ties, .95, method='linear')).sum()) == 0,
              'Strict greater-than tie convention mismatch')
    c.require(native.SCORE_FORMAT['q'] == .95 and native.SCORE_FORMAT['quantile_method'] == 'linear'
              and native.SCORE_FORMAT['positive'] == 'strict >', 'Frozen score format differs')
    return {'scope': 'Q metadata windows and invented scalar numbers only',
            'images_or_labels_read': 0, 'actual_Q_scores_read': 0,
            'actual_thresholds_computed': 0, 'model_operations': 0,
            'window_coverage_records': len(requests), 'strict_ties_checked': True,
            'Torch_imported': 'torch' in sys.modules}


def verify_preparation(identity=None, requests=None):
    if identity is None:
        install_guard(cpu=True)
        identity, requests = collect_identity()
    sealed = c.verify_manifest(CONTROL)
    c.require(set(sealed['files']) == {'protocol.json', 'Q_requests.json', 'CPU_validation.json'}
              and c.read_json(CONTROL / 'protocol.json') == protocol(identity, requests)
              and c.read_json(CONTROL / 'Q_requests.json') == requests
              and sealed['implementation_sha256'] == c.sha(__file__)
              and sealed['identity_digest'] == c.digest_json(identity), 'Calibration preparation differs')
    scalar_validation(requests)
    cfg = c.read_json(CONTROL / 'protocol.json')
    return {'prepared': True, 'models': 6, 'unique_Q_images': 200,
            'windows_per_model': cfg['windows_per_model'], 'new_model_operations': 0,
            'Q_images_or_labels_read': 0, 'actual_thresholds_computed': 0}


def prepare():
    install_guard(cpu=True)
    identity, requests = collect_identity()
    if (CONTROL / 'complete.json').exists():
        return verify_preparation(identity, requests)
    c.require(not CONTROL.exists() and not OUTPUT.exists(), 'Unknown/partial destination; preserve and investigate')
    resources = c.resource(2, MIN_DISK)
    validation = scalar_validation(requests)
    CONTROL.mkdir()
    c.save(CONTROL / 'protocol.json', protocol(identity, requests))
    c.save(CONTROL / 'Q_requests.json', requests)
    c.save(CONTROL / 'CPU_validation.json', {'resources_snapshot': resources, **validation})
    c.seal(CONTROL, {'implementation_sha256': c.sha(__file__), 'identity_digest': c.digest_json(identity)})
    return verify_preparation(identity, requests)


def load_prepared():
    identity, requests = collect_identity()
    verify_preparation(identity, requests)
    return identity, requests, c.read_json(CONTROL / 'protocol.json')


def binding(identity):
    return {'preparation_sha256': c.sha(CONTROL / 'complete.json'),
            'implementation_sha256': c.sha(__file__), 'identity_digest': c.digest_json(identity)}


def calibration_counts(scores):
    c.require(len(scores) == 200 and all(type(x) is float and np.isfinite(x) and 0 <= x <= 1 for x in scores),
              'Need all 200 finite original maxima; no partial denominator')
    tau = float(np.quantile(np.asarray(scores, dtype=np.float64), .95, method='linear'))
    fp = sum(score > tau for score in scores)
    return {'threshold': tau, 'q': .95, 'method': 'linear', 'comparison': 'strict >',
            'FP': fp, 'TN': 200 - fp, 'normal_denominator': 200,
            'same_pool_not_independent_validation': True}


def verify_results(identity=None, requests=None):
    if identity is None:
        install_guard(cpu=True)
        identity, requests, _ = load_prepared()
    c.require((OUTPUT / 'complete.json').exists(), 'Incomplete/unknown calibration; preserve, no automatic forwards')
    sealed = c.verify_manifest(OUTPUT)
    expected_binding = binding(identity)
    c.require(sealed['binding'] == expected_binding, 'Calibration result binding differs')
    summary = c.read_json(OUTPUT / 'summary.json')
    transaction = c.read_json(OUTPUT / 'transaction.json')
    cursor = c.read_json(OUTPUT / 'cursor.json')
    c.require(transaction['binding'] == expected_binding and cursor['phase'] == 'complete'
              and cursor['token'] == transaction['token'], 'Calibration transaction incomplete')
    expected_files = {'transaction.json', 'cursor.json', 'summary.json'}
    summaries = []
    for row in identity['models']:
        folder = OUTPUT / row['label']
        model_seal = c.verify_manifest(folder)
        c.require(model_seal['binding'] == expected_binding and model_seal['model'] == row,
                  'Model probability cache identity differs')
        scores = c.read_json(folder / 'scores.json')
        c.require(len(scores) == 200, 'Missing normal score records')
        model_summary = c.read_json(folder / 'summary.json')
        expected_model_files = {'scores.json', 'summary.json'}
        for request, score in zip(requests, scores):
            name = f"probabilities/{request['id']}.npy"
            expected_model_files.add(name)
            c.require(score['request'] == request and score['window_forwards'] == len(request['window_starts'])
                      and score['probability_sha256'] == c.sha(folder / name), 'Score source/order/hash differs')
            probability = np.load(folder / name, allow_pickle=False)
            c.require(probability.dtype == np.float32 and probability.shape == (request['height'], request['width'])
                      and np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
                      and float(probability.max()) == score['score'], 'Native probability / score mismatch')
        c.require(set(model_seal['files']) == expected_model_files, 'Unknown model cache file')
        counts = calibration_counts([s['score'] for s in scores])
        c.require(model_summary['calibration'] == counts and model_summary['model'] == row
                  and model_summary['parameter_hash_before'] == model_summary['parameter_hash_after'] ==
                  row['final_parameter_hash_recorded'] and model_summary['buffer_count'] == 0,
                  'Fixed threshold / read-only parameter record differs')
        summaries.append(model_summary)
        expected_files |= {f"{row['label']}/{name}" for name in expected_model_files | {'complete.json'}}
    c.require(set(sealed['files']) == expected_files and summary['models'] == summaries
              and summary['unique_normal_images'] == 200 and summary['full_image_evaluations'] == 1200,
              'Full calibration denominator/order differs')
    return {'complete_cache_verified': True, 'models': 6, 'unique_Q_images': 200,
            'calibration': [{'model': s['model']['label'], **s['calibration']} for s in summaries],
            'new_model_operations': 0, 'Q_images_or_labels_read': 0,
            'same_pool_not_independent_validation': True}


def write_cursor(value):
    temporary = OUTPUT / 'cursor.json.tmp'
    c.require(not temporary.exists(), 'Unknown cursor transaction; preserve and investigate')
    c.save(temporary, value)
    os.replace(temporary, OUTPUT / 'cursor.json')


def decode_normal(request):
    from PIL import Image
    path = c.DATA / request['image']
    c.require(c.sha(path) == request['image_sha256'], 'Q source file identity differs')
    with Image.open(path) as im:
        c.require(im.size == (request['width'], request['height']) and im.mode in ('L', 'RGB'),
                  'Q native dimensions/mode differ; no resizing')
        return np.asarray(im.convert('RGB'), dtype=np.float32) / np.float32(255)


def run():
    """Explicit inference only; never called by prepare/verify or this turn's preparation."""
    # Install the final Q-only reader boundary before source checks or model imports.
    requests = c.read_json(CONTROL / 'Q_requests.json')
    install_guard(cpu=False, q_paths=[c.DATA / r['image'] for r in requests])
    identity, requests, cfg = load_prepared()
    if (OUTPUT / 'complete.json').exists():
        return verify_results(identity, requests)
    c.require(not OUTPUT.exists(), 'Partial/unknown transaction: stop; no automatic resume or repeat forward')
    resources = c.resource(START_RAM, MIN_DISK)
    import torch
    from .model import make_model, state_hash
    c.require(torch.cuda.is_available(), 'CUDA required; no automatic CPU fallback')
    free_gpu = torch.cuda.mem_get_info()[0] / 2**30
    c.require(free_gpu >= FREE_VRAM, 'Need6GiB free VRAM before calibration')
    OUTPUT.mkdir()  # Exclusive directory creation is also the run ownership lock.
    token = uuid.uuid4().hex
    c.save(OUTPUT / 'transaction.json', {'token': token, 'binding': binding(identity),
                                       'resources': resources, 'free_VRAM_GiB': free_gpu})
    cursor = {'token': token, 'phase': 'before_model', 'model': None, 'image': None, 'y': None}
    write_cursor(cursor)
    start = time.monotonic()
    model_summaries = []
    try:
        for row in identity['models']:
            c.resource(START_RAM, MIN_DISK)
            c.require(torch.cuda.mem_get_info()[0] / 2**30 >= FREE_VRAM,
                      'Need6GiB free VRAM before each model; preserve transaction')
            folder = OUTPUT / row['label']
            folder.mkdir()
            (folder / 'probabilities').mkdir()
            cursor = {'token': token, 'phase': 'before_model_load', 'model': row['label'], 'image': None, 'y': None}
            write_cursor(cursor)
            checkpoint = c.ROOT / row['checkpoint']
            c.require(c.sha(checkpoint) == row['checkpoint_sha256'], 'Final checkpoint changed')
            model = make_model(row['seed'])
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            c.require(payload['identity'] == row['run_identity'] and payload['cursor'] == 10000,
                      'Only own final10000 checkpoint permitted')
            model.load_state_dict(payload['model'], strict=True)
            del payload
            c.require(state_hash(model) == row['final_parameter_hash_recorded'], 'Loaded final parameter identity differs')
            model = model.cuda().float().eval()
            c.require(len(list(model.buffers())) == 0 and not any(m.training for m in model.modules()),
                      'Expected read-only GN model with no buffers')
            before = state_hash(model)
            torch.cuda.reset_peak_memory_stats()
            scores = []
            model_start = time.monotonic()
            min_ram = c.d.available_ram()
            with torch.inference_mode():
                for request in requests:
                    snapshot = c.resource(MIN_RAM, MIN_DISK)
                    min_ram = min(min_ram, snapshot['available_RAM_GiB'])
                    rgb = decode_normal(request)
                    padded = np.pad(rgb, ((0, 0), (0, 256 - request['width']), (0, 0)), mode='reflect')

                    def windows():
                        for y in request['window_starts']:
                            cursor.update(phase='before_forward', image=request['id'], y=y)
                            write_cursor(cursor)  # Interrupted cursor is never used for automatic resume.
                            x = np.ascontiguousarray(padded[y:y + 256].transpose(2, 0, 1))
                            tensor = torch.from_numpy(x).unsqueeze(0).cuda().float()
                            probability = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy().astype(np.float32)
                            yield y, probability

                    probability = native.stitch_native(request['height'], request['width'], windows())
                    target = folder / 'probabilities' / f"{request['id']}.npy"
                    with target.open('xb') as stream:
                        np.save(stream, probability, allow_pickle=False)
                    scores.append({'request': request, 'score': float(probability.max()),
                                   'probability_sha256': c.sha(target),
                                   'window_forwards': len(request['window_starts'])})
                    cursor.update(phase='image_saved', image=request['id'], y=None)
                    write_cursor(cursor)
                    print(json.dumps({'model': row['label'], 'image': request['ordinal'] + 1, 'of': 200,
                                      'RAM_GiB': snapshot['available_RAM_GiB']}, ensure_ascii=False), flush=True)
            after = state_hash(model)
            c.require(before == after and len(list(model.buffers())) == 0, 'Inference changed parameters/buffers')
            summary = {'model': row, 'calibration': calibration_counts([s['score'] for s in scores]),
                       'parameter_hash_before': before, 'parameter_hash_after': after, 'buffer_count': 0,
                       'elapsed_scoring_seconds': time.monotonic() - model_start,
                       'minimum_image_boundary_RAM_GiB': min_ram,
                       'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                       'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
                       'window_forwards': cfg['windows_per_model']}
            c.save(folder / 'scores.json', scores)
            c.save(folder / 'summary.json', summary)
            c.seal(folder, {'binding': binding(identity), 'model': row})
            model_summaries.append(summary)
            del model, rgb, padded, probability
            torch.cuda.empty_cache()
        c.save(OUTPUT / 'summary.json', {'models': model_summaries, 'unique_normal_images': 200,
                                       'full_image_evaluations': 1200,
                                       'elapsed_seconds': time.monotonic() - start,
                                       'same_pool_not_independent_validation': True})
        write_cursor({'token': token, 'phase': 'complete'})
        c.seal(OUTPUT, {'binding': binding(identity)})
    except BaseException as error:
        if not (OUTPUT / 'failure.json').exists():
            c.save(OUTPUT / 'failure.json', {'token': token, 'cursor': cursor,
                                           'error': repr(error), 'action': 'preserve; no automatic repeat or resume'})
        raise
    return verify_results(identity, requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify-preparation', 'run', 'verify-results'))
    args = parser.parse_args()
    commands = {'prepare': prepare, 'verify-preparation': verify_preparation,
                'run': run, 'verify-results': verify_results}
    print(json.dumps(commands[args.command](), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
