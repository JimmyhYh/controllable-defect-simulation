"""Independent CPU E1004 metrics. prepare never opens GT or computes metrics.

The explicit run command alone authorizes the fixed GT inputs and actual metric
calculation. No model imports, forward calls, training, calibration or generation.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from . import real_evaluation as r
from . import metrics as frozen

c = r.c
CONTROL = c.RESULTS / 'real_metrics_preparation_v1'
OUTPUT = c.RESULTS / 'real_metrics_v1'
EVIDENCE = c.RESULTS / 'real_evaluation_verifications_v1/all_six_v1'
PROBE = c.RESULTS / 'probe_retry_metrics_v1'
PROBE_EVIDENCE = c.RESULTS / 'probe_retry_metrics_verification_v1'


def install_guard(*, gt=(), writes=()):
    forbidden = ('torch', 'torchvision', 'diffusers', 'transformers')
    c.require(not any(n.split('.')[0] in forbidden for n in sys.modules), 'Fresh CPU-only process required')
    allowed = {Path(p).resolve() for p in gt}
    destinations = tuple(Path(p).resolve() for p in writes)
    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('Model imports forbidden in CPU real metrics')
        if event in ('socket.connect', 'subprocess.Popen', 'pickle.find_class'):
            raise RuntimeError('No network/subprocess/pickle in CPU real metrics')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or '', args[2] or 0
            writing = any(v in str(mode) for v in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                c.require(any(path.is_relative_to(p) for p in destinations), 'Write outside independent metric destinations: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'data'):
                c.require(path == (c.DATA / 'split.json').resolve() or path in allowed,
                          'RGB/GT read outside explicitly authorized metric inputs: ' + str(path))
            elif path.is_relative_to(c.ROOT / 'results'):
                c.require(path == c.d.AUDIT or path.is_relative_to(c.RESULTS), 'Old result read forbidden')
    sys.addaudithook(audit)


def implementation():
    return c.sha(__file__)


def probe_reference():
    """Reuse the sealed small-pool AP/sklearn/ties crosscheck, never rerun it."""
    proof = c.verify_manifest(PROBE_EVIDENCE)
    complete = c.verify_manifest(PROBE)
    summary = c.read_json(PROBE / 'metrics.json')
    exact = c.read_json(PROBE / 'exact/metrics.json')
    c.require(proof['source_metric_complete_sha256'] == c.sha(PROBE / 'complete.json')
              and summary['gate_passed'] is True and summary['pixels'] == 704256
              and summary['positive_pixels'] == 26414 and summary['AP'] == exact['AP']
              and summary['absolute_difference'] == abs(summary['AP'] - summary['sklearn_AP'])
              and summary['absolute_difference'] <= 1e-10
              and summary['independent_distinct_scores'] == exact['distinct_scores']
              and summary['independent_tied_score_values'] == exact['tied_score_values']
              and summary['identity']['frozen_core']['metrics.py'] == c.sha(frozen.__file__),
              'Frozen probe crosscheck/formula identity differs')
    return {'metric_complete_sha256': c.sha(PROBE / 'complete.json'),
            'verification_complete_sha256': c.sha(PROBE_EVIDENCE / 'complete.json'),
            'summary_sha256': complete['files']['metrics.json'],
            'exact_metrics_sha256': complete['files']['exact/metrics.json'],
            'frozen_metrics_sha256': c.sha(frozen.__file__),
            'AP_sklearn_and_actual_ties_crosscheck': 'sealed evidence only; not recalculated'}


def source_context():
    """Only metadata, full-file hashes and NPY headers; no array or GT decode."""
    context = r.prepared_context()
    present = r.runtime_prefix(context, payload=False)
    c.require(present == list(r.ORDER), 'Need all six complete E caches; no successful intersection')
    refs = [r.verify_cached_model(label, context, payload=False) for label in r.ORDER]
    c.verify_manifest(EVIDENCE)
    evidence = c.read_json(EVIDENCE / 'verification.json')
    c.require(evidence['verified'] is True and evidence['E_complete_models'] == 6
              and evidence['model_order'] == list(r.ORDER)
              and evidence['preparation_complete_sha256'] == c.sha(r.CONTROL / 'complete.json')
              and evidence['runtime_binding_sha256'] == c.sha(r.OUTPUT / 'binding.json')
              and evidence['assembly_complete_sha256'] == c.sha(r.x.ASSEMBLY / 'complete.json'),
              'All-six E verification evidence differs')
    c.require(len(evidence['models']) == 6, 'Missing six-model evidence')
    for ref, row in zip(refs, evidence['models']):
        c.require(ref['label'] == row['label'] and ref['complete_sha256'] == row['source_complete_sha256']
                  and c.digest_json(ref['files']) == row['source_files_digest']
                  and ref['checkpoint_sha256'] == row['final_checkpoint_sha256']
                  and ref['threshold_reference'] == row['threshold_reference'], 'E source/evidence identity differs')
    split = c.read_json(c.DATA / 'split.json')
    inventory = c.read_json(c.INPUT / 'export_inventory.json')['files']
    requests = []
    for rgb, row in zip(context[1], split['E_official1004']):
        mask = row['mask']
        c.require(row['id'] == rgb['id'] and mask == f"test/{row['id']}_GT.png"
                  and (c.DATA / mask).resolve().is_relative_to(c.DATA.resolve())
                  and mask in inventory and len(inventory[mask]) == 64, 'GT path/source metadata differs')
        requests.append({'ordinal': rgb['ordinal'], 'id': rgb['id'], 'height': rgb['height'],
                         'width': rgb['width'], 'mask': mask, 'mask_sha256': inventory[mask]})
    c.require(len(requests) == r.IMAGES, 'GT metadata count differs')
    sources = {'E_preparation_complete_sha256': c.sha(r.CONTROL / 'complete.json'),
               'E_runtime_binding_sha256': c.sha(r.OUTPUT / 'binding.json'),
               'all_six_E_verification_complete_sha256': c.sha(EVIDENCE / 'complete.json'),
               'calibration_assembly_complete_sha256': c.sha(r.x.ASSEMBLY / 'complete.json'),
               'split_sha256': c.sha(c.DATA / 'split.json'),
               'export_inventory_sha256': c.sha(c.INPUT / 'export_inventory.json'),
               'E_models': refs, 'probe_crosscheck': probe_reference()}
    return context, requests, sources


def protocol(context):
    return {'schema': 'real-E1004-exact-CPU-metrics-v1', 'implementation_sha256': implementation(),
            'source_manifest_digest': c.digest_json(context[2]), 'GT_requests_digest': c.digest_json(context[1]),
            'model_order': list(r.ORDER), 'images_per_model': r.IMAGES, 'unique_E_images': r.IMAGES,
            'native_pixels_per_model': r.PIXELS,
            'GT': {'mode': ['1', 'L'], 'native_shape': 'exact HxW; no resize/dilation/soft labels',
                   'raw_binary_encoding': 'each image values subset of {0,1} OR {0,255}; nonzero maps exactly to1',
                   'source': 'fixed original GT path/hash; hash before decode',
                   'actual_class_counts_and_positive_pixels': 'unknown until explicit run decodes all1004',
                   'official_counts_to_check_at_run': {'anomaly': 110, 'normal': 894},
                   'normalized_cache': 'uint8 0/1, one decode per original GT; shared across all six models'},
            'AP': {'algorithm': 'frozen metrics.exact_ap', 'frozen_source_sha256': c.sha(frozen.__file__),
                   'scope': 'global original pixels, each pixel once; no subsampling or mean per-image AP',
                   'histogram_bins': frozen.BINS, 'chunk': frozen.CHUNK,
                   'bytes_per_model_two_tables': frozen.BINS * 8, 'bytes_six_models': frozen.BINS * 8 * 6,
                   'ties': 'all identical float32 values grouped; negative zero normalized to positive zero',
                   'verification': 'original probe sklearn/ties evidence + frozen formula/native N/actual P checks',
                   'no_positive_pixels': 'NA and stop; never positive evidence'},
            'image_metrics': {'AUROC': 'sklearn roc_auc_score on all1004 saved native maxima',
                              'threshold': 'original per-model Q summary path/hash; strict >, no recalibration',
                              'counts': 'TP/FN on actual anomaly denominator, FP/TN on actual normal denominator'},
            'pairs': {'seeds': list(c.SEEDS), 'delta': '(AP_B - AP_A)*100 percentage points',
                      'all_three': True, 'mean': 'arithmetic mean of three paired deltas', 'SD': 'sample ddof=1',
                      'judgement': 'all positive and mean>0: consistent preliminary; mixed mean>0: unstable; mean<=0: no average gain',
                      'limits': 'single category/fixed5/split; no significance/generalization/screen claim; disclose FP/FN tradeoffs'},
            'examples': 'first normal, first anomaly, smallest ID in union FP and union FN; deduplicate with all reasons',
            'resource': {'minimum_available_RAM_GiB': 3, 'minimum_free_disk_GiB': 80,
                         'checks': 'entry/GT image/AP input chunk/model boundary; frozen scan only has original entry gate, not continuous monitoring'},
            'output': str(OUTPUT),
            'cache': 'complete same identity only verifies; any partial/unknown/failure preserved, no resume/reaccumulate',
            'prepare': 'metadata/hash/NPY headers/invented scalars only; no GT bytes/arrays or actual metrics',
            'models_or_training_or_calibration': False}


def scalar_validation(requests):
    c.require(len(requests) == r.IMAGES and sum(row['height'] * row['width'] for row in requests) == r.PIXELS,
              'Native metadata denominator differs')
    c.require(all(set(row) == {'ordinal', 'id', 'height', 'width', 'mask', 'mask_sha256'} for row in requests),
              'Unexpected GT request metadata')
    # No image/GT arrays or real metric calls: invented arithmetic only.
    c.require(not (.5 > .5) and .75 > .5, 'Strict greater-than tie rule differs')
    dummy = np.asarray([1., 2., 3.], dtype=np.float64)
    c.require(float(dummy.mean()) == 2 and float(dummy.std(ddof=1)) == 1
              and abs((.8 - .7) * 100 - 10) < 1e-12, 'Invented paired percentage/SD convention differs')
    c.require(frozen.CHUNK == 1048576 and frozen.BINS == 0x3f800001, 'Frozen exact float32 algorithm differs')
    return {'GT_metadata_requests': r.IMAGES, 'native_pixels_per_model': r.PIXELS,
            'actual_GT_decodes': 0, 'actual_class_or_positive_pixel_counts': 'not computed',
            'actual_AP_AUROC_counts_or_paired_conclusions': False,
            'E_Q_array_payloads_parsed': 0, 'saved_score_and_threshold_scalar_metadata_read': True,
            'invented_scalar_strict_ties_percentage_mean_sample_SD': True,
            'Torch_imported': 'torch' in sys.modules, 'new_model_operations': 0}


def prepared_context():
    context = source_context()
    complete = c.verify_manifest(CONTROL)
    c.require(set(complete['files']) == {'protocol.json', 'source_manifest.json', 'GT_requests.json', 'CPU_validation.json'}
              and complete['implementation_sha256'] == implementation()
              and c.read_json(CONTROL / 'protocol.json') == protocol(context)
              and c.read_json(CONTROL / 'source_manifest.json') == context[2]
              and c.read_json(CONTROL / 'GT_requests.json') == context[1], 'Frozen real metric preparation differs')
    scalar_validation(context[1])
    return context


def prepare():
    install_guard(writes=(CONTROL,))
    resources = c.resource(3, 80)
    context = source_context()
    if (CONTROL / 'complete.json').exists():
        prepared_context()
        return {'prepared_cache_verified': True, **scalar_validation(context[1])}
    c.require(not CONTROL.exists() and not OUTPUT.exists(), 'Partial/unknown metric destination; preserve and diagnose')
    CONTROL.mkdir()
    for name, value in [('protocol.json', protocol(context)), ('source_manifest.json', context[2]),
                        ('GT_requests.json', context[1]),
                        ('CPU_validation.json', {'resources': resources, **scalar_validation(context[1])})]:
        c.save(CONTROL / name, value)
    c.seal(CONTROL, {'implementation_sha256': implementation()})
    prepared_context()
    return {'prepared': True, 'resources': resources, **scalar_validation(context[1])}


def verify_preparation():
    install_guard()
    context = prepared_context()
    return {'preparation_verified': True, **scalar_validation(context[1])}


def binding():
    return {'preparation_complete_sha256': c.sha(CONTROL / 'complete.json'),
            'protocol_sha256': c.sha(CONTROL / 'protocol.json'), 'implementation_sha256': implementation()}


def read_bound_bytes(path, expected_sha256):
    """Hash and consume one immutable byte snapshot, not two path opens."""
    payload = Path(path).read_bytes()
    c.require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'Metric input bytes changed: ' + str(path))
    return payload


def load_bound_array(path, expected_sha256):
    return np.load(io.BytesIO(read_bound_bytes(path, expected_sha256)), allow_pickle=False)


def read_bound_json(path, expected_sha256):
    return json.loads(read_bound_bytes(path, expected_sha256).decode('utf-8'))


def decode_gt(request):
    from PIL import Image
    path = c.DATA / request['mask']
    payload = read_bound_bytes(path, request['mask_sha256'])
    with Image.open(io.BytesIO(payload)) as im:
        c.require(im.mode in ('1', 'L') and im.size == (request['width'], request['height']),
                  'GT native mode/shape differs; no conversion/resize')
        raw = np.asarray(im)
        c.require(raw.ndim == 2 and (np.all((raw == 0) | (raw == 1)) or np.all((raw == 0) | (raw == 255))),
                  'GT must be exact binary {0,1} or {0,255}; preserve and diagnose')
        return (raw != 0).astype(np.uint8)


def prepare_labels(requests, cursor):
    folder = OUTPUT / 'labels'
    folder.mkdir()
    rows = []
    for request in requests:
        c.resource(3, 80)
        cursor.update(phase='before_GT_decode', image=request['id'], model=None)
        r.x.cursor_write(OUTPUT, cursor)
        target = decode_gt(request)
        path = folder / f"{request['id']}.npy"
        with path.open('xb') as stream:
            np.save(stream, target, allow_pickle=False)
        rows.append({'request': request, 'normalized_sha256': c.sha(path),
                     'positive_pixels': int(target.sum()), 'anomaly': bool(target.any())})
    p = sum(row['positive_pixels'] for row in rows)
    anomaly = sum(row['anomaly'] for row in rows)
    summary = {'images': r.IMAGES, 'native_pixels': r.PIXELS, 'positive_pixels': p,
               'anomaly_images': anomaly, 'normal_images': r.IMAGES - anomaly, 'rows': rows}
    c.save(folder / 'summary.json', summary)
    c.require(p > 0, 'Actual positive pixels0: AP=NA; stop without positive interpretation')
    c.require((anomaly, r.IMAGES - anomaly) == (110, 894), 'Actual GT counts differ from official110/894; stop, no subset')
    return summary


def verify_labels(context):
    labels = c.read_json(OUTPUT / 'labels/summary.json')
    c.require(labels['images'] == r.IMAGES and labels['native_pixels'] == r.PIXELS
              and 0 < labels['positive_pixels'] < r.PIXELS
              and (labels['anomaly_images'], labels['normal_images']) == (110, 894)
              and len(labels['rows']) == r.IMAGES, 'Actual cached GT denominators differ')
    for request, row in zip(context[1], labels['rows']):
        c.require(row['request'] == request and c.sha(OUTPUT / 'labels' / f"{request['id']}.npy") == row['normalized_sha256'],
                  'Normalized GT source/order differs')
    c.require(sum(row['positive_pixels'] for row in labels['rows']) == labels['positive_pixels']
              and sum(row['anomaly'] for row in labels['rows']) == labels['anomaly_images'], 'GT aggregate counts differ')
    return labels


def metric_identity(ref):
    return {'binding': binding(), 'model': ref['label'], 'E_complete_sha256': ref['complete_sha256'],
            'normalized_GT_summary_sha256': c.sha(OUTPUT / 'labels/summary.json'),
            'frozen_metrics_sha256': c.sha(frozen.__file__), 'threshold_reference': ref['threshold_reference']}


def blocks(ref, labels, cursor):
    for row in labels['rows']:
        request = row['request']
        c.resource(3, 80)
        cursor.update(phase='AP_input', model=ref['label'], image=request['id'])
        r.x.cursor_write(OUTPUT, cursor)
        name = f"probabilities/{request['id']}.npy"
        probability = load_bound_array(c.ROOT / ref['folder'] / name, ref['files'][name])
        target = load_bound_array(OUTPUT / 'labels' / f"{request['id']}.npy", row['normalized_sha256'])
        shape = (request['height'], request['width'])
        c.require(probability.dtype == np.float32 and probability.shape == target.shape == shape
                  and target.dtype == np.uint8 and np.all((target == 0) | (target == 1))
                  and int(target.sum()) == row['positive_pixels'], 'Native probability/GT pairing differs')
        p, y = probability.reshape(-1), target.reshape(-1)
        for start in range(0, p.size, frozen.CHUNK):
            c.resource(3, 80)
            yield p[start:start + frozen.CHUNK], y[start:start + frozen.CHUNK]
    cursor.update(phase='AP_scan_or_save', model=ref['label'], image=None)
    r.x.cursor_write(OUTPUT, cursor)


def image_metrics(ref, labels):
    from sklearn.metrics import roc_auc_score
    saved = read_bound_json(c.ROOT / ref['scores'], ref['files']['scores.json'])
    y = np.asarray([row['anomaly'] for row in labels['rows']], dtype=bool)
    scores = np.asarray([row['score'] for row in saved], dtype=np.float64)
    threshold_source = ref['threshold_reference']
    path = c.ROOT / threshold_source['summary']
    tau = read_bound_json(path, threshold_source['summary_sha256'])['calibration']['threshold']
    predicted = scores > tau
    return {'AUROC': float(roc_auc_score(y, scores)),
            'TP': int(np.sum(predicted & y)), 'FN': int(np.sum(~predicted & y)),
            'FP': int(np.sum(predicted & ~y)), 'TN': int(np.sum(~predicted & ~y)),
            'anomaly_denominator': labels['anomaly_images'], 'normal_denominator': labels['normal_images'],
            'threshold_reference': threshold_source,
            'FP_ids': [row['request']['id'] for row, flag in zip(labels['rows'], predicted & ~y) if flag],
            'FN_ids': [row['request']['id'] for row, flag in zip(labels['rows'], ~predicted & y) if flag]}


def paired_results(models):
    pairs = []
    for seed in c.SEEDS:
        a = next(row for row in models if row['model'] == f'A_{seed}')
        b = next(row for row in models if row['model'] == f'B_{seed}')
        pairs.append({'seed': seed, 'AP_A': a['pixel_AP'], 'AP_B': b['pixel_AP'],
                      'delta_AP_percentage_points': (b['pixel_AP'] - a['pixel_AP']) * 100,
                      'delta_FP_images': b['image_metrics']['FP'] - a['image_metrics']['FP'],
                      'delta_FN_images': b['image_metrics']['FN'] - a['image_metrics']['FN']})
    values = np.asarray([row['delta_AP_percentage_points'] for row in pairs], dtype=np.float64)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    judgement = 'consistent preliminary gain on this fixed split' if np.all(values > 0) and mean > 0 else (
        'unstable positive mean on this fixed split' if mean > 0 else 'no average gain observed on this fixed split')
    return {'pairs': pairs, 'mean_delta_AP_percentage_points': mean,
            'sample_SD_delta_AP_percentage_points': sd, 'ddof': 1, 'judgement': judgement,
            'limits': 'single category/fixed5/split; three initialization seeds, no population significance or screen evidence',
            'tradeoffs': 'positive delta_FP or delta_FN means worse corresponding fixed-threshold count; disclose all pairs'}


def selected_examples(models, labels):
    selections = {}
    candidates = [(next(row['request']['id'] for row in labels['rows'] if not row['anomaly']), 'first normal'),
                  (next(row['request']['id'] for row in labels['rows'] if row['anomaly']), 'first anomaly')]
    for field in ('FP_ids', 'FN_ids'):
        ids = sorted({item for model in models for item in model['image_metrics'][field]})
        if ids:
            candidates.append((ids[0], 'smallest ID in six-model ' + field + ' union'))
    for id_, reason in candidates:
        selections.setdefault(id_, []).append(reason)
    return {'examples': [{'id': id_, 'reasons': reasons} for id_, reasons in selections.items()],
            'empty_FP_union': not any(m['image_metrics']['FP_ids'] for m in models),
            'empty_FN_union': not any(m['image_metrics']['FN_ids'] for m in models),
            'selection_feedback_to_training_or_parameters': False}


def verify_results_internal(context):
    complete = c.verify_manifest(OUTPUT)
    c.require(complete['binding'] == binding(), 'Complete real metric identity differs')
    txn, cursor = c.read_json(OUTPUT / 'transaction.json'), c.read_json(OUTPUT / 'cursor.json')
    c.require(txn['binding'] == binding() and cursor == {'token': txn['token'], 'phase': 'complete'},
              'Real metric transaction incomplete')
    labels = verify_labels(context)
    result = c.read_json(OUTPUT / 'metrics.json')
    expected = {'transaction.json', 'cursor.json', 'labels/summary.json', 'metrics.json'} | {
        f"labels/{row['request']['id']}.npy" for row in labels['rows']}
    c.require(result['binding'] == binding() and result['unique_E_images'] == r.IMAGES
              and result['native_pixels_per_model'] == r.PIXELS
              and result['actual_positive_pixels_per_model'] == labels['positive_pixels']
              and [row['model'] for row in result['models']] == list(r.ORDER), 'Final metric denominator/order differs')
    for ref, row in zip(context[2]['E_models'], result['models']):
        folder = OUTPUT / ref['label'] / 'exact'
        sealed = c.verify_manifest(folder)
        exact = c.read_json(folder / 'metrics.json')
        transaction = c.read_json(folder / 'transaction.json')
        c.require(set(sealed['files']) == {'transaction.json', 'counts.uint32', 'positive.uint32', 'metrics.json'}
                  and sealed['identity'] == transaction['identity'] == metric_identity(ref)
                  and transaction['expected_n'] == exact['pixels'] == r.PIXELS
                  and transaction['expected_p'] == exact['positive_pixels'] == labels['positive_pixels']
                  and exact['exact_float32_scores'] is True and exact['histogram_bins'] == frozen.BINS
                  and exact['AP'] == row['pixel_AP'] and 0 <= exact['AP'] <= 1,
                  'Exact global AP identity/count/formula differs')
        for name in ('counts.uint32', 'positive.uint32'):
            c.require((folder / name).stat().st_size == frozen.BINS * 4, 'Exact histogram size differs')
        m = row['image_metrics']
        c.require(0 <= m['AUROC'] <= 1 and m['TP'] + m['FN'] == labels['anomaly_images']
                  and m['FP'] + m['TN'] == labels['normal_images']
                  and m['threshold_reference'] == ref['threshold_reference']
                  and len(m['FP_ids']) == m['FP'] and len(m['FN_ids']) == m['FN'], 'Image count/threshold source differs')
        expected |= {f"{ref['label']}/exact/{n}" for n in sealed['files']} | {f"{ref['label']}/exact/complete.json"}
    c.require(set(complete['files']) == expected and result['paired'] == paired_results(result['models'])
              and result['selected_examples'] == selected_examples(result['models'], labels),
              'Unknown result files or paired/example convention differs')
    return result


def run():
    requests = c.read_json(CONTROL / 'GT_requests.json')
    install_guard(gt=[c.DATA / row['mask'] for row in requests], writes=(OUTPUT,))
    context = prepared_context()
    if (OUTPUT / 'complete.json').exists():
        return {'complete_cache_verified': True, 'result': verify_results_internal(context),
                'AP_or_AUROC_recomputed': False, 'new_model_operations': 0}
    c.require(not OUTPUT.exists(), 'Partial/unknown metric transaction; preserve, never resume/reaccumulate')
    resources = c.resource(3, 80)
    OUTPUT.mkdir()
    token = uuid.uuid4().hex
    cursor = {'token': token, 'phase': 'before_GT_decode', 'image': None, 'model': None}
    c.save(OUTPUT / 'transaction.json', {'token': token, 'binding': binding(), 'resources': resources})
    r.x.cursor_write(OUTPUT, cursor)
    started = time.monotonic()
    try:
        labels = prepare_labels(requests, cursor)
        models = []
        for ref in context[2]['E_models']:
            c.resource(3, 80)
            cursor.update(phase='before_exact_AP', model=ref['label'], image=None)
            r.x.cursor_write(OUTPUT, cursor)
            # Frozen routine creates its own complete two-table transaction.
            exact = frozen.exact_ap(lambda: blocks(ref, labels, cursor), OUTPUT / ref['label'] / 'exact',
                                    metric_identity(ref), r.PIXELS, labels['positive_pixels'])
            image = image_metrics(ref, labels)
            models.append({'model': ref['label'], 'pixel_AP': exact['AP'], 'image_metrics': image,
                           'exact_complete_sha256': c.sha(OUTPUT / ref['label'] / 'exact/complete.json')})
            print(json.dumps({'model': ref['label'], 'completed_models': len(models), 'of': 6}, ensure_ascii=False), flush=True)
        result = {'schema': 'real-E1004-exact-metrics-v1', 'binding': binding(), 'models': models,
                  'unique_E_images': r.IMAGES, 'native_pixels_per_model': r.PIXELS,
                  'actual_positive_pixels_per_model': labels['positive_pixels'],
                  'actual_anomaly_images': labels['anomaly_images'], 'actual_normal_images': labels['normal_images'],
                  'paired': paired_results(models), 'selected_examples': selected_examples(models, labels),
                  'elapsed_seconds_before_final_seal': time.monotonic() - started,
                  'new_model_operations': 0, 'thresholds_recomputed': 0, 'Torch_imported': False}
        c.save(OUTPUT / 'metrics.json', result)
        r.x.cursor_write(OUTPUT, {'token': token, 'phase': 'complete'})
        c.seal(OUTPUT, {'binding': binding()})
    except BaseException as error:
        if not (OUTPUT / 'failure.json').exists():
            c.save(OUTPUT / 'failure.json', {'token': token, 'cursor': cursor, 'error': repr(error),
                                           'action': 'preserve all partial tables; no automatic resume/reaccumulate'})
        raise
    # Recheck all frozen sources after the long computation and final table seal.
    return {'finished': True, 'result': verify_results_internal(prepared_context()), 'new_model_operations': 0}


def verify_results():
    install_guard()
    return {'complete_cache_verified': True, 'result': verify_results_internal(prepared_context()),
            'AP_or_AUROC_recomputed': False, 'new_model_operations': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify-preparation', 'run', 'verify-results'))
    args = parser.parse_args()
    functions = {'prepare': prepare, 'verify-preparation': verify_preparation, 'run': run, 'verify-results': verify_results}
    print(json.dumps(functions[args.command](), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
