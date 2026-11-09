"""Independent stage-5 D_fit scoring preparation and explicit read-only runtime.

prepare / verify-preparation / inspect: stdlib metadata, opaque byte hashes and
virtual scalars only. score / verify-scores require separate explicit invocation.
No original-image, render, calibration, optimizer or fitting entry exists.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.abc
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent.parent
BASE = ROOT / 'results/mura_response/semu_applicability_v1'
PREP = BASE / 'preparation_v1'
FIT = BASE / 'fit_inputs_identity_v2'
NORMAL_PREP = BASE / 'normal_preparation_v1'
NORMAL_INPUTS = BASE / 'normal_inputs_v1'
NORMAL_SCORES = BASE / 'normal_scoring_v1'
OUT = BASE / 'fit_scoring_preparation_v1'
SCORES = BASE / 'fit_scores_v1'
MODELS = ('PatchCore', 'G1')
NORMAL_COMPLETE = {
    'PatchCore': '2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec',
    'G1': '75dd6b85f50bd6ebc98591aa90496e6c74dd61e039905fccfecec299b0c7bcec'}
TAU = {'PatchCore': 1.852951288223265, 'G1': 0.6587551027536391}
ASSETS = tuple(ROOT / rel for rel in (
    'models/checkpoints/wide_resnet50_2-95faca4d.pth',
    'models/weak_localization/patchcore/nnscorer_search_index.faiss',
    'models/weak_localization/patchcore/patchcore_params.pkl',
    'models/controlled_generation/formal_draem/G1_20261030/reconstruction.pth',
    'models/controlled_generation/formal_draem/G1_20261030/segmentation.pth'))
HEAVY = ('torch', 'torchvision', 'faiss', 'timm', 'patchcore',
         'segmentation_models_pytorch')
ARRAY_PACKAGES = ('numpy', 'scipy', 'PIL')
FORBIDDEN = ('mura_response.resources', 'mura_response.generation',
             'mura_response.generation_v2', 'controlled_generation', 'pod_probe',
             'localization_calibration', 'localization_training', 'detector')
MODE = None
CURRENT_MODEL = None
READ_ARRAY_FILES = set()
WRITE_ROOTS = set()
BLOCKED = []
COUNTERS = {'source_decodes': 0, 'array_opens': 0, 'model_loads': 0,
            'tensor_loads': 0, 'model_input_forwards': 0,
            'author_placeholder_forwards': 0, 'parameter_updates': 0,
            'render_calls': 0, 'real_calibrations': 0, 'statistical_fits': 0}


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no automatic retry, resume or substitution')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')


def matches(name, prefixes):
    return any(name == p or name.startswith(p + '.') for p in prefixes)


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = matches(fullname, FORBIDDEN)
        model = matches(fullname, HEAVY) and MODE != 'score'
        array = matches(fullname, ARRAY_PACKAGES) and MODE not in ('score', 'verify-scores')
        if forbidden or model or array:
            BLOCKED.append({'import': fullname})
            require(False, 'Import outside explicit stage-5 mode: ' + fullname)


def file_guard(event, args):
    if event in ('socket.connect', 'pickle.find_class'):
        BLOCKED.append({'event': event})
        require(False, 'Network/unrestricted pickle not allowed')
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
    writing = any(ch in mode for ch in 'wax+')
    if len(args) > 2 and isinstance(args[2], int):
        writing = writing or bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and not any(p.is_relative_to(folder) for folder in WRITE_ROOTS):
        BLOCKED.append({'write': str(p)})
        require(False, 'Write outside this independent transaction')
    ext = p.suffix.lower()
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        BLOCKED.append({'original': str(p)})
        require(False, 'No original images are allowed in stage-5 scoring')
    if ext in ('.npy', '.npz'):
        if writing:
            allowed = MODE == 'score' and p.parent == (SCORES / CURRENT_MODEL / 'heatmaps').resolve()
        else:
            allowed = p in READ_ARRAY_FILES
        if not allowed:
            BLOCKED.append({'array': str(p)})
            require(False, 'Array outside explicit frozen RGB/own heatmap allowlist')
        if not writing:
            COUNTERS['array_opens'] += 1
    if ext in ('.pth', '.pt', '.pkl', '.pickle', '.faiss', '.safetensors'):
        if p not in {a.resolve() for a in ASSETS} or writing:
            BLOCKED.append({'asset': str(p)})
            require(False, 'Unlisted model asset; preparation access is opaque byte hashing only')


def install_guards(mode, model):
    global MODE, CURRENT_MODEL
    MODE, CURRENT_MODEL = mode, model
    if mode == 'prepare':
        WRITE_ROOTS.add(OUT.resolve())
    elif mode == 'score':
        WRITE_ROOTS.add((SCORES / model).resolve())
    sys.meta_path.insert(0, ImportGuard())
    sys.addaudithook(file_guard)


def boundary():
    return {**COUNTERS,
            'model_packages_loaded': [n for n in sys.modules if matches(n, HEAVY)],
            'array_packages_loaded': [n for n in sys.modules if matches(n, ARRAY_PACKAGES)],
            'original_file_opens': 0, 'blocked_attempts': list(BLOCKED),
            'D_hold_S_bg_Q_cal_old_test_GT_reads': 0,
            'unrestricted_pickle_deserializations': 0,
            'counter_semantics': 'forward attempts before call; completed records prove success'}


def assert_static():
    b = boundary()
    require(not any(COUNTERS.values()) and not b['model_packages_loaded']
            and not b['array_packages_loaded'] and not BLOCKED,
            'CPU preparation crossed the array/model boundary')
    return b


def prior_module():
    # Reuse frozen stdlib-only checks without installing its guards or changing paths.
    from . import inference
    return inference


def closed_metadata(folder, expected, directories=()):
    """Only JSON bytes and stat for payloads; full payload proof is reused, not rerun."""
    prior = prior_module()
    prior.inventory(folder, expected, directories, complete=True)
    complete = read(folder / 'complete.json')
    require(set(complete['outputs']) == set(expected), 'Closed metadata inventory differs')
    json_hashes = {n: digest(folder / n) for n in expected if n.endswith('.json')}
    require(all(complete['outputs'][n] == h for n, h in json_hashes.items()),
            'Frozen JSON identity differs')
    stats = [[n, (folder/n).stat().st_size, (folder/n).stat().st_mtime_ns]
             for n in sorted(expected)]
    return {'complete_sha256': digest(folder/'complete.json'),
            'json_sha256': json_hashes,
            'payload_stat_sha256': hashlib.sha256(json.dumps(stats).encode()).hexdigest(),
            'payload_bytes_not_opened_in_preparation': True}


def source_context():
    prior = prior_module()
    proof = prior.verify_preparation()  # Isolated resources child; metadata/asset hashes only.
    require(proof['status'] == 'NORMAL_PREPARATION_CACHE_VERIFIED', 'Normal preparation not verified')
    cfg = read(NORMAL_PREP/'protocol.json')
    roles = read(PREP/'roles.json')['roles']['D_fit60']
    requests = [r for r in read(PREP/'requests.json')['rows'] if r['role'] == 'D_fit60']
    manifest = read(FIT/'manifest.json')
    unique_doc = read(FIT/'unique_inputs.json')
    unique = unique_doc['rows']
    normal_rows = read(NORMAL_INPUTS/'records.json')['rows']
    normal_names = {'started.json', 'sources.json', 'records.json', 'verification.json'}
    normal_names |= {r['file'] for r in normal_rows}
    normal_meta = closed_metadata(NORMAL_INPUTS, normal_names, ('images',))
    require(normal_meta['complete_sha256'] == 'bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da',
            'Frozen normal input complete differs')
    require(len(normal_rows) == 300 and [r['role'] for r in normal_rows] == ['Q200']*200+['F100']*100,
            'Normal role order differs')
    normal_by_hash = {r['working_RGB_sha256']: r for r in normal_rows}
    require(len(normal_by_hash) == 300, 'Normal working contents are not unique')
    normal_models = {}
    for model in MODELS:
        f = NORMAL_SCORES/model
        expected = {'started.json','sources.json','q_result.json','result.json','verification.json'}
        expected |= {f'probabilities/{r["id"]}.npy' for r in normal_rows}
        expected |= {f'records/{r["id"]}.json' for r in normal_rows}
        meta = closed_metadata(f, expected, ('probabilities','records'))
        require(meta['complete_sha256'] == NORMAL_COMPLETE[model], 'Normal workpoint complete changed')
        v, q, result = read(f/'verification.json'), read(f/'q_result.json'), read(f/'result.json')
        normal_complete = read(f/'complete.json')
        signature = {'model':model,'protocol_sha256':digest(NORMAL_PREP/'protocol.json'),
                     'normal_inputs_complete_sha256':normal_meta['complete_sha256']}
        require(read(f/'sources.json') == signature and read(f/'started.json')['identity'] == signature
                and v['identity'] == q['identity'] == result['identity'] == signature,
                'Normal model/source identity differs')
        require(v['passed'] and v['state_before'] == v['state_after']
                and q['threshold'] == TAU[model] and q['q'] == .95 and q['method'] == 'linear'
                and q['rule'] == 'strictly greater' and result['Q'] == q
                and result['F_scores_did_not_change_tau'], 'Normal state/tau proof differs')
        b = v['boundary']
        require(b['model_input_forwards'] == 300 and b['real_calibrations'] == 1
                and b['parameter_updates'] == b['source_decodes'] == b['render_calls'] == b['statistical_fits'] == 0
                and not b['blocked_attempts'], 'Saved normal operation proof differs')
        for i, row in enumerate(normal_rows):
            record = read(f/f'records/{row["id"]}.json')
            require(record['identity'] == signature and record['id'] == row['id']
                    and record['ordinal'] == i and record['role'] == row['role']
                    and record['working_RGB_sha256'] == row['working_RGB_sha256']
                    and record['probability_file'] == f'probabilities/{row["id"]}.npy'
                    and record['probability_sha256'] == normal_complete['outputs'][record['probability_file']]
                    and record['state_unchanged'], 'Normal per-input identity proof differs')
        report = PROJECT / f'报告/Mura响应研究第4阶段{model}正常工作点与核验.md'
        match = re.search(chr(96)*3 + r'json\n(.*?)\n' + chr(96)*3,
                          report.read_text(encoding='utf-8'), re.S)
        require(match is not None, 'Independent normal cache proof missing')
        independent = json.loads(match.group(1))
        require(independent['status'] == 'NORMAL_MODEL_CACHE_VERIFIED'
                and independent['model'] == model and independent['threshold'] == q['threshold']
                and independent['model_complete_sha256'] == meta['complete_sha256']
                and independent['tau_recomputed'] is False
                and independent['boundary']['model_input_forwards'] == 0
                and independent['boundary']['model_loads'] == 0
                and independent['boundary']['real_calibrations'] == 0,
                'Independent normal verification proof differs')
        normal_models[model] = {**meta, 'threshold':q['threshold'],
            'q_result_sha256':digest(f/'q_result.json'),
            'state':v['state_before'], 'independent_report_sha256':digest(report),
            'independent_proof':independent, 'Q_timing':q['timing'],
            'prior_full_payload_verification_reused':True}
    require(len(roles) == 60 and len(requests) == len(manifest['rows']) == 7560
            and len(unique) == 7305 and unique_doc['references'] == 7620,
            'Frozen D_fit denominators differ')
    frozen_ids = [r['id'] for r in requests]
    require(frozen_ids == [r['id'] for r in manifest['rows']], 'Request original order differs')
    role_by_id = {r['id']:r for r in roles}
    reference_rows = []
    # Baselines first, then all requests in the original manifest order.
    baseline_work = {}
    for row in manifest['rows']:
        bid = row['background_id']
        require(bid in role_by_id, 'Unfixed baseline ID')
        if bid in baseline_work:
            require(baseline_work[bid] == row['background_image'], 'Baseline identity varies')
        baseline_work[bid] = row['background_image']
    for row in roles:
        reference_rows.append({'id':row['id'],'kind':'baseline','background_id':row['id'],
                               **baseline_work[row['id']]})
    for frozen, row in zip(requests, manifest['rows']):
        require(all(frozen[k] == row[k] for k in ('id','background_id','geometry_id','polarity','amplitude')),
                'Frozen request semantics differ')
        require(row['role'] == 'D_fit60' and row['retained'], 'Request removed or foreign role')
        zero = row['measurement']['zero_actual_change']
        require(not zero or row['image'] == row['background_image'], 'Zero-change input does not reuse baseline')
        reference_rows.append({'id':row['id'],'kind':'request','background_id':row['background_id'],
            'geometry_id':row['geometry_id'], 'geometry_file':row['geometry_file'],
            'zero_actual_change':zero, 'measurement_defined':row['measurement']['defined'],
            **row['image']})
    refs_by_hash = collections.defaultdict(list)
    for row in reference_rows:
        refs_by_hash[row['working_RGB_sha256']].append(row['id'])
    fit_complete = read(FIT/'complete.json')
    unique_rows = []
    overlap = 0
    for i, row in enumerate(unique):
        h, file = row['working_RGB_sha256'], row['file']
        require(file == f'images/{h}.npy' and refs_by_hash[h] == row['references'],
                'Unique RGB/order/reference membership differs')
        shared = normal_by_hash.get(h)
        overlap += int(shared is not None)
        unique_rows.append({'ordinal':i, **row, 'file_sha256':fit_complete['outputs'][file],
                            'reuse_normal_id':shared['id'] if shared else None})
    require(len(refs_by_hash) == len(unique_rows) and sum(len(r['references']) for r in unique_rows) == 7620,
            'Unique-to-reference coverage differs')
    require(len({r['id'] for r in reference_rows}) == 7620, 'Duplicated reference ID')
    denominators = {'backgrounds':60,'baseline_references':60,'request_references':7560,
        'all_references':7620,'unique_float_RGB':7305,
        'zero_actual_change_requests':sum(r.get('zero_actual_change',False) for r in reference_rows),
        'nonzero_actual_change_requests':sum(r['kind']=='request' and not r['zero_actual_change'] for r in reference_rows),
        'undefined_measurement_requests':sum(r['kind']=='request' and not r['measurement_defined'] for r in reference_rows),
        'unique_overlap_with_Q_F':overlap,'new_input_forwards_per_model':len(unique_rows)-overlap,
        'new_input_forwards_two_models':2*(len(unique_rows)-overlap),
        'PatchCore_author_placeholder_forwards_separate':1,'G1_author_placeholder_forwards_separate':0,
        'request_variants_are_not_independent_backgrounds':True,
        'preparation_actual_model_input_forwards':0}
    require(denominators['zero_actual_change_requests'] == 315
            and denominators['nonzero_actual_change_requests'] == 7245, 'Saved residual-state counts differ')
    return {'normal_preparation_complete_sha256':digest(NORMAL_PREP/'complete.json'),
        'normal_protocol_sha256':digest(NORMAL_PREP/'protocol.json'),
        'normal_models':normal_models, 'normal_inputs':normal_meta,
        'fit_metadata':cfg['sources']['fit'],
        'frozen_sources':{'assets':cfg['sources']['assets'],
                          'runtime':cfg['sources']['runtime'],
                          'author_and_evidence':cfg['sources']['author_and_evidence']},
        'frozen_implementation':cfg['implementation'],
        'design_sha256':digest(PROJECT/'计划/Mura检测器响应与SEMU代理刻度研究设计.md'),
        'denominators':denominators}, unique_rows, reference_rows


def first_peak(heat_flat):
    require(len(heat_flat) > 0 and all(math.isfinite(float(x)) for x in heat_flat), 'Nonfinite virtual heatmap')
    return max(range(len(heat_flat)), key=lambda i: heat_flat[i])


def canonical_state(native, torch, modules):
    # Frozen runtime uses tuples for named buffer descriptors; saved JSON uses
    # lists. Compare all stages in the same JSON domain without changing data.
    return json.loads(json.dumps(native.state_signature(torch,modules),allow_nan=False))


def response_scalars(score, tau, heat_flat, mask_flat):
    """Contract for future CPU postprocessing; never called on real outputs here."""
    require(len(heat_flat) == len(mask_flat) and math.isfinite(score) and math.isfinite(tau),
            'Invalid virtual response domain')
    peak = first_peak(heat_flat)
    alarm = score > tau
    return {'H_alarm':int(alarm),'H_joint':int(alarm and bool(mask_flat[peak])),
            'first_flat_peak':peak}


def scalar_audit():
    runtime_like = {'PatchCore':{'parameters_sha256':'a'*64,
        'buffers':[('layer2.bn.running_mean','torch.float32',[512])]}}
    saved_like = json.loads(json.dumps(runtime_like))
    class VirtualNative:
        @staticmethod
        def state_signature(torch, modules):
            return runtime_like
    require(runtime_like != saved_like and canonical_state(VirtualNative,None,None) == saved_like,
            'Virtual native/JSON signature domain differs')
    tied = response_scalars(2.,1.,[9.,9.,0.],[False,True,False])
    inside = response_scalars(2.,1.,[0.,9.,0.],[False,True,False])
    equal = response_scalars(1.,1.,[0.,9.,0.],[False,True,False])
    patch = response_scalars(2.,1.,[.1,.2,.3],[False,False,True])
    require(tied == {'H_alarm':1,'H_joint':0,'first_flat_peak':0}
            and inside['H_joint'] == 1 and equal['H_alarm'] == 0 and patch['H_joint'] == 1,
            'Virtual strict/first-peak/model-score contract failed')
    require(7620-315 == 7305 and 60+7245 == 7305 and 7560-315 == 7245,
            'Virtual reference/dedup arithmetic failed')
    return {'virtual_scalars_only':True,'native_tuple_JSON_list_signature_compatible':True,
            'tied_peak_first_outside':tied,
            'inside_peak':inside,'strict_equality_not_alarm':equal,
            'PatchCore_score_not_heatmap_max':patch,
            'zero_actual_change_is_not_defect_FN':True,
            'no_real_response_or_threshold_computed':True}


def configuration(sources):
    prior = prior_module()
    return {'schema':'mura-fit-scoring-preparation-v1','stage':5,
        'implementation':{str(Path(__file__).resolve()):digest(__file__)},
        'frozen_dependency_implementation':sources['frozen_implementation'],
        'source_binding':sources,
        'model_order':list(MODELS),
        'input_order':'frozen unique_inputs rows: baselines first then first occurrence of requests',
        'input':'existing canonical HWC256x256x3 float32 C-contiguous [0,1]; no PNG, resizing, conversion or /255',
        'preprocessing':read(NORMAL_PREP/'protocol.json')['models'],
        'runtime':'eval/FP32/batch1/workers0; threads4; TF32 and cudnn benchmark off; construction seed42',
        'G1_BN':'49 stateless BN(22+27), empty buffers; frozen final10000-step double weights',
        'thresholds':{n:sources['normal_models'][n]['threshold'] for n in MODELS},
        'threshold_rule':'score strictly greater than saved own normal-workpoint tau; never recompute quantile',
        'state_checks':'initial full signature equals normal-workpoint saved state; unchanged after each unique input and before sealing',
        'prediction_reads':'RGB only; no M_geo, GT, residual or threshold parameter',
        'CPU_postprocessing_only_later':{
            'main_response':'H_joint = (native score > own tau) and first row-major heatmap argmax in fixed M_geo',
            'tie_rule':'first flattened C-order index; never choose an in-region tied peak',
            'paired':'score difference, fixed-M_geo max-response difference and baseline alarm/peak separately',
            'baseline':'one heatmap per background reused for all nine M_geo; not540 independent normal images',
            'zero_change':'retain315 zero-change requests, reuse baseline output; no defect FN interpretation',
            'measurement_domain':'all7560 requests retained; undefined/zero/weak states separate; no M_eff selection',
            'not_a_causal_detection_or_segmentation_accuracy':True},
        'normal_reuse':'same canonical RGB digest, fixed implementation/model/precision/state and saved tau identities; referenced normal record/heatmap must fully verify before use; no normal forward',
        'transaction':'one independent model per process; exclusive new directory; inflight before construction/each input; no partial continuation, automatic retry or intersection',
        'completed_cache_reuse':'strict complete inventory and all source/output identities; read-only verify only, no model/forward/calibration',
        'resources':{'CPU_RAM_GiB':2,'GPU_start_RAM_GiB':4.5,'GPU_start_free_GiB':4,'free_D_GiB':40},
        'GPU_peak_scope':'reset after CUDA admission before construction, all unique inputs; PyTorch counters not whole-device usage',
        'timing_scope':'owned construction plus scoring/state/IO transaction, excluding prior metadata and final hash seal',
        'denominators':sources['denominators'],
        'fit_output':str(SCORES),
        'actual_model_compatibility_speed_and_gpu_peaks_measured_this_stage':False,
        'no_detector_training_bank_fit_render_calibration_or_statistical_fit':True}


def prep_names():
    return {'sources.json','protocol.json','unique_inputs.json','references.json',
            'scalar_verification.json','verification.json'}


def verify_preparation():
    prior = prior_module()
    sources, unique, refs = source_context()
    prior.verify_closed(OUT, prep_names())
    require(read(OUT/'sources.json') == sources and read(OUT/'protocol.json') == configuration(sources)
            and read(OUT/'unique_inputs.json') == {'rows':unique}
            and read(OUT/'references.json') == {'rows':refs}
            and read(OUT/'scalar_verification.json') == scalar_audit(),
            'Fit preparation implementation/source/reference binding differs')
    v = read(OUT/'verification.json')
    require(v['status'] == 'FIT_SCORING_PREPARED' and v['boundary']['model_input_forwards'] == 0,
            'Saved preparation boundary differs')
    return {'status':'FIT_SCORING_PREPARATION_CACHE_VERIFIED','output':str(OUT),
            'complete_sha256':digest(OUT/'complete.json'),
            'denominators':sources['denominators'],'boundary':assert_static()}


def prepare():
    sources, unique, refs = source_context()
    prior = prior_module()
    resource = prior.resources(gpu=True)
    if OUT.exists():
        return verify_preparation()
    OUT.mkdir(exist_ok=False)
    try:
        write(OUT/'sources.json',sources)
        write(OUT/'protocol.json',configuration(sources))
        write(OUT/'unique_inputs.json',{'rows':unique})
        write(OUT/'references.json',{'rows':refs})
        write(OUT/'scalar_verification.json',scalar_audit())
        write(OUT/'verification.json',{'status':'FIT_SCORING_PREPARED',
            'boundary':assert_static(),'resources':resource,
            'array_model_compatibility_not_executed':True})
        prior.seal(OUT,prep_names())
    except BaseException as exc:
        if not (OUT/'failure.json').exists():
            write(OUT/'failure.json',{'error':type(exc).__name__,'message':str(exc),
                                    'boundary':boundary()})
        raise
    return {'status':'FIT_SCORING_PREPARED','output':str(OUT),'resources':resource,
            'denominators':sources['denominators'],'boundary':assert_static()}


def runtime_binding(model):
    # No actual source arrays or model are touched by this metadata verification.
    # assert_static() is still valid until the first explicit runtime import.
    verify_preparation()
    return {'model':model,'preparation_complete_sha256':digest(OUT/'complete.json'),
            'own_normal_model_complete_sha256':NORMAL_COMPLETE[model],
            'own_tau':TAU[model]}


def expected_score_names(unique):
    return {'started.json','sources.json','references.json','verification.json'} | {
        f'records/{r["working_RGB_sha256"]}.json' for r in unique} | {
        f'heatmaps/{r["working_RGB_sha256"]}.npy' for r in unique if r['reuse_normal_id'] is None}


def allow_runtime_arrays(model, unique):
    READ_ARRAY_FILES.update((FIT/r['file']).resolve() for r in unique)
    READ_ARRAY_FILES.update((SCORES/model/f'heatmaps/{r["working_RGB_sha256"]}.npy').resolve()
                            for r in unique if r['reuse_normal_id'] is None)
    READ_ARRAY_FILES.update((NORMAL_SCORES/model/f'probabilities/{r["reuse_normal_id"]}.npy').resolve()
                            for r in unique if r['reuse_normal_id'] is not None)


def load_rgb(row, np):
    from . import inputs
    payload = (FIT/row['file']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == row['file_sha256'], 'Frozen float RGB bytes differ')
    rgb = np.load(io.BytesIO(payload),allow_pickle=False)
    require(inputs.working_rgb_identity(rgb) == row['working_RGB_sha256'] and rgb.flags.c_contiguous,
            'Canonical RGB changed or needs implicit conversion')
    return rgb  # Alreadyfloat32/255 + injected float residual; NEVER divide again.


def validate_heat(model, score, heat, np):
    require(heat.dtype == np.float32 and heat.shape == (256,256)
            and np.isfinite(heat).all() and math.isfinite(score), 'Invalid native heatmap/score')
    if model == 'G1':
        require(float(heat.min()) >= 0 and float(heat.max()) <= 1
                and score == float(heat.max()), 'G1 softmax/max contract differs')


def reused_normal(model, row, signature, np):
    normal = NORMAL_SCORES/model
    saved = read(normal/f'records/{row["reuse_normal_id"]}.json')
    require(saved['working_RGB_sha256'] == row['working_RGB_sha256']
            and saved['state_unchanged'], 'Normal reuse RGB/state differs')
    path = normal/saved['probability_file']
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == saved['probability_sha256'],
            'Normal reused heatmap identity differs')
    heat = np.load(io.BytesIO(payload),allow_pickle=False)
    validate_heat(model,saved['score'],heat,np)
    return {'identity':signature,'ordinal':row['ordinal'],
        'working_RGB_sha256':row['working_RGB_sha256'],'score':saved['score'],
        'heatmap_file':str(path),'heatmap_sha256':saved['probability_sha256'],
        'reused_normal_id':row['reuse_normal_id'],'state_unchanged':True}


def score(model):
    signature = runtime_binding(model)
    unique = read(OUT/'unique_inputs.json')['rows']
    folder = SCORES/model
    if folder.exists():
        return verify_scores(model, signature=signature)
    prior = prior_module()
    start_resources = prior.resources(gpu=True)
    allow_runtime_arrays(model,unique)
    import numpy as np
    from . import inference_runtime as native
    require(not COUNTERS['model_loads'] and not COUNTERS['model_input_forwards'], 'Needs fresh model process')
    # A missing score root is an owned container; an unknown existing root is not adopted.
    if SCORES.exists():
        require(SCORES.is_dir() and not SCORES.is_symlink() and not SCORES.is_junction()
                and set(p.name for p in SCORES.iterdir()) <= set(MODELS), 'Unknown score container')
    else:
        SCORES.mkdir(exist_ok=False)
    folder.mkdir(exist_ok=False)
    (folder/'heatmaps').mkdir();(folder/'records').mkdir()
    started = time.perf_counter();done=[];samples=[start_resources]
    try:
        write(folder/'started.json',{'identity':signature,'scope':'one frozen model; no training or calibration'})
        write(folder/'sources.json',signature)
        write(folder/'references.json',read(OUT/'references.json'))
        write(folder/'inflight.json',{'identity':signature,'operation':'model_construction'})
        torch,modules,handle = native.build_model(sys.modules[__name__],model,read(NORMAL_PREP/'protocol.json'))
        initial = canonical_state(native,torch,modules)
        cfg = read(OUT/'protocol.json')
        require(initial == cfg['source_binding']['normal_models'][model]['state'],
                'Model initialization/BN state differs from its normal workpoint')
        (folder/'inflight.json').unlink()
        for row in unique:
            samples.append(prior.resources())
            h=row['working_RGB_sha256']
            if row['reuse_normal_id'] is not None:
                record=reused_normal(model,row,signature,np)
            else:
                rgb=load_rgb(row,np)
                write(folder/'inflight.json',{'identity':signature,'operation':'unique_RGB_forward',
                                             'ordinal':row['ordinal'],'working_RGB_sha256':h})
                COUNTERS['model_input_forwards']+=1
                score_value,heat=native.predict_rgb(torch,model,handle,rgb)
                torch.cuda.synchronize()
                validate_heat(model,score_value,heat,np)
                if model=='G1':
                    native.verify_stateless(sys.modules[__name__],torch,modules['reconstruction'],22)
                    native.verify_stateless(sys.modules[__name__],torch,modules['segmentation'],27)
                require(canonical_state(native,torch,modules)==initial,
                        'Parameters/buffers/training/gradient state changed')
                target=folder/f'heatmaps/{h}.npy'
                with target.open('xb') as stream: np.save(stream,heat,allow_pickle=False)
                record={'identity':signature,'ordinal':row['ordinal'],'working_RGB_sha256':h,
                    'score':score_value,'heatmap_file':f'heatmaps/{h}.npy',
                    'heatmap_sha256':digest(target),'reused_normal_id':None,'state_unchanged':True}
            write(folder/f'records/{h}.json',record)
            done.append(h)
            if (folder/'inflight.json').exists(): (folder/'inflight.json').unlink()
            print(json.dumps({'model':model,'unique_input':len(done),'of':len(unique),
                              'new_forward_attempts':COUNTERS['model_input_forwards']}),flush=True)
        expected=cfg['denominators']['new_input_forwards_per_model']
        require(COUNTERS['model_input_forwards']==expected and len(done)==7305
                and COUNTERS['author_placeholder_forwards']==(1 if model=='PatchCore' else 0)
                and COUNTERS['real_calibrations']==COUNTERS['parameter_updates']==0
                and canonical_state(native,torch,modules)==initial,'Final state/forward count differs')
        # Recheck frozen source hashes/metadata after the long run; no scalar/actual response analysis.
        current,_,_=source_context()
        require(current==read(OUT/'sources.json'),'Sources changed during owned scoring transaction')
        write(folder/'verification.json',{'passed':True,'identity':signature,
            'unique_inputs_completed':len(done),'denominators':cfg['denominators'],
            'state_before':initial,'state_after':canonical_state(native,torch,modules),
            'boundary':boundary(),'resources':samples,'elapsed_seconds':time.perf_counter()-started,
            'timing_scope':cfg['timing_scope'],'GPU_peak_scope':cfg['GPU_peak_scope'],
            'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'H_joint_or_statistical_fit_not_computed':True})
        prior.seal(folder,expected_score_names(unique),('heatmaps','records'))
    except BaseException as exc:
        if not (folder/'failure.json').exists():
            write(folder/'failure.json',{'error':type(exc).__name__,'message':str(exc),
                'boundary':boundary(),'completed_unique_inputs':len(done),'no_automatic_retry_or_resume':True})
        raise
    return {'status':'FIT_MODEL_SCORING_COMPLETE','model':model,'output':str(folder),
            'unique_inputs':len(done),'boundary':boundary(),'no_metrics_or_fitting':True}


def verify_scores(model, signature=None):
    if signature is None:
        signature=runtime_binding(model)
    unique=read(OUT/'unique_inputs.json')['rows'];cfg=read(OUT/'protocol.json');folder=SCORES/model
    allow_runtime_arrays(model,unique)
    prior=prior_module();prior.resources()
    prior.verify_closed(folder,expected_score_names(unique),('heatmaps','records'))
    require(read(folder/'sources.json')==signature and read(folder/'started.json')['identity']==signature
            and read(folder/'references.json')==read(OUT/'references.json'),'Fit score source/reference identity differs')
    import numpy as np
    for row in unique:
        # Source file bytes are verified, not loaded into a model or re-decoded.
        require(digest(FIT/row['file'])==row['file_sha256'],'Current frozen RGB source file differs')
        record=read(folder/f'records/{row["working_RGB_sha256"]}.json')
        require(record['identity']==signature and record['ordinal']==row['ordinal']
                and record['working_RGB_sha256']==row['working_RGB_sha256']
                and record['reused_normal_id']==row['reuse_normal_id'] and record['state_unchanged'],
                'Unique score record identity/order differs')
        if row['reuse_normal_id'] is not None:
            require(record==reused_normal(model,row,signature,np),'Reused normal cache record differs')
        else:
            expected=f'heatmaps/{row["working_RGB_sha256"]}.npy'
            require(record['heatmap_file']==expected,'Saved heatmap path differs')
            payload=(folder/expected).read_bytes()
            require(hashlib.sha256(payload).hexdigest()==record['heatmap_sha256'],'Saved native output identity differs')
            heat=np.load(io.BytesIO(payload),allow_pickle=False)
            validate_heat(model,record['score'],heat,np)
    v=read(folder/'verification.json');b=v['boundary']
    require(v['passed'] and v['identity']==signature and v['unique_inputs_completed']==7305
            and v['denominators']==cfg['denominators']
            and v['state_before']==v['state_after']==cfg['source_binding']['normal_models'][model]['state']
            and b['model_input_forwards']==cfg['denominators']['new_input_forwards_per_model']
            and b['author_placeholder_forwards']==(1 if model=='PatchCore' else 0)
            and b['real_calibrations']==b['parameter_updates']==b['render_calls']==b['statistical_fits']==0
            and not b['blocked_attempts'],'Saved fit state/count proof differs')
    require(not COUNTERS['model_loads'] and not COUNTERS['model_input_forwards']
            and not COUNTERS['real_calibrations'],'Cache reuse started new scoring/calibration')
    return {'status':'FIT_MODEL_CACHE_VERIFIED','model':model,'output':str(folder),
            'complete_sha256':digest(folder/'complete.json'),'denominators':cfg['denominators'],
            'tau_recomputed':False,'new_model_input_forwards':0,'boundary':boundary()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify-preparation','inspect','score','verify-scores'))
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    require((args.command in ('score','verify-scores')) == (args.model is not None),
            'Exactly one model is required only by explicit runtime modes')
    install_guards(args.command,args.model)
    if args.command=='prepare': result=prepare()
    elif args.command=='verify-preparation': result=verify_preparation()
    elif args.command=='inspect':
        sources,_,_=source_context()
        result={'status':'FIT_SCORING_STATIC_INSPECTION','denominators':sources['denominators'],
                'resources':prior_module().resources(gpu=True),'boundary':assert_static(),
                'preparation_exists':OUT.exists(),'actual_fit_scoring_not_started':not SCORES.exists()}
    elif args.command=='score': result=score(args.model)
    else: result=verify_scores(args.model)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

