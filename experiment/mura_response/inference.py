"""Stage-4 explicit preparation and normal-workpoint entry points.

Preparation uses stdlib metadata/file hashes only. Never import resources or
generation_v2 here: their permanent guards belong to separate processes.
Runtime image/array/model work is reachable only through an explicit CLI mode.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent.parent
BASE = ROOT / 'results/mura_response/semu_applicability_v1'
PREP = BASE / 'preparation_v1'
FIT = BASE / 'fit_inputs_identity_v2'
NORMAL_PREP = BASE / 'normal_preparation_v1'
NORMAL_INPUTS = BASE / 'normal_inputs_v1'
SCORES = BASE / 'normal_scoring_v1'
FIT_REPORT = PROJECT / '报告/Mura响应研究第3阶段独立身份适配与CPU取得结果.md'
PREP_SHA = '0fd936ea58f8c8a7d16c209c96d116daa0f34c6044d1b809781bd4b01f7c7ba5'
FIT_SHA = '3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593'
MODELS = ('PatchCore', 'G1')
HEAVY = ('torch', 'torchvision', 'faiss', 'timm', 'patchcore',
         'segmentation_models_pytorch')
OLD_RUNTIME = ('mura_response.resources', 'mura_response.generation',
               'mura_response.generation_v2', 'controlled_generation',
               'localization_calibration', 'localization_training', 'detector')
MODEL_ALLOWED = False
ARRAY_ROOTS = set()
ORIGINALS = set()
OPAQUE = set()
ACCESS = collections.Counter()
BLOCKED = []
COUNTERS = {'source_decodes': 0, 'array_opens': 0, 'model_loads': 0,
            'tensor_loads': 0, 'model_input_forwards': 0,
            'author_placeholder_forwards': 0, 'parameter_updates': 0,
            'render_calls': 0, 'real_calibrations': 0, 'statistical_fits': 0}


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state; no automatic retry, resume or substitution')


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        old = any(fullname == n or fullname.startswith(n + '.') for n in OLD_RUNTIME)
        heavy = any(fullname == n or fullname.startswith(n + '.') for n in HEAVY)
        if old or (heavy and not MODEL_ALLOWED):
            BLOCKED.append({'import': fullname})
            require(False, 'Import forbidden in this explicit process mode: ' + fullname)


def file_guard(event, args):
    if event == 'socket.connect':
        BLOCKED.append({'network': event})
        require(False, 'Network access is not part of frozen local scoring')
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
    writing = any(ch in mode for ch in 'wax+')
    if writing and not any(p.is_relative_to(folder) for folder in (NORMAL_PREP, NORMAL_INPUTS, SCORES)):
        BLOCKED.append({'write': str(p)})
        require(False, 'Write outside new stage-4 output roots')
    suffix = p.suffix.lower()
    if suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        if p not in ORIGINALS or writing:
            BLOCKED.append({'original': str(p)})
            require(False, 'Original image outside fixed Q/F acquisition')
        ACCESS[str(p)] += 1
        require(ACCESS[str(p)] == 1, 'Original file opened more than once in this acquisition')
    if suffix in ('.npy', '.npz'):
        if not any(p.is_relative_to(folder) for folder in ARRAY_ROOTS):
            BLOCKED.append({'array': str(p)})
            require(False, 'Input/probability array forbidden in this process mode')
        if not writing:
            COUNTERS['array_opens'] += 1
    if suffix in ('.pth', '.pt', '.pkl', '.pickle', '.faiss', '.safetensors'):
        if p not in OPAQUE or writing:
            BLOCKED.append({'model_payload': str(p)})
            require(False, 'Unlisted model payload; static access is byte hashing only')


def install_guards(model_allowed=False, array_roots=()):
    global MODEL_ALLOWED
    MODEL_ALLOWED = model_allowed
    ARRAY_ROOTS.update(Path(p).resolve() for p in array_roots)
    sys.meta_path.insert(0, ImportGuard())
    sys.addaudithook(file_guard)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')


def boundary():
    # Non-throwing capture is also safe in an exception handler.
    loaded = [n for n in sys.modules if any(n == m or n.startswith(m + '.') for m in HEAVY)]
    return {**COUNTERS, 'original_file_opens': sum(ACCESS.values()),
            'original_unique_paths': len(ACCESS), 'blocked_attempts': list(BLOCKED),
            'model_packages_loaded': loaded,
            'Q_F_acquisition_only_if_explicit': True,
            'D_hold_S_bg_Q_cal_old_test_GT_reads': 0, 'unrestricted_pickle_deserializations': 0,
            'forward_counter_semantics': 'attempts before call; only committed rows are completed outputs'}


def assert_static():
    b = boundary()
    require(not b['model_packages_loaded'] and not b['blocked_attempts']
            and not ACCESS and not any(COUNTERS.values()), 'Static preparation crossed data/model boundary')
    return b


def resources(gpu=False):
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'page_available',
                                            'virtual', 'virtual_available', 'extended')]
    state = Memory(); state.length = ctypes.sizeof(state)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(Memory)]; query.restype = ctypes.c_int
    require(query(ctypes.byref(state)), 'RAM query failed')
    info = {'available_RAM_GiB': state.available / 2**30,
            'free_D_GiB': shutil.disk_usage(ROOT).free / 2**30}
    require(info['available_RAM_GiB'] >= (4.5 if gpu else 2)
            and info['free_D_GiB'] >= 40, 'RAM/disk admission not met')
    if gpu:
        run = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
                              '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, encoding='utf-8', timeout=30)
        require(run.returncode == 0, 'GPU metadata query failed')
        rows = [r.strip().split(', ') for r in run.stdout.splitlines() if r.strip()]
        require(len(rows) == 1 and len(rows[0]) == 4, 'Expected one unambiguous GPU')
        name, driver, total, free = rows[0]
        info['GPU'] = {'name': name, 'driver': driver, 'total_MiB': int(total), 'free_MiB': int(free)}
        require(int(free) >= 4096, 'Need4GiB free GPU before model creation')
    return info


def runtime():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'executable_sha256': digest(sys.executable),
            'packages_metadata_only': {n: importlib.metadata.version(n) for n in
                ('numpy', 'scipy', 'Pillow', 'torch', 'torchvision', 'faiss-cpu', 'timm', 'tqdm')}}


def inventory(folder, names, directories=(), complete=False):
    require(folder.is_dir() and not folder.is_symlink() and not folder.is_junction(),
            'Output directory absent/redirected')
    members = list(folder.rglob('*'))
    expected = set(names) | ({'complete.json'} if complete else set())
    require({p.relative_to(folder).as_posix() for p in members if p.is_file()} == expected
            and {p.relative_to(folder).as_posix() for p in members if p.is_dir()} == set(directories)
            and all(not p.is_symlink() and not p.is_junction()
                    and p.resolve().is_relative_to(folder) for p in members), 'Strict directory inventory differs')


def seal(folder, names, directories=()):
    inventory(folder, names, directories)
    write(folder / 'complete.json', {'outputs': {n: digest(folder / n) for n in sorted(names)}})
    inventory(folder, names, directories, complete=True)


def verify_closed(folder, names, directories=()):
    require((folder / 'complete.json').is_file() and not (folder / 'failure.json').exists(),
            'Partial/unknown transaction; no automatic continuation')
    inventory(folder, names, directories, complete=True)
    complete = read(folder / 'complete.json')
    require(set(complete['outputs']) == set(names), 'Complete adopts unexpected output')
    require(all(digest(folder / n) == h for n, h in complete['outputs'].items()),
            'Complete file identity differs')


def linear_quantile(values, q=.95):
    require(values and all(math.isfinite(x) for x in values) and 0 <= q <= 1, 'Invalid finite scores')
    ordered = sorted(float(x) for x in values)
    pos = (len(ordered) - 1) * q; low = math.floor(pos); high = math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def wilson(fp, n):
    require(type(fp) is int and type(n) is int and 0 <= fp <= n and n > 0, 'Invalid Wilson denominator')
    z = 1.959963984540054; p = fp/n; den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    # The Wilson analytical endpoints are exact at fp=0/n; avoid cancellation crumbs.
    return [0. if fp == 0 else max(0., centre-half),
            1. if fp == n else min(1., centre+half)]


def scalar_audit():
    values = list(range(200)); tau = linear_quantile(values)
    require(abs(tau - 189.05) < 1e-12 and sum(s > tau for s in values) == 10,
            'Virtual q.95 linear scalar check failed')
    require(linear_quantile([.25]*200) == .25
            and sum(s > .25 for s in [.25]*200) == 0, 'Tie/strict comparison scalar check failed')
    require(wilson(0,100)[0] == 0 and wilson(100,100)[1] == 1, 'Wilson boundary check failed')
    return {'virtual_values_only': True, 'linear200_tau': tau, 'virtual_FP': 10,
            'constant_virtual_FP': 0, 'Wilson0_of100': wilson(0,100),
            'Wilson100_of100': wilson(100,100), 'actual_threshold_computed': False}


def fit_metadata(prep):
    require(digest(FIT / 'complete.json') == FIT_SHA, 'Frozen fit complete differs')
    complete = read(FIT / 'complete.json')
    audit = read(FIT / 'source_audit.json'); manifest = read(FIT / 'manifest.json')
    unique = read(FIT / 'unique_inputs.json'); verified = read(FIT / 'verification.json')
    names = {'started.json', 'sources.json', 'protocol.json', 'source_audit.json', 'coverage.json',
             'manifest.json', 'unique_inputs.json', 'verification.json'}
    names |= {f'geometry/{g["spec"]["id"]}.npz' for g in read(PREP / 'geometry.json')['rows']}
    names |= {f'residuals/{r["id"]}.npz' for r in prep['fit_requests']}
    names |= {f'images/{r["working_RGB_sha256"]}.npy' for r in unique['rows']}
    require(set(complete['outputs']) == names and len(names) == 14882, 'Fit expected inventory differs')
    inventory(FIT, names, ('images','geometry','residuals'), complete=True)
    json_hashes = {n: digest(FIT / n) for n in names if n.endswith('.json')}
    require(all(complete['outputs'][n] == h for n,h in json_hashes.items()), 'Fit JSON metadata changed')
    require(verified['passed'] and verified['summary'] == {**manifest['summary'],
        'backgrounds':60,'old_training_hash_only_paths':160,'geometry_arrays_verified':9,
        'all_request_arrays_verified':7560,'source_originals_reopened_for_verification':0,
        'render_calls_in_cache_verification':0,'references_declaration_verified':7620,
        'source_identity_separation_verified':True,
        'independence':'60 fixed backgrounds; variants not independent physical panels'}, 'Fit full-verification evidence differs')
    proof_text = FIT_REPORT.read_text(encoding='utf-8')
    match = re.search(r'```json\n(.*?)\n```', proof_text, re.S)
    require(match is not None, 'Independent stage3 proof absent')
    proof = json.loads(match.group(1))
    require(proof['status'] == 'FIT_INPUTS_IDENTITY_V2_CACHE_VERIFIED'
            and proof['complete_sha256'] == FIT_SHA and proof['frozen_files_verified'] == 14882
            and proof['current_invocation_boundary']['original_decodes']
                == proof['current_invocation_boundary']['render_calls'] == 0, 'Independent stage3 proof differs')
    for p,h in read(FIT / 'sources.json')['implementation'].items():
        require(digest(p) == h, 'Frozen fit implementation changed')
    stats = [[n,(FIT/n).stat().st_size,(FIT/n).stat().st_mtime_ns] for n in sorted(names)]
    return {'complete_sha256': FIT_SHA, 'json_sha256': json_hashes,
            'report_sha256': digest(FIT_REPORT), 'independent_proof': proof,
            'inventory_stat_sha256': hashlib.sha256(json.dumps(stats).encode()).hexdigest(),
            'payloads_not_rehashed_or_parsed_this_stage': True,
            'prior_full_array_verification_reused': True,
            'old_training_working_hashes': [r['working_RGB_sha256'] for r in audit['old_training']],
            'fit_working_hashes': [r['working_RGB_sha256'] for r in audit['backgrounds']]}


def context():
    child = subprocess.run([sys.executable,'-B','-X','utf8','-m','mura_response.resources','verify'],
                           cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
    require(child.returncode == 0, 'Isolated preparation verification failed: '+child.stderr[-1000:])
    require(json.loads(child.stdout)['status'] == 'PREPARATION_CACHE_VERIFIED'
            and digest(PREP/'complete.json') == PREP_SHA, 'Preparation identity differs')
    complete = read(PREP/'complete.json'); protocol = read(PREP/'protocol.json')
    roles = read(PREP/'roles.json'); source = read(PREP/'sources.json')
    prep = {'fit_requests':[r for r in read(PREP/'requests.json')['rows'] if r['role']=='D_fit60']}
    fit = fit_metadata(prep)
    actual_runtime = runtime()
    require(actual_runtime['python'] == protocol['runtime']['python']
            and actual_runtime['executable_sha256'] == protocol['runtime']['executable_sha256']
            and all(actual_runtime['packages_metadata_only'][n] == protocol['runtime']['packages_metadata_only'][n]
                    for n in ('numpy','scipy','Pillow','torch','torchvision')), 'Frozen runtime changed')
    asset_paths = [ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth',
                   ROOT/'models/weak_localization/patchcore/nnscorer_search_index.faiss',
                   ROOT/'models/weak_localization/patchcore/patchcore_params.pkl',
                   ROOT/'models/controlled_generation/formal_draem/G1_20261030/reconstruction.pth',
                   ROOT/'models/controlled_generation/formal_draem/G1_20261030/segmentation.pth']
    OPAQUE.update(p.resolve() for p in asset_paths)
    assets = {str(p): digest(p) for p in asset_paths}
    require(all(source['files'][p] == h for p,h in assets.items()), 'Frozen bank/final weight identity differs')
    extra = [ROOT/'localization_calibration.py',ROOT/'detector.py',ROOT/'vendor/DRAEM/model_unet.py',
             ROOT/'results/weak_localization/patchcore_bank.json',
             ROOT/'results/controlled_generation/formal_preparation/common_config.json',
             ROOT/'results/controlled_generation/formal_training/G1_20261030/training_result.json',
             ROOT/'results/controlled_generation/formal_training/G1_20261030/day8_verification.json']
    extra += sorted((ROOT/'vendor/patchcore-inspection/src/patchcore').glob('*.py'))
    extra.append(ROOT/'vendor/patchcore-inspection/LICENSE')
    extra_hashes = {str(p):digest(p) for p in extra}
    for p,h in extra_hashes.items():
        if p in source['files']:
            require(h == source['files'][p], 'Known frozen author/evidence source changed')
    bank = read(ROOT/'results/weak_localization/patchcore_bank.json')
    train = read(ROOT/'results/controlled_generation/formal_training/G1_20261030/training_result.json')
    gcheck = read(ROOT/'results/controlled_generation/formal_training/G1_20261030/day8_verification.json')
    require(bank['N_only'] and bank['normal_images']==80 and bank['input']==256
            and bank['dimensions']==1024 and bank['vectors']==819 and bank['coreset']==.01,
            'Frozen PatchCore bank contract differs')
    require(train['training_complete'] and train['condition']=='G1' and train['steps']==10000
            and train['seed']==20261030 and gcheck['verification_passed']
            and gcheck['verified_steps']==10000 and gcheck['checkpoint_count']==10,
            'G1 final-training verification differs')
    source_lines=[line for line in (ROOT/'README.md').read_text(encoding='utf-8').splitlines()
                  if 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad' in line]
    require(len(source_lines)==1 and 'amazon-science/patchcore-inspection' in source_lines[0],
            'Existing pinned PatchCore source record missing/ambiguous')
    normal_roles=roles['roles']['Q200']+roles['roles']['F100']
    require(len(roles['roles']['Q200'])==200 and len(roles['roles']['F100'])==100
            and len({r['id'] for r in normal_roles})==len({r['path'] for r in normal_roles})==300,
            'Normal role/reference denominator differs')
    return {'preparation':{**complete['outputs'],'complete.json':PREP_SHA},'fit':fit,
            'PatchCore_source_record':source_lines[0],
            'vendor_hash_scope':'observed local files bound now; not an upstream archive revalidation',
            'runtime':actual_runtime,'assets':assets,'author_and_evidence':extra_hashes,
            'dataset_root':roles['dataset_root'], 'Q200':roles['roles']['Q200'],
            'F100':roles['roles']['F100'],
            'G1_final_parameter_hashes':{'reconstruction':train['final_reconstruction_parameters_sha256'],
                                       'segmentation':train['final_segmentation_parameters_sha256']}}


def protocol(ctx):
    return {'schema':'mura-normal-workpoint-preparation-v1','stage':4,
            'implementation':{str(Path(__file__).resolve()):digest(__file__),
                              str(ROOT/'mura_response/inference_runtime.py'):digest(ROOT/'mura_response/inference_runtime.py')},
            'sources':ctx, 'model_order':list(MODELS),'role_order':['Q200','F100'],
            'source_acquisition':'independent shared CPU Q200 then F100, once per original, before any model',
            'work_preprocessing':'PIL RGB uint8 bilinear256 then float32/float32(255); unchanged contrast',
            'legacy_group':'opaque role/exclusion only; no guessed serialization',
            'new_source_digest':'decoded RGB1024 uint8 HWC C bytes SHA256, no header',
            'working_digest':'frozen inputs.array_identity dtype/shape/C header; never float requantization',
            'collision_gate':'Q/F internally, mutually, vs frozen D_fit60 and old N160; any collision stops before scoring',
            'models':{'PatchCore':{'score':'author maximum squared-L2 nearest-neighbour patch score',
                'heatmap':'author bilinear resize256 align_corners=False then scipy Gaussian sigma4; no extra postprocessing',
                'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
                'layers':['layer2','layer3'],'embedding':1024,'patchsize':3,'nn':1,
                'construction':'local ImageNetV1 weights, retain author single-ones dimension probe/BN initialization; snapshot after load; never bank fit',
                'score_is_not_heatmap_max':True},
              'G1':{'steps':10000,'score':'max raw anomaly softmax over256x256',
                'heatmap':'raw anomaly-channel float32 softmax; no pooling, smoothing, stretching or resize',
                'normalization':'none beyond float32/255','BN':'49 stateless BatchNorm2d,22 reconstruction+27 segmentation; empty buffers'}},
            'precision':'fp32','batch_size':1,'workers':0,'threads':4,'eval':True,
            'TF32':False,'cudnn_benchmark':False,'seed_for_construction':42,
            'quantile':.95,'quantile_method':'linear','alarm_rule':'score strictly greater than new_tau',
            'calibration_denominator':200,'F_check_denominator':100,'old_tau_unchanged':True,
            'Q_is_independent_validation':False,'F_tau_adaptation':False,
            'F_Wilson_z':1.959963984540054,'F_Wilson_confidence':.95,
            'state_checks':'full parameter/buffer digest after model construction, after every input and after all scoring; G1 fixed trained parameter hash',
            'timing':'Q200 separately timed after model load through last Q cache write, excludes quantile; whole model transaction recorded separately',
            'GPU_peak_scope':'reset after CUDA admission before model creation; includes constructor and all300 inputs; excludes earlier import/context admission',
            'transaction':'exclusive new per-model directory; inflight before each forward; no partial continuation; complete same identity read-only reuse',
            'unique_model_inputs':'each model Q200 then F100, once per unique RGB; all references retained; collisions stop before any forward',
            'resources':{'CPU_RAM_GiB':2,'GPU_start_RAM_GiB':4.5,'GPU_start_free_GiB':4,'free_D_GiB':40},
            'limits':'no detector training, rebuilding, GT, stimulus rendering or statistical response fitting'}


def prepare():
    ctx=context(); cfg=protocol(ctx); resource=resources(gpu=True)
    if NORMAL_PREP.exists():
        return verify_preparation()
    NORMAL_PREP.mkdir(exist_ok=False)
    try:
        write(NORMAL_PREP/'sources.json',ctx)
        write(NORMAL_PREP/'protocol.json',cfg)
        write(NORMAL_PREP/'verification.json',{'status':'NORMAL_WORKPOINT_PREPARED',
            'scalar_audit':scalar_audit(),'resources':resource,'boundary':assert_static(),
            'GPU_compatibility_or_speed_measured':False,'real_thresholds_exist':False})
        seal(NORMAL_PREP,{'sources.json','protocol.json','verification.json'})
    except BaseException as exc:
        if not (NORMAL_PREP/'failure.json').exists():
            write(NORMAL_PREP/'failure.json',{'error':type(exc).__name__,'message':str(exc),'boundary':boundary()})
        raise
    return {'status':'NORMAL_WORKPOINT_PREPARED','output':str(NORMAL_PREP),
            'resources':resource,'boundary':assert_static()}


def verify_preparation():
    ctx=context(); cfg=protocol(ctx)
    verify_closed(NORMAL_PREP,{'sources.json','protocol.json','verification.json'})
    require(read(NORMAL_PREP/'sources.json')==ctx and read(NORMAL_PREP/'protocol.json')==cfg,
            'Preparation sources/implementation/runtime differs')
    saved=read(NORMAL_PREP/'verification.json')
    require(saved['status']=='NORMAL_WORKPOINT_PREPARED' and saved['scalar_audit']==scalar_audit(),
            'Saved scalar/preparation evidence differs')
    return {'status':'NORMAL_PREPARATION_CACHE_VERIFIED','configuration_sha256':digest(NORMAL_PREP/'protocol.json'),
            'output':str(NORMAL_PREP),'boundary':boundary()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify-preparation','inspect',
        'acquire-normal-inputs','verify-normal-inputs','score-normal','verify-normal'))
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    require((args.command in ('score-normal','verify-normal')) == (args.model is not None),
            'Model argument must appear exactly for single-model scoring/verification')
    runtime_mode=args.command not in ('prepare','verify-preparation','inspect')
    install_guards(model_allowed=args.command=='score-normal',
                   array_roots=(NORMAL_INPUTS,SCORES) if runtime_mode else ())
    if args.command=='prepare': result=prepare()
    elif args.command=='verify-preparation':
        result=verify_preparation(); assert_static()
    elif args.command=='inspect':
        ctx=context(); result={'status':'NORMAL_STATIC_INSPECTION','resources':resources(gpu=True),
            'models':list(MODELS),'Q200':len(ctx['Q200']),'F100':len(ctx['F100']),
            'preparation_exists':NORMAL_PREP.exists(),'boundary':assert_static()}
    else:
        from . import inference_runtime
        commands={'acquire-normal-inputs':inference_runtime.acquire_inputs,
                  'verify-normal-inputs':inference_runtime.verify_inputs,
                  'score-normal':inference_runtime.score_normal,
                  'verify-normal':inference_runtime.verify_normal}
        result=commands[args.command](sys.modules[__name__],args.model)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
