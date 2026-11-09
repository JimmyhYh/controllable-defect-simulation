"""Stage3 only: final frozen-model probabilities; no training or performance metrics."""
import gc
import hashlib
import importlib.util
import io
import shutil
import time
from pathlib import Path

import numpy as np

from .data import ROOT, read_json, save_json, resources
from .holdout_preparation import FOLDER, PROTOCOL, REQUESTS
from .holdout_data import (CONFIG as GENERATION_CONFIG, MANIFEST, verify_generated,
                           array_hash, validate_RGB)
from .calibration import exclusive_file, image_score
from .probe_lr_decay_preparation import sha_file
from .probe_lr_decay_training import available_ram

CONFIG=FOLDER/'scoring_config.json'
CACHE=FOLDER/'scoring'


def input_sequence(manifest):
    sequence=[]
    for row in manifest['normal']:
        req=row['request'];path=f"generation/backgrounds/{req['id']}.npz"
        sequence.append({'id':req['id'],'input_class':'normal','relative_npz':path,'member':'B',
                         'npz_sha256':row['files_sha256'][path],'request_ordinal':None})
    for row in manifest['anomaly']:
        if not row['quality']['valid']:continue
        req=row['request'];path=f"generation/anomalies/{req['id']}.npz"
        sequence.append({'id':req['id'],'input_class':'anomaly','relative_npz':path,'member':'I',
            'npz_sha256':row['files_sha256'][path],'request_ordinal':req['ordinal'],
            **{key:req[key] for key in ('strength_id','polarity','template_id','pair_id')},'kind':req['spec']['kind']})
    if len(sequence)!=480 or len({x['id'] for x in sequence})!=480 or any(x['input_class']!='normal' for x in sequence[:20]):
        raise ValueError('Expected20 normals then460 shared valid anomalies')
    return sequence


def proposed_config():
    manifest=verify_generated();protocol=read_json(PROTOCOL)
    proof=read_json(FOLDER/'generation_verification.json')
    if proof['manifest_sha256']!=sha_file(MANIFEST) or proof['generation_config_sha256']!=sha_file(GENERATION_CONFIG):
        raise ValueError('Frozen stage2 completion proof differs')
    files=[Path(__file__),ROOT/'vendor/DRAEM/model_unet.py',ROOT/'controlled_generation/probe_lr_decay_training.py',
           ROOT/'controlled_generation/calibration.py']
    return {'schema':'shared-holdout-scoring-v1','protocol_sha256':sha_file(PROTOCOL),
        'requests_sha256':sha_file(REQUESTS),'generation_config_sha256':sha_file(GENERATION_CONFIG),
        'manifest_sha256':sha_file(MANIFEST),'generation_verification_sha256':sha_file(FOLDER/'generation_verification.json'),
        'implementation_sha256':{str(p.relative_to(ROOT)):sha_file(p) for p in files},
        'runtime_identity':protocol['runtime_identity'],'models':protocol['final_models'],
        'thresholds':protocol['threshold_results'],'score':protocol['score'],'preprocessing':protocol['preprocessing'],
        'normalization':protocol['normalization'],'conditions':['G0','G1'],'inputs':input_sequence(manifest),
        'device':'cuda','precision':'FP32','batch_size':1,'workers':0,'threads':4,
        'cudnn_benchmark':False,'TF32':False,'BN':'stateless; eval and inference_mode; no BN buffers',
        'expected_buffer_sha256':hashlib.sha256().hexdigest(),'expected_registered_buffer_count':0,
        'minimum_start_RAM_GiB':3.,'minimum_before_image_RAM_GiB':1.5,'minimum_free_GPU_GiB':2.,
        'minimum_free_disk_GiB':8.,'weights':'final10000 only','thresholds_changed':False,
        'partial_policy':'audit continuous committed prefix, stop on any incomplete condition; no automatic fill/re-forward',
        'scope':'scores and raw probabilities only; no AP/AUROC/recall/FP computation; no real evaluation'}


def validate_root_names(entries,owned_sequence=False):
    for name,is_dir in entries:
        if name in ('G0','G1') and is_dir:continue
        if name=='sequence.owner.lock' and not is_dir and owned_sequence:continue
        raise RuntimeError('Unknown or externally owned scoring root artifact; preserve and diagnose')


def check_scoring_root(owned_sequence=False):
    if (FOLDER/'scoring_preparation.owner.lock').exists():raise RuntimeError('Scoring preparation owned/interrupted')
    if CACHE.exists():
        if not CACHE.is_dir():raise RuntimeError('Scoring root is not a directory')
        validate_root_names([(p.name,p.is_dir()) for p in CACHE.iterdir()],owned_sequence)


def prepare_scoring():
    check_scoring_root()
    cfg=proposed_config()
    if CONFIG.exists():
        if read_json(CONFIG)!=cfg:raise ValueError('Scoring configuration differs; preserve')
    else:
        with exclusive_file(FOLDER/'scoring_preparation.owner.lock',{'operation':'prepare_scoring'}):
            if CONFIG.exists() or (CACHE.exists() and any(CACHE.iterdir())):
                raise RuntimeError('Unexpected preparation/cache state; preserve and diagnose')
            save_json(CONFIG,cfg)
    return cfg


def checked_config():
    check_scoring_root()
    cfg=read_json(CONFIG)
    if cfg!=proposed_config():raise ValueError('Scoring inputs/model/implementation identity differs')
    if (FOLDER/'scoring_preparation.owner.lock').exists():raise RuntimeError('Scoring preparation owned/interrupted')
    return cfg


def run_identity(condition):
    if condition not in ('G0','G1'):raise ValueError('Unknown model')
    return {'condition':condition,'configuration_sha256':sha_file(CONFIG)}


def load_RGB_only(item):
    # Only one image member is read. No normal_GT, M_eff, ROI, alpha, or residual.
    path=FOLDER/item['relative_npz']
    if sha_file(path)!=item['npz_sha256']:raise ValueError('Frozen inference input differs')
    with np.load(path,allow_pickle=False) as archive:rgb=archive[item['member']]
    validate_RGB(rgb)
    return rgb


def save_once(path,value):
    if path.exists() or path.with_suffix(path.suffix+'.tmp').exists():raise RuntimeError('Existing score artifact; no overwrite')
    save_json(path,value)


def save_probability(path,value):
    image_score(value)
    if path.exists() or path.with_suffix(path.suffix+'.tmp').exists():raise RuntimeError('Existing probability artifact')
    path.parent.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();np.save(buffer,value,allow_pickle=False)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:stream.write(buffer.getvalue())
    temporary.replace(path)


def cache_records(cfg,condition):
    folder=CACHE/condition;signature=run_identity(condition)
    if list(folder.rglob('*.tmp')) or (folder/'pending_forward.json').exists():
        raise RuntimeError('Uncertain forward/transaction; preserve every artifact; no automatic repeat')
    files=sorted((folder/'records').glob('*.json'));rows=[]
    if [p.name for p in files]!=[f'{k:04d}.json' for k in range(len(files))] or len(files)>480:
        raise ValueError('Committed cache is not a continuous prefix')
    expected_files=set()
    for k,path in enumerate(files):
        row=read_json(path);item=cfg['inputs'][k];probability_path=folder/'probabilities'/f'{k:04d}.npy'
        if (row['identity']!=signature or row['ordinal']!=k or row['input']!=item
                or row['probability_file']!=probability_path.relative_to(FOLDER).as_posix()
                or not row['state_unchanged'] or row['probability_sha256']!=sha_file(probability_path)):
            raise ValueError('Score row identity differs')
        probability=np.load(probability_path,allow_pickle=False)
        if image_score(probability)!=row['score']:raise ValueError('Score differs from saved raw maximum')
        if array_hash(load_RGB_only(item))!=row['input_array_sha256']:raise ValueError('Cached RGB input differs')
        expected_files.update((path.relative_to(folder).as_posix(),probability_path.relative_to(folder).as_posix()))
        rows.append(row)
    actual={p.relative_to(folder).as_posix() for sub in ('records','probabilities') for p in (folder/sub).rglob('*') if p.is_file()}
    if actual!=expected_files:raise RuntimeError('Unknown/orphaned score cache; no automatic repeat')
    allowed={'owner.lock','started.json','scores.json','records','probabilities'}
    if folder.exists() and any(p.name not in allowed for p in folder.iterdir()):raise RuntimeError('Unknown condition artifact')
    if files and not (folder/'started.json').exists():raise RuntimeError('Missing inference start boundary')
    if (folder/'started.json').exists() and read_json(folder/'started.json')['identity']!=signature:
        raise ValueError('Inference start identity differs')
    return rows


def verify_condition(cfg,condition,owned=False):
    folder=CACHE/condition
    if not owned and (folder/'owner.lock').exists():raise RuntimeError('Condition owned/interrupted; preserve')
    rows=cache_records(cfg,condition);result=read_json(folder/'scores.json')
    if (len(rows)!=480 or result['identity']!=run_identity(condition) or result['rows']!=rows
            or result['input_count']!=480 or result['normal_count']!=20 or result['anomaly_count']!=460
            or not result['complete'] or result['threshold']!=cfg['thresholds'][condition]['threshold']
            or result['threshold_changed'] or result['metric_computation_performed']
            or result['model_input_calls']!=480 or result['optimizer_creations']!=0
            or not result['final_state_unchanged']
            or result['final_parameters_sha256']!=cfg['models'][condition]['final_parameters_sha256']
            or result['subnetwork_forwards']!={'reconstruction':480,'segmentation':480}
            or result['BN_counts']!={'reconstruction':22,'segmentation':27}
            or result['buffer_sha256']!={key:cfg['expected_buffer_sha256'] for key in ('reconstruction','segmentation')}):
        raise ValueError('Final score cache is incomplete/mismatched')
    print('HOLDOUT_SCORE_CACHE_VERIFIED',condition,'480/480; no model loading/forward',flush=True)
    return result


def buffer_hash(model):
    digest=hashlib.sha256()
    for name,value in model.named_buffers():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(torch,models,cfg,condition,buffers):
    from .probe_lr_decay_training import parameters_hash,verify_stateless_bn
    for key,model in models.items():
        verify_stateless_bn(torch,model)
        if (any(module.training for module in model.modules()) or any(p.grad is not None for p in model.parameters())
                or parameters_hash(model)!=cfg['models'][condition]['final_parameters_sha256'][key]
                or buffer_hash(model)!=buffers[key]):raise ValueError('Parameter/gradient/BN/buffer state changed')


def predict_RGB(torch,rec,seg,rgb):
    # Pure scoring signature: no label, request, mask, threshold, or metrics.
    x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).cuda()
    with torch.inference_mode():
        reconstruction=rec(x)
        probability=torch.softmax(seg(torch.cat((reconstruction,x),dim=1)),dim=1)[0,1].cpu().numpy().copy()
    torch.cuda.synchronize()
    image_score(probability)
    return probability


def run_condition(cfg,condition):
    check_scoring_root(owned_sequence=True)
    folder=CACHE/condition;signature=run_identity(condition)
    if condition=='G1':verify_condition(cfg,'G0')
    with exclusive_file(folder/'owner.lock',signature):
        rows=cache_records(cfg,condition)
        if (folder/'scores.json').exists():return verify_condition(cfg,condition,owned=True)
        if rows or (folder/'started.json').exists():
            raise RuntimeError('Incomplete condition; preserve continuous prefix; no automatic filling/re-forward')
        before=resources();before['disk_free_GiB']=shutil.disk_usage(FOLDER).free/2**30
        if before.get('available_RAM_GiB',0)<3 or before['disk_free_GiB']<8:raise RuntimeError('Need3GiB RAM and8GiB disk before model loading')
        import torch
        from .formal_preparation import model_folder
        from .probe_lr_decay_training import configure_stateless_bn,verify_stateless_bn,parameters_hash
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]/2**30<2:
            raise RuntimeError('Need CUDA and2GiB free GPU; no fallback or forward')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        spec=importlib.util.spec_from_file_location('controlled_holdout_author_model',ROOT/'vendor/DRAEM/model_unet.py')
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        models={};minimum_ram=before['available_RAM_GiB'];calls=0;start=time.perf_counter()
        gpu_start={'name':torch.cuda.get_device_name(),'free_GiB':torch.cuda.mem_get_info()[0]/2**30}
        torch.cuda.reset_peak_memory_stats()
        try:
            models={'reconstruction':author.ReconstructiveSubNetwork(3,3).cuda(),
                    'segmentation':author.DiscriminativeSubNetwork(6,2).cuda()}
            for key,model in models.items():
                count=configure_stateless_bn(torch,model)
                if count!={'reconstruction':22,'segmentation':27}[key]:raise ValueError('BN structure differs')
                state=torch.load(model_folder(condition)/(key+'.pth'),map_location='cpu',weights_only=True)
                model.load_state_dict(state,strict=True);del state;model.eval();verify_stateless_bn(torch,model)
                if parameters_hash(model)!=cfg['models'][condition]['final_parameters_sha256'][key]:raise ValueError('Final loaded parameters differ')
            buffers={key:buffer_hash(model) for key,model in models.items()}
            if any(len(list(model.named_buffers()))!=cfg['expected_registered_buffer_count'] or buffers[key]!=cfg['expected_buffer_sha256']
                   for key,model in models.items()):raise ValueError('Unexpected registered buffer signature')
            assert_frozen(torch,models,cfg,condition,buffers)
            save_once(folder/'started.json',{'identity':signature,'note':'Before first forward; incomplete condition never auto-repeated'})
            for k,item in enumerate(cfg['inputs']):
                ram=available_ram();minimum_ram=min(minimum_ram,ram)
                if ram<1.5:raise RuntimeError('RAM below1.5GiB before image; preserve prefix')
                if shutil.disk_usage(FOLDER).free/2**30<8:raise RuntimeError('Disk below8GiB before image; preserve prefix')
                rgb=load_RGB_only(item);input_hash=array_hash(rgb)
                save_once(folder/'pending_forward.json',{'identity':signature,'ordinal':k,'input':item})
                tick=time.perf_counter();calls+=1
                probability=predict_RGB(torch,models['reconstruction'],models['segmentation'],rgb)
                assert_frozen(torch,models,cfg,condition,buffers)
                path=folder/'probabilities'/f'{k:04d}.npy';save_probability(path,probability)
                row={'identity':signature,'ordinal':k,'input':item,'input_array_sha256':input_hash,
                    'probability_file':path.relative_to(FOLDER).as_posix(),'probability_sha256':sha_file(path),
                    'score':image_score(probability),'state_unchanged':True,'scoring_and_state_check_seconds':time.perf_counter()-tick}
                save_once(folder/'records'/f'{k:04d}.json',row);(folder/'pending_forward.json').unlink();rows.append(row)
                if (k+1)%20==0:print('HOLDOUT_SCORED',condition,k+1,'/480',flush=True)
            assert_frozen(torch,models,cfg,condition,buffers)
            result={'identity':signature,'complete':True,'rows':rows,'input_count':480,'normal_count':20,'anomaly_count':460,
                'threshold':cfg['thresholds'][condition]['threshold'],'threshold_changed':False,'metric_computation_performed':False,
                'model_input_calls':calls,'subnetwork_forwards':{'reconstruction':calls,'segmentation':calls},
                'optimizer_creations':0,'final_state_unchanged':True,'BN_counts':{'reconstruction':22,'segmentation':27},
                'final_parameters_sha256':{key:parameters_hash(model) for key,model in models.items()},'buffer_sha256':buffers,
                'resources':{'start':before,'GPU_start':gpu_start,'sampled_min_available_RAM_GiB':minimum_ram,
                    'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
                    'elapsed_model_setup_scoring_state_check_seconds':time.perf_counter()-start,'RAM_sampling':'before each input; not peak monitoring'}}
            save_once(folder/'scores.json',result)
        finally:
            models.clear()
            if 'model' in locals():del model
            gc.collect();torch.cuda.empty_cache()
        return verify_condition(cfg,condition,owned=True)


def run_scoring():
    cfg=prepare_scoring()
    with exclusive_file(CACHE/'sequence.owner.lock',{'configuration_sha256':sha_file(CONFIG)}):
        for condition in cfg['conditions']:run_condition(cfg,condition)
        check_scoring_root(owned_sequence=True)
    print('HOLDOUT_BOTH_MODELS_SCORED; performance metrics deferred',flush=True)


def verify_scores():
    cfg=checked_config()
    if (CACHE/'sequence.owner.lock').exists():raise RuntimeError('Scoring sequence owned/interrupted; preserve')
    return {condition:verify_condition(cfg,condition) for condition in cfg['conditions']}
