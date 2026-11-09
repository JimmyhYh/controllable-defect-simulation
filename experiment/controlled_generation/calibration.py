"""Independent calibration preparation; Q_cal is opened only by explicit run_calibration."""
import gc
import importlib.util
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .data import ROOT, OUT, SPLIT, read_json, save_json, load_split
from .formal_preparation import CONFIG as TRAIN_CONFIG, model_folder, result_folder, runtime_identity
from .formal_training import identity as training_identity, checkpoint_header
from .probe_lr_decay_preparation import sha_file

PREP=OUT/'calibration_preparation'
CONFIG=PREP/'config.json'
CACHE=OUT/'calibration'


def image_score(probability):
    a=np.asarray(probability)
    if a.shape!=(256,256) or a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('Expected finite raw float32 anomaly softmax256x256 in[0,1]')
    return float(a.max())


def threshold_from_scores(scores):
    values=np.asarray(scores,dtype=np.float64)
    if values.shape!=(40,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Exactly40 finite normal image scores in[0,1] are required')
    return float(np.quantile(values,.95,method='linear'))


def alarm(score,threshold):
    if not np.isfinite([score,threshold]).all() or not 0<=score<=1 or not 0<=threshold<=1:
        raise ValueError('Invalid score/threshold')
    return bool(score>threshold)


def source_identity():
    names=('data.py','calibration.py','formal_preparation.py','formal_training.py',
           'formal_sampling.py','probe_lr_decay_preparation.py','probe_lr_decay_training.py')
    return {name:sha_file(Path(__file__).with_name(name)) for name in names}


def candidate():
    # Metadata and model/checkpoint bytes only; never opens a Q_cal image or score.
    train=read_json(TRAIN_CONFIG);split=load_split()
    if train['runtime_identity']!=runtime_identity() or train['split_sha256']!=sha_file(SPLIT):
        raise ValueError('Frozen runtime/split differs')
    for relative,sha in train['implementation_sha256'].items():
        if sha_file(ROOT/relative)!=sha:raise ValueError('Frozen formal implementation differs')
    for name,sha in train['author_files_sha256'].items():
        if sha_file(ROOT/'vendor/DRAEM'/name)!=sha:raise ValueError('Pinned author source differs')
    paths=split['Q_cal']
    if len(paths)!=40 or len(set(paths))!=40 or set(paths)&set(split['N_gen']+split['S_bg']):
        raise ValueError('Expected fixed disjoint Q_cal40 paths')
    for p in paths:
        rel=Path(p)
        if rel.is_absolute() or '..' in rel.parts or '/train/good/' not in p:
            raise ValueError('Unsafe/non-normal calibration metadata path')
    models={}
    for condition,proof_name in (('G0','day7_verification.json'),('G1','day8_verification.json')):
        folder=model_folder(condition);result_path=result_folder(condition)/'training_result.json'
        proof_path=result_folder(condition)/proof_name
        result=read_json(result_path);proof=read_json(proof_path)
        if (not proof['verification_passed'] or proof['checkpoint_count']!=10 or result['steps']!=10000
                or not result['training_complete'] or result['identity']!=training_identity(condition,train)):
            raise ValueError('Verified final10000 model identity differs')
        header=checkpoint_header(condition,result['identity'])
        if header is None or header['step']!=10000:raise ValueError('Final owned checkpoint missing')
        if set(result['weight_files'])!={'reconstruction.pth','segmentation.pth'}:
            raise ValueError('Both final weights required')
        bound={str(result_path.relative_to(ROOT)):sha_file(result_path)}
        for name,sha in result['weight_files'].items():
            p=folder/name
            if sha_file(p)!=sha:raise ValueError('Final weights changed')
            bound[str(p.relative_to(ROOT))]=sha
        for p,sha in bound.items():
            if proof['source_sha256'][p]!=sha:raise ValueError('Verification evidence differs')
        models[condition]={'source_files_sha256':bound,'verification_sha256':sha_file(proof_path),
            'final_parameters_sha256':{key:result['final_'+key+'_parameters_sha256'] for key in ('reconstruction','segmentation')},
            'initial_parameters_sha256':{key:result['initial_'+key+'_parameters_sha256'] for key in ('reconstruction','segmentation')}}
    if models['G0']['initial_parameters_sha256']!=models['G1']['initial_parameters_sha256']:
        raise ValueError('Common initial parameters differ')
    return {'schema':'controlled-calibration-v1','training_config_sha256':sha_file(TRAIN_CONFIG),
        'split_sha256':sha_file(SPLIT),'runtime_identity':runtime_identity(),'source_identity':source_identity(),
        'dataset_root':split['root'],'Q_cal':paths,'input_file_sha256':{p:split['source_sha256'][p] for p in paths},
        'models':models,'conditions':['G0','G1'],'sample_count':40,
        'preprocessing':'PIL RGB whole-image bilinear256; float32/255; no other normalization/augmentation',
        'normalization':train['normalization'],'device':'cuda','precision':'fp32','batch_size':1,'workers':0,'threads':4,
        'score':'maximum of raw anomaly-channel softmax over256x256; no pooling/smoothing/resizing/normalization',
        'quantile':.95,'quantile_method':'linear','alarm_rule':'score strictly greater than threshold',
        'scope':'Q_cal normal calibration only; no independent performance claim',
        'minimum_start_RAM_GiB':3.,'minimum_before_image_RAM_GiB':1.5,'minimum_free_GPU_GiB':2.,
        'cache_root':str(CACHE),'checkpoint':'final10000 only','training':False}


def prepare_calibration():
    expected=candidate()
    if CONFIG.exists():
        if read_json(CONFIG)!=expected:raise ValueError('Calibration configuration differs; preserve it')
    else:
        if PREP.exists() and any(PREP.iterdir()):raise ValueError('Partial preparation; preserve and diagnose')
        save_json(CONFIG,expected)
    print('CALIBRATION_PREPARED',{'models':2,'normal_paths':40,'Q_cal_images_read':0,'scores_read':0,'model_forwards':0},flush=True)
    return expected


def verify_preparation():
    cfg=read_json(CONFIG)
    if cfg!=candidate():raise ValueError('Calibration preparation identity differs')
    return cfg


def run_identity(condition):
    return {'condition':condition,'configuration_sha256':sha_file(CONFIG)}


def cache_rows(cfg,condition):
    """Called only during explicit calibration; preparation does not inspect score caches."""
    folder=CACHE/condition;signature=run_identity(condition);rows=[]
    if list(folder.rglob('*.tmp')):raise ValueError('Interrupted cache write; preserve and diagnose')
    expected_names={f'{k:03d}.json' for k in range(40)}
    if any(p.name not in expected_names for p in (folder/'images').glob('*')):
        raise ValueError('Unknown cache image artifact')
    for k,p in enumerate(cfg['Q_cal']):
        path=folder/'images'/f'{k:03d}.json'
        if not path.exists():
            if any((folder/'images'/f'{j:03d}.json').exists() for j in range(k+1,40)):
                raise ValueError('Cache is not a committed prefix')
            break
        row=read_json(path)
        if (row['identity']!=signature or row['ordinal']!=k or row['path']!=p
                or row['input_sha256']!=cfg['input_file_sha256'][p] or row['state_unchanged'] is not True):
            raise ValueError('Cached score source identity differs')
        alarm(row['score'],0.)
        rows.append(row)
    pending=folder/'inflight.json'
    if pending.exists():
        item=read_json(pending)
        if item['identity']!=signature or type(item['ordinal']) is not int:raise ValueError('Unknown inflight identity')
        if not rows or item['ordinal']!=len(rows)-1:
            raise ValueError('An uncommitted forward may have occurred; no automatic re-inference; preserve and diagnose')
    return rows,pending


def expected_result(cfg,condition,rows):
    scores=[r['score'] for r in rows];tau=threshold_from_scores(scores)
    return {'identity':run_identity(condition),'paths':cfg['Q_cal'],'scores':scores,'threshold':tau,
        'q':.95,'method':'linear','alarm_rule':'strictly greater','normal_image_denominator':40,
        'calibration_FP':sum(alarm(s,tau) for s in scores),'calibration_FP_denominator':40,
        'in_sample_calibration_not_validation':True,'image_records_sha256':{
            f'{k:03d}.json':sha_file(CACHE/condition/'images'/f'{k:03d}.json') for k in range(40)},
        'held_out_or_real_evaluation':False}


def qcal_source(cfg,relative):
    base=Path(cfg['dataset_root']).resolve();path=(base/relative).resolve()
    if not path.is_relative_to(base) or sha_file(path)!=cfg['input_file_sha256'][relative]:
        raise ValueError('Q_cal source bytes/path changed')
    return path


@contextmanager
def exclusive_file(path,signature):
    """Exclusive create, no PID guessing or automatic stale-lock deletion."""
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps({'identity':signature,'pid':os.getpid(),'token':uuid.uuid4().hex},sort_keys=True)
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError('Condition already owned or interrupted; preserve owner.lock and diagnose') from error
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:stream.write(payload)
        yield
    finally:
        if not path.exists() or path.read_text(encoding='utf-8')!=payload:
            raise RuntimeError('Ownership marker changed; preserve and diagnose')
        path.unlink()


def run_calibration(condition):
    """Future explicitly authorized scoring entry; never called by prepare/verify."""
    cfg=verify_preparation()
    if condition not in cfg['conditions']:raise ValueError('Unknown condition')
    with exclusive_file(CACHE/condition/'owner.lock',run_identity(condition)):
        return run_owned_calibration(cfg,condition)


def run_owned_calibration(cfg,condition):
    # Only called while this condition is exclusively owned, including cache-only reuse.
    rows,pending=cache_rows(cfg,condition);folder=CACHE/condition;output=folder/'result.json'
    # Recheck source bytes for cached scores too, without re-decoding or re-scoring them.
    for p in cfg['Q_cal']:qcal_source(cfg,p)
    if output.exists():
        if len(rows)!=40 or read_json(output)!=expected_result(cfg,condition,rows):
            raise ValueError('Final calibration cache differs')
        print('CALIBRATION_CACHE_VERIFIED',condition,flush=True);return read_json(output)
    if pending.exists():pending.unlink()  # Only a matching already committed row can reach here.
    if len(rows)<40:
        from .probe_lr_decay_training import available_ram,configure_stateless_bn,verify_stateless_bn,parameters_hash
        if available_ram()<cfg['minimum_start_RAM_GiB']:raise RuntimeError('Need3GiB free RAM before model loading')
        import torch
        from PIL import Image
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]/2**30<cfg['minimum_free_GPU_GiB']:
            raise RuntimeError('Need CUDA and2GiB free GPU; no fallback or forward')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        spec=importlib.util.spec_from_file_location('controlled_calibration_author_model',ROOT/'vendor/DRAEM/model_unet.py')
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        rec=author.ReconstructiveSubNetwork(3,3).cuda();seg=author.DiscriminativeSubNetwork(6,2).cuda()
        for name,model,expected_count in (('reconstruction',rec,22),('segmentation',seg,27)):
            if configure_stateless_bn(torch,model)!=expected_count:raise ValueError('BN structure differs')
            state=torch.load(model_folder(condition)/(name+'.pth'),map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state;model.eval();verify_stateless_bn(torch,model)
            if parameters_hash(model)!=cfg['models'][condition]['final_parameters_sha256'][name]:
                raise ValueError('Loaded final parameters differ')
        try:
            for k in range(len(rows),40):
                if available_ram()<cfg['minimum_before_image_RAM_GiB']:
                    raise RuntimeError('Low RAM before next image; preserve committed prefix')
                p=cfg['Q_cal'][k];source=qcal_source(cfg,p)
                with Image.open(source) as im:
                    rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),np.float32)/np.float32(255)
                x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).cuda()
                save_json(pending,{'identity':run_identity(condition),'ordinal':k})
                started=time.perf_counter()
                with torch.inference_mode():
                    reconstruction=rec(x)
                    probability=torch.softmax(seg(torch.cat((reconstruction,x),dim=1)),dim=1)
                    score=image_score(probability[0,1].cpu().numpy())
                torch.cuda.synchronize()
                for name,model in (('reconstruction',rec),('segmentation',seg)):
                    verify_stateless_bn(torch,model)
                    if any(p.grad is not None for p in model.parameters()) or parameters_hash(model)!=cfg['models'][condition]['final_parameters_sha256'][name]:
                        raise ValueError('Model state changed; pending forward preserved')
                row={'identity':run_identity(condition),'ordinal':k,'path':p,
                     'input_sha256':cfg['input_file_sha256'][p],'score':score,'state_unchanged':True,
                     'scoring_and_state_check_seconds':time.perf_counter()-started}
                save_json(folder/'images'/f'{k:03d}.json',row);pending.unlink();rows.append(row)
                print('CALIBRATION_SCORED',condition,k+1,'/40',flush=True)
        finally:
            del rec,seg;gc.collect();torch.cuda.empty_cache()
    rows,_=cache_rows(cfg,condition)
    result=expected_result(cfg,condition,rows)
    save_json(output,result)
    print('CALIBRATION_COMPLETE',condition,result['threshold'],flush=True)
    return result
