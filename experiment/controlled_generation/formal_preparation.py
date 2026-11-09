"""Prepare/audit all formal train-side requests on CPU; never starts training."""
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .data import ROOT, OUT, SPLIT, digest, load_split, normal_rgb, read_json, save_json
from .preparation import locked_json
from .probe_lr_decay_preparation import PROBE as PASSING_PROBE, CONFIG as PASSING_CONFIG, preparation_resources, sha_file
from .sampling import POLICY_PATH, AUDIT_PATH, validate_policy
from .formal_sampling import shared_sequence, training_sample, sample_fingerprint

FOLDER=OUT/'formal_preparation'
CONFIG=FOLDER/'common_config.json'
SEQUENCE=FOLDER/'common_sequence.npz'
INVENTORY=FOLDER/'texture_inventory.json'
AUDIT=FOLDER/'sampling_validity.json'
HASHES=FOLDER/'training_sample_hashes.npz'
MODEL_ROOT=ROOT/'models/controlled_generation/formal_draem'
RESULT_ROOT=OUT/'formal_training'


def model_folder(condition):
    if condition not in ('G0','G1'):raise ValueError('Unknown condition')
    return MODEL_ROOT/f'{condition}_20261030'


def result_folder(condition):
    if condition not in ('G0','G1'):raise ValueError('Unknown condition')
    return RESULT_ROOT/f'{condition}_20261030'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def runtime_identity():
    # Distribution metadata only: does not import torch or instantiate any network.
    return {'python':platform.python_version(),'python_implementation':platform.python_implementation(),
            'system':platform.system(),'machine':platform.machine(),
            'packages':{name:importlib.metadata.version(name) for name in
                        ('torch','torchvision','numpy','Pillow','opencv-python','scipy','imgaug')}}


def pinned_sources():
    return {f'controlled_generation/{name}':digest(Path(__file__).with_name(name)) for name in
            ('formal_sampling.py','formal_preparation.py','formal_training.py')}


def author_sources():
    expected={'data_loader.py':'92e6309293c8e36277471393d29f3cf13d007ccfb1fbacb3a5acfc4406c14ecd',
              'perlin.py':'b6ab97c1fd81d0da2513f99bda6fdc65d96536d7abe4dab49f55bf73a7d64fc1'}
    probe=read_json(PASSING_CONFIG)
    expected.update(probe['author']['files_sha256'])
    for name,sha in expected.items():
        if digest(ROOT/'vendor/DRAEM'/name)!=sha:raise ValueError('Pinned DRAEM source changed')
    return expected


def probe_evidence():
    # Config and finished files only. No network construction, forward or held-out access.
    from .probe_lr_decay_training import completed_lr_decay_status
    result=completed_lr_decay_status()
    verification=read_json(PASSING_PROBE/'第6天B4核验结果.json')
    key=str(PASSING_PROBE/'training_result.json')
    if (result['steps']!=1000 or not result['gate_passed'] or not verification['gate_recomputed']
            or verification['source_sha256'][key]!=digest(PASSING_PROBE/'training_result.json')
            or verification['global_pixel_AP_recomputed']!=result['training_pool_pixel_AP']):
        raise ValueError('Passing fixed1000 probe evidence differs')
    return {'config_sha256':digest(PASSING_CONFIG),'result_sha256':digest(PASSING_PROBE/'training_result.json'),
            'verification_sha256':digest(PASSING_PROBE/'第6天B4核验结果.json'),
            'fixed_training_pool_AP':result['training_pool_pixel_AP'],'not_real_performance':True}


def texture_inventory():
    labels=ROOT/'data/dtd/labels/train1.txt'
    if digest(labels)!='0a720c85ea6b52b7135a1b1dce63b58b2812a26be225e655b4903692c9380c81':
        raise ValueError('Frozen DTD train1 list changed')
    names=sorted(labels.read_text().splitlines())
    if len(names)!=1880 or len(set(names))!=1880:raise ValueError('DTD train1 count differs')
    hashes={}
    for name in names:
        relative=Path(name)
        if relative.is_absolute() or len(relative.parts)!=2 or '..' in relative.parts or relative.suffix.lower()!='.jpg':
            raise ValueError('Unexpected DTD relative path')
        hashes[name]=sha_file(ROOT/'data/dtd/images'/relative)
    return {'names':names,'sha256':hashes,'train1_label_sha256':digest(labels),
            'image_count':1880,'pool_metadata_sha256':canonical_hash(hashes)}


def candidate_config(inventory):
    split=load_split();g1=read_json(POLICY_PATH);validate_policy(g1)
    if not read_json(AUDIT_PATH)['ready_for_probe_preparation']:raise ValueError('G1 frozen sampling is not ready')
    probe=read_json(PASSING_CONFIG)
    return {'schema':'controlled-formal-common-v1','seed':20261030,'steps':10000,'conditions':['G0','G1'],
        'runtime_identity':runtime_identity(),
        'device':'cuda','precision':'fp32','batch_size':1,'workers':0,'threads':4,'input_size':256,
        'input':'whole RGB bilinear256 float32/255; no additional normalization or background augmentation',
        'optimizer':'Adam','learning_rate':1e-4,'weight_decay':0,'milestones':[8000,9000],'gamma':.2,
        'loss_coefficients':{'MSE':1,'SSIM':1,'Focal':1},'SSIM_behavior':'pinned author dynamic prediction range unchanged',
        'normalization':probe['normalization'],'author_files_sha256':author_sources(),
        'N_gen':split['N_gen'],'normal_source_sha256':{p:split['source_sha256'][p] for p in split['N_gen']},
        'split_sha256':digest(SPLIT),'G1_sampling':g1,'G1_policy_sha256':digest(POLICY_PATH),
        'G1_prior_audit_sha256':digest(AUDIT_PATH),'DTD_inventory_fingerprint':canonical_hash(inventory),
        'passing_probe':probe_evidence(),'implementation_sha256':pinned_sources(),
        'shared_requests':'5000 normal/anomaly pairs; each80-pair epoch PCG64 permutation of frozen N_gen80',
        'normal_request_count':5000,'anomaly_request_count':5000,'total_update_requests':10000,
        'random_streams':{'common':'SeedSequence([700001,seed,epoch]) PCG64',
                         'G0':'SeedSequence([700002,seed,anomaly_ordinal,attempt]) PCG64 plus isolated numpy/imgaug seeds',
                         'G1':'frozen SeedSequence([seed,anomaly_ordinal,attempt]) PCG64'},
        'G0_adapter':'exact author augmenter definitions/randAugmenter + NumPy Perlin AST bodies; local PCG64 scalar draws',
        'G0_lambda':'1-beta, beta uniform[0,.8); source texture uniform over DTD train1',
        'adaptations':['no author random normal switch; shared predetermined branch',
                       'no background rotation; same raw RGB target in both conditions',
                       'BGR texture augmentation then RGB conversion; float training arrays',
                       'actual float residual label instead of nominal Perlin region'],
        'validity':{'float_label_threshold':float(np.float32(1/255)),'minimum_effective_coverage':.90,
                    'outside_ROI_change':0,'max_attempts':5,'failed_request':'stop; no relabel as normal or background replacement'},
        'comparison':'whole generator method; masks and appearances differ',
        'checkpoint_interval':1000,'print_interval':100,'minimum_available_RAM_GiB':3.,
        'minimum_disk_per_condition_GiB':16.,'recommended_disk_for_both_GiB':32.,'recommended_free_GPU_GiB':4.,
        'initialization':'both author nets from scratch; same seed; G1 initial hashes must match finished G0 metadata',
        'model_root':str(MODEL_ROOT),'result_root':str(RESULT_ROOT),'manual_training_only':True,
        'held_out_scoring':False,'independent_real_validation':'pending'}


def verify_formal_preparation():
    cfg=read_json(CONFIG);inventory=read_json(INVENTORY)
    if cfg['implementation_sha256']!=pinned_sources() or cfg['author_files_sha256']!=author_sources():
        raise ValueError('Formal frozen code identity differs')
    if canonical_hash(inventory)!=cfg['DTD_inventory_fingerprint'] or texture_inventory()!=inventory:
        raise ValueError('DTD frozen inventory differs')
    expected=candidate_config(inventory)
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Formal common protocol differs')
    if (sha_file(SEQUENCE)!=cfg['sequence_sha256'] or sha_file(HASHES)!=cfg['sample_hashes_sha256']
            or digest(AUDIT)!=cfg['sampling_audit_sha256']):raise ValueError('Formal scientific input artifacts changed')
    audit=read_json(AUDIT)
    if not audit['ready_for_manual_training'] or audit['candidate_fingerprint']!=canonical_hash(expected):
        raise ValueError('Formal audit is not ready')
    with np.load(SEQUENCE,allow_pickle=False) as saved:sequence={k:saved[k] for k in saved.files}
    if any(not np.array_equal(sequence[k],v) for k,v in shared_sequence().items()):
        raise ValueError('Shared background/branch sequence changed')
    with np.load(HASHES,allow_pickle=False) as saved:sample_hashes={c:saved[c] for c in cfg['conditions']}
    if any(v.shape!=(5000,) or v.dtype!=np.dtype('S64') for v in sample_hashes.values()):
        raise ValueError('Anomaly sample fingerprints differ')
    split=load_split();backgrounds=[normal_rgb(split,p) for p in cfg['N_gen']]
    from .formal_training import learning_rate_for_step,schedule_state
    if [learning_rate_for_step(k) for k in (1,8000,8001,9000,9001,10000)]!=[1e-4,1e-4,2e-5,2e-5,4e-6,4e-6]:
        raise ValueError('Formal LR boundary differs')
    print('FORMAL_INPUT_VERIFIED',{'backgrounds':80,'DTD':1880,'requests_per_condition':10000,
                                  'optimizer_creations':0,'model_forwards':0},flush=True)
    return cfg,sequence,backgrounds,inventory,sample_hashes


def prepare_formal():
    if CONFIG.exists():return verify_formal_preparation()[0]
    if FOLDER.exists() and any(FOLDER.iterdir()):raise ValueError('Existing partial formal preparation; preserve and diagnose')
    inventory=texture_inventory();cfg=candidate_config(inventory);sequence=shared_sequence()
    split=load_split();backgrounds=[normal_rgb(split,p) for p in cfg['N_gen']]
    audit={'candidate_fingerprint':canonical_hash(cfg),'candidate_configuration':cfg,
           'planned_anomaly_requests_per_condition':5000,'normal_requests_per_condition':5000,
           'backgrounds':80,'conditions':{},'model_forwards':0,'optimizer_creations':0,
           'ready_for_manual_training':False,'scope':'N_gen/DTD train1 only'}
    sample_hashes={c:np.empty(5000,dtype='S64') for c in ('G0','G1')}
    started=time.perf_counter()
    for condition in ('G0','G1'):
        attempts_total=0;rejected=0;min_coverage=1.;positive_total=0;max_outside=0.;branches=Counter()
        for ordinal in range(5000):
            step=ordinal*2+1
            sample,info=training_sample(condition,cfg,sequence,step,backgrounds,inventory)
            attempts_total+=len(info['attempts']);rejected+=sum(not a['stats']['valid'] if 'stats' in a else 1 for a in info['attempts'])
            if sample is None:
                audit['conditions'][condition]={'accepted_requests':ordinal,'failed_ordinal':ordinal,
                    'failed_background':cfg['N_gen'][int(sequence['background_index'][step])],
                    'failed_request':info,'attempts':attempts_total,'rejected_attempts':rejected}
                audit['CPU_audit_seconds']=time.perf_counter()-started
                save_json(AUDIT,audit)
                raise ValueError('Formal request failed five attempts; preserve sampling_validity.json and diagnose')
            stats=sample['stats'];min_coverage=min(min_coverage,stats['effective_coverage'])
            max_outside=max(max_outside,stats['outside_ROI_max_float_residual']);positive_total+=stats['positive_pixels']
            branches[str(info['kind'])+'_'+str(info['polarity'])]+=1
            sample_hashes[condition][ordinal]=sample_fingerprint(sample).encode()
            if (ordinal+1)%500==0:print('FORMAL_CPU_AUDIT',condition,ordinal+1,'/5000',flush=True)
        audit['conditions'][condition]={'accepted_requests':5000,'failed_requests':0,'attempts':attempts_total,
            'rejected_attempts':rejected,'minimum_effective_coverage':min_coverage,'max_outside_ROI_change':max_outside,
            'positive_pixels_across_requests':positive_total,'type_polarity_counts':dict(branches)}
    audit['ready_for_manual_training']=True;audit['CPU_audit_seconds']=time.perf_counter()-started
    FOLDER.mkdir(parents=True,exist_ok=True)
    save_json(INVENTORY,inventory)
    with SEQUENCE.open('wb') as stream:np.savez_compressed(stream,**sequence)
    with HASHES.open('wb') as stream:np.savez_compressed(stream,**sample_hashes)
    save_json(AUDIT,audit)
    cfg.update(sequence_sha256=sha_file(SEQUENCE),sample_hashes_sha256=sha_file(HASHES),sampling_audit_sha256=digest(AUDIT))
    locked_json(CONFIG,cfg)
    save_json(FOLDER/'preparation_resources.json',preparation_resources())
    verify_formal_preparation()
    print('FORMAL_PREPARED; both conditions training steps=0; manual G0 training next',flush=True)
    return cfg
