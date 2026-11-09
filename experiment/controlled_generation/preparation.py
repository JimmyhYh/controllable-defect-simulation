"""Prepare/audit fixed train-side probe inputs; never invokes torch training."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .data import OUT, ROOT, SPLIT, digest, load_split, normal_rgb, read_json, resources, save_json
from .render import render
from .sampling import AUDIT_PATH, POLICY_PATH, generator_identity, prepare_sampling, validate_policy

PROBE = OUT / 'learnability_probe'
CONFIG = PROBE / 'config.json'
DATA = PROBE / 'inputs.npz'
INPUTS = PROBE / 'inputs.json'
COMMON = OUT / 'training_preparation/common_training_config.json'
MODEL_FOLDER = ROOT / 'models/controlled_generation/learnability_probe_20261029'


def pinned_author():
    original = read_json(ROOT / 'results/weak_localization/protocol.json')
    expected = {x['name']:x['sha256'] for x in original['source']['draem']['files']}
    files = {}
    for name in ('model_unet.py','loss.py'):
        actual = digest(ROOT / 'vendor/DRAEM' / name)
        if expected.get(name)!=actual:
            raise ValueError('Pinned author model or loss differs: '+name)
        files[name] = actual
    return {'folder':'vendor/DRAEM','files_sha256':files,'architecture':'ReconstructiveSubNetwork(3,3,base_width128); DiscriminativeSubNetwork(6,2,base_channels64)',
            'loss':'MSE reconstruction + author SSIM + author FocalLoss on softmax segmentation, coefficients1/1/1'}


def fixed_specs():
    return [
        {'kind':'mura_like','shape':'ellipse','units':'px256','center':[100,120],'axes':[32,20],'angle_deg':25,'amplitude':.04},
        {'kind':'mura_like','shape':'circle','units':'px256','center':[128,128],'radius':24,'amplitude':-.04},
        {'kind':'line_like','shape':'line','units':'px256','center':[128,128],'length':100,'width':4,'angle_deg':30,'amplitude':.04},
        {'kind':'line_like','shape':'line','units':'px256','center':[128,128],'length':100,'width':4,'angle_deg':60,'amplitude':-.04}]


def locked_json(path, value):
    if path.exists():
        if read_json(path)!=value:raise ValueError('Prepared scientific configuration differs; no overwrite: '+str(path))
    else:save_json(path,value)


def prepare_probe():
    sampling=prepare_sampling()
    if not sampling['ready_for_probe_preparation']:
        raise ValueError('Sampling is not ready; stop before training preparation')
    split=load_split();PROBE.mkdir(parents=True,exist_ok=True)
    common={'schema':'controlled-draem-paired-common-v1','sampling_policy_sha256':digest(POLICY_PATH),
            'sampling_audit_sha256':digest(AUDIT_PATH),'split_sha256':digest(SPLIT),'N_gen':split['N_gen'],
            'conditions':['G0','G1'],'comparison':'whole generator method; masks and appearances differ',
            'formal_seed_each':20261030,'formal_steps_each':10000,'input_size':256,'precision':'fp32',
            'batch_size':1,'workers':0,'optimizer':'Adam','learning_rate':1e-4,'weight_decay':0,
            'loss_coefficients':{'MSE':1,'SSIM':1,'Focal':1},'milestones':[8000,9000],'gamma':.2,
            'input':'float32 RGB256 /255; no additional normalization; no JPEG round-trip',
            'G0':'author Perlin/DTD train1, with common effective-change validity guard',
            'G1':'frozen controlled sampling, normal brightness feasibility conditioning',
            'same_across_conditions':['background sequence','normal/anomaly request sequence','model initialization','network','loss','budget'],
            'not_shared':['defect mask','defect texture/appearance'], 'training_labels':'actual float residual M_eff',
            'author':pinned_author(), 'formal_training_entrypoint_implemented':False,
            'progression_rule':'Formal training awaits user1000-step probe and passing its fixed gate; no automatic launch'}
    locked_json(COMMON,common)
    cfg={'schema':'controlled-eight-sample-learnability-v1','seed':20261029,'steps':1000,'device':'cuda',
         'precision':'fp32','batch_size':1,'workers':0,'optimizer':'Adam','learning_rate':1e-4,'weight_decay':0,
         'learning_rate_schedule':'constant for1000-step diagnostic only', 'checkpoint_interval':200,
         'print_interval':100,'input_size':256,'input':'float RGB256 /255; no additional normalization',
         'loss_coefficients':{'MSE':1,'SSIM':1,'Focal':1},'sample_order':'fixed round robin normal0,anomaly0,...,normal3,anomaly3;125 cycles',
         'normal_paths':split['N_gen'][:4],'anomaly_specs':fixed_specs(),'outline_seeds':[20261029+i for i in range(4)],
         'samples':8,'normal_samples':4,'mura_samples':2,'line_samples':2,'unique_backgrounds':4,
         'training_label':'max RGB abs float residual > float32(1/255); normals allzero',
         'eval':'all8 train-side samples once at final1000 in eval/inference_mode; raw anomaly softmax; exact sklearn pixelAP',
         'gate':{'finite_losses_and_gradients':True,'both_models_parameters_changed':True,
                 'last20_mean_loss_less_than_first20':True,'training_pool_pixel_AP_minimum':.80},
         'minimum_available_RAM_before_start_GiB':3.0,
         'RAM_gate_reason':'conservative guard using old ~2.94GiB peak working-set; new peak not measured',
         'sampling_policy_sha256':digest(POLICY_PATH),'sampling_audit_sha256':digest(AUDIT_PATH),
         'common_config_sha256':digest(COMMON),'split_sha256':digest(SPLIT),'author':pinned_author(),
         'generator_identity':generator_identity(),'training_implementation_sha256':digest(Path(__file__).with_name('training.py')),
         'preparation_implementation_sha256':digest(__file__), 'model_output':str(MODEL_FOLDER),
         'not_generalization':True,'no_held_out_scoring':True,'manual_training_only':True}
    locked_json(CONFIG,cfg)
    if INPUTS.exists():
        verify_probe();print('PROBE_INPUTS_CACHED 8; training not started',flush=True);return cfg
    if DATA.exists():raise ValueError('Unidentified probe arrays exist; preserve and diagnose')
    x=[];clean=[];masks=[];rows=[]
    for i,(relative,spec) in enumerate(zip(cfg['normal_paths'],cfg['anomaly_specs'])):
        rgb=normal_rgb(split,relative);generated=render(rgb,spec,cfg['outline_seeds'][i])
        if not generated['stats']['valid']:raise ValueError('Fixed probe specimen is invalid: '+relative)
        for anomaly in (False,True):
            x.append(generated['image'] if anomaly else rgb);clean.append(rgb)
            masks.append(generated['M_eff'] if anomaly else np.zeros((256,256),dtype=bool))
            row={'index':len(rows),'background':relative,'source_sha256':split['source_sha256'][relative],
                 'anomaly':anomaly,'kind':spec['kind'] if anomaly else 'normal','spec':spec if anomaly else None,
                 'outline_seed':cfg['outline_seeds'][i] if anomaly else None,
                 'positive_pixels':int(masks[-1].sum()),'stats':generated['stats'] if anomaly else None}
            rows.append(row)
            folder=PROBE/'samples'/f'{row["index"]:02d}';folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.rint(x[-1]*255).astype(np.uint8)).save(folder/'image.png')
            Image.fromarray(masks[-1].astype(np.uint8)*255).save(folder/'float_label.png')
    with DATA.open('wb') as stream:
        np.savez_compressed(stream,x=np.stack(x).astype(np.float32),clean=np.stack(clean).astype(np.float32),mask=np.stack(masks))
    save_json(INPUTS,{'config_sha256':digest(CONFIG),'arrays_sha256':digest(DATA),'rows':rows,
                     'positive_pixels':sum(row['positive_pixels'] for row in rows),
                     'total_pixels':8*256*256,'previews':'PNG for inspection only; training reads float inputs.npz',
                     'training_started':False})
    verify_probe();save_json(PROBE/'preparation_resources.json',resources())
    print('PROBE_PREPARED 8 images/4 backgrounds; optimizer steps=0',flush=True)
    return cfg


def verify_probe():
    cfg=read_json(CONFIG);metadata=read_json(INPUTS);split=load_split()
    validate_policy(read_json(POLICY_PATH))
    if not read_json(AUDIT_PATH)['ready_for_probe_preparation'] or digest(AUDIT_PATH)!=cfg['sampling_audit_sha256']:
        raise ValueError('Sampling evidence changed or not ready')
    identities=[(CONFIG,metadata['config_sha256']),(DATA,metadata['arrays_sha256']),(POLICY_PATH,cfg['sampling_policy_sha256']),
                (COMMON,cfg['common_config_sha256']),(SPLIT,cfg['split_sha256']),
                (Path(__file__),cfg['preparation_implementation_sha256']),
                (Path(__file__).with_name('training.py'),cfg['training_implementation_sha256'])]
    if any(digest(path)!=sha for path,sha in identities):raise ValueError('Prepared data/code/config identity differs')
    if cfg['normal_paths']!=split['N_gen'][:4] or cfg['generator_identity']!=generator_identity() or pinned_author()!=cfg['author']:
        raise ValueError('Probe source scope changed')
    with np.load(DATA,allow_pickle=False) as saved:
        x=saved['x'];clean=saved['clean'];mask=saved['mask']
    if x.shape!=(8,256,256,3) or clean.shape!=x.shape or mask.shape!=(8,256,256):raise ValueError('Incorrect input shapes')
    if x.dtype!=np.float32 or clean.dtype!=np.float32 or mask.dtype!=np.bool_:raise ValueError('Wrong float/label dtype')
    if not np.isfinite(x).all() or not np.isfinite(clean).all() or x.min()<0 or x.max()>1 or clean.min()<0 or clean.max()>1:
        raise ValueError('Invalid RGB range or nonfinite values')
    actual=np.max(np.abs(x-clean),axis=-1)>np.float32(1/255)
    if not np.array_equal(actual,mask):raise ValueError('Probe labels do not match actual float changes')
    for i,relative in enumerate(cfg['normal_paths']):
        rgb=normal_rgb(split,relative)
        result=render(rgb,cfg['anomaly_specs'][i],cfg['outline_seeds'][i])
        if (not np.array_equal(clean[2*i],rgb) or not np.array_equal(clean[2*i+1],rgb)
            or not np.array_equal(x[2*i],rgb) or not np.array_equal(x[2*i+1],result['image'])
            or not result['stats']['valid']):raise ValueError('Fixed probe input differs from CPU regeneration')
    result={'config_sha256':digest(CONFIG),'inputs_sha256':digest(DATA),'samples':8,'unique_backgrounds':4,
            'normal_samples':4,'mura_samples':2,'line_samples':2,'total_pixels':int(mask.size),
            'positive_pixels':int(mask.sum()),'negative_pixels':int(mask.size-mask.sum()),
            'all_float_labels_match_actual_residual':True,'source_and_generator_reconstruction_matches':True,
            'optimizer_steps_executed_by_preparation':0,'model_inferences_executed_by_preparation':0,
            'scope':'N_gen first4 only; no S_bg/Q_cal/real-evaluation images or scores'}
    save_json(PROBE/'input_audit.json',result)
    print('PROBE_INPUT_AUDIT',{'images':8,'backgrounds':4,'positive_pixels':int(mask.sum()),'optimizer_steps':0},flush=True)
    return cfg,metadata,(x,clean,mask)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','verify'])
    args=parser.parse_args()
    if args.command=='prepare':prepare_probe()
    else:verify_probe()
