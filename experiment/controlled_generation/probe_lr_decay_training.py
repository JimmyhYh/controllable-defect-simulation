"""Independent user-started1000-step LR-decay probe; all older probes untouched."""
import argparse
import ctypes
import gc
import hashlib
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .data import ROOT, read_json, save_json
from .probe_lr_decay_preparation import (CONFIG, DATA, INPUTS, MODEL_FOLDER, PROBE,
                                         sha_file as digest, verify_lr_decay_probe as verify_probe)


def available_ram():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise OSError('RAM query failed')
    return memory.available/2**30


def atomic_torch_save(torch,value,path):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)


def parameters_hash(model):
    h=hashlib.sha256()
    for name,parameter in model.named_parameters():
        h.update(name.encode());h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def weights_init(module):
    name=module.__class__.__name__
    if 'Conv' in name:module.weight.data.normal_(0,.02)
    elif 'BatchNorm' in name:module.weight.data.normal_(1,.02);module.bias.data.fill_(0)


def learning_rate_for_step(k):
    if type(k) is not int or not 1<=k<=1000:
        raise ValueError('Optimizer update index must be an integer in1..1000')
    return 1e-4 if k<=800 else 2e-5


def schedule_state(completed_steps):
    if type(completed_steps) is not int or not 0<=completed_steps<=1000:
        raise ValueError('Completed optimizer updates must be0..1000')
    last=learning_rate_for_step(completed_steps) if completed_steps else None
    next_rate=learning_rate_for_step(completed_steps+1) if completed_steps<1000 else None
    return {'completed_steps':completed_steps,'lr_used_last_step':last,'lr_for_next_step':next_rate,
            'optimizer_learning_rate':next_rate if next_rate is not None else last,
            'applied_milestones':[800] if completed_steps>=800 else []}


def validate_schedule_record(record,completed_steps):
    if record!=schedule_state(completed_steps):
        raise ValueError('Saved LR boundary/resume schedule differs; no optimizer update permitted')


def validate_optimizer_learning_rates(param_groups,completed_steps):
    expected=schedule_state(completed_steps)['optimizer_learning_rate']
    if not param_groups or any(group.get('lr')!=expected for group in param_groups):
        raise ValueError('Optimizer groups differ from owned checkpoint LR schedule')


def validate_lr_history(history,completed_steps):
    if len(history)!=completed_steps or any(rate!=learning_rate_for_step(i+1) for i,rate in enumerate(history)):
        raise ValueError('Saved actual learning-rate history differs')


def identity(cfg,metadata):
    return {'config_sha256':digest(CONFIG),'inputs_sha256':digest(DATA),'inputs_metadata_sha256':digest(INPUTS),
            'implementation_sha256':digest(__file__),'seed':cfg['seed'],'steps':cfg['steps'],
            'normalization':cfg['normalization'],'learning_rate_schedule':cfg['learning_rate_schedule'],
            'scope':'fixed8 train-side samples only; no formal G0/G1 or held-out inference'}


def resume_preflight(signature):
    """Detect incomplete save transactions before model creation or any update."""
    folder=(MODEL_FOLDER/'checkpoints').resolve();latest=MODEL_FOLDER/'latest.json'
    checkpoints=list(folder.glob('*.pt')) if folder.exists() else []
    partial=list(folder.glob('*.tmp')) if folder.exists() else []
    if MODEL_FOLDER.exists():partial+=list(MODEL_FOLDER.glob('*.tmp'))
    if partial:
        raise RuntimeError('Interrupted save transaction before training start: '+str(partial[0])+
                           '. Preserve files and diagnose; no optimizer update started.')
    if not latest.exists():
        if checkpoints or any(MODEL_FOLDER.glob('*.pth')):
            raise RuntimeError('Model/checkpoint file exists without committed latest.json. Save was incomplete; '
                               'preserve and diagnose before training, no optimizer update started.')
        return None
    header=read_json(latest)
    step=header.get('step')
    if not isinstance(step,int) or step not in range(200,1001,200) or header.get('identity')!=signature:
        raise ValueError('Committed probe checkpoint identity/step differs')
    validate_schedule_record(header.get('schedule_state'),step)
    expected=folder/f'step_{step:06d}.pt';checkpoint=(MODEL_FOLDER/header['filename']).resolve()
    if checkpoint!=expected or digest(checkpoint)!=header['checkpoint_sha256']:
        raise ValueError('Committed checkpoint path or checksum differs')
    permitted={f'step_{s:06d}.pt' for s in range(200,step+1,200)}
    orphan=[path for path in checkpoints if path.name not in permitted]
    if orphan:
        raise RuntimeError('Checkpoint committed before latest.json: '+str(orphan[0])+
                           '. Preserve and diagnose interrupted save before any training update.')
    return header


def completed_lr_decay_status():
    cfg,metadata,_=verify_probe();signature=identity(cfg,metadata)
    path=PROBE/'training_result.json'
    if not path.exists():
        result={'training_complete':False,'prepared_samples':8,'fixed_steps':1000,'awaiting_manual_user_training':True}
        row=resume_preflight(signature)
        if row is not None:
            result['saved_step']=row['step'];result['schedule_state']=row['schedule_state']
        else:result['saved_step']=0;result['schedule_state']=schedule_state(0)
        print('LR_DECAY_PROBE_STATUS',result,flush=True);return result
    result=read_json(path)
    if result['identity']!=signature or result['steps']!=1000:
        raise ValueError('Completed probe identity/step differs')
    validate_schedule_record(result.get('final_schedule_state'),1000)
    validate_lr_history(result['actual_learning_rate_per_step'],1000)
    for name,sha in result['weight_files'].items():
        if digest(MODEL_FOLDER/name)!=sha:raise ValueError('Completed weight identity differs')
    for name,sha in result['evaluation_artifact_sha256'].items():
        if digest(PROBE/name)!=sha:raise ValueError('Saved train-side evaluation artifact differs')
    print('LR_DECAY_PROBE_COMPLETE',{'steps':1000,'training_pool_pixel_AP':result['training_pool_pixel_AP'],
          'gate_passed':result['gate_passed'],'formal_training_automatically_started':False},flush=True)
    return result


def configure_stateless_bn(torch, model):
    count=0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats=False
            module.running_mean=None;module.running_var=None;module.num_batches_tracked=None
            count+=1
    if count==0:raise ValueError('No author BatchNorm2d found')
    return count


def verify_stateless_bn(torch, model):
    modules=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    if not modules or any(m.track_running_stats or m.running_mean is not None
                          or m.running_var is not None or m.num_batches_tracked is not None for m in modules):
        raise ValueError('Stateless BN contract differs')


def train_lr_decay_probe(seed=20261029):
    """User calls this explicit training entry. Preparation never calls it."""
    cfg,metadata,(images,targets,labels)=verify_probe()
    if seed!=cfg['seed'] or cfg['steps']!=1000:raise ValueError('Outside frozen probe seed/budget')
    signature=identity(cfg,metadata)
    if (PROBE/'training_result.json').exists():return completed_lr_decay_status()
    committed=resume_preflight(signature)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Frozen FP32 probe requires CUDA; no automatic CPU fallback')
    before=available_ram()
    if before<cfg['minimum_available_RAM_before_start_GiB']:
        raise RuntimeError(f'Only{before:.2f}GiB free RAM; free at least3.0GiB then rerun unchanged command. No model update started.')
    if shutil.disk_usage(ROOT).free/2**30<cfg['minimum_available_disk_before_start_GiB']:
        raise RuntimeError('Need at least8GiB free disk; preserve old artifacts. No model update started.')
    sys.path.insert(0,str(ROOT/'vendor/DRAEM'))
    from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork
    from loss import SSIM,FocalLoss
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    rec=ReconstructiveSubNetwork(3,3).cuda();seg=DiscriminativeSubNetwork(6,2).cuda()
    bn_counts={'rec':configure_stateless_bn(torch,rec),'seg':configure_stateless_bn(torch,seg)}
    rec.apply(weights_init);seg.apply(weights_init);rec.train();seg.train()
    initial_rec=parameters_hash(rec);initial_seg=parameters_hash(seg)
    if {'rec':initial_rec,'seg':initial_seg}!=cfg['expected_initial_parameters_sha256']:
        raise ValueError('Initial parameters differ from original same-seed probe; no optimizer update started')
    verify_stateless_bn(torch,rec);verify_stateless_bn(torch,seg)
    parameters=list(rec.parameters())+list(seg.parameters())
    optimizer=torch.optim.Adam(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    mse=torch.nn.MSELoss();ssim=SSIM();focal=FocalLoss()
    MODEL_FOLDER.mkdir(parents=True,exist_ok=True)
    checkpoint_folder=MODEL_FOLDER/'checkpoints';checkpoint_folder.mkdir(exist_ok=True)
    latest=MODEL_FOLDER/'latest.json'
    start=0;losses=[];gradient_maxima=[];lr_history=[];saved_elapsed=0.;resume_count=0;minimum_ram=before
    if committed is not None:
        header=committed;checkpoint=(MODEL_FOLDER/header['filename']).resolve()
        if (not checkpoint.is_relative_to(checkpoint_folder.resolve()) or header['identity']!=signature
                or digest(checkpoint)!=header['checkpoint_sha256']):raise ValueError('Owned probe checkpoint identity differs')
        # Only this locally produced, checksum-verified checkpoint is deserialized.
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        if state['identity']!=signature or state['step']!=header['step'] or not 0<state['step']<=1000:
            raise ValueError('Restored checkpoint has wrong identity/step')
        validate_schedule_record(state.get('schedule_state'),state['step'])
        if state['schedule_state']!=header['schedule_state']:raise ValueError('Header/checkpoint LR state differs')
        validate_lr_history(state['actual_learning_rate_per_step'],state['step'])
        validate_optimizer_learning_rates(state['optimizer']['param_groups'],state['step'])
        rec.load_state_dict(state['rec']);seg.load_state_dict(state['seg']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        start=state['step'];losses=state['losses'];gradient_maxima=state['gradient_maxima'];saved_elapsed=state['elapsed_seconds']
        if (state['initial_rec_parameters_sha256']!=initial_rec or state['initial_seg_parameters_sha256']!=initial_seg):
            raise ValueError('Restored initial parameter identity differs')
        lr_history=state['actual_learning_rate_per_step']
        validate_optimizer_learning_rates(optimizer.param_groups,start)
        resume_count=state['resume_count']+1;minimum_ram=state['minimum_available_RAM_GiB']
        if len(losses)!=start or len(gradient_maxima)!=start:raise ValueError('Incomplete restored numeric evidence')
        del state;gc.collect();print('LR_DECAY_PROBE_RESUMED',start,flush=True)
    elif any(checkpoint_folder.iterdir()) or any(MODEL_FOLDER.glob('*.pth')):
        raise ValueError('Unidentified partial probe model artifacts; preserve and diagnose instead of restarting')
    resumed_from_step=start
    segment_started=time.perf_counter()
    for step in range(start,1000):
        validate_optimizer_learning_rates(optimizer.param_groups,step)
        used_lr=learning_rate_for_step(step+1)
        i=step%8
        x=torch.from_numpy(np.ascontiguousarray(images[i].transpose(2,0,1))).unsqueeze(0).cuda()
        clean=torch.from_numpy(np.ascontiguousarray(targets[i].transpose(2,0,1))).unsqueeze(0).cuda()
        mask=torch.from_numpy(labels[i].astype(np.float32)[None,None]).cuda()
        optimizer.zero_grad(set_to_none=True)
        reconstructed=rec(x);probability=torch.softmax(seg(torch.cat((reconstructed,x),dim=1)),dim=1)
        loss=mse(reconstructed,clean)+ssim(reconstructed,clean)+focal(probability,mask)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop at step'+str(step+1))
        loss.backward()
        maxima=[p.grad.detach().abs().max() for p in parameters if p.grad is not None]
        if not maxima:raise RuntimeError('No gradients; stop before optimizer step')
        maximum=torch.stack(maxima).max()
        if not torch.isfinite(maximum):raise RuntimeError('Nonfinite gradients; stop before optimizer step')
        losses.append(float(loss.detach().cpu()));gradient_maxima.append(float(maximum.cpu()))
        lr_history.append(used_lr)
        optimizer.step()
        next_schedule=schedule_state(step+1)
        for group in optimizer.param_groups:group['lr']=next_schedule['optimizer_learning_rate']
        if (step+1)%cfg['print_interval']==0:
            torch.cuda.synchronize();minimum_ram=min(minimum_ram,available_ram())
            print({'probe_step':step+1,'of':1000,'loss':losses[-1],
                   'lr_used_this_step':used_lr,'lr_for_next_step':next_schedule['lr_for_next_step'],
                   'elapsed_seconds':saved_elapsed+time.perf_counter()-segment_started},flush=True)
        if (step+1)%cfg['checkpoint_interval']==0:
            current=checkpoint_folder/f'step_{step+1:06d}.pt'
            if current.exists():raise ValueError('Checkpoint step already exists; do not overwrite')
            state={'identity':signature,'step':step+1,'schedule_state':next_schedule,
                   'actual_learning_rate_per_step':lr_history,'rec':rec.state_dict(),'seg':seg.state_dict(),
                   'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'losses':losses,
                   'gradient_maxima':gradient_maxima,'initial_rec_parameters_sha256':initial_rec,
                   'initial_seg_parameters_sha256':initial_seg,'resume_count':resume_count,
                   'minimum_available_RAM_GiB':minimum_ram,'elapsed_seconds':saved_elapsed+time.perf_counter()-segment_started}
            atomic_torch_save(torch,state,current)
            save_json(latest,{'identity':signature,'step':step+1,'filename':str(current.relative_to(MODEL_FOLDER)),
                              'checkpoint_sha256':digest(current),'schedule_state':next_schedule})
            del state
    validate_lr_history(lr_history,1000)
    validate_optimizer_learning_rates(optimizer.param_groups,1000)
    verify_stateless_bn(torch,rec);verify_stateless_bn(torch,seg)
    torch.cuda.synchronize();elapsed_training=saved_elapsed+time.perf_counter()-segment_started
    if not all(torch.isfinite(p).all() for p in parameters):raise RuntimeError('Nonfinite final parameters')
    final_rec=parameters_hash(rec);final_seg=parameters_hash(seg)
    rec.eval();seg.eval();maps=np.empty((8,256,256),dtype=np.float32);reconstruction_mse=[];final_sample_diagnostics=[]
    with torch.inference_mode():
        for i in range(8):
            x=torch.from_numpy(np.ascontiguousarray(images[i].transpose(2,0,1))).unsqueeze(0).cuda()
            clean=torch.from_numpy(np.ascontiguousarray(targets[i].transpose(2,0,1))).unsqueeze(0).cuda()
            reconstructed=rec(x)
            probabilities=torch.softmax(seg(torch.cat((reconstructed,x),dim=1)),dim=1)
            p=probabilities[0,1]
            maps[i]=p.cpu().numpy();reconstruction_mse.append(float(mse(reconstructed,clean).cpu()))
            components={'MSE':reconstruction_mse[-1],'SSIM_loss':float(ssim(reconstructed,clean).cpu()),
                        'Focal':float(focal(probabilities,
                                            torch.from_numpy(labels[i].astype(np.float32)[None,None]).cuda()).cpu())}
            final_sample_diagnostics.append({'index':i,'losses':components,
                'reconstruction_range':[float(reconstructed.min().cpu()),float(reconstructed.max().cpu())],
                'fraction_reconstruction_channels_outside_0_1':float(((reconstructed<0)|(reconstructed>1)).float().mean().cpu()),
                'max_anomaly_softmax':float(p.max().cpu())})
    if not np.isfinite(maps).all() or not np.isfinite(reconstruction_mse).all():raise RuntimeError('Nonfinite final train-side evaluation')
    if not all(np.isfinite(list(row['losses'].values())).all() for row in final_sample_diagnostics):
        raise RuntimeError('Nonfinite final loss diagnostics')
    from sklearn.metrics import average_precision_score
    ap=float(average_precision_score(labels.reshape(-1).astype(np.uint8),maps.reshape(-1)))
    maps_path=PROBE/'final_train_heatmaps.npy'
    with maps_path.open('wb') as stream:np.save(stream,maps,allow_pickle=False)
    changed=final_rec!=initial_rec and final_seg!=initial_seg
    first20=float(np.mean(losses[:20]));last20=float(np.mean(losses[-20:]))
    finite=bool(np.isfinite(losses).all() and np.isfinite(gradient_maxima).all())
    passed=bool(finite and changed and last20<first20 and ap>=.80)
    atomic_torch_save(torch,rec.state_dict(),MODEL_FOLDER/'reconstruction.pth')
    atomic_torch_save(torch,seg.state_dict(),MODEL_FOLDER/'segmentation.pth')
    torch.cuda.synchronize()
    result={'identity':signature,'seed':seed,'steps':1000,'sample_counts':{'normal':4,'mura_like':2,'line_like':2},
            'unique_backgrounds':4,'total_pixels':int(labels.size),'positive_pixels':int(labels.sum()),
            'negative_pixels':int(labels.size-labels.sum()),'training_pool_pixel_AP':ap,
            'first20_mean_loss':first20,'last20_mean_loss':last20,'finite_losses_and_gradients':finite,
            'reconstruction_parameters_changed':final_rec!=initial_rec,'segmentation_parameters_changed':final_seg!=initial_seg,
            'initial_reconstruction_parameters_sha256':initial_rec,'initial_segmentation_parameters_sha256':initial_seg,
            'final_reconstruction_parameters_sha256':final_rec,'final_segmentation_parameters_sha256':final_seg,
            'gate_passed':passed,'losses':losses,'maximum_absolute_gradient_per_step':gradient_maxima,
            'reconstruction_mse_by_sample':reconstruction_mse,'eval_mode':'eval/inference_mode, stateless per-input BN, final1000 only',
            'normalization':cfg['normalization'],'batchnorm_module_counts':bn_counts,
            'normalization_running_buffers_absent':True,
            'baseline_training_pool_pixel_AP':cfg['source_training_pool_pixel_AP'],
            'AP_delta_vs_control_percentage_points':100*(ap-cfg['source_training_pool_pixel_AP']),
            'actual_learning_rate_per_step':lr_history,'final_schedule_state':schedule_state(1000),
            'final_sample_diagnostics':final_sample_diagnostics,
            'original_gate_changed':False,
            'elapsed_training_seconds':elapsed_training,'this_process_load_train_eval_save_seconds':time.perf_counter()-started,
            'resume_count':resume_count,'gpu':torch.cuda.get_device_name(0),
            'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'minimum_available_RAM_GiB':minimum_ram,
            'weight_files':{name:digest(MODEL_FOLDER/name) for name in ('reconstruction.pth','segmentation.pth')},
            'evaluation_artifact_sha256':{'final_train_heatmaps.npy':digest(maps_path)},
            'real_evaluation':False,'held_out_scoring':False,'formal_training_automatically_started':False,
            'scientific_scope':'fitting diagnostic on the same8 samples; not new model performance or generalization',
            'checkpoint_load_runtime_validated':resume_count>0,
            'optimizer_updates_after_resume':1000-resumed_from_step if resume_count else 0,
            'resume_path_runtime_validated':bool(resume_count>0 and resumed_from_step<1000)}
    save_json(PROBE/'training_result.json',result)
    print('LR_DECAY_PROBE_FINISHED',{'steps':1000,'training_pool_pixel_AP':ap,'gate_passed':passed,
          'next':'inspect saved result before separate formal training; no automatic next stage'},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['probe-train','probe-status'])
    parser.add_argument('--seed',type=int,default=20261029)
    args=parser.parse_args()
    if args.command=='probe-train':train_lr_decay_probe(args.seed)
    else:completed_lr_decay_status()
