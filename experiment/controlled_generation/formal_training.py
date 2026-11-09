"""Explicit user-started formal G0/G1; preparation/status never invokes training."""
import gc
import random
import shutil
import sys
import time

import numpy as np

from .data import ROOT, read_json, save_json
from .formal_preparation import (CONFIG, SEQUENCE, HASHES, model_folder, result_folder, verify_formal_preparation)
from .formal_sampling import training_sample, sample_fingerprint
from .probe_lr_decay_preparation import sha_file
from .probe_lr_decay_training import (available_ram, atomic_torch_save, parameters_hash, weights_init,
                                      configure_stateless_bn, verify_stateless_bn)


def learning_rate_for_step(k):
    if type(k) is not int or not 1<=k<=10000:raise ValueError('Update index must be1..10000')
    return 1e-4 if k<=8000 else 2e-5 if k<=9000 else 4e-6


def schedule_state(completed):
    if type(completed) is not int or not 0<=completed<=10000:raise ValueError('Completed updates must be0..10000')
    last=learning_rate_for_step(completed) if completed else None
    following=learning_rate_for_step(completed+1) if completed<10000 else None
    return {'completed_steps':completed,'lr_used_last_step':last,'lr_for_next_step':following,
            'optimizer_learning_rate':following if following is not None else last,
            'applied_milestones':[m for m in (8000,9000) if completed>=m]}


def validate_schedule(record,completed,groups=None):
    if record!=schedule_state(completed):raise ValueError('Owned checkpoint formal LR schedule differs')
    if groups is not None and (not groups or any(g['lr']!=record['optimizer_learning_rate'] for g in groups)):
        raise ValueError('All optimizer groups must match formal LR schedule')


def validate_history(history,completed):
    if len(history)!=completed or any(v!=learning_rate_for_step(k+1) for k,v in enumerate(history)):
        raise ValueError('Actual LR history differs')


def identity(condition,cfg):
    return {'condition':condition,'common_config_sha256':sha_file(CONFIG),'sequence_sha256':sha_file(SEQUENCE),
            'sample_hashes_sha256':sha_file(HASHES),'training_code_sha256':sha_file(__file__),
            'seed':cfg['seed'],'steps':cfg['steps'],'scope':'N_gen/DTD synthetic training only; no held-out scoring'}


def checkpoint_header(condition,signature):
    folder=model_folder(condition);checkpoints=folder/'checkpoints';latest=folder/'latest.json'
    if any(folder.glob('*.tmp')) or any(checkpoints.glob('*.tmp')):
        raise ValueError('Partial save transaction; preserve and diagnose before any optimizer update')
    existing=list(checkpoints.glob('*.pt'))
    if not latest.exists():
        if existing or any(folder.glob('*.pth')):raise ValueError('Unidentified partial models; no restart/overwrite')
        return None
    row=read_json(latest);step=row.get('step')
    if row['identity']!=signature or type(step) is not int or step not in range(1000,10001,1000):
        raise ValueError('Owned formal checkpoint identity/step differs')
    validate_schedule(row['schedule_state'],step)
    path=(folder/row['filename']).resolve();expected=(checkpoints/f'step_{step:06d}.pt').resolve()
    if path!=expected or sha_file(path)!=row['checkpoint_sha256']:raise ValueError('Checkpoint path/hash differs')
    if any(p.name not in {f'step_{k:06d}.pt' for k in range(1000,step+1,1000)} for p in existing):
        raise ValueError('Uncommitted future/orphan checkpoint; preserve and diagnose')
    return row


def completed_formal_status(condition):
    cfg,*_=verify_formal_preparation();signature=identity(condition,cfg)
    path=result_folder(condition)/'training_result.json'
    if not path.exists():
        header=checkpoint_header(condition,signature)
        result={'training_complete':False,'condition':condition,'saved_step':header['step'] if header else 0,
                'fixed_steps':10000,'manual_training_pending':True}
        print('FORMAL_STATUS',result,flush=True);return result
    result=read_json(path)
    if result['identity']!=signature or result['steps']!=10000:raise ValueError('Completed formal identity differs')
    header=checkpoint_header(condition,signature)
    if header is None or header['step']!=10000:raise ValueError('Completed result requires owned complete10000 checkpoint')
    if set(result['weight_files'])!={'reconstruction.pth','segmentation.pth'}:
        raise ValueError('Both exact final formal weights are required')
    if (len(result['losses'])!=10000 or len(result['maximum_absolute_gradient_per_step'])!=10000
            or not np.isfinite(result['losses']).all() or not np.isfinite(result['maximum_absolute_gradient_per_step']).all()
            or result['sampling']['normal_updates']!=5000 or result['sampling']['anomaly_updates']!=5000):
        raise ValueError('Completed formal training evidence differs')
    validate_history(result['actual_learning_rate_per_step'],10000)
    validate_schedule(result['final_schedule_state'],10000)
    for name,sha in result['weight_files'].items():
        if sha_file(model_folder(condition)/name)!=sha:raise ValueError('Final formal weights changed')
    print('FORMAL_COMPLETE',{'condition':condition,'steps':10000,'held_out_scoring':False},flush=True)
    return result


def train_formal(condition,seed=20261030):
    """Only this explicit user-called entry constructs models/optimizer and updates weights."""
    cfg,sequence,backgrounds,inventory,fingerprints=verify_formal_preparation()
    if condition not in cfg['conditions'] or seed!=cfg['seed'] or cfg['steps']!=10000:
        raise ValueError('Outside frozen formal condition/seed/budget')
    signature=identity(condition,cfg);output=result_folder(condition);folder=model_folder(condition)
    if (output/'training_result.json').exists():return completed_formal_status(condition)
    committed=checkpoint_header(condition,signature)
    g0=None
    if condition=='G1':
        g0=completed_formal_status('G0')
        if g0.get('steps')!=10000:raise ValueError('G1 waits for complete G0; no models or optimizer created')
    before=available_ram()
    if before<cfg['minimum_available_RAM_GiB']:raise RuntimeError('Need at least3GiB free RAM; no updates started')
    if shutil.disk_usage(ROOT).free/2**30<cfg['minimum_disk_per_condition_GiB']:
        raise RuntimeError('Need at least16GiB free disk for this condition; preserve old files')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Frozen formal training requires CUDA; no CPU fallback')
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
    if g0 is not None and (initial_rec!=g0['initial_reconstruction_parameters_sha256']
                          or initial_seg!=g0['initial_segmentation_parameters_sha256']):
        raise ValueError('G1 initialization differs from G0; no optimizer updates started')
    parameters=list(rec.parameters())+list(seg.parameters())
    optimizer=torch.optim.Adam(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    mse=torch.nn.MSELoss();ssim=SSIM();focal=FocalLoss()
    folder.mkdir(parents=True,exist_ok=True);checkpoints=folder/'checkpoints';checkpoints.mkdir(exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    start=0;losses=[];gradients=[];lr_history=[];saved_elapsed=0.;resume_count=0;minimum_ram=before
    sampling={'normal_updates':0,'anomaly_updates':0,'generation_attempts':0,'rejected_attempts':0,'kind_polarity_counts':{}}
    if committed is not None:
        path=folder/committed['filename']
        # Local user-produced, fixed owned path and verified SHA256 only.
        state=torch.load(path,map_location='cuda',weights_only=False)
        if state['identity']!=signature or state['step']!=committed['step']:raise ValueError('Restored formal identity differs')
        validate_schedule(state['schedule_state'],state['step'],state['optimizer']['param_groups'])
        if state['schedule_state']!=committed['schedule_state']:raise ValueError('Header/state LR differs')
        validate_history(state['actual_learning_rate_per_step'],state['step'])
        if state['initial_rec_parameters_sha256']!=initial_rec or state['initial_seg_parameters_sha256']!=initial_seg:
            raise ValueError('Restored initialization differs')
        rec.load_state_dict(state['rec']);seg.load_state_dict(state['seg']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        start=state['step'];losses=state['losses'];gradients=state['gradients'];lr_history=state['actual_learning_rate_per_step']
        sampling=state['sampling'];saved_elapsed=state['elapsed_seconds'];resume_count=state['resume_count']+1
        minimum_ram=min(before,state['minimum_available_RAM_GiB'])
        if (len(losses)!=start or len(gradients)!=start or sampling['normal_updates']!=start//2
                or sampling['anomaly_updates']!=start//2):raise ValueError('Incomplete restored training evidence')
        del state;gc.collect();print('FORMAL_RESUMED',condition,start,flush=True)
    resumed_from_step=start;segment_started=time.perf_counter()
    for step in range(start,10000):
        validate_schedule(schedule_state(step),step,optimizer.param_groups)
        sample,info=training_sample(condition,cfg,sequence,step,backgrounds,inventory)
        if sample is None:raise RuntimeError('Five invalid synthesis attempts; stop without replacing background/branch')
        if info['anomaly']:
            ordinal=info['ordinal']
            if sample_fingerprint(sample).encode()!=fingerprints[condition][ordinal]:
                raise ValueError('Actual training input differs from frozen CPU audit; no update at this request')
        x=torch.from_numpy(np.ascontiguousarray(sample['image'].transpose(2,0,1))).unsqueeze(0).cuda()
        clean=torch.from_numpy(np.ascontiguousarray(sample['background'].transpose(2,0,1))).unsqueeze(0).cuda()
        mask=torch.from_numpy(sample['M_eff'].astype(np.float32)[None,None]).cuda()
        optimizer.zero_grad(set_to_none=True)
        reconstructed=rec(x);probability=torch.softmax(seg(torch.cat((reconstructed,x),dim=1)),dim=1)
        loss=mse(reconstructed,clean)+ssim(reconstructed,clean)+focal(probability,mask)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss before update'+str(step+1))
        loss.backward()
        maxima=[p.grad.detach().abs().max() for p in parameters if p.grad is not None]
        if not maxima:raise RuntimeError('No gradients before optimizer update')
        maximum=torch.stack(maxima).max()
        if not torch.isfinite(maximum):raise RuntimeError('Nonfinite gradient before optimizer update')
        losses.append(float(loss.detach().cpu()));gradients.append(float(maximum.cpu()))
        used_lr=learning_rate_for_step(step+1);lr_history.append(used_lr)
        optimizer.step();post_schedule=schedule_state(step+1)
        for group in optimizer.param_groups:group['lr']=post_schedule['optimizer_learning_rate']
        if info['anomaly']:
            sampling['anomaly_updates']+=1;sampling['generation_attempts']+=len(info['attempts'])
            sampling['rejected_attempts']+=len(info['attempts'])-1
            key=str(info['kind'])+'_'+str(info['polarity'])
            sampling['kind_polarity_counts'][key]=sampling['kind_polarity_counts'].get(key,0)+1
        else:sampling['normal_updates']+=1
        if (step+1)%cfg['print_interval']==0:
            torch.cuda.synchronize();minimum_ram=min(minimum_ram,available_ram())
            print({'condition':condition,'step':step+1,'of':10000,'loss':losses[-1],
                   'lr_used_this_step':used_lr,'lr_for_next_step':post_schedule['lr_for_next_step'],
                   'elapsed_seconds':saved_elapsed+time.perf_counter()-segment_started},flush=True)
        if (step+1)%cfg['checkpoint_interval']==0:
            current=checkpoints/f'step_{step+1:06d}.pt'
            if current.exists():raise ValueError('Do not overwrite committed checkpoint')
            state={'identity':signature,'step':step+1,'schedule_state':post_schedule,'rec':rec.state_dict(),'seg':seg.state_dict(),
                'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'losses':losses,'gradients':gradients,
                'actual_learning_rate_per_step':lr_history,'sampling':sampling,'initial_rec_parameters_sha256':initial_rec,
                'initial_seg_parameters_sha256':initial_seg,'resume_count':resume_count,'minimum_available_RAM_GiB':minimum_ram,
                'elapsed_seconds':saved_elapsed+time.perf_counter()-segment_started}
            atomic_torch_save(torch,state,current)
            save_json(folder/'latest.json',{'identity':signature,'step':step+1,'schedule_state':post_schedule,
                      'filename':str(current.relative_to(folder)),'checkpoint_sha256':sha_file(current)})
            del state
    validate_history(lr_history,10000);validate_schedule(schedule_state(10000),10000,optimizer.param_groups)
    verify_stateless_bn(torch,rec);verify_stateless_bn(torch,seg)
    if not all(torch.isfinite(p).all() for p in parameters):raise RuntimeError('Nonfinite final formal weights')
    torch.cuda.synchronize();elapsed=saved_elapsed+time.perf_counter()-segment_started
    atomic_torch_save(torch,rec.state_dict(),folder/'reconstruction.pth')
    atomic_torch_save(torch,seg.state_dict(),folder/'segmentation.pth')
    result={'identity':signature,'condition':condition,'seed':seed,'steps':10000,'training_complete':True,
        'initial_reconstruction_parameters_sha256':initial_rec,'initial_segmentation_parameters_sha256':initial_seg,
        'final_reconstruction_parameters_sha256':parameters_hash(rec),'final_segmentation_parameters_sha256':parameters_hash(seg),
        'losses':losses,'maximum_absolute_gradient_per_step':gradients,'actual_learning_rate_per_step':lr_history,
        'final_schedule_state':schedule_state(10000),'sampling':sampling,'batchnorm_module_counts':bn_counts,
        'normalization':cfg['normalization'],'elapsed_training_seconds':elapsed,'resume_count':resume_count,
        'resumed_from_step':resumed_from_step,'optimizer_updates_after_resume':10000-resumed_from_step if resume_count else 0,
        'gpu':torch.cuda.get_device_name(0),'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'minimum_available_RAM_GiB':minimum_ram,
        'weight_files':{name:sha_file(folder/name) for name in ('reconstruction.pth','segmentation.pth')},
        'held_out_scoring':False,'real_evaluation':False,'automatic_calibration_or_next_condition':False}
    save_json(output/'training_result.json',result)
    print('FORMAL_TRAINING_FINISHED',{'condition':condition,'steps':10000,'next':'inspect saved result; no automatic next stage'},flush=True)
    return result
