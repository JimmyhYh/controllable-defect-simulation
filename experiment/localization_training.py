"""Fixed-budget formal DRAEM training with exact owned-checkpoint recovery."""
import copy
import gc
import hashlib
import json
import random
import time
import numpy as np
import torch
from localization_data import ROOT, OUT, save, digest
from localization_resources import frozen_resources, MODEL_ROOT
from localization_synthesis import PairStream, blend
from localization_train import ReconstructiveSubNetwork, DiscriminativeSubNetwork, weights_init, SSIM, FocalLoss, process_ram, available_ram

def atomic_torch_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp)
    tmp.replace(path)

def train(condition='A',seed=20261020):
    cfg,split,identity=frozen_resources()
    if seed not in cfg['paired_seeds'] or condition not in ('A','B'):
        raise ValueError('Outside first-round scope')
    folder=MODEL_ROOT/f'draem_{condition}_{seed}'
    summary=OUT/f'training_{condition}_{seed}.json'
    if summary.exists():
        done=json.loads(summary.read_text(encoding='utf-8'))
        if done['identity'] != identity or done['steps'] != cfg['steps'] or any(digest(folder/n)!=h for n,h in done['weight_files'].items()):
            raise ValueError('Final model identity differs')
        print(json.dumps({'training_cached':True,'condition':condition,'steps':done['steps']}),flush=True)
        return
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'resume.pt';meta=folder/'resume.json'
    if checkpoint.exists()!=meta.exists():
        raise ValueError('Partial checkpoint transaction; inspect before recovery')
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    wall_start=time.perf_counter()
    rec=ReconstructiveSubNetwork(3,3).cuda();seg=DiscriminativeSubNetwork(6,2).cuda()
    rec.apply(weights_init);seg.apply(weights_init)
    initial_hash=hashlib.sha256(rec.encoder.block1[0].weight.detach().cpu().numpy().tobytes()).hexdigest()
    reference=json.loads((OUT/'probe_results.json').read_text(encoding='utf-8'))['results'][0]['initial_first_conv_sha256']
    if initial_hash != reference:
        raise ValueError('Formal initial weights differ from paired initialization')
    params=list(rec.parameters())+list(seg.parameters())
    optimizer=torch.optim.Adam(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['gamma'])
    stream=PairStream(split,seed+cfg['data_seed_offset'])
    mse=torch.nn.MSELoss();ssim=SSIM();focal=FocalLoss()
    start=0;elapsed=0.0;loss_points=[];active=0;replaced=0;minimum_ram=available_ram();resume_count=0
    # A rolling digest is saved as bytes rather than pickling a hashlib object.
    trace=''
    if checkpoint.exists():
        header=json.loads(meta.read_text(encoding='utf-8'))
        if header['identity']!=identity or header['condition']!=condition or header['seed']!=seed or digest(checkpoint)!=header['checkpoint_sha256']:
            raise ValueError('Owned checkpoint identity mismatch')
        # Only this locally produced, checksummed checkpoint is deserialized.
        state=torch.load(checkpoint,weights_only=False,map_location='cuda')
        rec.load_state_dict(state['rec']);seg.load_state_dict(state['seg'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        stream.rng.bit_generator.state=state['stream_rng'];stream.ordinal=state['stream_ordinal']
        stream.generator=state['generator']
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        start=state['step'];elapsed=state['elapsed_seconds'];loss_points=state['loss_points']
        active=state['active'];replaced=state['replaced'];trace=state['trace']
        minimum_ram=state['minimum_ram'];resume_count=state['resume_count']+1
        del state;gc.collect()
        print(f'RESUMED {condition} at step {start}',flush=True)
    segment_start=time.perf_counter()
    for step in range(start,cfg['steps']):
        sample=stream.next()
        strength=cfg['lambda_weak'] if condition=='B' and sample['replace'] else sample['lambda_A']
        x=torch.from_numpy(np.ascontiguousarray(blend(sample,strength).transpose(2,0,1))).unsqueeze(0).cuda()
        clean=torch.from_numpy(np.ascontiguousarray(sample['background'].transpose(2,0,1))).unsqueeze(0).cuda()
        mask=torch.from_numpy(sample['mask'].astype(np.float32)[None,None]).cuda()
        optimizer.zero_grad(set_to_none=True)
        reconstructed=rec(x);prob=torch.softmax(seg(torch.cat((reconstructed,x),1)),dim=1)
        loss=mse(reconstructed,clean)+ssim(reconstructed,clean)+focal(prob,mask)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step+1}')
        loss.backward()
        if (step+1)%100==0 and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError(f'Nonfinite gradients at {step+1}')
        optimizer.step();scheduler.step()
        active+=int(sample['active']);replaced+=int(condition=='B' and sample['replace'])
        component={'metadata':sample['metadata'],'active':sample['active'],'replace':sample['replace'],'lambda_A':sample['lambda_A'],
                   'B':hashlib.sha256(sample['background'].tobytes()).hexdigest(),
                   'T':hashlib.sha256(sample['texture'].tobytes()).hexdigest(),
                   'M':hashlib.sha256(sample['mask'].tobytes()).hexdigest()}
        trace=hashlib.sha256((trace+json.dumps(component,sort_keys=True)).encode()).hexdigest()
        if (step+1)%100==0:
            torch.cuda.synchronize();minimum_ram=min(minimum_ram,available_ram())
            print(json.dumps({'condition':condition,'step':step+1,'of':cfg['steps'],'loss':float(loss.detach().cpu()),'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':elapsed+time.perf_counter()-segment_start}),flush=True)
        if (step+1)%1000==0:
            loss_points.append({'step':step+1,'loss':float(loss.detach().cpu()),'lr_after_step':optimizer.param_groups[0]['lr']})
            current_elapsed=elapsed+time.perf_counter()-segment_start
            state={'identity':identity,'condition':condition,'seed':seed,'step':step+1,
                   'rec':rec.state_dict(),'seg':seg.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'stream_rng':copy.deepcopy(stream.rng.bit_generator.state),'stream_ordinal':stream.ordinal,'generator':stream.generator,
                   'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
                   'python_rng':random.getstate(),'elapsed_seconds':current_elapsed,'loss_points':loss_points,
                   'active':active,'replaced':replaced,'trace':trace,'minimum_ram':minimum_ram,'resume_count':resume_count}
            atomic_torch_save(state,checkpoint)
            save(meta,{'identity':identity,'condition':condition,'seed':seed,'step':step+1,'checkpoint_sha256':digest(checkpoint)})
            del state
    torch.cuda.synchronize()
    if not all(torch.isfinite(p).all() for p in params):raise RuntimeError('Final parameters not finite')
    atomic_torch_save(rec.state_dict(),folder/'reconstruction.pth')
    atomic_torch_save(seg.state_dict(),folder/'segmentation.pth')
    result={'condition':condition,'seed':seed,'steps':cfg['steps'],'identity':identity,
        'implementation_sha256':digest(__file__),'initial_first_conv_sha256':initial_hash,
        'component_trace_sha256':trace,'active_anomalies':active,'replaced_anomalies':replaced,
        'loss_points':loss_points,'elapsed_training_seconds':elapsed+time.perf_counter()-segment_start,
        'this_process_load_train_save_seconds':time.perf_counter()-wall_start,'resume_count':resume_count,
        'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'process_ram':process_ram(),'minimum_available_ram_sampled_every100_steps_GiB':minimum_ram,
        'loss_finite_checked_every_step':True,'gradient_finite_checked_every100_steps':True,'final_weights_finite':True,
        'optimizer':'Adam1e-4 weight_decay0','final_lr':optimizer.param_groups[0]['lr'],
        'checkpoint_rule':'fixed final10000, no held-out selection','held_out_scoring':False,
        'weight_files':{n:digest(folder/n) for n in ['reconstruction.pth','segmentation.pth']}}
    save(summary,result)
    print(json.dumps({'completed':True,'condition':condition,'steps':cfg['steps'],'elapsed_seconds':result['elapsed_training_seconds']}),flush=True)
