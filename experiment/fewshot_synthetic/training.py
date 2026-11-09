"""Explicit human-started training only. All Torch imports are inside run()."""
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
from . import common as c
from .sampling import load_window, verify_pairs


def gpu_free_gib():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    values=[float(line.strip())/1024 for line in result.stdout.splitlines() if line.strip()]
    c.require(len(values)==1,'Expected one GPU; no automatic device selection')
    return values[0]


def folders(condition,seed):
    label=f'probe_{seed}' if condition=='probe' else f'{condition}_{seed}'
    return c.MODELS/label,c.RESULTS/label


def checkpoint_inventory(folder, interval, steps, own_active=False):
    allowed={'run.json'}
    if own_active: allowed.add('active.json')
    completed=[]
    for meta in sorted(folder.glob('checkpoint_*.json')):
        row=c.read_json(meta); step=row['step']
        c.require(step % interval==0 and 1<=step<=steps,'Invalid checkpoint step')
        name=f'checkpoint_{step:05d}.pt'
        c.require(meta.name==f'checkpoint_{step:05d}.json' and c.sha(folder/name)==row['sha256'],'Checkpoint identity differs')
        allowed.update((meta.name,name)); completed.append(step)
    c.require(completed==list(range(interval,(max(completed) if completed else 0)+1,interval)), 'Missing checkpoint interval')
    c.require({p.name for p in folder.iterdir()}==allowed,'Unknown/partial checkpoint files; preserve and investigate')
    return completed


def train(condition,seed):
    state={}
    try:
        return _train(condition,seed,state)
    except KeyboardInterrupt:
        if state:
            # Release only an unambiguous own transaction with a full saved point.
            try:
                folder=state['folder']; probe=condition=='probe'
                saved=checkpoint_inventory(folder,200 if probe else 1000,1000 if probe else 10000,True)
                if saved and c.read_json(folder/'active.json')['token']==state['token']:
                    (folder/'active.json').unlink()
            except Exception:
                pass
        raise
    except BaseException as error:
        if state:
            failure=state['folder']/'failure.json'
            if not failure.exists():
                c.save(failure,{'error_type':type(error).__name__,'message':str(error),
                                'last_completed_update':state.get('last_completed_update',0),
                                'minimum_observed_RAM_GiB':state.get('minimum_RAM_GiB'),
                                'token':state['token'],'rule':'stop; diagnosis required; no automatic resume'})
            # Retain active marker as well; never clear an uncertain/error state.
        raise


def _train(condition,seed,state):
    """Calling this function authorizes updates; CLI names explicitly say train."""
    c.require(condition in ('probe','A','B'),'Unknown training condition')
    probe=condition=='probe'
    c.require(seed==(c.PROBE_SEED if probe else seed) and (probe or seed in c.SEEDS),'Seed differs from protocol')
    split,identity=c.source_metadata()
    c.guard(split,cpu=False)
    verify_pairs(split,identity)
    cfg=c.read_json(c.PREP/'common_config.json')
    model_dir,result_dir=folders(condition,seed)
    if (result_dir/'complete.json').exists():
        return verify_run(condition,seed,split,identity)
    if result_dir.exists():
        raise RuntimeError('Partial prediction/result transaction; preserve, do not repeat unclear forward')
    if not probe:
        from .metrics import verify_probe_metrics
        verify_probe_metrics(split,identity)
        order=[(x,s) for s in c.SEEDS for x in ('A','B')]
        for prior_cond,prior_seed in order[:order.index((condition,seed))]:
            verify_run(prior_cond,prior_seed,split,identity)
    resources=c.resource(3,80)
    free_gpu=gpu_free_gib(); c.require(free_gpu>=6,'Need6GiB free VRAM before model initialization')
    if model_dir.exists():
        c.require(not (model_dir/'active.json').exists() and not (model_dir/'failure.json').exists(),
                  'Active/stale/error training transaction; preserve and diagnose, no updates')
        checkpoint_inventory(model_dir,200 if probe else 1000,1000 if probe else 10000)
    else:
        model_dir.mkdir(parents=True)
    token=uuid.uuid4().hex
    c.save(model_dir/'active.json',{'token':token,'pid':os.getpid(),'condition':condition,'seed':seed,
                                  'preparation':c.sha(c.PREP/'complete.json'),'implementation':c.identities()})
    state.update(folder=model_dir,token=token,last_completed_update=0,minimum_RAM_GiB=resources['available_RAM_GiB'])
    # This point is reached only by an explicitly manual training command.
    from .model import make_model,state_hash,loss_value
    import torch
    c.require(torch.cuda.is_available(),'CUDA required; no automatic CPU fallback')
    model=make_model(seed)
    initial_hash=state_hash(model)
    model=model.cuda().float()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    steps,interval=(1000,200) if probe else (10000,1000)
    run_identity={'input_identity':identity,'preparation':c.sha(c.PREP/'complete.json'),'config':c.sha(c.PREP/'common_config.json'),
                  'implementation':c.identities(),'environment':c.environment(),'condition':condition,'seed':seed,'initial_hash':initial_hash}
    if (model_dir/'run.json').exists():
        c.require(c.read_json(model_dir/'run.json')['identity']==run_identity,'Existing run identity differs')
        completed=checkpoint_inventory(model_dir,interval,steps,True)
        c.require(bool(completed),'No complete own checkpoint; ambiguous start preserved')
    else:
        c.save(model_dir/'run.json',{'identity':run_identity,'resources':resources,'free_VRAM_GiB':free_gpu})
        completed=[]
    cursor=0; history=[]; resource_samples=[]; minimum=resources['available_RAM_GiB']; accumulated=0.; resume_count=0
    if completed:
        cp=torch.load(model_dir/f'checkpoint_{completed[-1]:05d}.pt',map_location='cpu',weights_only=False)
        c.require(cp['identity']==run_identity and cp['cursor']==completed[-1], 'Own checkpoint identity/cursor differs')
        c.require(cp['LR_history']==[c.lr_at(i,probe) for i in range(1,cp['cursor']+1)],'Own checkpoint LR prefix differs')
        model.load_state_dict(cp['model'],strict=True); optimizer.load_state_dict(cp['optimizer'])
        random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng']); torch.set_rng_state(cp['torch_rng'])
        torch.cuda.set_rng_state_all(cp['cuda_rng'])
        cursor=cp['cursor']; history=cp['LR_history']; resource_samples=cp['resource_samples']; minimum=cp['minimum_RAM_GiB']
        accumulated=cp['elapsed_seconds']; resume_count=cp['resume_count']+1
        del cp
    state.update(last_completed_update=cursor,minimum_RAM_GiB=minimum)
    if not probe and condition=='B':
        a_result=c.read_json(folders('A',seed)[1]/'result.json')
        c.require(a_result['initial_hash']==initial_hash,'Paired initial parameter identity differs')
    real_lookup={r['id']:r for role in ('R_real5','N_train1885') for r in split[role]}
    inventory=c.read_json(c.INPUT/'export_inventory.json')
    sequence=c.read_json(c.PREP/'probe.json') if probe else c.read_json(c.PREP/str(seed)/'sequence.json')
    def input_at(index):
        if probe:
            epoch,offset=divmod(index,12)
            gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026101912,seed,epoch])))
            item=sequence[int(gen.permutation(12)[offset])]
            path=c.PREP/item['file']
            with np.load(path,allow_pickle=False) as pack:
                return tuple(np.array(pack[n]) for n in ('image','target','valid'))
        item=sequence[index]
        if condition=='B' and item['role']=='intervention':
            path=c.PREP/str(seed)/'synthetic'/f"{item['synthetic_ordinal']:04d}.npz"
            with np.load(path,allow_pickle=False) as pack:
                return tuple(np.array(pack[n]) for n in ('image','target','valid'))
        return load_window(real_lookup[item['id']],item['y'],item['flip_x'],item['flip_y'],item['role']=='real',inventory)
    started=time.monotonic(); model.train()
    try:
        for update in range(cursor+1,steps+1):
            rgb,target,valid=input_at(update-1)
            x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).cuda()
            y=torch.from_numpy(target.astype(np.int64)).unsqueeze(0).cuda()
            v=torch.from_numpy(valid.astype(bool)).unsqueeze(0).cuda()
            lr=c.lr_at(update,probe)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss,ce,dice=loss_value(logits,y,v)
            c.require(bool(torch.isfinite(loss)),'Nonfinite loss; stop without changing protocol')
            loss.backward()
            c.require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'Nonfinite gradient')
            optimizer.step(); history.append(lr)
            state['last_completed_update']=update
            del x,y,v,logits,loss
            if update%100==0:
                ram=c.d.available_ram(); minimum=min(minimum,ram)
                state['minimum_RAM_GiB']=minimum
                resource_samples.append({'step':update,'RAM_GiB':ram})
                print({'condition':condition,'step':update,'of':steps,'CE':float(ce.detach()),'DiceLoss':float(dice.detach()),
                       'lr':lr,'elapsed_seconds':accumulated+time.monotonic()-started,'RAM_GiB':ram},flush=True)
                c.require(ram>=3,'RAM sample below3GiB: stop; do not silently change resources')
            if update%interval==0:
                payload={'identity':run_identity,'cursor':update,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                         'LR_history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                         'resource_samples':resource_samples,'minimum_RAM_GiB':minimum,
                         'elapsed_seconds':accumulated+time.monotonic()-started,'resume_count':resume_count}
                name=f'checkpoint_{update:05d}.pt'; temporary=model_dir/(name+'.tmp')
                c.require(not temporary.exists() and not (model_dir/name).exists(),'Checkpoint save collision')
                torch.save(payload,temporary); os.replace(temporary,model_dir/name)
                c.save(model_dir/f'checkpoint_{update:05d}.json',{'step':update,'sha256':c.sha(model_dir/name),'identity':run_identity})
                del payload
    except BaseException:
        # Keep completed own checkpoints. No opportunistic save or automatic retry.
        raise
    final_hash=state_hash(model)
    result_dir.mkdir(parents=True)
    c.save(result_dir/'prediction_transaction.json',{'identity':run_identity,'final_hash':final_hash,'purpose':'probe final12 only' if probe else 'no Q/E prediction'})
    if probe:
        (result_dir/'probabilities').mkdir()
        model.eval()
        with torch.inference_mode():
            for i,item in enumerate(sequence):
                with np.load(c.PREP/item['file'],allow_pickle=False) as pack:
                    rgb=pack['image']
                x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).cuda()
                probability=torch.softmax(model(x),dim=1)[0,1].cpu().numpy().astype(np.float32)
                np.save(result_dir/'probabilities'/f'{i:02d}.npy',probability,allow_pickle=False)
        c.require(state_hash(model)==final_hash and len(list(model.buffers()))==0,'Final inference changed model state')
    summary={'condition':condition,'seed':seed,'step':steps,'input_identity':identity,'initial_hash':initial_hash,'final_hash':final_hash,
             'implementation':c.identities(),'preparation':run_identity['preparation'],'all_finite':True,'min_available_RAM_GiB':minimum,
             'resource_samples':resource_samples,'elapsed_update_seconds':accumulated+time.monotonic()-started,
             'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
             'resume_count':resume_count,'LR_history':history,'checkpoint_steps':list(range(interval,steps+1,interval)),
             'exposures':{'real':5000,'common_normal':2500,'intervention':2500} if not probe else {'fixed_pool':12},
             'normal_intervention':2500 if condition=='A' else 0,'synthetic_intervention':2500 if condition=='B' else 0,
             'gate':'CPU probe-metrics must be run separately; not an independent score' if probe else 'formal final weight; no Q/E'}
    c.save(result_dir/'result.json',summary)
    c.require(c.read_json(model_dir/'active.json')['token']==token,'Own run ownership changed; preserve and diagnose')
    (model_dir/'active.json').unlink()
    c.seal(result_dir,{'identity':run_identity,'model_files':{p.name:c.sha(p) for p in sorted(model_dir.iterdir())}})
    print({'finished':condition,'step':steps,'next':'inspect saved result; no automatic next training or scoring'},flush=True)
    return summary


def verify_run(condition,seed,split=None,identity=None):
    if split is None:
        split,identity=c.source_metadata(); c.guard(split,cpu=True)
    model_dir,result_dir=folders(condition,seed)
    complete=c.verify_manifest(result_dir)
    run_identity=complete['identity']
    c.require(run_identity['input_identity']==identity and run_identity['implementation']==c.identities()
              and run_identity['preparation']==c.sha(c.PREP/'complete.json') and run_identity['environment']==c.environment(),'Run identity differs')
    interval,steps=(200,1000) if condition=='probe' else (1000,10000)
    c.require(checkpoint_inventory(model_dir,interval,steps)==list(range(interval,steps+1,interval)),'Not all final checkpoints present')
    c.require({p.name:c.sha(p) for p in model_dir.iterdir()}==complete['model_files'],'Saved model file identity differs')
    result=c.read_json(result_dir/'result.json')
    c.require(result['step']==steps and result['condition']==condition and result['seed']==seed
              and result['LR_history']==[c.lr_at(i,condition=='probe') for i in range(1,steps+1)]
              and result['all_finite'] is True,'Final run summary differs')
    return {'verified':True,'condition':condition,'seed':seed,'step':steps,'new_forward_or_update':0,'result':result}


def status(condition,seed):
    model_dir,result_dir=folders(condition,seed)
    if (result_dir/'complete.json').exists(): return verify_run(condition,seed)
    if result_dir.exists(): return {'status':'partial final prediction transaction; stop and investigate'}
    if not model_dir.exists(): return {'status':'not started','condition':condition,'seed':seed}
    if (model_dir/'active.json').exists() or (model_dir/'failure.json').exists():
        return {'status':'active or failed transaction; no resume before diagnosis'}
    return {'status':'own checkpoints present; manual resume only; not yet completion',
            'checkpoint_steps':checkpoint_inventory(model_dir,200 if condition=='probe' else 1000,1000 if condition=='probe' else 10000)}
