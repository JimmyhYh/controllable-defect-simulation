"""Explicit future model operations. Importing this module creates no model/tensor."""
from __future__ import annotations
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time
from . import training as t


class ResourceStop(RuntimeError):
    pass


def setup_runtime(folder,bound):
    import tempfile
    folder=t.c.no_redirect(folder)
    if folder.exists():
        t.c.require(t.load(folder/'owner.json')==bound,"Unknown/foreign runtime temporary directory")
        t.c.tree(folder)  # Reject redirected descendants; temporary caches are not scientific results.
    else:
        folder.mkdir(parents=True,exist_ok=False); t.save(folder/'owner.json',bound)
    os.environ.update({'TEMP':str(folder),'TMP':str(folder),'TORCH_HOME':str(folder/'torch'),'HF_HOME':str(folder/'hf'),
                       'XDG_CACHE_HOME':str(folder/'xdg'),'TORCHINDUCTOR_CACHE_DIR':str(folder/'inductor'),
                       'TRITON_CACHE_DIR':str(folder/'triton'),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    tempfile.tempdir=str(folder)


def forward_binding():
    return {'preparation_complete_sha256':t.sha(t.PREP/'complete.json'),'implementation':t.implementation(),
            'input_complete_sha256':t.INPUT_SHA,'seed':t.c.SEEDS[0],'compatibility_sha256':t.sha(t.INPUT/'compatibility.json'),'inputs':25}


def verify_forward():
    names={'binding.json','transaction.json','result.json'}|{f'probability_{i:02d}.npy' for i in range(25)}
    t.verify_seal(t.FORWARD,names,forward_binding())
    r=t.load(t.FORWARD/'result.json')
    expected=t.load(t.INPUT/'compatibility.json')['inputs']
    t.c.require(r['binding']==forward_binding() and r['forward_calls']==25 and r['gate_passed'] and len(r['items'])==25,"Incomplete compatibility gate")
    t.c.require(r['initial_state_hash']==r['final_state_hash'] and r['initial_buffers_hash']==r['final_buffers_hash'],"Compatibility state/buffers changed")
    t.c.require([x['input'] for x in r['items']]==expected and all(x['finite'] and x['shape']==[256,256] and 0<=x['minimum']<=x['maximum']<=1 for x in r['items']),"Compatibility output metadata differs")
    return r


def check_forward():
    protocol=t.runtime_preflight()
    if t.FORWARD.exists(): return verify_forward()
    snapshot=t.resources(model=True,full_round=True); bound=forward_binding(); runtime=t.c.LARGE/'runtime_v1/forward'
    t.c.require(not runtime.exists(),"Unknown/partial compatibility runtime transaction")
    t.guard(cpu=False,destinations=(t.FORWARD,runtime),temporary=runtime)
    t.FORWARD.mkdir(parents=True,exist_ok=False)
    t.save(t.FORWARD/'binding.json',bound)
    t.save(t.FORWARD/'transaction.json',{'resources':snapshot,'mode':'explicit eval/inference FP32; each of25 once; no optimizer'})
    try:
        setup_runtime(runtime,bound)
        import numpy as np
        import torch
        from pretrained_segmentation.model import build_model,state_hash,buffers_hash
        from .training_data import TrainingData
        data=TrainingData(); started=time.monotonic()
        model,initial=build_model(t.c.SEEDS[0],{'compatibility':protocol['model_compatibility']}); model.to('cuda'); model.eval()
        before,buffers=state_hash(model.state_dict()),buffers_hash(model); items=[]
        t.c.require(buffers==initial['initial_buffers_hash'],"Compatibility initial BN changed")
        with torch.inference_mode():
            for i in range(25):
                current=t.c.resource_snapshot()
                t.c.require(current['available_RAM_GiB']>=2 and current['E_free_GiB']>=1,"Compatibility resource boundary stop; preserve")
                pack,meta=data.compatibility(i)
                logits=model(torch.from_numpy(pack['image']).unsqueeze(0).to('cuda'))
                t.c.require(logits.shape==(1,2,256,256) and bool(torch.isfinite(logits).all()),"Invalid compatibility logit")
                p=torch.softmax(logits,dim=1)[0,1].detach().cpu().numpy().astype(np.float32,copy=False)
                t.c.require(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and pack['valid'].any(),"Invalid compatibility probability/domain")
                with (t.FORWARD/f'probability_{i:02d}.npy').open('xb') as f:
                    np.save(f,p,allow_pickle=False); f.flush(); os.fsync(f.fileno())
                items.append({'index':i,'input':meta,'shape':list(p.shape),'finite':True,'minimum':float(p.min()),'maximum':float(p.max()),'valid_max':float(p[pack['valid']].max())})
                del logits,pack,p
        after,after_buffers=state_hash(model.state_dict()),buffers_hash(model)
        t.c.require(before==after and buffers==after_buffers,"Read-only compatibility changed parameters/BN")
        result={'binding':bound,'forward_calls':25,'items':items,'gate_passed':True,'initialization':initial,
                'initial_state_hash':before,'final_state_hash':after,'initial_buffers_hash':buffers,'final_buffers_hash':after_buffers,
                'elapsed_seconds':time.monotonic()-started,'optimizer_created':False,'training_updates':0,'AP_or_calibration':False,'resources':snapshot}
        t.save(t.FORWARD/'result.json',result)
        names={'binding.json','transaction.json','result.json'}|{f'probability_{i:02d}.npy' for i in range(25)}
        t.seal(t.FORWARD,names,bound)
        return verify_forward()
    except Exception as exc:
        if not (t.FORWARD/'failure.json').exists(): t.save(t.FORWARD/'failure.json',{'error':repr(exc),'rule':'preserve unknown/partial forwards; no automatic retry'})
        raise


def exposure(history):
    by_kind=Counter(); by_id=Counter(); pixels={k:{'updates':0,'valid_pixels':0,'positive_pixels':0} for k in ['real','normal','synthetic']}
    for entry in history:
        kind=entry['kind']; by_kind[kind]+=1
        if kind=='real': by_id[entry['source']['id']]+=1
        for key,value in [('updates',1),('valid_pixels',entry['valid_pixels']),('positive_pixels',entry['positive_pixels'])]: pixels[kind][key]+=value
    return {'kinds':dict(by_kind),'R5':dict(by_id),'pixels':pixels}


def verify_history(history,bound):
    rows=t.load(t.INPUT/'requests.json'); native={r['id']:r for r in t.load(t.INPUT/'native_inputs.json')}
    materials=t.load(t.INPUT/'synthetic_verification.json')['items']
    group=bound['group']
    for i,entry in enumerate(history):
        row=rows[i]; ref=row.get('common_input',row.get(group+'_input'))
        if ref['kind']=='synthetic': n,p=materials[ref['ordinal']]['valid_pixels'],materials[ref['ordinal']]['effective_pixels']
        else:
            item=native[ref['id']]; n=256*item['width']; p=next(w['positive_pixels'] for w in item['windows'] if w['y']==ref['y'])
        t.c.require(entry['update']==i+1 and entry['request_sha256']==t.stable_hash(row) and entry['input_identity']==row['input_identity_'+group]
                    and entry['source']==ref and entry['kind']==ref['kind'] and entry['valid_pixels']==n and entry['positive_pixels']==p,"Actual request/input/pixel exposure prefix differs")
        t.c.require(entry['lr']==t.lr_at(i+1) and entry['finite'] is True and all(math.isfinite(entry[k]) for k in ['loss','CE','Dice']),"LR/loss prefix differs")


def verify_checkpoint(folder,bound):
    t.verify_seal(folder,t.CHECK_FILES,bound)
    row,history=t.load(Path(folder)/'checkpoint.json'),t.load(Path(folder)/'history.json')
    done=row['step']
    t.c.require(type(done) is int and 1<=done<=10000 and len(history)==done and row['binding']==bound,"Checkpoint prefix/cursor differs")
    verify_history(history,bound)
    t.c.require(row['next_lr']==t.next_lr(done) and row['buffers_hash']==row['initial_buffers_hash'] and row['exposures']==exposure(history),"Checkpoint schedule/BN/exposure differs")
    t.c.require(row['optimizer_groups']==[{'name':n,'lr':lr} for n,lr in zip(['encoder','decoder_head'],t.lr_at(done))],"Saved optimizer groups differ")
    return row,history


def model_tree(group,seed,final=False,completed=False):
    run,model,_=t.paths(group,seed); bound=t.binding(group,seed)
    t.c.tree(model)
    expected={'initialization.json'}|({'final_weights.safetensors'} if final else set())|({'complete.json'} if completed else set())
    t.c.require({p.name for p in model.iterdir() if p.is_file()}==expected,"Unknown/partial model-root file")
    dirs={p.name for p in model.iterdir() if p.is_dir()}
    t.c.require(dirs<= {'checkpoints','resource_stops'},"Unknown model-root directory")
    initial=t.load(model/'initialization.json'); checkpoints=[]
    for dirname in dirs:
        members=list((model/dirname).iterdir())
        t.c.require(members and all(p.is_dir() for p in members),"Unknown/empty checkpoint container")
        for folder in members:
            row,_=verify_checkpoint(folder,bound); step=row['step']
            t.c.require(folder.name==f'step_{step:05d}' and step%(1000 if dirname=='checkpoints' else 100)==0
                        and (dirname!='resource_stops' or step%1000!=0),"Checkpoint directory/cursor differs")
            t.c.require(row['initial_buffers_hash']==initial['initial_buffers_hash'],"Checkpoint initial BN differs")
            checkpoints.append((step,folder))
    return checkpoints


def verify_run(group,seed):
    run,model,_=t.paths(group,seed); bound=t.binding(group,seed)
    t.verify_seal(run,{'binding.json','transaction.json','result.json'},bound)
    initial=t.load(model/'initialization.json'); result=t.load(run/'result.json')
    t.c.require(t.load(run/'binding.json')==bound,"Final saved run binding differs")
    model_tree(group,seed,final=True,completed=True)
    required={'initialization.json','final_weights.safetensors'}
    for step in range(1000,10001,1000):
        folder=model/'checkpoints'/f'step_{step:05d}'
        verify_checkpoint(folder,bound)
        required|={f'checkpoints/step_{step:05d}/{n}' for n in t.CHECK_FILES|{'complete.json'}}
    if (model/'resource_stops').exists(): required|={p.relative_to(model).as_posix() for p in (model/'resource_stops').rglob('*') if p.is_file()}
    t.verify_seal(model,required,bound)
    last,history=verify_checkpoint(model/'checkpoints/step_10000',bound)
    t.c.require(result['binding']==bound and result['step']==10000 and result['all_finite'] and result['exposures']==exposure(history),"Final budget/exposure differs")
    expected=t.load(t.INPUT/'summary.json')['exposures']
    t.c.require(result['exposures']['kinds']==expected[group] and result['exposures']['R5']==expected['R5_by_group'][group]
                and result['exposures']['pixels']==expected['pixel_exposure_by_group_role'][group],"Final actual exposure differs from input preparation")
    t.c.require(t.sha(model/'final_weights.safetensors')==t.sha(model/'checkpoints/step_10000/model.safetensors')==result['final_weight_sha256'],"Final weight serialization differs")
    t.c.require(result['initial_buffers_hash']==result['final_buffers_hash']==initial['initial_buffers_hash']==last['buffers_hash']
                and result['final_state_hash']==last['state_hash'],"Final state/BN differs")
    return {'training_complete':True,'group':group,'seed':seed,'step':10000,'checkpoints':list(range(1000,10001,1000)),
            'final_weight_sha256':result['final_weight_sha256'],'exposures':result['exposures'],'GPU_resume_measured':False,'Q_E_calibration_or_metrics':False}


def save_checkpoint(folder,model,optimizer,history,bound,initial,samples,elapsed,peaks):
    import torch
    from safetensors.torch import save_file
    from pretrained_segmentation.model import state_hash,buffers_hash
    from pretrained_segmentation.training import rng_state
    t.c.require(not folder.exists(),"Existing/partial own checkpoint; preserve")
    folder.mkdir(parents=True,exist_ok=False)
    state={n:v.detach().cpu().contiguous().clone() for n,v in model.state_dict().items()}
    save_file(state,str(folder/'model.safetensors')); torch.save(optimizer.state_dict(),folder/'optimizer.pt')
    t.save(folder/'rng.json',rng_state()); t.save(folder/'history.json',history)
    done=len(history)
    row={'binding':bound,'step':done,'state_hash':state_hash(state),'buffers_hash':buffers_hash(model),'initial_buffers_hash':initial['initial_buffers_hash'],
         'next_lr':t.next_lr(done),'optimizer_groups':[{'name':g['name'],'lr':g['lr']} for g in optimizer.param_groups],
         'exposures':exposure(history),'resource_samples':samples,'elapsed_seconds':elapsed,
         'peak_cuda_allocated_MiB':max(peaks[0],torch.cuda.max_memory_allocated()/2**20),'peak_cuda_reserved_MiB':max(peaks[1],torch.cuda.max_memory_reserved()/2**20),
         'GPU_resume_measured':False}
    t.c.require(row['buffers_hash']==initial['initial_buffers_hash'],"Encoder BN buffers changed")
    t.save(folder/'checkpoint.json',row); t.seal(folder,t.CHECK_FILES,bound)


def train(group,seed,resume_from=None):
    protocol=t.runtime_preflight(); forward=verify_forward()
    run,model_path,runtime=t.paths(group,seed); bound=t.binding(group,seed)
    if (run/'complete.json').exists() or (model_path/'complete.json').exists():
        t.c.require(resume_from is None,"Complete result is read-only; no restore")
        return verify_run(group,seed)
    index=t.ORDER.index((group,seed))
    for previous in t.ORDER[:index]: verify_run(*previous)
    snapshot=t.resources(model=True,full_round=index==0 and resume_from is None)
    restore=None
    if resume_from is None:
        t.c.require(not run.exists() and not model_path.exists() and not runtime.exists(),"Unknown/partial run; no automatic start or restore")
    else:
        t.c.require(t.load(run/'binding.json')==bound,"Own restore binding differs")
        t.c.verify_closed_tree(run,{'binding.json','transaction.json'})
        restore=t.c.no_redirect(resume_from)
        candidates=model_tree(group,seed)
        t.c.require(restore.is_relative_to(model_path) and candidates and (max(s for s,_ in candidates),restore) in candidates,"Restore explicit latest own complete point only")
        row,_=verify_checkpoint(restore,bound)
        t.c.require(row['step']<10000,"Final checkpoint with partial finalization; preserve and diagnose")
    t.guard(cpu=False,destinations=(run,model_path,runtime),temporary=runtime)
    if restore is None:
        run.mkdir(parents=True,exist_ok=False); model_path.mkdir(parents=True,exist_ok=False)
        t.save(run/'binding.json',bound); t.save(run/'transaction.json',{'manual_entry':True,'resources':snapshot,'forward_complete_sha256':t.sha(t.FORWARD/'complete.json')})
    started=time.monotonic()
    try:
        setup_runtime(runtime,bound)
        import torch
        from safetensors.torch import load_file,save_file
        from pretrained_segmentation.model import build_model,make_optimizer,loss_value,set_training_mode,state_hash,buffers_hash
        from pretrained_segmentation.training import restore_rng
        from .training_data import TrainingData
        data=TrainingData(); model,initial=build_model(seed,{'compatibility':protocol['model_compatibility']}); model.to('cuda')
        if group=='B':
            a=t.load(t.paths('A',seed)[1]/'initialization.json')
            t.c.require(initial==a,"Paired full initial parameters/BN differ")
        elif seed==t.c.SEEDS[0]:
            t.c.require(initial==forward['initialization'],"Formal initialization differs from compatibility construction")
        else:
            for old in t.c.SEEDS[:t.c.SEEDS.index(seed)]:
                a=t.load(t.paths('A',old)[1]/'initialization.json')
                t.c.require(initial['initial_state_hash']!=a['initial_state_hash'] and initial['encoder_loaded_hash']==a['encoder_loaded_hash'],"Different seed decoder/encoder identity differs")
        optimizer=make_optimizer(model); history=[]; samples=[]; done=0; prior=0.; peaks=(0.,0.)
        if restore is None: t.save(model_path/'initialization.json',initial)
        else:
            row,history=verify_checkpoint(restore,bound)
            t.c.require(t.load(model_path/'initialization.json')==initial,"Fresh own initialization differs")
            model.load_state_dict(load_file(str(restore/'model.safetensors'),device='cuda'),strict=True)
            optimizer.load_state_dict(torch.load(restore/'optimizer.pt',map_location='cpu',weights_only=True))
            t.c.require(state_hash(model.state_dict())==row['state_hash'] and buffers_hash(model)==initial['initial_buffers_hash'],"Restored parameters/BN differ")
            t.c.require([{'name':g['name'],'lr':g['lr']} for g in optimizer.param_groups]==row['optimizer_groups'],"Restored optimizer groups differ")
            t.c.require(all(int(s['step'].item())==row['step'] for s in optimizer.state.values()),"Restored Adam update counter differs")
            restore_rng(t.load(restore/'rng.json')); done=row['step']; samples=row['resource_samples']; prior=row['elapsed_seconds']
            peaks=(row['peak_cuda_allocated_MiB'],row['peak_cuda_reserved_MiB'])
        torch.cuda.reset_peak_memory_stats()
        for update in range(done+1,10001):
            pack,request,ref=data.request(group,update)
            x=torch.from_numpy(pack['image']).unsqueeze(0).to('cuda'); y=torch.from_numpy(pack['target']).unsqueeze(0).to('cuda')
            valid=torch.from_numpy(pack['valid']).unsqueeze(0).to('cuda'); rates=t.lr_at(update)
            for opt,rate in zip(optimizer.param_groups,rates): opt['lr']=rate
            set_training_mode(model); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss,ce,dice=loss_value(logits,y,valid)
            t.c.require(bool(torch.isfinite(loss)),"Nonfinite loss")
            loss.backward()
            t.c.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad),"Missing/nonfinite gradient")
            optimizer.step()
            t.c.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"Nonfinite updated parameters")
            history.append({'update':update,'request_sha256':t.stable_hash(request),'input_identity':request['input_identity_'+group],
                            'source':ref,'kind':ref['kind'],'valid_pixels':int(pack['valid'].sum()),'positive_pixels':int(pack['target'][pack['valid']].sum()),
                            'lr':rates,'loss':float(loss.detach()),'CE':float(ce.detach()),'Dice':float(dice.detach()),'finite':True})
            del x,y,valid,logits,loss,ce,dice,pack
            low=False
            if update%100==0:
                current=t.c.resource_snapshot(); current['step']=update; samples.append(current); low=current['available_RAM_GiB']<2
                print({'group':group,'seed':seed,'step':update,'of':10000,'loss':history[-1]['loss'],'LR':rates,'elapsed_seconds':prior+time.monotonic()-started},flush=True)
            if update%1000==0 or low:
                folder=model_path/('checkpoints' if update%1000==0 else 'resource_stops')/f'step_{update:05d}'
                save_checkpoint(folder,model,optimizer,history,bound,initial,samples,prior+time.monotonic()-started,peaks)
            if low: raise ResourceStop('RAM below2GiB at completed update boundary; full own checkpoint saved, explicit restore only')
        torch.cuda.synchronize(); final={n:v.detach().cpu().contiguous().clone() for n,v in model.state_dict().items()}
        save_file(final,str(model_path/'final_weights.safetensors'))
        result={'binding':bound,'step':10000,'all_finite':True,'exposures':exposure(history),'initial_buffers_hash':initial['initial_buffers_hash'],
                'final_buffers_hash':buffers_hash(model),'final_state_hash':state_hash(final),'final_weight_sha256':t.sha(model_path/'final_weights.safetensors'),
                'elapsed_seconds':prior+time.monotonic()-started,'resumed_from':str(restore) if restore else None,'resource_samples':samples,
                'sampled_min_RAM_GiB':min(s['available_RAM_GiB'] for s in samples),
                'peak_cuda_allocated_MiB':max(peaks[0],torch.cuda.max_memory_allocated()/2**20),'peak_cuda_reserved_MiB':max(peaks[1],torch.cuda.max_memory_reserved()/2**20),
                'Q_E_calibration_or_metrics':False,'GPU_resume_measured':False}
        t.save(run/'result.json',result)
        model_tree(group,seed,final=True)
        names={p.relative_to(model_path).as_posix() for p in model_path.rglob('*') if p.is_file()}
        t.seal(model_path,names,bound); t.seal(run,{'binding.json','transaction.json','result.json'},bound)
        return verify_run(group,seed)
    except (KeyboardInterrupt,ResourceStop):
        print('Stopped; preserve. Ctrl+C makes no extra checkpoint; explicit latest-own restore only.',flush=True)
        raise
    except Exception as exc:
        if not (run/'failure.json').exists(): t.save(run/'failure.json',{'error':repr(exc),'rule':'preserve, no automatic retry/restore/changed settings'})
        raise
