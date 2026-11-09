"""Explicit future CPU acquisition / read-only GPU scoring implementation.

No heavy import or execution occurs on importing this module. All entry points
are called only by inference.py's explicitly selected runtime CLI mode.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import time


def input_binding(c):
    c.verify_preparation()
    return {'protocol_sha256':c.digest(c.NORMAL_PREP/'protocol.json'),
            'preparation_complete_sha256':c.digest(c.NORMAL_PREP/'complete.json')}


def image_rows(cfg):
    return [(role,row) for role in ('Q200','F100') for row in cfg['sources'][role]]


def save_array(path, value, np):
    with path.open('xb') as stream:
        np.save(stream,value,allow_pickle=False)


def input_names(cfg):
    return {'started.json','sources.json','records.json','verification.json'} | {
        f'images/{row["id"]}.npy' for _,row in image_rows(cfg)}


def collision_report(records, cfg):
    byhash=collections.defaultdict(list)
    old=set(cfg['sources']['fit']['old_training_working_hashes'])
    fit=set(cfg['sources']['fit']['fit_working_hashes'])
    collisions=[]
    for row in records:
        h=row['working_RGB_sha256'];byhash[h].append(row['id'])
        if h in old or h in fit:
            collisions.append({'id':row['id'],'working_RGB_sha256':h,
                               'vs_old_training':h in old,'vs_D_fit':h in fit})
    collisions.extend({'within_Q_F':ids,'working_RGB_sha256':h}
                      for h,ids in byhash.items() if len(ids)>1)
    return {'Q200':sum(r['role']=='Q200' for r in records),
            'F100':sum(r['role']=='F100' for r in records),
            'unique_working_RGB':len(byhash),'collisions':collisions,
            'D_hold_working_collisions_not_checked':True}


def acquire_inputs(c, model=None):
    signature=input_binding(c);cfg=c.read(c.NORMAL_PREP/'protocol.json');c.resources()
    if c.NORMAL_INPUTS.exists():
        return verify_inputs(c)
    c.require(not c.ACCESS and not any(c.COUNTERS.values()),'Acquisition needs a fresh process')
    import numpy as np
    from PIL import Image
    from . import inputs
    root=Path(cfg['sources']['dataset_root']).resolve()
    ordered=image_rows(cfg)
    c.require(len(ordered)==300 and len({r['path'] for _,r in ordered})==300,'Fixed normal source denominator differs')
    c.ORIGINALS.update((root/r['path']).resolve() for _,r in ordered)
    c.NORMAL_INPUTS.mkdir(exist_ok=False);(c.NORMAL_INPUTS/'images').mkdir(exist_ok=False)
    records=[];started=time.perf_counter();samples=[]
    try:
        c.write(c.NORMAL_INPUTS/'started.json',{'identity':signature,'scope':'CPU Q200 then F100 only'})
        c.write(c.NORMAL_INPUTS/'sources.json',signature)
        for role,frozen in ordered:
            samples.append(c.resources())
            p=root/frozen['path']
            c.require(p.resolve().is_relative_to(root) and p.is_file()
                      and not p.is_symlink() and not p.is_junction(), 'Missing/redirected Q/F original')
            payload=p.read_bytes()
            c.require(len(payload)==frozen['bytes']
                      and hashlib.sha256(payload).hexdigest()==frozen['file_sha256'],'Q/F original bytes differ')
            with Image.open(io.BytesIO(payload)) as im:
                c.COUNTERS['source_decodes']+=1
                c.require(list(reversed(im.size))==frozen['original_size_hw']==[1024,1024]
                          and im.mode==frozen['original_mode'],'Q/F original domain differs')
                rgb=im.convert('RGB');raw=np.ascontiguousarray(np.asarray(rgb,np.uint8))
                raw_hash=hashlib.sha256(raw.tobytes(order='C')).hexdigest()
                work=np.asarray(rgb.resize((256,256),Image.Resampling.BILINEAR),np.float32)/np.float32(255)
            h=inputs.working_rgb_identity(work)
            relative=f'images/{frozen["id"]}.npy';save_array(c.NORMAL_INPUTS/relative,work,np)
            records.append({'id':frozen['id'],'ordinal':frozen['ordinal'],'role':role,
                'path':frozen['path'],'file_sha256':frozen['file_sha256'],
                'legacy_rgb_group_id':frozen['rgb_sha256'], 'decoded_RGB_u8_sha256':raw_hash,
                'decoded_RGB_u8_domain':'RGB1024x1024x3,uint8,HWC,C bytes,no header',
                'working_RGB_sha256':h,'file':relative,'decode_count':1})
        collisions=collision_report(records,cfg)
        c.write(c.NORMAL_INPUTS/'records.json',{'rows':records,'collision_check':collisions})
        c.require(not collisions['collisions'],'Canonical Q/F collision; no replacement or scoring')
        expected={str((root/r['path']).resolve()):1 for _,r in ordered}
        c.require(dict(c.ACCESS)==expected and c.COUNTERS['source_decodes']==300,
                  'Actual normal decode/access denominator differs')
        c.write(c.NORMAL_INPUTS/'verification.json',{'passed':True,'identity':signature,
            'actual_original_opens':dict(c.ACCESS),'boundary':c.boundary(),
            'collision_check':collisions,'resources':samples,
            'elapsed_seconds':time.perf_counter()-started,'no_model_forward':True})
        c.require(input_binding(c)==signature,'Preparation changed during acquisition')
        c.seal(c.NORMAL_INPUTS,input_names(cfg),('images',))
    except BaseException as exc:
        if not (c.NORMAL_INPUTS/'failure.json').exists():
            c.write(c.NORMAL_INPUTS/'failure.json',{'error':type(exc).__name__,'message':str(exc),
                'boundary':c.boundary(),'completed_source_records':records,
                'actual_original_opens':dict(c.ACCESS),'no_automatic_retry':True})
        raise
    return {'status':'NORMAL_INPUTS_COMPLETE','output':str(c.NORMAL_INPUTS),
            'collision_check':collisions,'boundary':c.boundary()}


def verify_inputs(c, model=None):
    signature=input_binding(c);cfg=c.read(c.NORMAL_PREP/'protocol.json');c.resources()
    c.verify_closed(c.NORMAL_INPUTS,input_names(cfg),('images',))
    c.require(c.read(c.NORMAL_INPUTS/'sources.json')==signature
              and c.read(c.NORMAL_INPUTS/'started.json')['identity']==signature,'Normal input source identity differs')
    import numpy as np
    from . import inputs
    record_doc=c.read(c.NORMAL_INPUTS/'records.json');rows=record_doc['rows'];ordered=image_rows(cfg)
    c.require(len(rows)==len(ordered)==300,'Normal input denominator differs')
    for (role,frozen),row in zip(ordered,rows):
        c.require(row['id']==frozen['id'] and row['ordinal']==frozen['ordinal'] and row['role']==role
                  and row['path']==frozen['path'] and row['file_sha256']==frozen['file_sha256']
                  and row['legacy_rgb_group_id']==frozen['rgb_sha256'] and row['decode_count']==1
                  and row['file']==f'images/{frozen["id"]}.npy','Normal source order/identity differs')
        c.require(len(row['decoded_RGB_u8_sha256'])==64
                  and all(ch in '0123456789abcdef' for ch in row['decoded_RGB_u8_sha256'])
                  and row['decoded_RGB_u8_domain']=='RGB1024x1024x3,uint8,HWC,C bytes,no header',
                  'Saved raw RGB domain/digest differs')
        rgb=np.load(c.NORMAL_INPUTS/row['file'],allow_pickle=False)
        c.require(inputs.working_rgb_identity(rgb)==row['working_RGB_sha256'],'Saved normal work input differs')
    collision=collision_report(rows,cfg)
    verified=c.read(c.NORMAL_INPUTS/'verification.json')
    c.require(not collision['collisions'] and collision==record_doc['collision_check']==verified['collision_check']
              and verified['passed'] and verified['identity']==signature,'Normal input coverage/verification differs')
    expected_opens={str((Path(cfg['sources']['dataset_root'])/r['path']).resolve()):1 for _,r in ordered}
    b=verified['boundary']
    c.require(verified['actual_original_opens']==expected_opens
              and b['source_decodes']==b['original_file_opens']==b['original_unique_paths']==300
              and b['model_input_forwards']==b['author_placeholder_forwards']==b['tensor_loads']==0
              and not b['model_packages_loaded'] and not b['blocked_attempts'], 'Saved acquisition boundary differs')
    c.require(c.COUNTERS['source_decodes']==c.COUNTERS['render_calls']==0 and not c.ACCESS,
              'Cache verification reopened originals or generated inputs')
    return {'status':'NORMAL_INPUTS_CACHE_VERIFIED','normal_complete_sha256':c.digest(c.NORMAL_INPUTS/'complete.json'),
            'identity':signature,'collision_check':collision,'boundary':c.boundary()}


def parameters_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_signature(torch, modules):
    signatures={}
    for name,model in modules.items():
        h=hashlib.sha256()
        for key,value in model.state_dict().items():
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        cbuf=[(n,str(v.dtype),list(v.shape)) for n,v in model.named_buffers()]
        signatures[name]={'parameters_sha256':parameters_hash(model),'state_dict_sha256':h.hexdigest(),
                          'buffers':cbuf,'training_modules':[n for n,m in model.named_modules() if m.training],
                          'gradients_present':any(p.grad is not None for p in model.parameters())}
    return signatures


def verify_stateless(c, torch, model, expected):
    bn=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    c.require(len(bn)==expected and all(not m.track_running_stats and m.running_mean is None
              and m.running_var is None and m.num_batches_tracked is None for m in bn)
              and not list(model.named_buffers()),'G1 stateless BN or empty-buffer contract differs')


def build_model(c, name, cfg):
    # The only caller is the explicitly selected score-normal mode.
    import numpy as np
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4*2**30,
              'Need4GiB free CUDA before model creation; no CPU fallback')
    torch.cuda.reset_peak_memory_stats()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c.COUNTERS['model_loads']+=1
    if name=='PatchCore':
        import faiss
        from torchvision.models import wide_resnet50_2
        src=c.ROOT/'vendor/patchcore-inspection/src'
        sys.path.insert(0,str(src))
        import patchcore.common
        import patchcore.patchcore
        import patchcore.sampler
        class UnicodeNN(patchcore.common.FaissNN):
            def load(self,filename):
                self.search_index=self._index_to_gpu(faiss.deserialize_index(
                    np.frombuffer(Path(filename).read_bytes(),np.uint8)))
        backbone=wide_resnet50_2(weights=None)
        c.COUNTERS['tensor_loads']+=1
        backbone.load_state_dict(torch.load(c.ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth',
                                            map_location='cpu',weights_only=True),strict=True)
        backbone.name='wideresnet50';device=torch.device('cuda')
        model=patchcore.patchcore.PatchCore(device)
        def placeholder(_module,_inputs):
            c.COUNTERS['author_placeholder_forwards']+=1
        hook=backbone.register_forward_pre_hook(placeholder)
        try:
            model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],device=device,
                input_shape=(3,256,256),pretrain_embed_dimension=1024,target_embed_dimension=1024,
                patchsize=3,anomaly_score_num_nn=1,
                featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.01,device),
                nn_method=UnicodeNN(on_gpu=False,num_workers=4))
        finally:
            hook.remove()
        bank=c.ROOT/'models/weak_localization/patchcore'
        c.require({p.name for p in bank.iterdir()}=={'nnscorer_search_index.faiss','patchcore_params.pkl'},
                  'Unexpected bank side material; do not load optional detection features')
        model.anomaly_scorer.load(str(bank))
        c.require(model.anomaly_scorer.nn_method.search_index.ntotal==819
                  and model.anomaly_scorer.nn_method.search_index.d==1024,'Loaded FAISS bank contract differs')
        model.eval();model.forward_modules.eval()
        c.require(all(p.dtype==torch.float32 for p in model.parameters()),'PatchCore parameters are not FP32')
        c.require(c.COUNTERS['author_placeholder_forwards']==1,'Author initialization probe count differs')
        return torch,{'PatchCore':model},model
    spec=importlib.util.spec_from_file_location('mura_response_g1_author',c.ROOT/'vendor/DRAEM/model_unet.py')
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    rec=author.ReconstructiveSubNetwork(3,3).cuda();seg=author.DiscriminativeSubNetwork(6,2).cuda()
    for key,model,n in [('reconstruction',rec,22),('segmentation',seg,27)]:
        for m in model.modules():
            if isinstance(m,torch.nn.BatchNorm2d):
                m.track_running_stats=False;m.running_mean=None;m.running_var=None;m.num_batches_tracked=None
        c.COUNTERS['tensor_loads']+=1
        model.load_state_dict(torch.load(c.ROOT/f'models/controlled_generation/formal_draem/G1_20261030/{key}.pth',
                                        map_location='cpu',weights_only=True),strict=True)
        model.eval();verify_stateless(c,torch,model,n)
        c.require(all(p.dtype==torch.float32 for p in model.parameters()),'G1 parameters are not FP32')
        c.require(parameters_hash(model)==cfg['sources']['G1_final_parameter_hashes'][key],
                  'Loaded G1 final parameters differ')
    return torch,{'reconstruction':rec,'segmentation':seg},(rec,seg)


def predict_rgb(torch, name, handle, rgb):
    """Consumes only RGB, no IDs, roles, masks, GT or thresholds."""
    import numpy as np
    x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0)
    with torch.inference_mode():
        if name=='PatchCore':
            mean=torch.tensor([.485,.456,.406])[None,:,None,None]
            std=torch.tensor([.229,.224,.225])[None,:,None,None]
            scores,maps=handle.predict((x-mean)/std)
            return float(scores[0]),np.asarray(maps[0],dtype=np.float32)
        rec,seg=handle;x=x.cuda()
        probability=torch.softmax(seg(torch.cat((rec(x),x),dim=1)),dim=1)[0,1]
        heat=probability.cpu().numpy()
        return float(heat.max()),heat


def scoring_binding(c, name, normal_verified):
    return {'model':name,'protocol_sha256':c.digest(c.NORMAL_PREP/'protocol.json'),
            'normal_inputs_complete_sha256':normal_verified['normal_complete_sha256']}


def score_names(records):
    return {'started.json','sources.json','q_result.json','result.json','verification.json'} | {
        f'probabilities/{r["id"]}.npy' for r in records} | {f'records/{r["id"]}.json' for r in records}


def counts(scores, tau):
    fp=sum(s>tau for s in scores)
    return {'FP':fp,'TN':len(scores)-fp,'normal_denominator':len(scores),'FPR':fp/len(scores)}


def q_timing(started, finished):
    elapsed=finished-started
    if elapsed < 0:
        raise ValueError('Nonmonotonic Q timing')
    return {'Q200_scoring_and_cache_seconds':elapsed,
            'Q200_seconds_per_input_reference':elapsed/200,
            'Q200_input_references':200,
            'scope':'after model load through last Q cache write; includes per-image state checks and IO; excludes quantile'}


def score_normal(c, name):
    import numpy as np
    from . import inputs
    normal_verified=verify_inputs(c);cfg=c.read(c.NORMAL_PREP/'protocol.json')
    records=c.read(c.NORMAL_INPUTS/'records.json')['rows'];signature=scoring_binding(c,name,normal_verified)
    folder=c.SCORES/name
    if folder.exists():
        return verify_normal(c,name)
    # Resource admission and all CPU identities precede a new model transaction.
    start_resources=c.resources(gpu=True)
    c.require(not c.COUNTERS['model_input_forwards'] and not c.COUNTERS['model_loads'],
              'Each model needs its own fresh process')
    c.SCORES.mkdir(exist_ok=True);folder.mkdir(exist_ok=False)
    (folder/'probabilities').mkdir(exist_ok=False);(folder/'records').mkdir(exist_ok=False)
    started=time.perf_counter();actual=[];samples=[start_resources]
    try:
        c.write(folder/'started.json',{'identity':signature,'scope':'one model Q200 then F100; no training'})
        c.write(folder/'sources.json',signature)
        # Mark loading itself as an owned transaction: constructor may contain the author probe.
        c.write(folder/'inflight.json',{'identity':signature,'operation':'model_construction'})
        torch,modules,handle=build_model(c,name,cfg)
        initial=state_signature(torch,modules)
        c.require(all(not s['training_modules'] and not s['gradients_present'] for s in initial.values()),
                  'Loaded model is not eval/read-only')
        (folder/'inflight.json').unlink()
        q_started=time.perf_counter()
        for i,row in enumerate(records):
            samples.append(c.resources())
            if i==200:
                # Freeze once, before F100; never recompute on cache reuse.
                scores=[r['score'] for r in actual]
                tau=float(np.quantile(np.asarray(scores,np.float64),.95,method='linear'))
                c.COUNTERS['real_calibrations']+=1
                q_result={'identity':signature,'threshold':tau,'q':.95,'method':'linear',
                    'rule':'strictly greater','Q_counts':counts(scores,tau),'in_sample_not_validation':True,
                    'timing':q_timing(q_started,q_finished),
                    'constant_scores':len(set(scores))==1,
                    'threshold_at_score_extreme':tau in (min(scores),max(scores)),
                    'G1_threshold_at_probability_endpoint':name=='G1' and tau in (0.,1.)}
                c.write(folder/'q_result.json',q_result)
            rgb=np.load(c.NORMAL_INPUTS/row['file'],allow_pickle=False)
            c.require(inputs.working_rgb_identity(rgb)==row['working_RGB_sha256'],'Scoring input changed')
            c.write(folder/'inflight.json',{'identity':signature,'operation':'input_forward',
                                         'ordinal':i,'id':row['id']})
            c.COUNTERS['model_input_forwards']+=1
            score,heat=predict_rgb(torch,name,handle,rgb)
            torch.cuda.synchronize()
            c.require(heat.dtype==np.float32 and heat.shape==(256,256) and np.isfinite(heat).all()
                      and np.isfinite(score),'Invalid native model output')
            if name=='G1':
                c.require(float(heat.min())>=0 and float(heat.max())<=1 and score==float(heat.max()),
                          'G1 raw probability/max contract differs')
                verify_stateless(c,torch,modules['reconstruction'],22)
                verify_stateless(c,torch,modules['segmentation'],27)
            c.require(state_signature(torch,modules)==initial,'Parameters/buffers/training/grad state changed')
            target=folder/f'probabilities/{row["id"]}.npy';save_array(target,heat,np)
            output={'identity':signature,'id':row['id'],'ordinal':i,'role':row['role'],
                    'working_RGB_sha256':row['working_RGB_sha256'],'score':score,
                    'probability_file':f'probabilities/{row["id"]}.npy',
                    'probability_sha256':c.digest(target),'state_unchanged':True}
            c.write(folder/f'records/{row["id"]}.json',output)
            actual.append(output);(folder/'inflight.json').unlink()
            if i==199:
                q_finished=time.perf_counter()
            print(json.dumps({'model':name,'role':row['role'],'image':i+1,'of':300}),flush=True)
        q=c.read(folder/'q_result.json');tau=q['threshold'];f_scores=[r['score'] for r in actual[200:]]
        fcounts=counts(f_scores,tau)
        c.write(folder/'result.json',{'identity':signature,'Q':q,'F100':fcounts,
            'F100_Wilson95':c.wilson(fcounts['FP'],100),'F_scores_did_not_change_tau':True,
            'independent_normal_image_count':300,'model_input_forwards':300,
            'physical_groups_known':False,'no_detector_training':True})
        c.require(c.COUNTERS['model_input_forwards']==300 and c.COUNTERS['real_calibrations']==1
                  and state_signature(torch,modules)==initial,'Final model/threshold/count state differs')
        c.require(input_binding(c)==normal_verified['identity'],'Bound implementation changed before sealing')
        c.write(folder/'verification.json',{'passed':True,'identity':signature,'state_before':initial,
            'state_after':state_signature(torch,modules),'boundary':c.boundary(),'resources':samples,
            'elapsed_seconds':time.perf_counter()-started,
            'timing_scope':'owned model transaction before construction through all scoring/state checks; excludes prior CPU input verification and final file-hash sealing',
            'GPU_peak_scope':cfg['GPU_peak_scope'],
            'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20})
        c.seal(folder,score_names(records),('probabilities','records'))
    except BaseException as exc:
        if not (folder/'failure.json').exists():
            c.write(folder/'failure.json',{'error':type(exc).__name__,'message':str(exc),
                'boundary':c.boundary(),'completed_image_count':len(actual),
                'resources':samples,'no_automatic_retry_resume':True})
        raise
    return {'status':'NORMAL_MODEL_SCORING_COMPLETE','model':name,'output':str(folder),
            'Q':q['Q_counts'],'F100':fcounts,'threshold':tau,'boundary':c.boundary()}


def verify_normal(c, name):
    import numpy as np
    normal=verify_inputs(c);cfg=c.read(c.NORMAL_PREP/'protocol.json')
    records=c.read(c.NORMAL_INPUTS/'records.json')['rows'];folder=c.SCORES/name
    signature=scoring_binding(c,name,normal)
    c.verify_closed(folder,score_names(records),('probabilities','records'))
    c.require(c.read(folder/'sources.json')==signature
              and c.read(folder/'started.json')['identity']==signature,'Normal scoring identity differs')
    q=c.read(folder/'q_result.json');result=c.read(folder/'result.json');v=c.read(folder/'verification.json')
    c.require(q['identity']==signature and q['q']==.95 and q['method']=='linear'
              and q['rule']=='strictly greater' and result['identity']==signature
              and result['Q']==q and v['passed'] and v['identity']==signature,'Saved normal result identity differs')
    tau=q['threshold'];all_scores=[]
    c.require(np.isfinite(tau),'Nonfinite saved threshold')
    for i,row in enumerate(records):
        record=c.read(folder/f'records/{row["id"]}.json')
        expected=f'probabilities/{row["id"]}.npy'
        c.require(record['identity']==signature and record['id']==row['id'] and record['ordinal']==i
                  and record['role']==row['role'] and record['working_RGB_sha256']==row['working_RGB_sha256']
                  and record['probability_file']==expected and record['state_unchanged']
                  and c.digest(folder/expected)==record['probability_sha256'],'Saved score row identity differs')
        heat=np.load(folder/expected,allow_pickle=False);score=record['score']
        c.require(heat.dtype==np.float32 and heat.shape==(256,256) and np.isfinite(heat).all()
                  and np.isfinite(score),'Saved native output shape/finiteness differs')
        if name=='G1':
            c.require(float(heat.min())>=0 and float(heat.max())<=1 and score==float(heat.max()),
                      'Saved G1 probability/max differs')
        all_scores.append(score)
    qcounts=counts(all_scores[:200],tau);fcounts=counts(all_scores[200:],tau)
    c.require(qcounts==q['Q_counts'] and fcounts==result['F100']
              and result['F100_Wilson95']==c.wilson(fcounts['FP'],100)
              and result['F_scores_did_not_change_tau']
              and result['independent_normal_image_count']==300 and result['model_input_forwards']==300,
              'Saved frozen-threshold denominators/counts differ')
    b=v['boundary']
    c.require(v['state_before']==v['state_after'] and b['model_input_forwards']==300
              and b['real_calibrations']==1 and b['source_decodes']==b['original_file_opens']==0
              and b['render_calls']==b['parameter_updates']==b['statistical_fits']==0
              and not b['blocked_attempts'],'Saved model/state operation evidence differs')
    c.require(c.COUNTERS['source_decodes']==c.COUNTERS['model_input_forwards']
              ==c.COUNTERS['real_calibrations']==0 and not c.ACCESS,'Read-only cache reuse performed new work')
    return {'status':'NORMAL_MODEL_CACHE_VERIFIED','model':name,'threshold':tau,
            'Q':qcounts,'F100':fcounts,'F100_Wilson95':result['F100_Wilson95'],
            'tau_recomputed':False,'boundary':c.boundary()}
