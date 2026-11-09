"""Independent CPU metrics protocol for the final saved retry; no model calls."""
import os
from pathlib import Path
import sys

import numpy as np
from . import common as c
from . import probe_retry as r
from . import metrics as frozen

CONTROL=c.RESULTS/'probe_retry_metrics_preparation_v1'
OUTPUT=c.RESULTS/'probe_retry_metrics_v1'
RUN=r.NEW_RESULTS/f'probe_{r.SEED}'
WEIGHTS=r.NEW_MODELS/f'probe_{r.SEED}'
N,P=704256,26414
GROUPS={'real':(5,294400,24482),'synthetic':(3,175616,1932),'normal':(4,234240,0)}


def guard():
    c.require(not any(n.split('.')[0] in ('torch','torchvision','diffusers','transformers') for n in sys.modules),'CPU metrics requires fresh non-model process')
    def audit(event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','torchvision','diffusers','transformers'):
            raise RuntimeError('Model package import forbidden')
        if event in ('subprocess.Popen','socket.connect','pickle.find_class'):
            raise RuntimeError('No subprocess/network/pickle in CPU metrics')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            mode,flags=args[1] or '',args[2] or 0
            writing=any(x in str(mode) for x in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            if writing:
                c.require(path.is_relative_to(CONTROL) or path.is_relative_to(OUTPUT),'Write outside independent metric destinations')
    sys.addaudithook(audit)


def input_identity():
    """Full existing retry verifier hashes weights, never deserializes them."""
    # Pass the already verified roles explicitly: the standalone old verifier
    # otherwise installs its CPU-only PREP write restriction permanently.
    split,source_identity=c.source_metadata()
    r.verify_preparation()
    r.verify_binding()
    with r.retry_context():
        checked=r.t.verify_run('probe',r.SEED,split,source_identity)
    meta=checked['result']
    c.require(meta['initial_hash']==c.read_json(r.CONTROL/'protocol.json')['expected_initial_parameter_sha256'],
              'Retry initial parameter identity differs')
    c.require(meta['step']==1000 and meta['checkpoint_steps']==[200,400,600,800,1000], 'Final1000/checkpoints differ')
    items=c.read_json(c.PREP/'probe.json')
    c.require(len(items)==12,'Probe population differs')
    files={}
    for path in (c.PREP/'complete.json',c.PREP/'common_config.json',c.PREP/'probe.json',
                 r.CONTROL/'complete.json',r.CONTROL/'protocol.json',r.NEW_RESULTS/'binding.json',
                 RUN/'complete.json',RUN/'result.json',WEIGHTS/'run.json'):
        files[path.relative_to(c.ROOT).as_posix()]=c.sha(path)
    for step in (200,400,600,800,1000):
        for suffix in ('json','pt'):
            path=WEIGHTS/f'checkpoint_{step:05d}.{suffix}'
            files[path.relative_to(c.ROOT).as_posix()]=c.sha(path)
    for i,item in enumerate(items):
        for path in (c.PREP/item['file'],RUN/'probabilities'/f'{i:02d}.npy'):
            files[path.relative_to(c.ROOT).as_posix()]=c.sha(path)
        c.require(c.sha(c.PREP/item['file'])==item['sha256'],'Probe input differs')
    c.require(sum(x['valid_pixels'] for x in items)==N and sum(x['positive_pixels'] for x in items)==P,'Probe denominators differ')
    for kind,expected in GROUPS.items():
        group=[x for x in items if x['kind']==kind]
        c.require((len(group),sum(x['valid_pixels'] for x in group),sum(x['positive_pixels'] for x in group))==expected,'Group denominators differ')
    identity={'files':files,'frozen_core':c.identities(),'retry_adapter_sha256':r.source_hash(),
              'metric_adapter_sha256':c.sha(__file__),'environment':c.environment(),
              'initial_parameter_hash_recorded':meta['initial_hash'],'final_parameter_hash_recorded':meta['final_hash'],
              'final_checkpoint_sha256':c.sha(WEIGHTS/'checkpoint_01000.pt')}
    return identity,items,meta


def config(identity):
    return {'schema':'probe-retry-cpu-metrics-v1','identity':identity,'steps':1000,'seed':r.SEED,
            'images':12,'valid_pixels':N,'positive_pixels':P,'group_denominators':{k:list(v) for k,v in GROUPS.items()},
            'probability':'saved original float32 anomaly softmax; no smoothing/scaling/forward',
            'validity':'exclude padding; same target and effective mask from frozen12',
            'AP':'exact global AP over all valid pixels; no subsampling or mean per-image AP',
            'algorithm':'frozen metrics.exact_ap; all nonnegative float32 bit values, ties grouped',
            'histogram_bins':frozen.BINS,'chunk':frozen.CHUNK,'histogram_bytes':2*frozen.BINS*4,
            'sklearn_absolute_tolerance':1e-10,'ties':'independent np.unique on normalized float32 bits; compare distinct and tied score values',
            'group_AP':'sklearn exact pooled valid pixels per group; normal0 positives => NA, no FP/recall without threshold',
            'gate':{'AP_min':.8,'minimum_sampled_RAM_GiB':3,'all_losses_gradients_finite':True,'crosschecks_required':True},
            'min_available_RAM_GiB':3,'min_free_disk_GiB':80,'output':str(OUTPUT),
            'cache':'complete same identity verifies only; partial or unknown output preserved, never reaccumulate',
            'meaning':'training-side fitting; no independent real detection/screen evidence; no automatic formal admission'}


def blocks(items):
    """No AP here: only shape/range/label checks and all valid native pixels."""
    for i,item in enumerate(items):
        with np.load(c.PREP/item['file'],allow_pickle=False) as pack:
            c.require(set(pack.files)=={'image','target','valid'},'Unexpected probe NPZ fields')
            target,valid=pack['target'],pack['valid']
        probability=np.load(RUN/'probabilities'/f'{i:02d}.npy',allow_pickle=False)
        c.require(probability.dtype==np.float32 and probability.shape==target.shape==valid.shape==(256,256),'Cached probability/label shape or dtype differs')
        c.require(np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1,'Invalid raw float32 probabilities')
        c.require(np.all((target==0)|(target==1)) and np.all((valid==0)|(valid==1)),'Invalid binary label/validity')
        valid=valid.astype(bool)
        c.require(not np.any(target[~valid]) and int(valid.sum())==item['valid_pixels']
                  and int(target[valid].sum())==item['positive_pixels'],'Fixed per-image valid population differs')
        yield probability[valid],target[valid]


def prepare():
    guard()
    identity,items,meta=input_identity()
    if (CONTROL/'complete.json').exists(): return verify_preparation(identity)
    c.require(not CONTROL.exists() and not OUTPUT.exists(),'Unknown/partial metric preparation/output; preserve and investigate')
    resources=c.resource(3,80)
    counts=[(len(s),int(y.sum())) for s,y in blocks(items)]
    c.require(sum(x[0] for x in counts)==N and sum(x[1] for x in counts)==P,'Actual cached population differs')
    CONTROL.mkdir()
    c.save(CONTROL/'protocol.json',config(identity))
    c.save(CONTROL/'input_validation.json',{'per_image_counts':counts,'images':12,'valid_pixels':N,'positive_pixels':P,
                                           'resources':resources,'Torch_imported':False,'AP_calculated':False,'new_model_operations':0})
    c.seal(CONTROL,{'identity':identity})
    return verify_preparation(identity)


def verify_preparation(identity=None):
    guard()
    if identity is None: identity,_,_=input_identity()
    complete=c.verify_manifest(CONTROL)
    c.require(set(complete['files'])=={'protocol.json','input_validation.json'} and complete['identity']==identity
              and c.read_json(CONTROL/'protocol.json')==config(identity),'Metric preparation identity differs')
    validation=c.read_json(CONTROL/'input_validation.json')
    c.require(validation['images']==12 and validation['valid_pixels']==N and validation['positive_pixels']==P
              and validation['AP_calculated'] is False,'Input validation differs')
    return {'metric_preparation_verified':True,'images':12,'valid_pixels':N,'positive_pixels':P,
            'AP_calculated':False,'new_model_operations':0,'metric_output_exists':OUTPUT.exists()}


def validate_summary(summary,exact,meta,identity):
    c.require(summary['identity']==identity and summary['step']==1000 and summary['images']==12,'Summary identity differs')
    c.require(exact['pixels']==summary['pixels']==N and exact['positive_pixels']==summary['positive_pixels']==P,'Metric denominators differ')
    ap=exact['AP']; reference=summary['sklearn_AP']
    c.require(isinstance(ap,(int,float)) and np.isfinite(ap) and 0<=ap<=1 and np.isfinite(reference)
              and summary['AP']==ap and abs(reference-ap)<=1e-10
              and summary['absolute_difference']==abs(reference-ap),'AP crosscheck differs')
    c.require(summary['independent_distinct_scores']==exact['distinct_scores']
              and summary['independent_tied_score_values']==exact['tied_score_values'],'Actual ties crosscheck differs')
    c.require(set(summary['groups'])==set(GROUPS),'Missing group')
    for kind,expected in GROUPS.items():
        g=summary['groups'][kind]
        c.require((g['images'],g['pixels'],g['positive_pixels'])==expected,'Group denominator differs')
        c.require(g['AP'] is None if expected[2]==0 else isinstance(g['AP'],(int,float)) and np.isfinite(g['AP']) and 0<=g['AP']<=1,'Group AP/NA invalid')
    gate=bool(ap>=.8 and meta['min_available_RAM_GiB']>=3 and meta['all_finite'] is True)
    c.require(type(summary['gate_passed']) is bool and summary['gate_passed']==gate,'Gate inconsistent with final evidence')


def compute():
    guard()
    identity,items,meta=input_identity(); verify_preparation(identity)
    if (OUTPUT/'complete.json').exists(): return verify_results(identity,meta)
    c.require(not OUTPUT.exists(),'Partial/unknown AP transaction; preserve, do not recalculate or reaccumulate')
    resources=c.resource(3,80)
    from sklearn.metrics import average_precision_score
    OUTPUT.mkdir()
    c.save(OUTPUT/'transaction.json',{'identity':identity,'preparation_sha256':c.sha(CONTROL/'complete.json'),'resources':resources})
    exact=frozen.exact_ap(lambda:blocks(items),OUTPUT/'exact',identity,N,P)
    pairs=list(blocks(items))
    scores=np.concatenate([x[0] for x in pairs]); labels=np.concatenate([x[1] for x in pairs])
    reference=float(average_precision_score(labels,scores))
    normalized=scores.copy(); normalized[normalized==0]=np.float32(0)
    _,counts=np.unique(normalized.view(np.uint32),return_counts=True)
    groups={}
    for kind in GROUPS:
        indices=[i for i,x in enumerate(items) if x['kind']==kind]
        s=np.concatenate([pairs[i][0] for i in indices]); y=np.concatenate([pairs[i][1] for i in indices])
        groups[kind]={'images':len(indices),'pixels':len(y),'positive_pixels':int(y.sum()),
                      'AP':float(average_precision_score(y,s)) if y.any() else None}
    summary={'identity':identity,'step':1000,'images':12,'AP':exact['AP'],'sklearn_AP':reference,
             'absolute_difference':abs(reference-exact['AP']),'pixels':N,'positive_pixels':P,'groups':groups,
             'independent_distinct_scores':len(counts),'independent_tied_score_values':int(np.sum(counts>1)),
             'gate_passed':bool(exact['AP']>=.8 and meta['min_available_RAM_GiB']>=3 and meta['all_finite'] is True),
             'meaning':'training-side fitting only; real/synthetic subsets are trained inputs, not real held-out evidence'}
    validate_summary(summary,exact,meta,identity)
    c.save(OUTPUT/'metrics.json',summary)
    c.seal(OUTPUT,{'identity':identity,'preparation_sha256':c.sha(CONTROL/'complete.json')})
    return summary


def verify_results(identity=None,meta=None):
    guard()
    if identity is None: identity,_,meta=input_identity()
    verify_preparation(identity)
    complete=c.verify_manifest(OUTPUT)
    expected={'transaction.json','metrics.json','exact/transaction.json','exact/counts.uint32',
              'exact/positive.uint32','exact/metrics.json','exact/complete.json'}
    c.require(set(complete['files'])==expected and complete['identity']==identity
              and complete['preparation_sha256']==c.sha(CONTROL/'complete.json'),'Complete metric cache identity/namespace differs')
    inner=c.verify_manifest(OUTPUT/'exact')
    c.require(inner['identity']==identity,'Exact AP identity differs')
    summary=c.read_json(OUTPUT/'metrics.json'); exact=c.read_json(OUTPUT/'exact'/'metrics.json')
    validate_summary(summary,exact,meta,identity)
    return summary
