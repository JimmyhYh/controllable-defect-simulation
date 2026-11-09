"""CPU-only exact AP over every float32 bit value, including ties; no Torch."""
from pathlib import Path
import time
import numpy as np
from . import common as c

BINS=0x3f800001
CHUNK=1048576


def exact_ap(block_factory, destination, identity, expected_n, expected_p=None):
    """Factory yields score float32 and binary label chunks, each <= CHUNK.

    Each of the two disk tables has one uint32 per nonnegative float32 score
    from +0 through 1. These are exact distinct scores, not approximate bins.
    An unfinished transaction is never accumulated or restarted implicitly.
    """
    destination=Path(destination)
    c.require('torch' not in __import__('sys').modules,'Exact CPU metric requires isolated process without Torch')
    if (destination/'complete.json').exists():
        complete=c.verify_manifest(destination)
        c.require(complete['identity']==identity,'Metric input identity differs')
        result=c.read_json(destination/'metrics.json')
        c.require(result['pixels']==expected_n and (expected_p is None or result['positive_pixels']==expected_p),
                  'Requested AP denominators differ from complete cache')
        return result
    c.require(not destination.exists(),'Incomplete/unknown AP transaction; preserve and investigate')
    resources=c.resource(3,80)
    destination.mkdir(parents=True)
    c.save(destination/'transaction.json',{'identity':identity,'expected_n':expected_n,'expected_p':expected_p,'resources':resources})
    started=time.monotonic()
    total=np.memmap(destination/'counts.uint32',dtype='<u4',mode='w+',shape=(BINS,))
    positive=np.memmap(destination/'positive.uint32',dtype='<u4',mode='w+',shape=(BINS,))
    n=pcount=0
    for scores,labels in block_factory():
        scores=np.asarray(scores); labels=np.asarray(labels)
        c.require(scores.dtype==np.float32 and scores.shape==labels.shape and scores.size<=CHUNK,'Invalid AP chunk/dtype')
        s=scores.reshape(-1).copy(); y=labels.reshape(-1)
        c.require(np.isfinite(s).all() and np.all((s>=0)&(s<=1)) and np.all((y==0)|(y==1)),'Invalid score/label range')
        s[s==0]=np.float32(0)
        bits=s.view(np.uint32)
        keys,counts=np.unique(bits,return_counts=True)
        c.require(np.all(total[keys].astype(np.uint64)+counts < 2**32),'Histogram count overflow')
        total[keys]+=counts.astype(np.uint32)
        pk,pc=np.unique(bits[y.astype(bool)],return_counts=True)
        c.require(np.all(positive[pk].astype(np.uint64)+pc < 2**32),'Positive count overflow')
        positive[pk]+=pc.astype(np.uint32)
        n+=len(s); pcount+=int(y.sum())
    total.flush(); positive.flush()
    c.require(n==expected_n and (expected_p is None or pcount==expected_p),'AP population count differs')
    tp=np.uint64(0); seen=np.uint64(0); ap=np.float64(0); distinct=ties=0
    for end in range(BINS,0,-CHUNK):
        start=max(0,end-CHUNK)
        t=np.array(total[start:end][::-1],dtype=np.uint64)
        p=np.array(positive[start:end][::-1],dtype=np.uint64)
        nonzero=t>0; t=t[nonzero]; p=p[nonzero]
        if not len(t): continue
        cum_t=np.cumsum(t,dtype=np.uint64)+seen; cum_p=np.cumsum(p,dtype=np.uint64)+tp
        if pcount:
            ap+=np.sum((p.astype(np.float64)/pcount)*(cum_p/cum_t),dtype=np.float64)
        seen=cum_t[-1]; tp=cum_p[-1]; distinct+=len(t); ties+=int(np.sum(t>1))
    c.require(int(seen)==n and int(tp)==pcount,'AP scan totals differ')
    del total,positive
    result={'AP':float(ap) if pcount else None,'pixels':n,'positive_pixels':pcount,'distinct_scores':distinct,
            'tied_score_values':ties,'exact_float32_scores':True,'histogram_bins':BINS,'elapsed_seconds':time.monotonic()-started}
    c.save(destination/'metrics.json',result)
    c.seal(destination,{'identity':identity})
    return result


def probe_metrics():
    """Stage3 CPU entry, only saved final12 probe probabilities; never a model."""
    from .sampling import verify_pairs
    split,identity=c.source_metadata(); c.guard(split,cpu=False)
    verify_pairs(split,identity)
    from .training import verify_run
    verify_run('probe',c.PROBE_SEED,split,identity)
    run=c.RESULTS/f'probe_{c.PROBE_SEED}'
    complete=c.verify_manifest(run)
    meta=c.read_json(run/'result.json')
    c.require(meta['step']==1000 and meta['input_identity']==identity,'Probe final identity differs')
    probe=c.read_json(c.PREP/'probe.json')
    score_paths=[run/'probabilities'/f'{i:02d}.npy' for i in range(12)]
    def values():
        for item,path in zip(probe,score_paths):
            with np.load(c.PREP/item['file'],allow_pickle=False) as pack:
                valid=pack['valid'].astype(bool); y=pack['target'][valid]
            s=np.load(path,allow_pickle=False)
            c.require(s.shape==(256,256),'Probe probability size differs')
            yield s[valid],y
    n=sum(item['valid_pixels'] for item in probe); p=sum(item['positive_pixels'] for item in probe)
    metric_identity={'training_complete':c.sha(run/'complete.json'),'preparation_complete':c.sha(c.PREP/'complete.json'),
                     'implementation':c.identities()}
    destination=c.RESULTS/f'probe_metrics_{c.PROBE_SEED}'
    if (destination/'complete.json').exists():
        return verify_probe_metrics(split,identity)
    c.require(not destination.exists(),'Partial/unknown probe metric transaction; preserve and investigate')
    result=exact_ap(values,destination/'exact',metric_identity,n,p)
    # An independently sorted small-pool reference; not a replacement metric.
    from sklearn.metrics import average_precision_score
    pairs=list(values()); scores=np.concatenate([v[0] for v in pairs]); labels=np.concatenate([v[1] for v in pairs])
    reference=float(average_precision_score(labels,scores))
    c.require(abs(reference-result['AP'])<=1e-10,'Exact AP disagrees with independent sklearn')
    normalized=scores.copy(); normalized[normalized==0]=np.float32(0)
    _,counts=np.unique(normalized.view(np.uint32),return_counts=True)
    c.require(result['distinct_scores']==len(counts) and result['tied_score_values']==int(np.sum(counts>1)),
              'Actual score tie counts disagree with independently sorted small pool')
    groups={}
    for kind in ('real','synthetic','normal'):
        indices=[i for i,item in enumerate(probe) if item['kind']==kind]
        gy=np.concatenate([pairs[i][1] for i in indices]); gs=np.concatenate([pairs[i][0] for i in indices])
        groups[kind]={'images':len(indices),'pixels':int(len(gy)),'positive_pixels':int(gy.sum()),
                      'AP':float(average_precision_score(gy,gs)) if gy.any() else None}
    summary={'step':1000,'AP':result['AP'],'sklearn_AP':reference,'absolute_difference':abs(reference-result['AP']),
             'pixels':n,'positive_pixels':p,'groups':groups,'independent_distinct_scores':len(counts),
             'independent_tied_score_values':int(np.sum(counts>1)),
             'gate_passed':bool(result['AP']>=.8 and meta['min_available_RAM_GiB']>=3 and meta['all_finite']),
             'meaning':'training-side fitting only; no Q/E or real held-out evidence','identity':metric_identity}
    path=destination/'gate.json'
    if path.exists(): c.require(c.read_json(path)==summary,'Existing probe gate differs')
    else: c.save(path,summary)
    c.seal(destination,{'identity':metric_identity})
    return summary


def validate_gate(summary, exact, meta):
    c.require(type(summary.get('gate_passed')) is bool and summary['gate_passed'] is True,'Probe gate not passed')
    c.require(summary['step']==1000 and meta['step']==1000 and meta['all_finite'] is True
              and meta['min_available_RAM_GiB']>=3 and exact['AP'] is not None and exact['AP']>=.8,
              'Probe final fitting/resource criteria not met')
    c.require(summary['AP']==exact['AP'] and summary['pixels']==exact['pixels']
              and summary['positive_pixels']==exact['positive_pixels']
              and abs(summary['sklearn_AP']-exact['AP'])<=1e-10
              and summary['absolute_difference']==abs(summary['sklearn_AP']-exact['AP'])
              and summary['independent_distinct_scores']==exact['distinct_scores']
              and summary['independent_tied_score_values']==exact['tied_score_values'], 'Probe metric evidence differs')


def verify_probe_metrics(split=None, identity=None):
    if split is None:
        split,identity=c.source_metadata(); c.guard(split,cpu=False)
    c.require('torch' not in __import__('sys').modules,'Probe metric admission is CPU-only')
    from .training import verify_run
    verify_run('probe',c.PROBE_SEED,split,identity)
    run=c.RESULTS/f'probe_{c.PROBE_SEED}'
    destination=c.RESULTS/f'probe_metrics_{c.PROBE_SEED}'
    complete=c.verify_manifest(destination)
    expected={'training_complete':c.sha(run/'complete.json'),'preparation_complete':c.sha(c.PREP/'complete.json'),
              'implementation':c.identities()}
    c.require(complete['identity']==expected,'Sealed probe metric identity differs')
    exact_complete=c.verify_manifest(destination/'exact')
    c.require(exact_complete['identity']==expected,'Exact AP identity differs')
    summary=c.read_json(destination/'gate.json'); exact=c.read_json(destination/'exact'/'metrics.json')
    c.require(summary['identity']==expected,'Gate identity differs')
    probe=c.read_json(c.PREP/'probe.json')
    c.require(exact['pixels']==sum(x['valid_pixels'] for x in probe)
              and exact['positive_pixels']==sum(x['positive_pixels'] for x in probe),'Fixed12 denominators differ')
    validate_gate(summary,exact,c.read_json(run/'result.json'))
    return summary
