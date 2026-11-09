"""Stage4: CPU metrics from frozen float caches. Never imports Torch or runs models."""
import copy
import gc
import importlib.metadata
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ROOT, read_json, save_json, resources
from .holdout_preparation import FOLDER
from .holdout_inference import (CONFIG as SCORE_CONFIG, CACHE as SCORE_CACHE,
                                checked_config, verify_condition)
from .holdout_data import MANIFEST
from .calibration import exclusive_file
from .probe_lr_decay_preparation import sha_file
from .probe_lr_decay_training import available_ram

CONFIG=FOLDER/'metrics_config.json'
RESULT=FOLDER/'metrics.json'
STARTED=FOLDER/'metrics_started.json'
OWNER=FOLDER/'metrics.owner.lock'


def require_CPU():
    if 'torch' in sys.modules:raise RuntimeError('Stage4 requires a fresh CPU process without Torch')


def validate_metric_namespace(entries,owned=False):
    allowed={CONFIG.name,RESULT.name,STARTED.name,OWNER.name,'metrics_verification.json'}
    for name,regular_file in entries:
        if name not in allowed or not regular_file:raise RuntimeError('Unknown/partial metrics artifact; preserve and diagnose')
        if name==OWNER.name and not owned:raise RuntimeError('Metrics owned/interrupted; preserve')


def check_metric_namespace(owned=False):
    validate_metric_namespace([(p.name,p.is_file() and not p.is_symlink()) for p in FOLDER.glob('metrics*')],owned)


def inputs_and_config():
    require_CPU();cfg=checked_config()
    scores={c:verify_condition(cfg,c) for c in cfg['conditions']}
    proof=read_json(FOLDER/'scoring_verification.json')
    if (not proof['scoring_complete'] or proof['scoring_config_sha256']!=sha_file(SCORE_CONFIG)
            or proof['manifest_sha256']!=sha_file(MANIFEST)):
        raise ValueError('Stage3 completion identity differs')
    for c in cfg['conditions']:
        if proof['scores_sha256'][c]!=sha_file(SCORE_CACHE/c/'scores.json'):raise ValueError('Frozen score result differs')
    if any(a['input']!=b['input'] or a['input_array_sha256']!=b['input_array_sha256']
           for a,b in zip(scores['G0']['rows'],scores['G1']['rows'])):raise ValueError('Models did not use identical RGB inputs')
    paths=[Path(__file__),ROOT/'controlled_generation/holdout_inference.py',ROOT/'controlled_generation/holdout_data.py']
    proposed={'schema':'shared-holdout-metrics-v1','device':'CPU','scoring_config_sha256':sha_file(SCORE_CONFIG),
        'manifest_sha256':sha_file(MANIFEST),'scoring_verification_sha256':sha_file(FOLDER/'scoring_verification.json'),
        'scores_sha256':{c:sha_file(SCORE_CACHE/c/'scores.json') for c in cfg['conditions']},
        'implementation_sha256':{str(p.relative_to(ROOT)):sha_file(p) for p in paths},
        'packages':{key:importlib.metadata.version(key) for key in ('numpy','scipy','scikit-learn')},
        'thresholds':{c:cfg['thresholds'][c]['threshold'] for c in cfg['conditions']},
        'alarm':'score strictly greater than per-model frozen threshold',
        'strength_order':['a04','a02'],'type_order':['mura_like','line_like'],'polarity_order':['bright','dark'],
        'template_order':['M_circle','M_ellipse','M_blob','L_edge','L_diagonal','L_polyline'],
        'pixel_AP':'sklearn average_precision_score; all pixels of selected anomalies plus same20 normals; no sampling or image averaging',
        'image_AUROC':'sklearn roc_auc_score of original raw maximum scores',
        'labels':'saved float M_eff, normal_GT all zero; no PNG labels or pixel threshold on predictions',
        'empty_groups':'null/NA with reason; never substitute0 or1',
        'normal_reuse':'same20 normal maps and scores in every group; no extra independent normals',
        'exact_set_reuse':'same model and identical ordered anomaly IDs reuse exact metric, including paired set if identical',
        'minimum_available_RAM_GiB':3.,'sort_estimate_bytes_per_pixel':112,'extra_available_RAM_reserve_GiB':1.,
        'minimum_free_disk_GiB':8.,'independence':'single training seed and at most20 backgrounds; descriptive, no CI or significance tests',
        'bias':'G1 renderer domain; shared appearance/labels/geometries favor G1; not real-screen evidence'}
    return cfg,scores,read_json(MANIFEST),proposed


def strict_detection(scores,threshold):
    values=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid image scores')
    return values>threshold


def requested_groups(cfg):
    anomalies=[x for x in cfg['inputs'] if x['input_class']=='anomaly']
    return {s:[x for x in anomalies if x['strength_id']==s] for s in ('a04','a02')}


def image_metrics(normals,anomalies,threshold,normal_paths):
    nd=strict_detection([r['score'] for r in normals],threshold)
    ad=strict_detection([r['score'] for r in anomalies],threshold)
    n,a=len(normals),len(anomalies)
    FP,TP=int(nd.sum()),int(ad.sum())
    y=np.r_[np.zeros(n,dtype=np.uint8),np.ones(a,dtype=np.uint8)]
    values=[r['score'] for r in normals+anomalies]
    return {'normal_images':n,'valid_anomaly_images':a,'TP':TP,'FN':a-TP,'FP':FP,'TN':n-FP,
        'recall':TP/a if a else None,'normal_false_positive_rate':FP/n if n else None,
        'image_AUROC':float(roc_auc_score(y,values)) if n and a else None,
        'image_AUROC_NA_reason':None if n and a else 'both normal and anomaly classes required',
        'recall_NA_reason':None if a else 'no valid anomalies',
        'FP_inputs':[{'id':r['input']['id'],'background_path':normal_paths[r['input']['id']]} for r,flag in zip(normals,nd) if flag],
        'FN_ids':[r['input']['id'] for r,flag in zip(anomalies,ad) if not flag]}


class Evaluator:
    def __init__(self,cfg,scores,manifest):
        self.cfg=cfg;self.scores=scores;self.manifest=manifest;self.reuse={};self.resources=[];self.AP_calls=0
        self.normal_items=cfg['inputs'][:20]
        self.normal_paths={r['request']['id']:r['request']['background_path'] for r in manifest['normal']}
        self.rows={c:{r['input']['id']:r for r in scores[c]['rows']} for c in ('G0','G1')}

    def labels(self,item):
        with np.load(FOLDER/item['relative_npz'],allow_pickle=False) as archive:
            value=archive['normal_GT' if item['input_class']=='normal' else 'M_eff']
        if value.shape!=(256,256) or value.dtype!=bool:raise ValueError('Expected saved bool float-label mask')
        if item['input_class']=='normal' and value.any():raise ValueError('Normal GT is not zero')
        return value

    def evaluate(self,condition,items):
        key=(condition,tuple(x['id'] for x in items))
        if key in self.reuse:return copy.deepcopy(self.reuse[key])
        normals=[self.rows[condition][x['id']] for x in self.normal_items]
        anomalies=[self.rows[condition][x['id']] for x in items]
        result=image_metrics(normals,anomalies,self.cfg['thresholds'][condition]['threshold'],self.normal_paths)
        count=(20+len(items))*65536;before=available_ram()
        required=max(3.,count*112/2**30+1.)
        if before<required:raise RuntimeError(f'Exact AP needs{required:.3f}GiB available by fixed estimate; only{before:.3f}; stop, no pixel sampling')
        if shutil.disk_usage(FOLDER).free/2**30<8:raise RuntimeError('Need8GiB free disk before exact AP')
        target=np.empty(count,dtype=bool);probability=np.empty(count,dtype=np.float32)
        try:
            for k,item in enumerate(self.normal_items+items):
                row=self.rows[condition][item['id']]
                path=FOLDER/row['probability_file']
                if sha_file(path)!=row['probability_sha256']:raise ValueError('Probability changed during metrics')
                pred=np.load(path,mmap_mode='r',allow_pickle=False)
                if pred.shape!=(256,256) or pred.dtype!=np.float32 or not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:
                    raise ValueError('Malformed raw probability map')
                segment=slice(k*65536,(k+1)*65536)
                probability[segment]=pred.reshape(-1);target[segment]=self.labels(item).reshape(-1)
                del pred
            positives=int(np.count_nonzero(target));tick=time.perf_counter()
            ap=None
            if items and positives:
                ap=float(average_precision_score(target,probability));self.AP_calls+=1
            result.update(pixel_AP=ap,pixel_AP_NA_reason=None if ap is not None else 'no valid anomalies or positive pixels',
                          total_pixels=count,positive_pixels=positives,negative_pixels=count-positives,
                          anomaly_ids=[x['id'] for x in items],normal_ids=[x['id'] for x in self.normal_items])
            self.resources.append({'model':condition,'anomalies':len(items),'pixels':count,'positive_pixels':positives,
                'available_RAM_before_GiB':before,'required_available_RAM_GiB':required,'available_RAM_after_AP_GiB':available_ram(),
                'AP_sort_seconds':time.perf_counter()-tick})
        finally:
            del target,probability;gc.collect()
        self.reuse[key]=copy.deepcopy(result)
        print('HOLDOUT_EXACT_METRIC',condition,len(items),'anomalies +20normal;',count,'pixels',flush=True)
        return result

    def counts(self,condition,items,planned):
        rows=[self.rows[condition][x['id']] for x in items]
        detected=strict_detection([r['score'] for r in rows],self.cfg['thresholds'][condition]['threshold'])
        return {'planned':planned,'valid':len(items),'invalid':planned-len(items),'TP':int(detected.sum()),
                'FN':int(len(items)-detected.sum()),'recall':float(detected.mean()) if len(items) else None,
                'recall_NA_reason':None if len(items) else 'no valid anomalies'}


def paired_sets(manifest,groups):
    planned=defaultdict(dict)
    for row in manifest['anomaly']:planned[row['pair_id']][row['strength_id']]=row
    both=[];counts={'both_valid':0,'only_a04_valid':0,'only_a02_valid':0,'both_invalid':0}
    for pair_id,pair in planned.items():
        a=pair['a04']['quality']['valid'];b=pair['a02']['quality']['valid']
        counts['both_valid' if a and b else 'only_a04_valid' if a else 'only_a02_valid' if b else 'both_invalid']+=1
        if a and b:both.append(pair_id)
    selected={s:[x for x in groups[s] if x['pair_id'] in set(both)] for s in ('a04','a02')}
    return {'planned_pairs':len(planned),**counts,'pair_ids':both},selected


def compare(a,b):
    return {'pixel_AP_G1_minus_G0_pp':100*(b['pixel_AP']-a['pixel_AP']) if a['pixel_AP'] is not None and b['pixel_AP'] is not None else None,
        'image_AUROC_G1_minus_G0_pp':100*(b['image_AUROC']-a['image_AUROC']) if a['image_AUROC'] is not None and b['image_AUROC'] is not None else None,
        **{key+'_G1_minus_G0':b[key]-a[key] for key in ('TP','FN','FP','TN')}}


def calculate(cfg,scores,manifest):
    groups=requested_groups(cfg);pair_summary,paired=paired_sets(manifest,groups);e=Evaluator(cfg,scores,manifest)
    output={'generation':manifest['summary'],'shared_pair_summary':pair_summary,'models':{},'differences':{}}
    for c in ('G0','G1'):
        strengths={};threshold=cfg['thresholds'][c]['threshold']
        for strength in ('a04','a02'):
            items=groups[strength]
            overall=e.evaluate(c,items);types={kind:e.evaluate(c,[x for x in items if x['kind']==kind]) for kind in ('mura_like','line_like')}
            brightness={kind:{direction:e.counts(c,[x for x in items if x['kind']==kind and x['polarity']==direction],60)
                for direction in ('bright','dark')} for kind in ('mura_like','line_like')}
            templates={name:{direction:e.counts(c,[x for x in items if x['template_id']==name and x['polarity']==direction],20)
                for direction in ('bright','dark')} for name in ('M_circle','M_ellipse','M_blob','L_edge','L_diagonal','L_polyline')}
            first_valid={kind:{direction:next((x['id'] for x in sorted(items,key=lambda x:x['id']) if x['kind']==kind and x['polarity']==direction),None)
                for direction in ('bright','dark')} for kind in ('mura_like','line_like')}
            strengths[strength]={'overall':overall,'types':types,'brightness':brightness,'templates':templates,
                'predeclared_example_ids':{'first_valid':first_valid,
                    'first_normal_FP':min((x['id'] for x in overall['FP_inputs']),default=None),
                    'first_anomaly_FN':min(overall['FN_ids'],default=None)}}
        strong={x['pair_id']:e.rows[c][x['id']] for x in paired['a04']};weak={x['pair_id']:e.rows[c][x['id']] for x in paired['a02']}
        joint={'both_detected':0,'only_a04_detected':0,'only_a02_detected':0,'both_missed':0};deltas=[]
        for pair_id in pair_summary['pair_ids']:
            a,b=strong[pair_id]['score']>threshold,weak[pair_id]['score']>threshold
            joint['both_detected' if a and b else 'only_a04_detected' if a else 'only_a02_detected' if b else 'both_missed']+=1
            deltas.append(weak[pair_id]['score']-strong[pair_id]['score'])
        distribution={'count':len(deltas),'min':float(np.min(deltas)),'median':float(np.median(deltas)),'max':float(np.max(deltas)),
            'mean':float(np.mean(deltas)),'p05':float(np.quantile(deltas,.05,method='linear')),'p95':float(np.quantile(deltas,.95,method='linear'))} if deltas else {'count':0,'NA_reason':'no common valid pairs'}
        output['models'][c]={'threshold':threshold,'strengths':strengths,
            'paired':{'strength_metrics':{s:e.evaluate(c,paired[s]) for s in ('a04','a02')},'joint_detections':joint,
                      'weak_minus_strong_raw_max_score':distribution}}
    for s in ('a04','a02'):
        a,b=output['models']['G0']['strengths'][s],output['models']['G1']['strengths'][s]
        output['differences'][s]={'overall':compare(a['overall'],b['overall']),
            'types':{kind:compare(a['types'][kind],b['types'][kind]) for kind in ('mura_like','line_like')}}
    for c in ('G0','G1'):
        for s in ('a04','a02'):
            q=output['models'][c]['strengths'][s]['overall']
            if (q['normal_images'],q['valid_anomaly_images'],q['total_pixels'],q['positive_pixels'])!=(20,230,16384000,219208):
                raise ValueError('Actual frozen main denominators differ')
    output['execution']={'exact_AP_calls':e.AP_calls,'unique_model_and_anomaly_sets':len(e.reuse),
                         'group_resources':e.resources,'model_forwards':0,'Torch_imported':'torch' in sys.modules}
    return output


def verify_metrics():
    require_CPU();check_metric_namespace();cfg,scores,manifest,expected=inputs_and_config()
    if OWNER.exists():raise RuntimeError('Metrics owned/interrupted; preserve')
    if any(FOLDER.glob('metrics*.tmp')):raise RuntimeError('Partial metric transaction; preserve')
    if read_json(CONFIG)!=expected:raise ValueError('Metric implementation/configuration differs')
    identity={'metrics_config_sha256':sha_file(CONFIG)}
    result=read_json(RESULT);started=read_json(STARTED)
    if started['identity']!=identity or result['identity']!=identity or not result['complete']:
        raise ValueError('Metric result incomplete/mismatched')
    if started.get('completed_metrics_sha256')!=sha_file(RESULT):raise ValueError('Metric output not bound to complete transaction')
    if result['execution']['Torch_imported'] or result['execution']['model_forwards']!=0:raise ValueError('Outside CPU metric scope')
    print('HOLDOUT_METRICS_CACHE_VERIFIED; no metric recomputation or model',flush=True)
    return result


def run_metrics():
    require_CPU();check_metric_namespace()
    if OWNER.exists():raise RuntimeError('Metrics owned/interrupted; preserve')
    cfg,scores,manifest,expected=inputs_and_config()
    if RESULT.exists():return verify_metrics()
    with exclusive_file(OWNER,{'operation':'shared_holdout_CPU_metrics'}):
        check_metric_namespace(owned=True)
        if STARTED.exists() or any(FOLDER.glob('metrics*.tmp')):raise RuntimeError('Incomplete metric transaction; preserve and diagnose')
        if CONFIG.exists() and read_json(CONFIG)!=expected:raise ValueError('Metric config differs; no overwrite')
        initial=resources();initial['disk_free_GiB']=shutil.disk_usage(FOLDER).free/2**30
        if initial['available_RAM_GiB']<3 or initial['disk_free_GiB']<8:raise RuntimeError('Need3GiB free RAM and8GiB disk; metrics not started')
        if not CONFIG.exists():save_json(CONFIG,expected)
        identity={'metrics_config_sha256':sha_file(CONFIG)}
        save_json(STARTED,{'identity':identity,'note':'Incomplete CPU metric transaction is preserved, never overwritten'})
        tick=time.perf_counter();result=calculate(cfg,scores,manifest)
        result.update(identity=identity,complete=True,scope=expected['bias'],independent_real_validation='pending; not read',
                      resources_start=initial,elapsed_metrics_seconds=time.perf_counter()-tick)
        require_CPU();save_json(RESULT,result)
        save_json(STARTED,{'identity':identity,'completed_metrics_sha256':sha_file(RESULT)})
        print('HOLDOUT_METRICS_COMPLETE; exact all-pixel CPU evaluation',flush=True)
    return result
