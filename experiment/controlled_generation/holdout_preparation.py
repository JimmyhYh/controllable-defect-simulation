"""Day9 metadata only: no image bytes, renderer imports, generation, or inference."""
import copy
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .data import ROOT, OUT, SPLIT, load_split, read_json, save_json
from .calibration import (CONFIG as CAL_CONFIG, CACHE as CAL_CACHE,
                          verify_preparation as verify_calibration, exclusive_file)
from .probe_lr_decay_preparation import sha_file

FOLDER=OUT/'shared_holdout_v1'
PROTOCOL=FOLDER/'protocol.json'
REQUESTS=FOLDER/'requests.json'
ROOT_SEED=20261031
THRESHOLDS={'G0':0.6124536782503127,'G1':0.7155899673700332}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def templates():
    cases=[
        ('M_circle','mura_like',{'shape':'circle','center':[64,64],'radius':18}),
        ('M_ellipse','mura_like',{'shape':'ellipse','center':[176,148],'axes':[32,16],'angle_deg':35}),
        ('M_blob','mura_like',{'shape':'anostyler_blob','center':[128,128],'size':[56,40]}),
        ('L_edge','line_like',{'shape':'line','center':[128,24],'length':120,'width':3,'angle_deg':0}),
        ('L_diagonal','line_like',{'shape':'line','center':[132,140],'length':100,'width':5,'angle_deg':60}),
        ('L_polyline','line_like',{'shape':'polyline','points':[[48,176],[112,152],[176,184]],'width':5})]
    return [{'template_index':t,'template_id':name,'spec':{'kind':kind,'units':'px256',**spec}}
            for t,(name,kind,spec) in enumerate(cases)]


def static_geometry(spec):
    """Continuous bounds/segment scalars only; never rasterizes or calls render."""
    shape=spec['shape']
    if spec['units']!='px256':raise ValueError('Expected px256')
    if shape=='circle':
        extent=np.repeat(float(spec['radius']),2);center=np.asarray(spec['center'],float)
        if spec['radius']<=1:raise ValueError('Radius below supported scale')
        lo,hi=center-extent,center+extent
    elif shape=='ellipse':
        axes=np.asarray(spec['axes'],float);center=np.asarray(spec['center'],float)
        if np.any(axes<=1):raise ValueError('Axes below supported scale')
        angle=math.radians(spec['angle_deg']);a,b=axes
        extent=np.sqrt([(a*math.cos(angle))**2+(b*math.sin(angle))**2,
                        (a*math.sin(angle))**2+(b*math.cos(angle))**2])
        lo,hi=center-extent,center+extent
    elif shape=='anostyler_blob':
        size=np.asarray(spec['size'],float)
        if np.any(size<4) or np.any(size!=np.rint(size)):raise ValueError('Invalid integer blob size')
        lo=np.rint(np.asarray(spec['center'],float)-(size-1)/2);hi=lo+size-1
    elif shape in ('line','polyline'):
        if spec['width']<=1:raise ValueError('Line below supported width')
        if shape=='line':
            angle=math.radians(spec['angle_deg']);direction=np.array([math.cos(angle),math.sin(angle)])
            center=np.asarray(spec['center'],float)
            points=np.stack((center-direction*spec['length']/2,center+direction*spec['length']/2))
        else:points=np.asarray(spec['points'],float)
        if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<3):raise ValueError('Degenerate segment')
        lo=points.min(0)-spec['width']/2;hi=points.max(0)+spec['width']/2
    else:raise ValueError('Unexpected template shape')
    if not np.isfinite([lo,hi]).all() or np.any(lo<0) or np.any(hi>255):
        raise ValueError('Static bounds exceed working frame')
    return {'continuous_or_requested_box_xyxy':[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],
            'rasterization_checked':False,'image_validity_checked':False}


def make_requests(backgrounds):
    if len(backgrounds)!=20 or len(set(backgrounds))!=20:raise ValueError('Expected fixed S_bg20')
    normal=[];anomaly=[];table=templates()
    for b,path in enumerate(backgrounds):
        normal.append({'id':f's{b:02d}_normal','background_index':b,'background_path':path})
        for entry in table:
            t=entry['template_index']
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([ROOT_SEED,b,t])))
            seed=int(rng.integers(0,2**31-1))
            for polarity,sign in (('bright',1),('dark',-1)):
                for strength,amplitude in (('a04',.04),('a02',.02)):
                    spec=copy.deepcopy(entry['spec']);spec['amplitude']=sign*amplitude
                    anomaly.append({'id':f's{b:02d}_t{t:02d}_{polarity}_{strength}',
                        'ordinal':len(anomaly),'background_index':b,'background_path':path,
                        'normal_id':f's{b:02d}_normal','template_index':t,'template_id':entry['template_id'],
                        'geometry_group_id':f's{b:02d}_t{t:02d}','pair_id':f's{b:02d}_t{t:02d}_{polarity}',
                        'polarity':polarity,'strength_id':strength,'geometry_seed':seed,'spec':spec})
    return {'schema':'shared-holdout-predeclared-requests-v1','root_seed':ROOT_SEED,
        'seed_rule':'PCG64(SeedSequence([20261031,b,t])).integers(0,2**31-1); upper exclusive; no polarity/strength component',
        'order':'background, template, bright/dark, a04/a02; do not lexically reorder',
        'templates':table,'normal':normal,'anomaly':anomaly,'observed_validity':'not generated or checked'}


def verify_request_scalars(requests,backgrounds):
    if requests!=make_requests(backgrounds):raise ValueError('Request values/order/seeds differ')
    rows=requests['anomaly'];normal=requests['normal']
    if len(rows)!=480 or len(normal)!=20 or len({r['id'] for r in rows+normal})!=500:
        raise ValueError('Planned request counts/IDs differ')
    groups={};pairs={}
    for r in rows:
        groups.setdefault(r['geometry_group_id'],[]).append(r)
        pairs.setdefault(r['pair_id'],[]).append(r)
    if len(groups)!=120 or len(pairs)!=240:raise ValueError('Pair/group count differs')
    for group in groups.values():
        specs=[{k:v for k,v in r['spec'].items() if k!='amplitude'} for r in group]
        if len(group)!=4 or len({r['geometry_seed'] for r in group})!=1 or any(s!=specs[0] for s in specs):
            raise ValueError('Geometry seed/spec not shared by four variants')
    for pair in pairs.values():
        if len(pair)!=2 or {r['strength_id'] for r in pair}!={'a04','a02'}:
            raise ValueError('Strength pairing differs')
    counts=Counter((r['strength_id'],r['spec']['kind'],r['polarity']) for r in rows)
    if len(counts)!=8 or set(counts.values())!={60}:raise ValueError('Type/polarity/strength denominators differ')
    return {'planned_normal':20,'planned_anomaly':480,'per_strength':240,
            'per_strength_type':120,'per_strength_type_polarity':60,'geometry_groups':120,'strength_pairs':240,
            'static_geometry':{e['template_id']:static_geometry(e['spec']) for e in requests['templates']},
            'actual_valid_anomaly_count':None,'actual_positive_pixel_count':None}


def source_identity():
    files=[Path(__file__),ROOT/'controlled_generation/data.py',ROOT/'controlled_generation/render.py',
           ROOT/'controlled_generation/geometry.py',ROOT/'controlled_generation/calibration.py']
    return {str(p.relative_to(ROOT)):sha_file(p) for p in files}


def candidate():
    # Existing frozen calibration verification hashes model/metadata only, never images or render.
    cal=verify_calibration();split=load_split();backgrounds=split['S_bg']
    if len(backgrounds)!=20 or len(set(backgrounds))!=20 or set(backgrounds)&set(split['N_gen']+split['Q_cal']):
        raise ValueError('Fixed S_bg20 split differs')
    for p in backgrounds:
        rel=Path(p)
        if rel.is_absolute() or '..' in rel.parts or '/train/good/' not in p:
            raise ValueError('Unexpected S_bg metadata path')
    requests=make_requests(backgrounds);summary=verify_request_scalars(requests,backgrounds)
    calibration_proof_path=CAL_CACHE/'calibration_verification.json';proof=read_json(calibration_proof_path)
    if not proof['calibration_complete'] or proof['configuration_sha256']!=sha_file(CAL_CONFIG):
        raise ValueError('Actual calibration evidence differs')
    results={}
    for condition in ('G0','G1'):
        path=CAL_CACHE/condition/'result.json';sha=sha_file(path);result=read_json(path)
        if (proof['source_sha256'][str(path.relative_to(ROOT))]!=sha or result['threshold']!=THRESHOLDS[condition]
                or result['threshold']!=proof['results'][condition]['threshold'] or result['q']!=.95
                or result['method']!='linear' or result['alarm_rule']!='strictly greater'
                or result['normal_image_denominator']!=40
                or result['identity']!={'condition':condition,'configuration_sha256':sha_file(CAL_CONFIG)}):
            raise ValueError('Frozen actual threshold identity differs')
        results[condition]={'path':str(path),'sha256':sha,'threshold':result['threshold']}
    generation_protocol=read_json(OUT/'generation_protocol.json')
    protocol={'schema':'shared-holdout-metadata-v1','root_seed':ROOT_SEED,'split_sha256':sha_file(SPLIT),
        'S_bg':backgrounds,'dataset_root':split['root'],
        'expected_background_file_sha256':{p:split['source_sha256'][p] for p in backgrounds},
        'background_bytes_verified':False,'preparation_implementation':source_identity(),
        'calibration_config_sha256':sha_file(CAL_CONFIG),'calibration_verification_sha256':sha_file(calibration_proof_path),
        'final_models':cal['models'],'threshold_results':results,'runtime_identity':cal['runtime_identity'],
        'metric_package':{'scikit-learn':importlib.metadata.version('scikit-learn')},
        'upstream_mask_identity':generation_protocol['anostyler'],'requests_canonical_sha256':canonical_hash(requests),
        'planned_denominators':summary,'preprocessing':cal['preprocessing'],'normalization':cal['normalization'],
        'score':cal['score'],'alarm_rule':'score strictly greater than per-model frozen threshold',
        'generation_rules':{'attempts_per_request':1,'retry':False,'replace_background':False,
            'change_location':False,'brightness_capacity_rescue':False,'invalid_as_normal':False,
            'residual_definition':'D=max_RGB(abs(I-B)); actual float32 arrays after clipping',
            'effective_mask_definition':'M_eff=(D>float32(1/255)); strictly greater',
            'finite_RGB_range':'B and I must be finite float32 RGB in [0,1]',
            'nonempty_masks':['M_roi','M_eff'],
            'model_input':'anomaly I and normal B float32 arrays; no PNG/JPEG round trip',
            'normal_GT':'all zero; one normal input per fixed background',
            'float32_label_threshold':float(np.float32(1/255)),'minimum_effective_coverage':.90,
            'coverage_definition':'count(M_eff & M_roi)/count(M_roi)',
            'outside_ROI_change':0,'outside_ROI_positive_pixels':0,
            'invalid_for_detection_metrics':'exclude and report generation denominator separately',
            'missing_source_or_unexpected_geometry_error':'stop; do not drop a background as ordinary invalid'},
        'metric_rules':{'normal_denominator':20,'primary':'a04 all valid plus all20 normals',
            'secondary':'a02 all valid plus same20 normals','pixel_AP':'exact global; all pixels; no sampling or per-image averaging',
            'pixel_denominator':'(20+valid_anomaly_count)*65536; positive count from actual M_eff',
            'image_AUROC':'valid anomalies vs20 normals; NA without both classes',
            'recall_denominator':'valid_anomaly_count','normal_FP_denominator':20,
            'type_groups':'each type valid anomalies plus same20 normals; do not add normal counts across tables',
            'paired_strength':'both strengths valid only; max240 pairs; record unmatched failures',
            'model_comparison':'same shared valid set; no model-specific omission',
            'empty_positive_or_anomaly_group':'NA, not0 or1','independent_repetitions':False},
        'bias':'G1-renderer-domain evaluation; sharing inputs does not remove renderer preference',
        'independent_real_validation':'pending','state':'metadata only; no generation or scoring',
        'future_generation_implementation':'separate holdout_data.py and generation identity; do not edit this frozen module'}
    return protocol,requests


def verify_preparation():
    expected,requests=candidate();saved=read_json(PROTOCOL)
    expected['requests_file_sha256']=sha_file(REQUESTS)
    if saved!=expected or read_json(REQUESTS)!=requests:raise ValueError('Frozen holdout preparation identity differs')
    print('HOLDOUT_METADATA_VERIFIED',{'backgrounds':20,'normal':20,'anomaly':480,'image_bytes_read':0,
                                     'render_calls':0,'model_forwards':0},flush=True)
    return saved,requests


def prepare_holdout():
    protocol,requests=candidate()
    with exclusive_file(FOLDER/'preparation.owner.lock',{'operation':'holdout_metadata_preparation'}):
        if PROTOCOL.exists():
            if not REQUESTS.exists():raise ValueError('Missing requests; preserve partial preparation')
            expected=dict(protocol,requests_file_sha256=sha_file(REQUESTS))
            if read_json(PROTOCOL)!=expected or read_json(REQUESTS)!=requests:
                raise ValueError('Existing metadata differs; do not overwrite')
        else:
            other=[p for p in FOLDER.iterdir() if p.name!='preparation.owner.lock']
            if other:raise ValueError('Unidentified/partial holdout artifacts; preserve and diagnose')
            save_json(REQUESTS,requests)
            protocol['requests_file_sha256']=sha_file(REQUESTS)
            save_json(PROTOCOL,protocol)
    return verify_preparation()
