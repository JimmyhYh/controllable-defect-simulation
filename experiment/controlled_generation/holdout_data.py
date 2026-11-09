"""Explicit CPU-only stage2 generation; no model, scoring, calibration or training."""
import hashlib
import io
import platform
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import ROOT, read_json, save_json, resources
from .calibration import exclusive_file
from .holdout_preparation import (FOLDER, PROTOCOL, REQUESTS, canonical_hash,
                                  verify_preparation)
from .probe_lr_decay_preparation import sha_file
from .geometry import describe
from .render import render

CONFIG=FOLDER/'generation_config.json'
STARTED=FOLDER/'generation_started.json'
OWNER=FOLDER/'generation.owner.lock'
MANIFEST=FOLDER/'manifest.json'
DATA=FOLDER/'generation'


def proposed_config(protocol):
    return {'schema':'shared-holdout-generation-v1','protocol_sha256':sha_file(PROTOCOL),
        'requests_sha256':sha_file(REQUESTS),
        'implementation_sha256':{str(Path(__file__).relative_to(ROOT)):sha_file(Path(__file__))},
        'frozen_dependencies':protocol['preparation_implementation'],
        'python':platform.python_version(),'runtime_identity':protocol['runtime_identity'],
        'generation_rules':protocol['generation_rules'],'device':'CPU','root_seed':20261031,
        'minimum_start_RAM_GiB':3.,'minimum_free_disk_GiB':8.,
        'failure_policy':'retain started marker and partial files; no automatic resume or re-render',
        'storage':'background B and zero normal GT once; each request I,M_roi,alpha,D,M_eff,D_png_codes,M_eff_png; npz float arrays and PNG previews',
        'png_rule':'round-to-nearest-even uint8 RGB; D_png_codes=max_RGB(abs(I8-B8)); M_eff_png=(D_png_codes>1)',
        'image_input':'RGB whole-image PIL bilinear256 float32/255; no model input PNG round trip'}


def check_no_leftovers():
    if (FOLDER/'preparation.owner.lock').exists():
        raise RuntimeError('Preparation ownership unresolved; preserve and diagnose')
    if any(FOLDER.rglob('*.tmp')):
        raise RuntimeError('Uncommitted temporary artifacts; preserve and diagnose')


def check_resources():
    value=resources();value['disk_free_GiB']=shutil.disk_usage(FOLDER).free/2**30
    if value.get('available_RAM_GiB',0)<3. or value['disk_free_GiB']<8.:
        raise RuntimeError('Need at least3GiB available RAM and8GiB disk; no generation started')
    return value


def source_path(protocol,relative):
    if relative not in protocol['S_bg']:raise ValueError('Only fixed S_bg sources allowed')
    root=Path(protocol['dataset_root']).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Source escapes fixed dataset root')
    return path


def verify_sources(protocol):
    # Hash bytes without decoding, including on complete-cache verification.
    for relative in protocol['S_bg']:
        path=source_path(protocol,relative)
        if not path.is_file() or sha_file(path)!=protocol['expected_background_file_sha256'][relative]:
            raise ValueError('Missing/mismatched S_bg source; stop, never drop or replace background')


def load_background_once(protocol,relative):
    raw=source_path(protocol,relative).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=protocol['expected_background_file_sha256'][relative]:
        raise ValueError('Source changed between preflight and decode')
    with Image.open(io.BytesIO(raw)) as image:
        background=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/np.float32(255)
    validate_RGB(background)
    return background


def validate_RGB(value):
    if value.shape!=(256,256,3) or value.dtype!=np.float32 or not np.isfinite(value).all() or value.min()<0 or value.max()>1:
        raise ValueError('Nonfinite/malformed RGB input or output; execution error, not ordinary invalid request')


def array_hash(value):
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def analyze(background,image,roi,alpha,amplitude):
    """Actual saved-array arithmetic, no renderer/rasterizer; also used for cache audit."""
    validate_RGB(background);validate_RGB(image)
    if roi.shape!=(256,256) or roi.dtype!=bool or not roi.any():raise ValueError('Malformed/empty ROI; stop')
    if alpha.shape!=(256,256) or alpha.dtype!=np.float32 or not np.isfinite(alpha).all() or alpha.min()<0 or alpha.max()>1:
        raise ValueError('Malformed alpha; stop')
    if np.any(alpha[~roi]!=0):raise ValueError('Alpha outside ROI; frozen renderer needs investigation')
    unclipped=background+np.float32(amplitude)*alpha[...,None]
    if not np.array_equal(image,np.clip(unclipped,0,1).astype(np.float32)):
        raise ValueError('Output differs from frozen signed-additive construction')
    residual=np.max(np.abs(image-background),axis=-1)
    effective=residual>np.float32(1/255)
    outside=float(residual[~roi].max(initial=0))
    if outside!=0 or np.any(effective&~roi):raise ValueError('Unexpected outside-ROI change; stop and diagnose')
    before8=np.rint(background*255).astype(np.uint8);after8=np.rint(image*255).astype(np.uint8)
    residual8=np.max(np.abs(after8.astype(np.int16)-before8.astype(np.int16)),axis=-1)
    effective8=residual8>1;coverage=float((effective&roi).sum()/roi.sum())
    reasons=[]
    if not effective.any():reasons.append('effective_mask_empty')
    if coverage<.90:reasons.append('effective_coverage_below0.90')
    stats={'valid':not reasons,'invalid_reasons':reasons,'roi':describe(roi),'effective':describe(effective),
        'png_effective':describe(effective8),'effective_coverage':coverage,
        'png_effective_coverage':float((effective8&roi).sum()/roi.sum()),
        'roi_residual_min_median_max':[float(residual[roi].min()),float(np.median(residual[roi])),float(residual[roi].max())],
        'roi_clipped_channel_fraction':float(((unclipped<0)|(unclipped>1))[roi].mean()),
        'outside_ROI_max_float_residual':outside,'outside_ROI_max_png_codes':int(residual8[~roi].max(initial=0)),
        'float_png_mask_disagreement_pixels':int(np.count_nonzero(effective!=effective8)),
        'png_only_positive_pixels':int((effective8&~effective).sum()),
        'float_only_positive_pixels':int((effective&~effective8).sum()),
        'max_abs_float_vs_png_residual_error':float(np.max(np.abs(residual-residual8.astype(np.float32)/np.float32(255))))}
    return {'D':residual,'M_eff':effective,'D_png_codes':residual8,'M_eff_png':effective8},stats,after8


def write_bytes_once(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix+'.tmp').exists():
        raise RuntimeError('Output already exists; no overwrite/re-generation')
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:stream.write(raw)
    temporary.replace(path)


def save_arrays(path,**arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);write_bytes_once(path,buffer.getvalue())


def save_png(path,pixels):
    buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');write_bytes_once(path,buffer.getvalue())


def relative(path):return path.relative_to(FOLDER).as_posix()


def files_record(*paths):return {relative(path):sha_file(path) for path in paths}


def verify_record_metadata(row,request,protocol,normal=False):
    if row['request']!=request:raise ValueError('Record order/source/request identity differs')
    folder='backgrounds' if normal else 'anomalies'
    expected={f"generation/{folder}/{request['id']}{suffix}" for suffix in ('.npz','.png')}
    if set(row['files_sha256'])!=expected:raise ValueError('Record must bind exactly its own NPZ and PNG')
    if normal:
        if row['source_sha256']!=protocol['expected_background_file_sha256'][request['background_path']]:
            raise ValueError('Normal source hash metadata differs')
    else:
        expected_fields={key:request[key] for key in ('id','ordinal','template_id','polarity','strength_id','pair_id')}
        expected_fields['kind']=request['spec']['kind']
        if any(row[key]!=value for key,value in expected_fields.items()):
            raise ValueError('Derived ordinal/group/pair metadata differs from frozen request')


def load_arrays(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key] for key in archive.files}


def summary(rows):
    groups={}
    for fields in (('strength_id',),('strength_id','kind'),('strength_id','kind','polarity'),('strength_id','template_id','polarity')):
        values=defaultdict(list)
        for row in rows:values['/'.join(row[k] for k in fields)].append(row)
        for key,group in values.items():
            valid=[row for row in group if row['quality']['valid']]
            groups[key]={'planned':len(group),'valid':len(valid),'invalid':len(group)-len(valid),
                'failure_reason_counts':dict(Counter(reason for row in group for reason in row['quality']['invalid_reasons'])),
                'valid_positive_pixels':sum(row['quality']['effective']['pixels'] for row in valid)}
    paired=defaultdict(dict)
    for row in rows:paired[row['pair_id']][row['strength_id']]=row
    counts=Counter();ratios=[];different_masks=0
    for pair in paired.values():
        strong,weak=pair['a04'],pair['a02'];a=strong['quality']['valid'];b=weak['quality']['valid']
        counts['both_valid' if a and b else 'only_a04_valid' if a else 'only_a02_valid' if b else 'both_invalid']+=1
        if a and b:
            ratios.append(weak['quality']['roi_residual_min_median_max'][2]/strong['quality']['roi_residual_min_median_max'][2])
            different_masks+=strong['effective_array_sha256']!=weak['effective_array_sha256']
    valid=sum(row['quality']['valid'] for row in rows)
    return {'normal_inputs':20,'planned_anomalies':480,'generation_attempts':480,'valid_anomalies':valid,
        'invalid_anomalies':480-valid,'groups':groups,
        'strength_pairs':{'planned':240,**{key:counts[key] for key in ('both_valid','only_a04_valid','only_a02_valid','both_invalid')},
            'both_valid_different_M_eff':different_masks,
            'both_valid_weak_over_strong_max_residual_min_median_max':[float(np.min(ratios)),float(np.median(ratios)),float(np.max(ratios))] if ratios else None},
        'all_requests_float_PNG_disagreement_pixels':sum(row['quality']['float_png_mask_disagreement_pixels'] for row in rows),
        'maximum_actual_float_residual':max(row['quality']['roi_residual_min_median_max'][2] for row in rows),
        'future_normal_FP_denominator':20,'future_anomaly_recall_denominator':'valid count per strength/subgroup',
        'future_pixel_denominators':{s:(20+groups[s]['valid'])*65536 for s in ('a04','a02')},
        'scoring_performed':False,'independent_backgrounds_upper_bound':20}


def verify_generated(protocol=None,requests=None,config=None,owned=False):
    if not owned and OWNER.exists():raise RuntimeError('Generation owned/interrupted; preserve marker and diagnose')
    check_no_leftovers()
    if protocol is None:protocol,requests=verify_preparation()
    expected=proposed_config(protocol)
    if config is None:config=read_json(CONFIG)
    if config!=expected or read_json(CONFIG)!=expected:raise ValueError('Generation implementation/config identity differs')
    identity={'generation_config_sha256':sha_file(CONFIG)}
    if read_json(STARTED)['identity']!=identity:raise ValueError('Generation start identity differs')
    verify_sources(protocol)
    manifest=read_json(MANIFEST)
    if manifest['identity']!=identity or not manifest['complete']:raise ValueError('Manifest incomplete or mismatched')
    normals=manifest['normal'];rows=manifest['anomaly']
    if len(normals)!=20 or len(rows)!=480:raise ValueError('Incomplete planned set')
    for row,request in zip(normals,requests['normal']):verify_record_metadata(row,request,protocol,normal=True)
    for row,request in zip(rows,requests['anomaly']):verify_record_metadata(row,request,protocol)
    expected_files=set();geometry={}
    for b,normal in enumerate(normals):
        request=requests['normal'][b]
        if normal['request']!=request:raise ValueError('Normal order/source identity differs')
        for name,sha in normal['files_sha256'].items():
            path=FOLDER/name
            if path.resolve().parent!= (DATA/'backgrounds').resolve() or sha_file(path)!=sha:raise ValueError('Normal file identity differs')
            expected_files.add(name)
        path=DATA/'backgrounds'/f"{request['id']}.npz"
        arrays=load_arrays(path)
        if set(arrays)!={'B','normal_GT'}:raise ValueError('Background archive fields differ')
        background=arrays['B'];validate_RGB(background)
        if arrays['normal_GT'].shape!=(256,256) or arrays['normal_GT'].dtype!=bool or arrays['normal_GT'].any():raise ValueError('Normal GT differs')
        if array_hash(background)!=normal['background_array_sha256']:raise ValueError('Background array identity differs')
        for offset,row in enumerate(rows[b*24:(b+1)*24]):
            req=requests['anomaly'][b*24+offset]
            for name,sha in row['files_sha256'].items():
                path=FOLDER/name
                if path.resolve().parent!=(DATA/'anomalies').resolve() or sha_file(path)!=sha:raise ValueError('Generated file identity differs')
                expected_files.add(name)
            arrays=load_arrays(DATA/'anomalies'/f"{req['id']}.npz")
            if set(arrays)!={'I','M_roi','alpha','D','M_eff','D_png_codes','M_eff_png'}:raise ValueError('Anomaly archive fields differ')
            labels,quality,_=analyze(background,arrays['I'],arrays['M_roi'],arrays['alpha'],req['spec']['amplitude'])
            if quality!=row['quality'] or any(not np.array_equal(arrays[k],v) or arrays[k].dtype!=v.dtype for k,v in labels.items()):
                raise ValueError('Saved residual/labels/validity differ')
            group=(array_hash(arrays['M_roi']),array_hash(arrays['alpha']))
            if group!=tuple(row['geometry_array_sha256']):raise ValueError('Geometry array identity differs')
            if array_hash(arrays['M_eff'])!=row['effective_array_sha256']:raise ValueError('Effective mask identity differs')
            if geometry.setdefault(req['geometry_group_id'],group)!=group:raise ValueError('Four-variant geometry differs')
    actual_files={relative(p) for p in DATA.rglob('*') if p.is_file()}
    if actual_files!=expected_files or len(actual_files)!=1000:raise ValueError('Unknown/missing generation files')
    if len(geometry)!=120 or summary(rows)!=manifest['summary']:raise ValueError('Summary/group identity differs')
    if manifest['execution_counts']!={'source_decodes':20,'render_calls':480,'retries':0,'model_forwards':0,'optimizer_creations':0}:
        raise ValueError('Execution count contract differs')
    print('HOLDOUT_GENERATED_VERIFIED',manifest['summary']['valid_anomalies'],'/480; normal20; no render or source decode during verification',flush=True)
    return manifest


def generate_holdout():
    protocol,requests=verify_preparation();check_no_leftovers()
    expected=proposed_config(protocol)
    with exclusive_file(OWNER,{'operation':'shared_holdout_generation','protocol_sha256':sha_file(PROTOCOL)}):
        if MANIFEST.exists():return verify_generated(protocol,requests,owned=True)
        if STARTED.exists() or DATA.exists():raise RuntimeError('Partial/uncertain generation; preserve all outputs; no automatic resume or re-render')
        allowed={'protocol.json','requests.json','preparation_verification.json',CONFIG.name,OWNER.name,
                 '第9天阶段1准备结果.md'}
        if any(p.name not in allowed for p in FOLDER.iterdir()):raise RuntimeError('Unknown pre-generation artifact; preserve and diagnose')
        if CONFIG.exists() and read_json(CONFIG)!=expected:raise ValueError('Existing generation config differs')
        start_resources=check_resources();verify_sources(protocol)
        if not CONFIG.exists():save_json(CONFIG,expected)
        identity={'generation_config_sha256':sha_file(CONFIG)}
        save_json(STARTED,{'identity':identity,'note':'Persistent one-shot boundary; incomplete generation is never automatically repeated'})
        start=time.perf_counter();normals=[];rows=[];geometry={};decode_count=0;render_count=0
        minimum_RAM=start_resources['available_RAM_GiB'];minimum_disk=start_resources['disk_free_GiB']
        for b,request in enumerate(requests['normal']):
            current=check_resources();minimum_RAM=min(minimum_RAM,current['available_RAM_GiB']);minimum_disk=min(minimum_disk,current['disk_free_GiB'])
            background=load_background_once(protocol,request['background_path']);decode_count+=1
            archive=DATA/'backgrounds'/f"{request['id']}.npz";png=archive.with_suffix('.png')
            save_arrays(archive,B=background,normal_GT=np.zeros((256,256),bool));save_png(png,np.rint(background*255).astype(np.uint8))
            normals.append({'request':request,'background_array_sha256':array_hash(background),
                'source_sha256':protocol['expected_background_file_sha256'][request['background_path']],
                'files_sha256':files_record(archive,png)})
            for req in requests['anomaly'][b*24:(b+1)*24]:
                # Exactly one call per predeclared request; no retry/rescue branch.
                render_count+=1;output=render(background,req['spec'],seed=req['geometry_seed'])
                labels,quality,after8=analyze(background,output['image'],output['M_roi'],output['alpha'],req['spec']['amplitude'])
                if not np.array_equal(output['background'],background) or output['stats']['valid']!=quality['valid']:
                    raise ValueError('Frozen renderer contract differs')
                for old,new in (('residual','D'),('M_eff','M_eff'),('M_eff_png','M_eff_png')):
                    if not np.array_equal(output[old],labels[new]):raise ValueError('Renderer actual labels differ')
                group=(array_hash(output['M_roi']),array_hash(output['alpha']))
                if geometry.setdefault(req['geometry_group_id'],group)!=group:raise ValueError('Shared geometry/alpha differ')
                archive=DATA/'anomalies'/f"{req['id']}.npz";png=archive.with_suffix('.png')
                save_arrays(archive,I=output['image'],M_roi=output['M_roi'],alpha=output['alpha'],**labels);save_png(png,after8)
                rows.append({'id':req['id'],'ordinal':req['ordinal'],'request':req,'kind':req['spec']['kind'],
                    'template_id':req['template_id'],'polarity':req['polarity'],'strength_id':req['strength_id'],'pair_id':req['pair_id'],
                    'quality':quality,'renderer_stats':output['stats'],'geometry_array_sha256':list(group),
                    'effective_array_sha256':array_hash(labels['M_eff']),'files_sha256':files_record(archive,png)})
            print('HOLDOUT_BACKGROUND_COMPLETED',b+1,'/20; requests',len(rows),'/480',flush=True)
        if decode_count!=20 or render_count!=480 or len(rows)!=480:raise ValueError('Incomplete execution')
        manifest={'schema':'shared-holdout-generated-inputs-v1','identity':identity,'complete':True,
            'normal':normals,'anomaly':rows,'summary':summary(rows),
            'execution_counts':{'source_decodes':decode_count,'render_calls':render_count,'retries':0,'model_forwards':0,'optimizer_creations':0},
            'resources':{'start':start_resources,'sampled_min_available_RAM_GiB':minimum_RAM,'sampled_min_disk_free_GiB':minimum_disk,
                         'elapsed_generation_seconds':time.perf_counter()-start,'sampling':'before each background; not peak monitoring'},
            'bias':protocol['bias'],'independent_real_validation':'pending; not read','thresholds':protocol['threshold_results']}
        save_json(MANIFEST,manifest)
        return verify_generated(protocol,requests,expected,owned=True)
