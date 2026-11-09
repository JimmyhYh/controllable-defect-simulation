"""CPU-only frozen PatchCore response extraction; no predictor or fitting path.

Reads score/heatmap caches and nine fixed M_geo masks, never RGB or residuals.
Unknown/partial output transactions are preserved; complete ones only verify.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent.parent
BASE = ROOT/'results/mura_response/semu_applicability_v1'
FIT = BASE/'fit_inputs_identity_v2'
SCORE = ROOT/'results/mura_capability/week1_v1/patchcore_fit_scoring_v1'
MINI = SCORE.parent/'patchcore_minimal_runtime_v1'
OUT = SCORE.parent/'patchcore_fit_response_v1'
PREP, RUN = OUT/'preparation', OUT/'run'
TAU = 1.852951288223265
SCORE_HASH = '501b547812f6cb941e8f4f8ed7f92046c8451e330d5698302fb77d7578428f7e'
FIT_HASH = '3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593'
MINI_HASH = '51472f150d69d2d206fd26990a56ea95d35d6a9440ce1e7212c0e4bdffd038b6'
DESIGN = PROJECT/'计划/Mura检测器响应与SEMU代理刻度研究设计.md'
EVIDENCE = PROJECT/'报告/PatchCore独立D_fit全量只读评分执行与核验.md'
AMPLITUDES = (.001,.002,.004,.008,.016,.032,.064)
FORBIDDEN = ('torch','torchvision','faiss','timm','patchcore','scipy','sklearn','statsmodels',
    'controlled_generation','mura_response.inference','mura_response.inference_runtime',
    'mura_response.generation','mura_response.generation_v2','mura_response.resources',
    'mura_capability.patchcore_fit_scoring','mura_capability.patchcore_minimal')
ARRAY_READS = Counter()
DECODE = {'heatmaps':0,'geometry_masks':0}
ALLOWED_ARRAYS = set()
BLOCKED = []


def require(ok,message):
    if not ok:raise RuntimeError(message+'; preserve state; no automatic retry or substitution')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def safe(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink() and not p.is_junction()
            and p.absolute()==p.resolve(),'Missing/redirected source: '+str(p))
    return p


def inventory(folder,files,dirs=()):
    require(folder.is_dir() and not folder.is_symlink() and not folder.is_junction(),'Missing/redirected cache')
    members=list(folder.rglob('*'))
    require(not any(p.is_symlink() or p.is_junction() for p in members),'Redirected cache member')
    require({p.relative_to(folder).as_posix() for p in members if p.is_file()}==set(files)
            and {p.relative_to(folder).as_posix() for p in members if p.is_dir()}==set(dirs),
            'Unknown/partial cache inventory')


def closed(folder,names):
    inventory(folder,set(names)|{'complete.json'})
    outputs=read(folder/'complete.json')['outputs']
    require(set(outputs)==set(names) and all(digest(safe(folder/n))==h for n,h in outputs.items()),
            'Saved output identity differs')


def seal(folder):
    require(not any((folder/n).exists() for n in ('complete.json','failure.json','inflight.json')),
            'Cannot seal partial/failed output')
    write(folder/'complete.json',{'outputs':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}})


def resources():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','page','page_available','virtual','virtual_available','extended')]
    s=Memory();s.length=ctypes.sizeof(s)
    q=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
    q.argtypes=[ctypes.POINTER(Memory)];q.restype=ctypes.c_int
    require(q(ctypes.byref(s)),'RAM query failed')
    result={'available_RAM_GiB':s.available/2**30,'D_free_GiB':shutil.disk_usage(ROOT).free/2**30}
    require(result['available_RAM_GiB']>=2 and result['D_free_GiB']>=40,'CPU RAM/disk admission not met')
    return result


def matches(name):return any(name==p or name.startswith(p+'.') for p in FORBIDDEN)


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if matches(fullname):
            BLOCKED.append({'import':fullname});require(False,'Model/fitting/generation import forbidden')


def install_boundary():
    own=OUT.resolve();temps={Path(v).resolve() for k in ('TEMP','TMP') if (v:=os.environ.get(k))}
    def deny(event):BLOCKED.append(event);require(False,str(event))
    def writable(p):return p.is_relative_to(own) or any(p.is_relative_to(t) for t in temps)
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','pickle.find_class','os.system'):
            deny({'event':event})
        if event=='subprocess.Popen':
            f=sys._getframe(1);good=False
            while f:
                if (f.f_code.co_name=='_syscmd_ver' and Path(f.f_code.co_filename).resolve()==
                    (Path(sys.base_prefix)/'Lib/platform.py').resolve() and
                    f.f_locals.get('cmd') in ('ver','command /c ver','cmd /c ver')):good=True;break
                f=f.f_back
            if not good:deny({'subprocess':str(args[:2])})
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename'):
            for value in (args[:2] if event=='os.rename' else args[:1]):
                if isinstance(value,(str,bytes)) and not writable(Path(os.fsdecode(value)).resolve()):
                    deny({'mutation':str(value)})
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        raw=os.fsdecode(args[0])
        if raw.upper() in ('NUL','\\\\.\\NUL','\\\\?\\NUL'):return
        p=Path(raw).resolve();ext=p.suffix.lower()
        mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=any(x in mode for x in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not writable(p):deny({'write':str(p)})
            return
        if ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.pth','.pt','.pkl','.pickle','.faiss','.safetensors'):
            deny({'RGB_GT_or_model_asset':str(p)})
        if ext in ('.npy','.npz'):
            if p not in ALLOWED_ARRAYS:deny({'unlisted_array':str(p)})
            ARRAY_READS[str(p)]+=1
    sys.meta_path.insert(0,ImportGuard());sys.addaudithook(audit)


def boundary():
    loaded=[n for n in sys.modules if matches(n)]
    require(not loaded and not BLOCKED,'CPU boundary crossed')
    return {'model_or_fitting_packages_loaded':loaded,'blocked':list(BLOCKED),
        'heatmap_decodes':DECODE['heatmaps'],'geometry_mask_decodes':DECODE['geometry_masks'],
        'array_file_opens':dict(ARRAY_READS),'model_forwards':0,'RGB_GT_reads':0,
        'parameter_updates':0,'calibrations':0,'statistical_fits':0,'renders':0}


def source_context():
    hashes={str(Path(__file__).resolve()):digest(__file__)}
    def bind(path,expected=None):
        path=safe(path);h=digest(path)
        require(expected is None or h==expected,'Source hash differs: '+str(path))
        hashes[str(path)]=h
    bind(SCORE/'run/complete.json',SCORE_HASH);sc=read(SCORE/'run/complete.json')['outputs']
    unique=read(SCORE/'preparation/unique_inputs.json')['rows']
    refs=read(SCORE/'preparation/references.json')['rows']
    sp=read(SCORE/'preparation/sources.json')
    for folder in (SCORE/'preparation',MINI/'preparation'):
        outputs=read(folder/'complete.json')['outputs'];inventory(folder,set(outputs)|{'complete.json'})
        bind(folder/'complete.json')
        for n,h in outputs.items():require(n.endswith('.json') and Path(n).name==n,'Foreign prep member');bind(folder/n,h)
    require(digest(SCORE/'preparation/complete.json')=='ff3ae176f3677d2fa994b66e8f22b035d92a0308480cc2c35f08661c19ba7538',
            'Scoring preparation identity differs')
    bind(ROOT/'mura_capability/patchcore_fit_scoring.py','8382d936fec12ab0e45afcd4245da698c3b66daf14ed109efc9253b0e2e2494d')
    bind(ROOT/'mura_capability/patchcore_minimal.py','479fd19ac5f2694d5ab81dd9b0722a095241288c985d2c0834fc74f7acf62924')
    expected={'started.json','references.json','result.json','verification.json'}|{
        'records/'+r['working_RGB_sha256']+'.json' for r in unique}|{
        'heatmaps/'+r['working_RGB_sha256']+'.npy' for r in unique if not r['reuse_minimal']}
    require(set(sc)==expected,'Scoring manifest coverage differs')
    inventory(SCORE/'run',expected|{'complete.json'},('records','heatmaps'))
    for n,h in sc.items():
        if n.endswith('.json'):bind(SCORE/'run'/n,h)
    bind(MINI/'run/complete.json',MINI_HASH);mc=read(MINI/'run/complete.json')['outputs']
    inventory(MINI/'run',set(mc)|{'complete.json'})
    for n,h in mc.items():
        if n.endswith('.json'):bind(MINI/'run'/n,h)
    result=read(SCORE/'run/result.json');verification=read(SCORE/'run/verification.json')
    require(result['status']=='FULL_FIT_SCORING_COMPLETE' and result['counters']=={
        'model_loads':1,'tensor_loads':1,'author_placeholder_forwards':1,'model_input_forwards':7303}
        and result['reused']==2 and result['tau']==TAU and not result['CPU_response_statistics_performed']
        and result['no_training_generation_calibration_fitting'] and verification['passed']
        and verification['state_before']==verification['state_after']==sp['minimal_sources']['normal_state'],
        'Frozen score/model/BN/tau proof differs')
    bind(EVIDENCE)
    match=re.search(r'```json\s*\n(.*?)\n```',EVIDENCE.read_text(encoding='utf-8'),re.S)
    require(match is not None,'Independent full verification proof missing')
    proof=json.loads(match.group(1))
    require(proof['status']=='FULL_FIT_COMPLETE_CACHE_VERIFIED' and proof['complete_sha256']==SCORE_HASH
        and proof['denominators']==result['denominators'] and proof['new_model_input_forwards']==0
        and proof['new_source_RGB_reads']==0 and proof['tau_recomputed'] is False
        and proof['CPU_response_statistics_performed'] is False,
        'Independent scoring verification identity differs')
    bind(FIT/'complete.json',FIT_HASH);fc=read(FIT/'complete.json')['outputs']
    for n,h in fc.items():
        if n.endswith('.json'):bind(FIT/n,h)
    gen=read(FIT/'sources.json')
    for n,h in gen['preparation'].items():bind(BASE/'preparation_v1'/n,h)
    bind(DESIGN,gen['legacy_and_supplement']['files'][str(DESIGN)])
    geom=read(BASE/'preparation_v1/geometry.json')['rows']
    manifest=read(FIT/'manifest.json')['rows'];audit=read(FIT/'source_audit.json')['backgrounds']
    roles=read(BASE/'preparation_v1/roles.json')['roles']['D_fit60']
    requests=[r for r in read(BASE/'preparation_v1/requests.json')['rows'] if r['role']=='D_fit60']
    require(len(unique)==7305 and len(refs)==7620 and len(manifest)==len(requests)==7560
        and len(roles)==len(audit)==60 and len(geom)==9,'Original denominators differ')
    require(read(SCORE/'run/references.json')['rows']==refs
        and [r['id'] for r in refs[:60]]==[r['id'] for r in roles]==[r['id'] for r in audit]
        and [r['id'] for r in refs[60:]]==[r['id'] for r in manifest]==[r['id'] for r in requests],
        'Reference/role/request original order differs')
    bybg={r['id']:r for r in audit};gids=[g['spec']['id'] for g in geom]
    require(len(set(gids))==9,'Duplicate geometry')
    for g in geom:
        path=FIT/f'geometry/{g["spec"]["id"]}.npz';ALLOWED_ARRAYS.add(path.resolve());bind(path,fc[path.relative_to(FIT).as_posix()])
    byhash=defaultdict(list)
    for r in refs:byhash[r['working_RGB_sha256']].append(r['id'])
    records={}
    for i,row in enumerate(unique):
        h=row['working_RGB_sha256'];rec=read(SCORE/'run/records'/f'{h}.json');reuse=row['reuse_minimal']
        path=MINI/'run'/reuse['heatmap'] if reuse else SCORE/'run/heatmaps'/f'{h}.npy'
        require(row['ordinal']==rec['ordinal']==i and row['references']==byhash[h]
            and rec['working_RGB_sha256']==h and rec['input_file_sha256']==row['file_sha256']
            and rec['state_unchanged'] and rec['tau']==TAU and rec['heatmap_path']==str(path)
            and rec['heatmap_sha256']==(mc[path.name] if reuse else sc[path.relative_to(SCORE/'run').as_posix()])
            and rec['source']==('minimal' if reuse else 'new_forward')
            and rec['reused_minimal_id']==(reuse['id'] if reuse else None),'Record/RGB/heatmap binding differs')
        if reuse:require(rec['score']==reuse['score'],'Minimal reused score differs')
        safe(path);ALLOWED_ARRAYS.add(path.resolve());records[h]=rec
    require(len(records)==len(byhash)==7305 and sum(len(v) for v in byhash.values())==7620,
            'Unique-to-reference coverage differs')
    for spec,row,ref in zip(requests,manifest,refs[60:]):
        require(all(row[k]==v for k,v in spec.items()) and row['retained'] and row['role']=='D_fit60'
            and row['image']['working_RGB_sha256']==ref['working_RGB_sha256']
            and row['background_image']==bybg[row['background_id']]['image'], 'Request semantics differ')
        validate_measurement(row['measurement'],bybg[row['background_id']]['measurements'][row['geometry_id']])
    runtime=gen['runtime'];require(sys.version.split()[0]==runtime['python']
        and Path(sys.executable).resolve()==Path(runtime['executable']).resolve()
        and importlib.metadata.version('numpy')==runtime['packages']['numpy'],'CPU environment differs')
    bind(sys.executable,runtime['executable_sha256'])
    source={'files':hashes,'score_complete_sha256':SCORE_HASH,'input_complete_sha256':FIT_HASH,
        'independent_verification':proof,'model_state':verification['state_before'],'tau':TAU,
        'input_residual_summary':read(FIT/'manifest.json')['summary'],'geometry_ids':gids,
        'model_assets_not_reopened':True,'scoring_denominators':result['denominators']}
    return source,unique,refs,manifest,audit,geom,records


def validate_measurement(m,b):
    require(all(m[k]==v for k,v in b.items()) and m['defined']==(m['Lb']>1e-6)
        and m['outside_max']==0 and not m['admission_uses_effective_mask']
        and m['zero_actual_change']==(m['nonzero_pixels']==0)==(m['residual_max']==0),
        'Frozen measurement/background conditions differ')
    require(0<=m['M_eff_1_over_255_pixels']<=m['nonzero_pixels']<=m['area_pixels'], 'Residual counts differ')
    if m['defined']:
        c=100*abs(m['DeltaY'])/m['Lb']
        require(math.isclose(m['C_pair_pct'],c,rel_tol=1e-12,abs_tol=1e-12),'Paired contrast formula differs')
        for k in (.2,.4,.8):
            expected=c/(1.97/(m['area_pixels']*k*k)**.33+.72)
            require(math.isclose(m['SEMU_proxy'][str(k)],expected,rel_tol=1e-12,abs_tol=1e-12),
                    'Frozen SEMU proxy formula differs')
    else:require(m['C_pair_pct'] is None and all(v is None for v in m['SEMU_proxy'].values()),
                 'Undefined proxy was regularized')


def response_fields(heat,score,masks,gids,np,tau=TAU):
    require(math.isfinite(score) and np.isfinite(heat).all(),'Nonfinite response')
    flat=heat.reshape(-1,order='C');peak=int(np.argmax(flat));alarm=int(score>tau)
    regions={}
    for gid in gids:
        mask=masks[gid]
        require(mask.dtype==np.bool_ and mask.shape==heat.shape and bool(mask.any()),'Invalid fixed M_geo')
        inside=bool(mask.reshape(-1,order='C')[peak])
        regions[gid]={'peak_in_M_geo':inside,'H_joint':int(alarm and inside),
                      'region_max':float(heat[mask].max())}
    return {'score':score,'H_alarm':alarm,'first_flat_peak':peak,
        'peak_xy':[peak%heat.shape[1],peak//heat.shape[1]],'global_heat_max':float(flat[peak]),
        'global_peak_tie_pixels':int(np.count_nonzero(flat==flat[peak])),'by_geometry':regions}


def paired(response,baseline,gid):
    return {'score_delta':response['score']-baseline['score'],
            'region_max_delta':response['by_geometry'][gid]['region_max']-baseline['by_geometry'][gid]['region_max']}


def rates(rows):
    n=len(rows);alarm=sum(r['H_alarm'] for r in rows);joint=sum(r['H_joint'] for r in rows)
    return {'n':n,'H_alarm_sum':alarm,'H_joint_sum':joint,
            'H_alarm_rate':alarm/n if n else None,'H_joint_rate':joint/n if n else None,
            'zero_actual':sum(bool(r.get('zero_actual_change',False)) for r in rows),
            'nonzero_actual':sum(r.get('zero_actual_change') is False for r in rows),
            'weak_nonzero_empty_old_M_eff':sum(r.get('weak_nonzero_empty_old_M_eff',False) for r in rows),
            'undefined':sum(r.get('measurement_defined') is False for r in rows)}


def scalar_audit(np):
    mask=np.array([[False,True],[False,False]],dtype=bool)
    tie=response_fields(np.array([[9,9],[0,0]],np.float32),2.,{'g':mask},['g'],np,1.)
    inside=response_fields(np.array([[0,9],[0,0]],np.float32),2.,{'g':mask},['g'],np,1.)
    equal=response_fields(np.array([[0,9],[0,0]],np.float32),1.,{'g':mask},['g'],np,1.)
    native=response_fields(np.array([[0,.2],[0,0]],np.float32),2.,{'g':mask},['g'],np,1.)
    require(tie['first_flat_peak']==0 and tie['global_peak_tie_pixels']==2 and tie['by_geometry']['g']['H_joint']==0
        and inside['by_geometry']['g']['H_joint']==1 and equal['H_alarm']==0 and native['H_alarm']==1,
        'First C-peak/strict tau/native score artificial check failed')
    require(paired(tie,tie,'g')=={'score_delta':0.,'region_max_delta':0.}
        and rates([])['H_joint_rate'] is None,'Zero-pair/empty-group convention differs')
    undefined={'Lb':0.,'defined':False,'area_pixels':1,'background_std_Y':0.,'background_relative_std':None}
    validate_measurement({**undefined,'outside_max':0.,'admission_uses_effective_mask':False,
        'zero_actual_change':True,'nonzero_pixels':0,'residual_max':0.,'M_eff_1_over_255_pixels':0,
        'C_pair_pct':None,'SEMU_proxy':{'0.2':None,'0.4':None,'0.8':None}},undefined)
    return {'passed':True,'first_flat_tie_outside_not_rescued':True,'strict_equal_not_alarm':True,
            'native_score_not_heatmax':True,'zero_pair_exact':True,'empty_rate_NA':True,'undefined_retained':True}


def prepare():
    context=source_context();source=context[0];snapshot=resources()
    if OUT.exists():check_preparation();return {'status':'CPU_RESPONSE_PREPARATION_VERIFIED'}
    import numpy as np
    numeric=scalar_audit(np)
    PREP.mkdir(parents=True,exist_ok=False)
    write(PREP/'sources.json',source)
    write(PREP/'protocol.json',{'tau':TAU,'H_alarm':'native score strictly greater than original tau',
        'H_joint':'H_alarm and first C-flat global heatmap peak in fixed M_geo',
        'pairing':'score(I)-score(B); max(heat(I)[M_geo])-max(heat(B)[M_geo])',
        'read_only_heatmaps':7305,'fixed_masks':9,'all_references':7620,'requests':7560,'baselines':60,
        'baseline_views':540,'zero_requests':315,'nonzero_requests':7245,'weak_nonzero':2070,
        'all_groups':'original 9 geometry x bright/dark x 7 amplitudes; all background and residual groups',
        'no_fit_or_model_or_recalibration':True,'resources_at_prepare':snapshot})
    write(PREP/'scalar_validation.json',numeric);seal(PREP)
    return {'status':'CPU_RESPONSE_PREPARATION_COMPLETE','resources':snapshot,'scalar_validation':numeric}


def check_preparation():
    require(OUT.is_dir() and not OUT.is_symlink() and not OUT.is_junction()
        and {p.name for p in OUT.iterdir()}<={'preparation','run'},'Unknown output root')
    closed(PREP,{'sources.json','protocol.json','scalar_validation.json'})
    context=source_context();require(context[0]==read(PREP/'sources.json'),'Bound CPU sources changed')
    cfg=read(PREP/'protocol.json')
    require(cfg['tau']==TAU and cfg['no_fit_or_model_or_recalibration']
        and cfg['read_only_heatmaps']==7305 and cfg['fixed_masks']==9 and cfg['baseline_views']==540,
        'Frozen CPU protocol differs')
    return context


def array_identity(a):
    header=f'{a.dtype.str}|{",".join(map(str,a.shape))}|C\n'.encode('ascii')
    return hashlib.sha256(header+a.tobytes(order='C')).hexdigest()


def derive(context,np):
    source,unique,refs,manifest,audit,geom,records=context;gids=source['geometry_ids'];masks={}
    for g in geom:
        gid=g['spec']['id'];payload=(FIT/f'geometry/{gid}.npz').read_bytes()
        require(hashlib.sha256(payload).hexdigest()==source['files'][str(FIT/f'geometry/{gid}.npz')],
            'Fixed geometry NPZ changed')
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            require(set(z.files)=={'M_geo','alpha'},'Unknown geometry payload')
            mask=z['M_geo'];DECODE['geometry_masks']+=1
        require(mask.dtype==np.bool_ and mask.shape==(256,256) and mask.flags.c_contiguous
            and int(mask.sum())==g['area_pixels'] and array_identity(mask)==g['M_geo_sha256'],
            'Fixed geometry mask content/area differs')
        masks[gid]=mask
    needed=defaultdict(set)
    for r in refs:needed[r['working_RGB_sha256']].update(gids if r['kind']=='baseline' else [r['geometry_id']])
    computed={};unique_rows=[]
    for i,u in enumerate(unique):
        if i%250==0:resources()
        h=u['working_RGB_sha256'];rec=records[h];payload=Path(rec['heatmap_path']).read_bytes()
        require(hashlib.sha256(payload).hexdigest()==rec['heatmap_sha256'],'Original heatmap changed')
        heat=np.load(io.BytesIO(payload),allow_pickle=False);DECODE['heatmaps']+=1
        require(heat.dtype==np.float32 and heat.shape==(256,256) and np.isfinite(heat).all()
            and float(heat.min())==rec['heat_min'] and float(heat.max())==rec['heat_max'],
            'Original heatmap shape/finiteness/range differs')
        fields=response_fields(heat,rec['score'],masks,[g for g in gids if g in needed[h]],np)
        row={'ordinal':i,'working_RGB_sha256':h,'references':u['references'],
             'heatmap_sha256':rec['heatmap_sha256'],**fields}
        computed[h]=row;unique_rows.append(row)
        if (i+1)%1000==0 or i+1==len(unique):print(json.dumps({'CPU_heatmaps_processed':i+1,'of':len(unique)}),flush=True)
    baselines=[];views=[];bybg={}
    for r in audit:
        row={'id':r['id'],'working_RGB_sha256':r['image']['working_RGB_sha256'],
             'measurement_by_geometry':r['measurements'],**computed[r['image']['working_RGB_sha256']]}
        row['id']=r['id'];baselines.append(row);bybg[r['id']]=row
        for gid in gids:
            require(r['measurements'][gid]['area_pixels']==int(masks[gid].sum()),'Baseline area differs')
            views.append({'background_id':r['id'],'geometry_id':gid,'score':row['score'],
                'H_alarm':row['H_alarm'],'first_flat_peak':row['first_flat_peak'],
                **row['by_geometry'][gid],'measurement':r['measurements'][gid]})
    requests=[]
    for raw in manifest:
        h=raw['image']['working_RGB_sha256'];resp=computed[h];base=bybg[raw['background_id']];gid=raw['geometry_id'];m=raw['measurement']
        require(m['area_pixels']==int(masks[gid].sum()),'Measurement/M_geo area differs')
        differences=paired(resp,base,gid)
        if m['zero_actual_change']:
            require(h==raw['background_image']['working_RGB_sha256'] and differences=={'score_delta':0.,'region_max_delta':0.}
                and resp['first_flat_peak']==base['first_flat_peak'],'Zero-change pair differs')
        row={**raw,'measurement_defined':m['defined'],'zero_actual_change':m['zero_actual_change'],
            'weak_nonzero_empty_old_M_eff':not m['zero_actual_change'] and m['M_eff_1_over_255_pixels']==0,
            'score':resp['score'],'H_alarm':resp['H_alarm'],'first_flat_peak':resp['first_flat_peak'],
            'peak_xy':resp['peak_xy'],'global_peak_tie_pixels':resp['global_peak_tie_pixels'],
            **resp['by_geometry'][gid],**differences,
            'baseline':{'score':base['score'],'H_alarm':base['H_alarm'],'first_flat_peak':base['first_flat_peak'],
                        'peak_xy':base['peak_xy'],**base['by_geometry'][gid]}}
        requests.append(row)
    require(DECODE['heatmaps']==7305 and DECODE['geometry_masks']==9 and len(requests)==7560
        and len(baselines)==60 and len(views)==540,'Actual CPU decode/response denominators differ')
    return {'unique_responses.json':{'rows':unique_rows},'requests.json':{'rows':requests},
        'baselines.json':{'rows':baselines},'baseline_views.json':{'rows':views},
        'coverage.json':coverage(unique_rows,requests,baselines,views,gids)}


def coverage(unique,requests,baselines,views,gids):
    states={'all':requests,'zero_actual':[r for r in requests if r['zero_actual_change']],
        'nonzero_actual':[r for r in requests if not r['zero_actual_change']],
        'weak_nonzero_empty_old_M_eff':[r for r in requests if r['weak_nonzero_empty_old_M_eff']],
        'old_M_eff_nonempty':[r for r in requests if r['measurement']['M_eff_1_over_255_pixels']>0],
        'undefined_measurement':[r for r in requests if not r['measurement_defined']]}
    require(len(states['zero_actual'])==315 and len(states['nonzero_actual'])==7245
        and len(states['weak_nonzero_empty_old_M_eff'])==2070,'Residual coverage differs')
    cells=[{'geometry_id':g,'polarity':p,'amplitude':a,**rates([r for r in requests if
        r['geometry_id']==g and r['polarity']==p and r['amplitude']==a])}
        for g in gids for p in ('bright','dark') for a in AMPLITUDES]
    require(len(cells)==126 and all(c['n']==60 for c in cells),'Fixed subgroup coverage differs')
    return {'denominators':{'backgrounds':60,'baseline_views_not_independent':540,'request_references':7560,
            'all_references':7620,'unique_heatmaps':7305,'zero_actual':315,'nonzero_actual':7245,
            'weak_nonzero_empty_old_M_eff':2070,'undefined':len(states['undefined_measurement'])},
        'baseline_alarm':{'backgrounds':60,'H_alarm_sum':sum(r['H_alarm'] for r in baselines)},
        'baseline_geometry_views':rates(views),'request_states':{k:rates(v) for k,v in states.items()},
        'by_geometry':[{'geometry_id':g,**rates([r for r in requests if r['geometry_id']==g])} for g in gids],
        'by_polarity':[{'polarity':p,**rates([r for r in requests if r['polarity']==p])} for p in ('bright','dark')],
        'by_amplitude':[{'amplitude':a,**rates([r for r in requests if r['amplitude']==a])} for a in AMPLITUDES],
        'all_fixed_cells':cells,'by_background':[{'background_id':b['id'],**rates([r for r in requests if r['background_id']==b['id']])} for b in baselines],
        'unique_heatmaps_with_tied_global_peak':sum(r['global_peak_tie_pixels']>1 for r in unique),
        'no_fitted_probability_model_or_p0':True,'no_confidence_intervals_or_significance_tests':True,
        'response_counts_are_not_true_defect_accuracy_or_causal_POD':True}


OUTPUTS={'started.json','unique_responses.json','requests.json','baselines.json','baseline_views.json','coverage.json','verification.json'}


def run():
    context=check_preparation()
    if RUN.exists():return verify()
    snapshot=resources();RUN.mkdir(exist_ok=False)
    write(RUN/'started.json',{'preparation_sha256':digest(PREP/'complete.json'),'resources':snapshot,'CPU_only':True})
    write(RUN/'inflight.json',{'operation':'fixed_cached_response_extraction'})
    started=time.perf_counter()
    try:
        import numpy as np
        outputs=derive(context,np)
        for name,doc in outputs.items():write(RUN/name,doc)
        require(source_context()[0]==context[0],'CPU sources changed during processing')
        write(RUN/'verification.json',{'passed':True,'boundary':boundary(),'resources_after':resources(),
            'elapsed_seconds':time.perf_counter()-started,'no_new_model_or_statistical_fit':True})
        (RUN/'inflight.json').unlink();seal(RUN)
    except BaseException as exc:
        if not (RUN/'failure.json').exists():write(RUN/'failure.json',{
            'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'decoded':dict(DECODE),'no_automatic_retry':True})
        raise
    c=outputs['coverage.json']
    return {'status':'CPU_RESPONSE_EXTRACTION_COMPLETE','output':str(RUN),
            'denominators':c['denominators'],'baseline_alarm':c['baseline_alarm'],
            'request_states':c['request_states']}


def verify():
    context=check_preparation();closed(RUN,OUTPUTS);resources()
    require(read(RUN/'started.json')['preparation_sha256']==digest(PREP/'complete.json'),'Saved CPU binding differs')
    v=read(RUN/'verification.json');b=v['boundary']
    require(v['passed'] and v['no_new_model_or_statistical_fit'] and b['heatmap_decodes']==7305
        and b['geometry_mask_decodes']==9 and not b['blocked'] and not b['model_or_fitting_packages_loaded']
        and all(b[k]==0 for k in ('model_forwards','RGB_GT_reads','parameter_updates','calibrations','statistical_fits','renders')),
        'Saved CPU boundary differs')
    import numpy as np
    reconstructed=derive(context,np)
    require(all(read(RUN/name)==doc for name,doc in reconstructed.items()),'Full CPU response recomputation differs')
    boundary()
    return {'status':'CPU_RESPONSE_COMPLETE_CACHE_VERIFIED','complete_sha256':digest(RUN/'complete.json'),
        'denominators':reconstructed['coverage.json']['denominators'],'full_saved_response_comparison':True,
        'new_model_forwards':0,'source_RGB_reads':0,'statistical_fits':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify-preparation','run','verify'))
    args=p.parse_args();install_boundary()
    if args.command=='verify-preparation':
        check_preparation();result={'status':'CPU_RESPONSE_PREPARATION_VERIFIED','boundary':boundary()}
    else:result={'prepare':prepare,'run':run,'verify':verify}[args.command]()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
