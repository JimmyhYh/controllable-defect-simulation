"""Independent CPU statistical preparation and explicit HUMAN fitting entry.

prepare/verify-preparation never parse real request rows or derive real moments,
p0, coefficients or predictions. fit requires --allow-statistical-fit. No holdout,
image, heatmap, detector, generation, calibration or training entry exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent.parent
RESPONSE=ROOT/'results/mura_capability/week1_v1/patchcore_fit_response_v1'
OUT=RESPONSE.parent/'patchcore_fit_analysis_v1'
PREP,RUN=OUT/'preparation',OUT/'run'
DESIGN=PROJECT/'计划/Mura检测器响应与SEMU代理刻度研究设计.md'
EVIDENCE=PROJECT/'报告/PatchCore独立D_fit缓存响应后处理结果.md'
RESPONSE_HASH='69f7c691b241a4175f4a8c1f2442fddc1ea6de978cc44572ee278f73d652adae'
RESPONSE_PREP_HASH='77b63940d93e789613e665e49454ad6dbe1280c8335dab2285ade47421e68d94'
TAU=1.852951288223265
L2=.01
OPTIONS={'maxiter':2000,'gtol':1e-8,'ftol':1e-12,'maxcor':10,'maxls':20,'maxfun':15000}
SPECS=[{'id':'C-only','kind':'C-only','k':None},
       {'id':'C+area','kind':'C+area','k':None},
       {'id':'SEMU-only_k0.4','kind':'SEMU-only','k':.4},
       {'id':'C+area+context','kind':'C+area+context','k':None},
       {'id':'SEMU-only_k0.2','kind':'SEMU-only','k':.2},
       {'id':'SEMU-only_k0.8','kind':'SEMU-only','k':.8}]
FORBIDDEN=('torch','torchvision','timm','faiss','patchcore','sklearn','statsmodels',
    'controlled_generation','mura_response','mura_capability.patchcore_fit_response',
    'mura_capability.patchcore_fit_scoring','mura_capability.patchcore_minimal')
MODE=None
ALLOW_FIT=False
COUNTERS={'actual_request_payloads_parsed':0,'actual_standardizations':0,
          'actual_p0_computations':0,'optimizer_calls':0}
BLOCKED=[]


def require(ok,message):
    if not ok:raise RuntimeError(message+'; preserve state; no automatic retry or configuration change')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def safe(p):
    p=Path(p)
    require(p.is_file() and not p.is_symlink() and not p.is_junction()
        and p.absolute()==p.resolve(),'Missing/redirected file: '+str(p))
    return p


def closed(folder,names):
    require(folder.is_dir() and not folder.is_symlink() and not folder.is_junction(),'Missing/redirected cache')
    require({p.name for p in folder.iterdir()}==set(names)|{'complete.json'}
        and all(p.is_file() and not p.is_symlink() and not p.is_junction() for p in folder.iterdir()),
        'Unknown/partial cache members')
    outputs=read(folder/'complete.json')['outputs']
    require(set(outputs)==set(names) and all(digest(safe(folder/n))==h for n,h in outputs.items()),
        'Cache hash differs')


def seal(folder):
    require(not any((folder/n).exists() for n in ('complete.json','failure.json','inflight.json')),
        'Cannot seal incomplete/failed statistical transaction')
    write(folder/'complete.json',{'outputs':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}})


def resources():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
            ('total','available','page','page_available','virtual','virtual_available','extended')]
    s=Memory();s.length=ctypes.sizeof(s);q=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
    q.argtypes=[ctypes.POINTER(Memory)];q.restype=ctypes.c_int
    require(q(ctypes.byref(s)),'RAM query failed')
    out={'available_RAM_GiB':s.available/2**30,'D_free_GiB':shutil.disk_usage(ROOT).free/2**30}
    require(out['available_RAM_GiB']>=2 and out['D_free_GiB']>=40,'CPU RAM/disk admission not met')
    return out


def matches(name):return any(name==p or name.startswith(p+'.') for p in FORBIDDEN)


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if matches(fullname) or ((fullname=='scipy' or fullname.startswith('scipy.')) and MODE!='fit'):
            BLOCKED.append({'import':fullname});require(False,'Import outside explicit CPU fitting mode')


def install_boundary():
    own=OUT.resolve();temps={Path(v).resolve() for k in ('TEMP','TMP') if (v:=os.environ.get(k))}
    def allowed(p):return p.is_relative_to(own) or any(p.is_relative_to(t) for t in temps)
    def deny(item):BLOCKED.append(item);require(False,str(item))
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
                if isinstance(value,(str,bytes)) and not allowed(Path(os.fsdecode(value)).resolve()):
                    deny({'mutation':str(value)})
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        raw=os.fsdecode(args[0])
        if raw.upper() in ('NUL','\\\\.\\NUL','\\\\?\\NUL'):return
        path=Path(raw).resolve();ext=path.suffix.lower()
        mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=any(x in mode for x in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not allowed(path):deny({'write':str(path)})
            return
        if ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.npy','.npz','.pth','.pt','.pkl','.pickle','.faiss','.safetensors'):
            deny({'image_array_or_model_asset':str(path)})
    sys.meta_path.insert(0,ImportGuard());sys.addaudithook(audit)


def boundary():
    loaded=[n for n in sys.modules if matches(n)]
    require(not loaded and not BLOCKED,'CPU scope crossed')
    return {'counters':dict(COUNTERS),'blocked':list(BLOCKED),'model_packages_loaded':loaded,
        'RGB_GT_heatmap_reads':0,'D_hold_reads':0,'detector_forwards':0,'calibrations':0,'renders':0}


def source_context():
    hashes={str(Path(__file__).resolve()):digest(__file__)}
    def bind(p,expected=None):
        p=safe(p);h=digest(p);require(expected is None or h==expected,'Source hash differs: '+str(p));hashes[str(p)]=h
    bind(RESPONSE/'run/complete.json',RESPONSE_HASH)
    bind(RESPONSE/'preparation/complete.json',RESPONSE_PREP_HASH)
    for folder in (RESPONSE/'preparation',RESPONSE/'run'):
        outputs=read(folder/'complete.json')['outputs'];closed(folder,outputs)
        for n,h in outputs.items():
            require(n.endswith('.json') and Path(n).name==n,'Foreign response member');bind(folder/n,h)
    # requests.json and unique_responses.json were only opaque-byte hashed above.
    # Real request decoding is restricted to load_actual_rows in fit/verify-fit.
    src=read(RESPONSE/'preparation/sources.json');cfg=read(RESPONSE/'preparation/protocol.json')
    cv=read(RESPONSE/'run/coverage.json');v=read(RESPONSE/'run/verification.json')
    require(cfg['tau']==src['tau']==TAU and v['passed'] and v['no_new_model_or_statistical_fit']
        and v['boundary']['statistical_fits']==v['boundary']['model_forwards']==0,
        'Frozen response/tau/boundary proof differs')
    require(cv['denominators']['backgrounds']==60 and cv['denominators']['request_references']==7560
        and cv['denominators']['zero_actual']==315 and cv['denominators']['undefined']==0
        and cv['request_states']['all']['H_joint_sum']==3107
        and len(cv['by_background'])==60 and all(r['n']==126 and r['undefined']==0 for r in cv['by_background']),
        'Common-defined/background/label admission metadata differs')
    bind(DESIGN,src['files'][str(DESIGN)])
    bind(ROOT/'mura_capability/patchcore_fit_response.py','886479105d76679c6ab442766f8a31e01b9de6191bd7718c739276db1c5128c7')
    bind(EVIDENCE)
    match=re.search(r'```json\s*\n(.*?)\n```',EVIDENCE.read_text(encoding='utf-8'),re.S)
    require(match is not None,'Independent response verification proof missing')
    proof=json.loads(match.group(1))
    require(proof['status']=='CPU_RESPONSE_COMPLETE_CACHE_VERIFIED'
        and proof['complete_sha256']==RESPONSE_HASH and proof['full_saved_response_comparison']
        and proof['new_model_forwards']==proof['source_RGB_reads']==proof['statistical_fits']==0,
        'Independent full response proof differs')
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy')}
    require(versions=={'numpy':'1.26.4','scipy':'1.16.2'} and sys.version.split()[0]=='3.12.7'
        and Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve(),'Existing CPU environment differs')
    bind(sys.executable,'137c474a645a9471ab3a369084e7cd573cd8b7ff8f81cb85363fffdf5904106c')
    dist=importlib.metadata.distribution('scipy')
    for name in ('scipy/optimize/_minimize.py','scipy/optimize/_lbfgsb_py.py',
                 'scipy/optimize/_lbfgsb.cp312-win_amd64.pyd'):
        bind(dist.locate_file(name))
    return {'files':hashes,'response_complete_sha256':RESPONSE_HASH,'original_tau':TAU,
        'independent_response_proof':proof,'versions':versions,
        'background_order':[r['background_id'] for r in cv['by_background']],
        'geometry_order':src['geometry_ids'],
        'admission_counts':{'backgrounds':60,'common_defined_requests':7560,'zero_actual':315,
                            'H_joint_ones':3107,'H_joint_zeros':4453},
        'real_request_rows_parsed_during_preparation':False}


def protocol():
    return {'target':'H_joint','H_alarm':'descriptive only; never alternate fitting target',
        'tau':TAU,'model_specs':SPECS,'shared_p0':'sum_b(mean_j(H_joint[b,j]))/number_of_backgrounds',
        'eligibility':'all 7560 common-defined D_fit requests; include315 zero; all60 backgrounds',
        'weights':'each background total1/B; row weight1/(B*n_b); no class reweighting',
        'raw_features':{'C-only':['log1p(C_pair_pct)'],
            'C+area':['u','log(A_geo)','u*log(A_geo)'],
            'SEMU-only':['log1p(SEMU_proxy[k])'],
            'C+area+context':['u','log(A_geo)','u*log(A_geo)','log(Lb)',
                            'background_std_Y/Lb','is_bright','is_ellipse_h','is_ellipse_d']},
        'interaction':'raw u*v before centering/scaling; not product of standardized u and v',
        'standardization':'background-weighted D_fit population mean/std(ddof0); continuous only',
        'constant_continuous':'exact raw max==min -> transformed0; saved scale0; no epsilon',
        'indicators':'bright1/dark0; ellipse_h1/otherwise0; ellipse_d1/otherwise0; remain0/1',
        'objective':'sum_i(w_i*logaddexp(0,(1-2*y_i)*z_i)) + 0.01/2*sum(beta[1:]**2)',
        'intercept':'first column1, unpenalized','L2':L2,'optimizer':'SciPy L-BFGS-B',
        'analytic_gradient':True,'initialization':'all coefficients0; no randomness',
        'options':OPTIONS,'inherited_SciPy_defaults':['maxcor10','maxls20','maxfun15000'],
        'stop':'one-class/<40 backgrounds/nonfinite/invalid scale/optimizer not successful; preserve/no retry',
        'rank':'record rank/condition; unpenalized design rank alone is not failure under fixed L2',
        'all_required':'4 main candidates + 2 SEMU-only sensitivities + shared p0; no best-model selection',
        'transaction':'own complete or preserve partial/unknown; no resume; no optimizer in verification',
        'D_hold':'no loading/evaluation entry; separate authorization only after complete fit freeze',
        'resources':{'RAM_GiB':2,'D_free_GiB':40},'fitting_requires_user_manual_launch':True}


def background_weights(ids,np):
    counts=Counter(ids);require(bool(counts),'Empty background set')
    w=np.asarray([1./(len(counts)*counts[b]) for b in ids],dtype=np.float64)
    require(np.isfinite(w).all() and bool((w>0).all()) and abs(float(w.sum())-1)<1e-12,'Invalid background weights')
    return w,counts


def raw_features(rows,spec,np):
    values=[]
    names={'C-only':['u'],'C+area':['u','v','u_times_v'],
        'SEMU-only':['log1p_SEMU'],
        'C+area+context':['u','v','u_times_v','log_Lb','relative_background_std','is_bright','is_ellipse_h','is_ellipse_d']}[spec['kind']]
    for r in rows:
        m=r['measurement'];c=m['C_pair_pct'];a=m['area_pixels'];lb=m['Lb']
        require(m['defined'] and c is not None and c>=0 and a>0 and lb>1e-6
            and m['background_relative_std'] is not None,'Request not common-defined')
        u=math.log1p(c);v=math.log(a)
        if spec['kind']=='C-only':x=[u]
        elif spec['kind']=='C+area':x=[u,v,u*v]
        elif spec['kind']=='SEMU-only':
            semu=m['SEMU_proxy'][str(spec['k'])]
            require(semu is not None and semu>=0,'Undefined sensitivity proxy');x=[math.log1p(semu)]
        else:x=[u,v,u*v,math.log(lb),m['background_relative_std'],int(r['polarity']=='bright'),
                int(r['geometry_id'].endswith('ellipse_h')),int(r['geometry_id'].endswith('ellipse_d'))]
        values.append(x)
    raw=np.asarray(values,dtype=np.float64)
    require(raw.shape==(len(rows),len(names)) and np.isfinite(raw).all(),'Nonfinite raw feature matrix')
    return raw,names,[n.startswith('is_') for n in names]


def standardized(raw,names,binary,w,np):
    out=raw.copy();transforms=[]
    for j,name in enumerate(names):
        x=raw[:,j]
        if binary[j]:
            require(bool(np.isin(x,[0.,1.]).all()),'Indicator not0/1')
            transforms.append({'name':name,'kind':'indicator','mean':None,'scale':None,'constant':False});continue
        constant=bool(x.max()==x.min())
        mean=float(x[0]) if constant else float(np.dot(w,x))
        scale=0. if constant else math.sqrt(float(np.dot(w,np.square(x-mean))))
        require(math.isfinite(mean) and math.isfinite(scale) and (constant or scale>0),
                'Invalid nonconstant standardization; no epsilon rescue')
        out[:,j]=0. if constant else (x-mean)/scale
        transforms.append({'name':name,'kind':'continuous','mean':mean,'scale':scale,'constant':constant})
    require(np.isfinite(out).all(),'Nonfinite standardized features')
    return np.column_stack((np.ones(len(raw)),out)),transforms


def sigmoid(z,np):
    p=np.empty_like(z,dtype=np.float64);positive=z>=0
    p[positive]=1./(1.+np.exp(-z[positive]));e=np.exp(z[~positive]);p[~positive]=e/(1.+e)
    return p


def objective(beta,x,y,w,np,l2=L2):
    z=x@beta;p=sigmoid(z,np)
    loss=float(np.dot(w,np.logaddexp(0.,(1.-2.*y)*z))+.5*l2*np.dot(beta[1:],beta[1:]))
    grad=x.T@(w*(p-y));grad[1:]+=l2*beta[1:]
    require(math.isfinite(loss) and np.isfinite(grad).all(),'Nonfinite objective/gradient')
    return loss,grad


def numerical_validation():
    import numpy as np
    ids=['a','b','b','c','c','c'];w,counts=background_weights(ids,np)
    require(all(abs(float(w[np.asarray(ids)==b].sum())-1/3)<1e-12 for b in counts),'Unequal-group weight check failed')
    rows=[]
    for i,(c,a) in enumerate(zip((0.,.1,2.,1.,3.,7.),(49,53,201,441,453,49))):
        rows.append({'measurement':{'defined':True,'C_pair_pct':c,'area_pixels':a,'Lb':.1+i*.1,
            'background_relative_std':i*.02,'SEMU_proxy':{str(k):c/(1.97/(a*k*k)**.33+.72) for k in (.2,.4,.8)}},
            'geometry_id':('r04_circle','r04_ellipse_h','r08_ellipse_d')[i%3],
            'polarity':'bright' if i%2 else 'dark'})
    y=np.array([0,1,0,0,1,1],np.float64);errors={}
    for spec in SPECS:
        raw,names,binary=raw_features(rows,spec,np);x,t=standardized(raw,names,binary,w,np)
        beta=np.linspace(-.25,.35,x.shape[1]);loss,g=objective(beta,x,y,w,np);eps=1e-6;finite=[]
        for j in range(len(beta)):
            plus=beta.copy();minus=beta.copy();plus[j]+=eps;minus[j]-=eps
            finite.append((objective(plus,x,y,w,np)[0]-objective(minus,x,y,w,np)[0])/(2*eps))
        err=float(np.max(np.abs(g-np.asarray(finite))));require(err<1e-7,'Artificial analytic gradient check failed')
        plain=objective(beta,x,y,w,np,l2=0.)
        require(abs((loss-plain[0])-.005*float(np.dot(beta[1:],beta[1:])))<1e-12
            and abs(g[0]-plain[1][0])<1e-12,'Intercept/penalty convention differs')
        errors[spec['id']]=err
    raw=np.column_stack((np.arange(6.),np.full(6,7.),np.array([0,1,0,1,0,1],float)))
    x,t=standardized(raw,['vary','constant','is_binary'],[False,False,True],w,np)
    require(bool((x[:,2]==0).all()) and t[1]['constant'] and bool((x[:,3]==raw[:,2]).all()),
            'Constant/indicator handling differs')
    extreme=sigmoid(np.array([-1000.,0.,1000.]),np)
    require(bool(np.isfinite(extreme).all()) and extreme.tolist()==[0.,.5,1.],'Stable sigmoid check failed')
    # This is p0 on explicit artificial labels only, not on D_fit.
    artificial_p0=float(np.dot(w,y))
    require(abs(artificial_p0-sum(float(y[np.asarray(ids)==b].mean()) for b in counts)/3)<1e-12,
            'Artificial p0/group weighting differs')
    require(not any(COUNTERS.values()),'Preparation touched actual fitting data/optimizer')
    return {'passed':True,'synthetic_only':True,'optimizer_called':False,
            'analytic_gradient_max_abs_errors':errors,'unequal_background_weights':True,
            'constant_column_zero':True,'indicators_unscaled':True,'intercept_unpenalized':True,
            'raw_interaction_before_standardization':True,'stable_sigmoid_extremes':True,
            'artificial_p0_agrees_with_mean_of_background_means':True}


def prepare():
    source=source_context();snapshot=resources()
    if OUT.exists():return verify_preparation()
    numeric=numerical_validation();require(not any(COUNTERS.values()),'Actual fitting during preparation')
    PREP.mkdir(parents=True,exist_ok=False)
    write(PREP/'sources.json',source);write(PREP/'protocol.json',protocol())
    write(PREP/'numerical_validation.json',numeric)
    write(PREP/'verification.json',{'status':'STATISTICAL_FIT_PREPARED_NOT_EXECUTED',
                                  'boundary':boundary(),'resources':snapshot})
    seal(PREP)
    return {'status':'STATISTICAL_FIT_PREPARED_NOT_EXECUTED','output':str(PREP),
            'boundary':boundary(),'resources':snapshot,'numerical_validation':numeric}


def verify_preparation():
    require(OUT.is_dir() and not OUT.is_symlink() and not OUT.is_junction()
        and {p.name for p in OUT.iterdir()}<={'preparation','run'},'Unknown output root')
    closed(PREP,{'sources.json','protocol.json','numerical_validation.json','verification.json'})
    require(source_context()==read(PREP/'sources.json') and protocol()==read(PREP/'protocol.json'),
            'Bound fitting protocol/source differs')
    b=read(PREP/'verification.json')['boundary']
    require(all(v==0 for v in b['counters'].values()) and not b['blocked'] and not b['model_packages_loaded']
        and read(PREP/'numerical_validation.json')['passed'],'Saved preparation crossed actual-data boundary')
    return {'status':'STATISTICAL_PREPARATION_CACHE_VERIFIED','complete_sha256':digest(PREP/'complete.json'),
            'actual_fit_executed':False,'current_counters':dict(COUNTERS)}


def load_actual_rows(source):
    require(MODE in ('fit','verify-fit'),'Real request payload/actual transforms forbidden in preparation')
    COUNTERS['actual_request_payloads_parsed']+=1
    rows=read(RESPONSE/'run/requests.json')['rows']
    backgrounds=source['background_order'];gids=source['geometry_order']
    expected=[f'{b}_{g}_{p}_a{i}' for b in backgrounds for g in gids for p in ('bright','dark') for i in range(7)]
    require(len(rows)==7560 and [r['id'] for r in rows]==expected,'Actual row coverage/order differs')
    require(len(backgrounds)==60 and len(set(backgrounds))==60 and len(backgrounds)>=40,'Insufficient backgrounds')
    zero=0
    for r in rows:
        m=r['measurement'];require(r['measurement_defined'] and m['defined'] and r['H_joint'] in (0,1)
            and r['H_alarm'] in (0,1) and r['H_joint']<=r['H_alarm'] and r['retained']
            and r['role']=='D_fit60' and r['background_id'] in backgrounds,'Invalid target/common-defined input')
        require(m['C_pair_pct'] is not None and m['Lb']>1e-6 and m['area_pixels']>0
            and m['background_relative_std'] is not None and m['background_relative_std']>=0,
            'Invalid common-defined feature domain; do not drop rows')
        for k in (.2,.4,.8):
            require(m['SEMU_proxy'][str(k)] is not None and math.isfinite(m['SEMU_proxy'][str(k)])
                and m['SEMU_proxy'][str(k)]>=0,'Undefined sensitivity feature; no subset rescue')
        zero+=int(r['zero_actual_change'])
        if r['zero_actual_change']:require(r['H_joint']==0,'Zero-control target differs')
    require(zero==315 and len({r['H_joint'] for r in rows})==2,'Zero count/one-class target: stop before fitting')
    return rows


def fitted_data(rows,spec,w,np):
    raw,names,binary=raw_features(rows,spec,np)
    COUNTERS['actual_standardizations']+=1
    x,t=standardized(raw,names,binary,w,np)
    return x,t


def solve_one(x,y,w,spec,transforms,np):
    require(MODE=='fit' and ALLOW_FIT,'Optimizer requires explicit human fit flag')
    from scipy.optimize import minimize
    COUNTERS['optimizer_calls']+=1
    result=minimize(lambda beta:objective(beta,x,y,w,np),np.zeros(x.shape[1],np.float64),
                    method='L-BFGS-B',jac=True,options=OPTIONS.copy())
    beta=np.asarray(result.x,np.float64);loss,g=objective(beta,x,y,w,np)
    rank=int(np.linalg.matrix_rank(x));condition=float(np.linalg.cond(x))
    return {'spec':spec,'transforms':transforms,'coefficients':beta.tolist(),
        'coefficient_order':['intercept']+[t['name'] for t in transforms],
        'objective':loss,'gradient':g.tolist(),'gradient_inf_norm':float(np.abs(g).max()),
        'optimizer_success':bool(result.success),'status':int(result.status),'message':str(result.message),
        'iterations':int(result.nit),'function_evaluations':int(result.nfev),
        'design_rank':rank,'design_columns':int(x.shape[1]),
        'design_condition':condition if math.isfinite(condition) else None,
        'design_condition_status':'finite' if math.isfinite(condition) else 'singular_or_infinite'}


FIT_OUTPUTS={'started.json','data_binding.json','constant_reference.json','models.json',
             'fitted_predictions.json','verification.json'}


def fit():
    require(MODE=='fit' and ALLOW_FIT,'Fit must be manually launched with --allow-statistical-fit')
    verify_preparation()
    if RUN.exists():return verify_fit()
    source=read(PREP/'sources.json');snapshot=resources();RUN.mkdir(exist_ok=False)
    write(RUN/'started.json',{'preparation_sha256':digest(PREP/'complete.json'),
          'human_fit_flag':True,'resources':snapshot,'CPU_statistical_fitting_not_detector_training':True})
    write(RUN/'inflight.json',{'operation':'manual_statistical_fit'})
    started=time.perf_counter();models=[];current='data_admission'
    try:
        import numpy as np
        rows=load_actual_rows(source);y=np.asarray([r['H_joint'] for r in rows],np.float64)
        w,counts=background_weights([r['background_id'] for r in rows],np)
        require(set(counts)==set(source['background_order']) and all(n==126 for n in counts.values()),'Group counts differ')
        COUNTERS['actual_p0_computations']+=1;p0=float(np.dot(w,y))
        require(0<p0<1 and math.isfinite(p0),'One-class/invalid p0; stop')
        binding={'request_ids':[r['id'] for r in rows],'backgrounds':source['background_order'],
            'background_counts':dict(counts),'row_weights':w.tolist(),'target':'H_joint',
            'nominal_rows':7560,'common_defined_rows':len(rows),'zero_actual':315}
        write(RUN/'data_binding.json',binding)
        write(RUN/'constant_reference.json',{'p0':p0,'same_weights_and_rows_as_all_models':True,
              'D_fit_only_not_validation':True})
        predictions={'request_ids':binding['request_ids'],'target':y.tolist(),'p0':[p0]*len(rows),'models':{}}
        for spec in SPECS:
            current=spec['id'];resources();x,t=fitted_data(rows,spec,w,np)
            model=solve_one(x,y,w,spec,t,np);models.append(model)
            require(model['optimizer_success'],'Optimizer not converged for '+current)
            values=sigmoid(x@np.asarray(model['coefficients'],np.float64),np)
            require(np.isfinite(values).all(),'Nonfinite fitted probabilities')
            predictions['models'][current]=values.tolist()
            print(json.dumps({'candidate':current,'optimizer_success':True,'iterations':model['iterations']}),flush=True)
        require(COUNTERS['optimizer_calls']==6 and len(models)==6,'Missing predeclared fitting candidate')
        write(RUN/'models.json',{'models':models,'constant_reference':'constant_reference.json'})
        write(RUN/'fitted_predictions.json',predictions)
        require(source_context()==source,'Sources changed during fitting')
        write(RUN/'verification.json',{'status':'ALL_DECLARED_D_FIT_MODELS_FROZEN',
            'boundary':boundary(),'elapsed_seconds':time.perf_counter()-started,'resources_after':resources(),
            'D_fit_only_not_holdout_or_real_validation':True,'D_hold_not_automatically_authorized':True})
        (RUN/'inflight.json').unlink();seal(RUN)
    except BaseException as exc:
        if not (RUN/'failure.json').exists():write(RUN/'failure.json',{'error':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'current':current,
            'partial_or_failed_models':models,'counters':dict(COUNTERS),
            'D_hold_blocked':True,'no_automatic_retry_resume_or_best_weight':True})
        raise
    return {'status':'ALL_DECLARED_D_FIT_MODELS_FROZEN','output':str(RUN),'optimizer_calls':6,
            'not_D_hold_validation':True}


def verify_fit():
    verify_preparation();closed(RUN,FIT_OUTPUTS);resources()
    source=read(PREP/'sources.json');started=read(RUN/'started.json')
    require(started['preparation_sha256']==digest(PREP/'complete.json') and started['human_fit_flag'],
            'Saved fitting admission differs')
    import numpy as np
    rows=load_actual_rows(source);y=np.asarray([r['H_joint'] for r in rows],np.float64)
    w,counts=background_weights([r['background_id'] for r in rows],np)
    b=read(RUN/'data_binding.json');constant=read(RUN/'constant_reference.json')
    COUNTERS['actual_p0_computations']+=1
    p0=float(np.dot(w,y));require(constant['p0']==p0 and b['row_weights']==w.tolist()
        and b['request_ids']==[r['id'] for r in rows] and b['background_counts']==dict(counts),
        'Saved row/weight/p0 differs')
    saved=read(RUN/'models.json')['models'];pred=read(RUN/'fitted_predictions.json')
    require(len(saved)==6 and set(pred['models'])=={s['id'] for s in SPECS}
        and pred['request_ids']==b['request_ids'] and pred['target']==y.tolist() and pred['p0']==[p0]*len(rows),
        'Saved model/prediction coverage differs')
    for spec,model in zip(SPECS,saved):
        x,t=fitted_data(rows,spec,w,np);beta=np.asarray(model['coefficients'],np.float64)
        require(model['spec']==spec and model['transforms']==t and beta.shape==(x.shape[1],)
            and np.isfinite(beta).all() and model['optimizer_success'],'Saved coefficients/transform/status differs')
        value,g=objective(beta,x,y,w,np)
        require(math.isclose(value,model['objective'],rel_tol=1e-10,abs_tol=1e-10)
            and np.allclose(g,model['gradient'],rtol=1e-10,atol=1e-10)
            and pred['models'][spec['id']]==sigmoid(x@beta,np).tolist(),'Saved objective/gradient/prediction differs')
    v=read(RUN/'verification.json')
    require(v['status']=='ALL_DECLARED_D_FIT_MODELS_FROZEN' and v['boundary']['counters']['optimizer_calls']==6
        and not v['boundary']['blocked'] and not v['boundary']['model_packages_loaded']
        and COUNTERS['optimizer_calls']==0,'Saved fitting scope or verification optimizer counter differs')
    boundary()
    return {'status':'STATISTICAL_FIT_COMPLETE_CACHE_VERIFIED','complete_sha256':digest(RUN/'complete.json'),
        'new_optimizer_calls':0,'actual_statistics_recomputed_for_verification':True,
        'D_hold_or_real_reads':0,'detector_forwards':0}


def main():
    global MODE,ALLOW_FIT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify-preparation','fit','verify-fit'))
    p.add_argument('--allow-statistical-fit',action='store_true')
    args=p.parse_args();MODE=args.command;ALLOW_FIT=args.allow_statistical_fit
    if MODE=='fit' and not ALLOW_FIT:p.error('fit requires explicit --allow-statistical-fit; user manual launch only')
    install_boundary()
    functions={'prepare':prepare,'verify-preparation':verify_preparation,'fit':fit,'verify-fit':verify_fit}
    print(json.dumps(functions[MODE](),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
