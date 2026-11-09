"""Independent frozen D_fit scoring; reuse two complete minimal outputs.

No training, generation, calibration, response statistics or partial resumption.
All real model work occurs only in the explicit run command.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from . import patchcore_minimal as m

ROOT, BASE, FIT, TAU = m.ROOT, m.BASE, m.FIT, m.TAU
OUT = ROOT / 'results/mura_capability/week1_v1/patchcore_fit_scoring_v1'
PREP, RUN = OUT / 'preparation', OUT / 'run'
MINIMAL_CODE = '479fd19ac5f2694d5ab81dd9b0722a095241288c985d2c0834fc74f7acf62924'
MINIMAL_COMPLETE = '51472f150d69d2d206fd26990a56ea95d35d6a9440ce1e7212c0e4bdffd038b6'
COUNTERS = {'model_loads': 0, 'tensor_loads': 0,
            'author_placeholder_forwards': 0, 'model_input_forwards': 0}
EVENTS = {'source_RGB_reads': [], 'NUL_opens': 0, 'temporary_write_opens': [],
          'system_queries': [], 'blocked': []}
require, read, write, digest = m.require, m.read, m.write, m.digest


def source_binding():
    require(digest(m.__file__) == MINIMAL_CODE and digest(m.RUN/'complete.json') == MINIMAL_COMPLETE,
            'Minimal implementation/result identity differs')
    m.verify_preparation()
    minimal_sources = read(m.PREP/'sources.json')
    m.closed(m.RUN, {'started.json', 'state.json', 'result.json'} |
             {r['id']+'.npy' for r in minimal_sources['pair']})
    previous = read(m.RUN/'result.json'); states = read(m.RUN/'state.json')
    require(previous['status'] == 'MINIMAL_TWO_INPUTS_COMPLETE'
            and states['before'] == states['after'] == minimal_sources['normal_state']
            and previous['counters'] == {'model_loads':1,'tensor_loads':1,
                'author_placeholder_forwards':1,'model_input_forwards':2}
            and not previous['events']['blocked'], 'Minimal saved model/boundary differs')
    require(read(m.RUN/'started.json')['preparation_sha256'] == digest(m.PREP/'complete.json'),
            'Minimal run/preparation binding differs')
    metadata = BASE/'fit_scoring_preparation_v1'
    unique = read(metadata/'unique_inputs.json')['rows']
    refs = read(metadata/'references.json')['rows']
    roles = read(BASE/'preparation_v1/roles.json')['roles']['D_fit60']
    manifest = read(FIT/'manifest.json')['rows']
    frozen = read(BASE/'preparation_v1/requests.json')['rows']
    frozen = [r for r in frozen if r['role']=='D_fit60']
    require(len(unique)==7305 and len(refs)==7620 and len(roles)==60
            and len(manifest)==len(frozen)==7560, 'D_fit denominators differ')
    require([r['id'] for r in manifest]==[r['id'] for r in frozen]
            and [r['id'] for r in refs[:60]]==[r['id'] for r in roles]
            and [r['id'] for r in refs[60:]]==[r['id'] for r in frozen], 'Frozen order differs')
    by_background = {}
    for row, spec in zip(manifest, frozen):
        require(all(row[k]==spec[k] for k in ('id','background_id','geometry_id','polarity','amplitude')),
                'Request semantic identity differs')
        old = by_background.setdefault(row['background_id'], row['background_image'])
        require(old==row['background_image'], 'Background image reference differs')
    expected_refs = [{'id':r['id'],'kind':'baseline','background_id':r['id'],
                      **by_background[r['id']]} for r in roles]
    expected_refs += [{'id':r['id'],'kind':'request','background_id':r['background_id'],
        'geometry_id':r['geometry_id'],'geometry_file':r['geometry_file'],
        'zero_actual_change':r['measurement']['zero_actual_change'],
        'measurement_defined':r['measurement']['defined'],**r['image']} for r in manifest]
    require(expected_refs==refs and len({r['id'] for r in refs})==7620, 'Full reference map differs')
    ref_ids = defaultdict(list)
    for r in refs:
        ref_ids[r['working_RGB_sha256']].append(r['id'])
    input_complete = read(FIT/'complete.json')['outputs']
    reused = {r['working_RGB_sha256']:r for r in previous['rows']}
    require(len(reused)==len(previous['rows'])==2, 'Minimal reuse count differs')
    for saved, original in zip(previous['rows'],minimal_sources['pair']):
        require(all(saved[k]==v for k,v in original.items()) and saved['state_unchanged']
                and saved['tau']==TAU and math.isfinite(saved['score'])
                and saved['above_tau']==(saved['score']>TAU)
                and digest(m.RUN/saved['heatmap'])==saved['heatmap_sha256'], 'Reuse identity differs')
    planned = []
    for i,row in enumerate(unique):
        h=row['working_RGB_sha256']
        require(row['ordinal']==i and row['file']==f'images/{h}.npy'
                and row['file_sha256']==input_complete[row['file']]
                and row['references']==ref_ids[h] and row['reuse_normal_id'] is None,
                'Unique RGB/order/coverage/hash differs')
        require((FIT/row['file']).is_file(), 'Frozen input missing')
        reuse = reused.get(h)
        if reuse:
            require(row['file']==reuse['file'] and row['file_sha256']==reuse['file_sha256'],
                    'Reused RGB bytes differ')
        planned.append({**row, 'reuse_minimal':reuse})
    require(set(ref_ids)=={r['working_RGB_sha256'] for r in planned}
            and sum(len(r['references']) for r in planned)==7620
            and sum(r['reuse_minimal'] is not None for r in planned)==2, 'Dedup/reuse coverage differs')
    denominators = {'backgrounds':60,'baseline_references':60,'request_references':7560,
        'all_references':7620,'unique_float_RGB':7305,'reused_minimal_outputs':2,
        'new_model_input_forwards':7303,'new_author_ones_probes':1,
        'zero_actual_change_requests':sum(r['measurement']['zero_actual_change'] for r in manifest),
        'nonzero_actual_change_requests':sum(not r['measurement']['zero_actual_change'] for r in manifest),
        'undefined_measurement_requests':sum(not r['measurement']['defined'] for r in manifest),
        'independent_background_count_not_references':60}
    require(denominators['zero_actual_change_requests']==315
            and denominators['nonzero_actual_change_requests']==7245
            and denominators['undefined_measurement_requests']==0,'Residual metadata coverage differs')
    return {'minimal_sources':minimal_sources,'minimal_preparation_sha256':digest(m.PREP/'complete.json'),
        'minimal_complete_sha256':MINIMAL_COMPLETE,'own_implementation_sha256':digest(__file__),
        'denominators':denominators,'tau':TAU}, planned, refs


def check_preparation():
    require(OUT.is_dir() and not OUT.is_symlink() and not OUT.is_junction()
            and {p.name for p in OUT.iterdir()} <= {'preparation','run'}, 'Unknown output root')
    m.closed(PREP, {'sources.json','protocol.json','unique_inputs.json','references.json'})
    sources, unique, refs = source_binding()
    require(read(PREP/'sources.json')==sources and read(PREP/'unique_inputs.json')['rows']==unique
            and read(PREP/'references.json')['rows']==refs, 'Preparation/input/source changed')
    cfg=read(PREP/'protocol.json')
    require(cfg['denominators']==sources['denominators'] and cfg['tau']==TAU
            and cfg['score_rule']=='strictly greater' and cfg['state_check']=='full signature after every new input'
            and cfg['CPU_response_statistics'] is False, 'Frozen scoring protocol differs')
    return sources, unique, refs


def prepare():
    sources, unique, refs=source_binding(); snapshot=m.resources(model=True)
    if OUT.exists():
        check_preparation()
        return {'status':'FULL_FIT_PREPARATION_VERIFIED','denominators':sources['denominators']}
    # Full input decode verification is consumed from prior frozen evidence; only
    # the two reused model heatmaps are inspected here, never any new input RGB.
    m.verify()
    PREP.mkdir(parents=True,exist_ok=False)
    write(PREP/'sources.json',sources)
    write(PREP/'unique_inputs.json',{'rows':unique})
    write(PREP/'references.json',{'rows':refs})
    write(PREP/'protocol.json',{'model':'PatchCore','tau':TAU,'score_rule':'strictly greater',
        'denominators':sources['denominators'],'state_check':'full signature after every new input',
        'input':'original HWC float32 [0,1], no PNG/no extra division/no rendering',
        'runtime':'original native.build_model/predict_rgb; eval/FP32/batch1/workers0; seed42',
        'CPU_response_statistics':False,'partial_transaction':'preserve and stop, never resume/retry',
        'startup_RAM_GiB':4.5,'during_run_RAM_GiB':2,'free_D_GiB':40,
        'initial_free_GPU_GiB':4,'resources_at_prepare':snapshot})
    m.seal(PREP)
    return {'status':'FULL_FIT_PREPARATION_COMPLETE','denominators':sources['denominators'],
            'resources':snapshot}


def install_boundary(unique):
    allowed={(FIT/r['file']).resolve() for r in unique}
    reuse_heat={(m.RUN/r['reuse_minimal']['heatmap']).resolve() for r in unique if r['reuse_minimal']}
    assets={p.resolve() for p in (ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth',
        ROOT/'models/weak_localization/patchcore/nnscorer_search_index.faiss',
        ROOT/'models/weak_localization/patchcore/patchcore_params.pkl')}
    temp_roots={Path(v).resolve() for k in ('TEMP','TMP') if (v:=os.environ.get(k))}
    platform_source=(Path(sys.base_prefix)/'Lib/platform.py').resolve()
    ownroot=RUN.resolve()
    def own(p):return p.is_relative_to(ownroot)
    def temporary(p):return any(p.is_relative_to(t) for t in temp_roots)
    def deny(message):
        EVENTS['blocked'].append(message);require(False,message)
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','pickle.find_class'):
            deny('Network/unrestricted pickle denied: '+event)
        if event=='subprocess.Popen':
            f=sys._getframe(1);good=False
            while f:
                if f.f_code.co_name=='_syscmd_ver' and Path(f.f_code.co_filename).resolve()==platform_source and f.f_locals.get('cmd') in ('ver','command /c ver','cmd /c ver'):
                    good=True;break
                f=f.f_back
            if not good:deny('Unlisted subprocess: '+str(args[:2]))
            EVENTS['system_queries'].append({'executable':str(args[0]),'command':str(args[1])})
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename'):
            for item in (args[:2] if event=='os.rename' else args[:1]):
                if isinstance(item,(str,bytes)):
                    p=Path(os.fsdecode(item)).resolve()
                    if not own(p) and not temporary(p):deny('Mutation outside own result/temp: '+str(p))
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        raw=os.fsdecode(args[0])
        if raw.upper() in ('NUL','\\\\.\\NUL','\\\\?\\NUL'):
            EVENTS['NUL_opens']+=1;return
        p=Path(raw).resolve();ext=p.suffix.lower()
        mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not own(p) and not temporary(p):deny('Write outside own result/temp: '+str(p))
            if temporary(p):EVENTS['temporary_write_opens'].append(str(p))
            return
        if ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):deny('No original RGB/GT reads: '+str(p))
        if ext in ('.npy','.npz') and not own(p):
            if p in allowed:EVENTS['source_RGB_reads'].append(str(p))
            elif p not in reuse_heat:deny('Unlisted input/probability array: '+str(p))
        if ext in ('.pth','.pt','.pkl','.pickle','.faiss','.safetensors') and p not in assets:
            deny('Unlisted model asset: '+str(p))
    sys.addaudithook(audit)


def validate_heat(score,heat,np):
    require(math.isfinite(score) and heat.dtype==np.float32 and heat.shape==(256,256)
            and np.isfinite(heat).all(),'Native output shape/finiteness differs')


def names(unique):
    return {'started.json','references.json','result.json','verification.json'} | {
        'records/'+r['working_RGB_sha256']+'.json' for r in unique} | {
        'heatmaps/'+r['working_RGB_sha256']+'.npy' for r in unique if not r['reuse_minimal']}


def closed_run(unique):
    require(RUN.is_dir() and not RUN.is_symlink() and not RUN.is_junction(),'Missing/redirected run')
    require({p.name for p in RUN.iterdir() if p.is_dir()}=={'records','heatmaps'},'Unknown run directories')
    expected=names(unique)
    actual={p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()}
    require(actual==expected|{'complete.json'} and not any(p.is_symlink() or p.is_junction() for p in RUN.rglob('*')),
            'Partial/unknown/redirected transaction; no automatic continuation')
    outputs=read(RUN/'complete.json')['outputs']
    require(set(outputs)==expected and all(digest(m.safe_file(RUN/n))==h for n,h in outputs.items()),
            'Saved output hash/coverage differs')


def run():
    sources,unique,refs=check_preparation()
    if RUN.exists():return verify()
    snapshot=m.resources(model=True)
    require(not any(COUNTERS.values()),'Fresh model process required')
    RUN.mkdir(exist_ok=False);(RUN/'records').mkdir();(RUN/'heatmaps').mkdir()
    write(RUN/'started.json',{'preparation_sha256':digest(PREP/'complete.json'),'pid':os.getpid(),
        'resources':snapshot,'temporary_environment':{k:os.environ.get(k) for k in ('TEMP','TMP','TMPDIR')}})
    write(RUN/'references.json',{'rows':refs})
    install_boundary(unique)
    started=time.perf_counter();completed=0;reused_count=0;current='model_construction';timings=[]
    try:
        write(RUN/'inflight.json',{'operation':current})
        import numpy as np
        from mura_response import inference_runtime as native
        torch,modules,handle=native.build_model(SimpleNamespace(ROOT=ROOT,COUNTERS=COUNTERS,require=require),'PatchCore',{})
        def state():return json.loads(json.dumps(native.state_signature(torch,modules),allow_nan=False))
        initial=state()
        require(initial==sources['minimal_sources']['normal_state'],'Model/BN differs from normal/minimal state')
        (RUN/'inflight.json').unlink();constructed=time.perf_counter()
        for row in unique:
            current=row['working_RGB_sha256'];reuse=row['reuse_minimal']
            m.resources()  # Original full-scoring protocol: start4.5GiB, runtime2GiB.
            if reuse:
                heatpath=m.RUN/reuse['heatmap'];payload=heatpath.read_bytes()
                require(hashlib.sha256(payload).hexdigest()==reuse['heatmap_sha256'],'Reuse heatmap changed')
                heat=np.load(io.BytesIO(payload),allow_pickle=False);score=reuse['score']
                validate_heat(score,heat,np);reused_count+=1
                record={'heatmap_path':str(heatpath),'heatmap_sha256':reuse['heatmap_sha256'],
                        'source':'minimal','reused_minimal_id':reuse['id']}
            else:
                payload=(FIT/row['file']).read_bytes()
                require(hashlib.sha256(payload).hexdigest()==row['file_sha256'],'Frozen input file changed')
                rgb=np.load(io.BytesIO(payload),allow_pickle=False)
                require(rgb.dtype==np.float32 and rgb.shape==(256,256,3) and rgb.flags.c_contiguous
                        and np.isfinite(rgb).all() and 0<=float(rgb.min())<=float(rgb.max())<=1
                        and m.rgb_identity(rgb)==current,'Canonical RGB hash/domain differs')
                write(RUN/'inflight.json',{'operation':'unique_RGB_forward','ordinal':row['ordinal'],
                                          'working_RGB_sha256':current})
                COUNTERS['model_input_forwards']+=1;t=time.perf_counter()
                score,heat=native.predict_rgb(torch,'PatchCore',handle,rgb);torch.cuda.synchronize()
                validate_heat(score,heat,np)
                require(state()==initial,'Parameters/BN/training/grad changed after input')
                target=RUN/'heatmaps'/f'{current}.npy'
                with target.open('xb') as stream:np.save(stream,heat,allow_pickle=False)
                record={'heatmap_path':str(target),'heatmap_sha256':digest(target),
                        'source':'new_forward','reused_minimal_id':None}
                timings.append(time.perf_counter()-t)
            record.update(ordinal=row['ordinal'],working_RGB_sha256=current,
                input_file_sha256=row['file_sha256'],score=score,tau=TAU,
                state_unchanged=True,heat_min=float(heat.min()),heat_max=float(heat.max()))
            write(RUN/'records'/f'{current}.json',record)
            if (RUN/'inflight.json').exists():(RUN/'inflight.json').unlink()
            completed+=1
            if completed==1 or completed%25==0 or completed==len(unique):
                print(json.dumps({'unique_completed':completed,'unique_total':7305,
                    'new_input_forwards':COUNTERS['model_input_forwards'],'new_total':7303,
                    'reused':reused_count,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        require(COUNTERS=={'model_loads':1,'tensor_loads':1,'author_placeholder_forwards':1,
                          'model_input_forwards':7303} and reused_count==2 and completed==7305,
                'Actual scoring/reuse denominators differ')
        expected_reads=[str((FIT/r['file']).resolve()) for r in unique if not r['reuse_minimal']]
        require(EVENTS['source_RGB_reads']==expected_reads and not EVENTS['blocked'],'Actual RGB read order differs')
        require(state()==initial,'Final model state changed')
        write(RUN/'result.json',{'status':'FULL_FIT_SCORING_COMPLETE','denominators':sources['denominators'],
            'counters':COUNTERS,'reused':reused_count,'tau':TAU,'elapsed_seconds':time.perf_counter()-started,
            'construction_seconds':constructed-started,'new_input_scoring_state_IO_seconds':sum(timings),
            'CPU_response_statistics_performed':False,'no_training_generation_calibration_fitting':True})
        write(RUN/'verification.json',{'passed':True,'state_before':initial,'state_after':state(),
            'events':EVENTS,'resources_after':m.resources(),'GPU':torch.cuda.get_device_name(),
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'original_RGB_GT_reads':0,'new_RGB_reads':len(EVENTS['source_RGB_reads'])})
        require(source_binding()[0]==sources,'Provenance changed during scoring')
        m.seal(RUN)
    except BaseException as exc:
        if not (RUN/'failure.json').exists():
            write(RUN/'failure.json',{'error':type(exc).__name__,'message':str(exc),
                'traceback':traceback.format_exc(),'current':current,'completed_unique':completed,
                'reused':reused_count,'counters':COUNTERS,'events':EVENTS,'no_automatic_retry_resume':True})
        raise
    return {'status':'FULL_FIT_SCORING_COMPLETE','output':str(RUN),'denominators':sources['denominators'],
            'counters':COUNTERS}


def verify():
    sources,unique,refs=check_preparation();closed_run(unique)
    import numpy as np
    result=read(RUN/'result.json');v=read(RUN/'verification.json')
    require(result['status']=='FULL_FIT_SCORING_COMPLETE' and result['denominators']==sources['denominators']
            and result['counters']=={'model_loads':1,'tensor_loads':1,'author_placeholder_forwards':1,'model_input_forwards':7303}
            and result['reused']==2 and result['tau']==TAU and not result['CPU_response_statistics_performed']
            and result['no_training_generation_calibration_fitting'] and v['passed']
            and v['state_before']==v['state_after']==sources['minimal_sources']['normal_state'],
            'Saved full result/state/counters differ')
    require(read(RUN/'references.json')['rows']==refs
            and read(RUN/'started.json')['preparation_sha256']==digest(PREP/'complete.json'),
            'Saved reference/source binding differs')
    require(not v['events']['blocked'] and v['original_RGB_GT_reads']==0 and v['new_RGB_reads']==7303
            and v['events']['source_RGB_reads']==[str((FIT/r['file']).resolve()) for r in unique if not r['reuse_minimal']],
            'Saved input boundary differs')
    for row in unique:
        h=row['working_RGB_sha256'];record=read(RUN/'records'/f'{h}.json');reuse=row['reuse_minimal']
        expected_path=m.RUN/reuse['heatmap'] if reuse else RUN/'heatmaps'/f'{h}.npy'
        require(record['ordinal']==row['ordinal'] and record['working_RGB_sha256']==h
                and record['input_file_sha256']==row['file_sha256'] and record['tau']==TAU
                and record['state_unchanged'] and record['heatmap_path']==str(expected_path)
                and record['source']==('minimal' if reuse else 'new_forward')
                and record['reused_minimal_id']==(reuse['id'] if reuse else None)
                and digest(m.safe_file(expected_path))==record['heatmap_sha256'],'Saved record identity differs')
        if reuse:require(record['score']==reuse['score'] and record['heatmap_sha256']==reuse['heatmap_sha256'],
                         'Reused output value differs')
        heat=np.load(expected_path,allow_pickle=False);validate_heat(record['score'],heat,np)
        require(float(heat.min())==record['heat_min'] and float(heat.max())==record['heat_max'],'Heat range differs')
    require(not any(COUNTERS.values()) and 'torch' not in sys.modules,'Verify must be a fresh CPU process')
    return {'status':'FULL_FIT_COMPLETE_CACHE_VERIFIED','complete_sha256':digest(RUN/'complete.json'),
            'denominators':sources['denominators'],'new_model_input_forwards':0,
            'new_source_RGB_reads':0,'tau_recomputed':False,'CPU_response_statistics_performed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify-preparation','run','verify'))
    args=parser.parse_args()
    if args.command=='verify-preparation':
        sources,_,_=check_preparation();result={'status':'FULL_FIT_PREPARATION_VERIFIED','denominators':sources['denominators']}
    else:result={'prepare':prepare,'run':run,'verify':verify}[args.command]()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
