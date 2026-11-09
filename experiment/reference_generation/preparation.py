"""Day1 CPU preparation and read-only identity/geometry verification."""
import ctypes
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .coordinates import VARIANTS, build_geometry
from .reference import prepare_reference, read_mask, read_rgb, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/reference_generation/v1'
SOURCES = ['results/weak_localization/dataset_manifest.json',
           'data/weak_localization/split.json', 'data/controlled_generation/split.json']
CORE = ['__init__.py', 'reference.py', 'coordinates.py', 'preparation.py']
CANDIDATES = ['019', '024', '020', '023', '021']
REASONS = {
    '019': 'Dark blue-gray context; visible diffuse pale patch centered on annotation.',
    '024': 'Similar dark context and pale rounded diffuse patch; edge brightening remains in context.',
    '020': 'Dark blue-gray context with low-contrast diffuse pale patch matching annotation.',
    '023': 'Similar dark context and diffuse pale patch; fifth-stage development candidate, not independent evidence.',
    '021': 'Similar dark context and weak diffuse pale patch; physical source relationships unknown.'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
            (x,ctypes.c_ulonglong) for x in ('total','available','page_total','page_available',
                                          'virtual_total','virtual_available','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx failed')
    return {'python':platform.python_version(), 'device':'CPU',
            'total_RAM_GiB':m.total/2**30, 'available_RAM_GiB':m.available/2**30,
            'free_disk_GiB':shutil.disk_usage(ROOT).free/2**30,
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','Pillow','opencv-python')},
            'GPU_readonly_query':subprocess.check_output(['nvidia-smi',
                '--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],
                text=True).strip(), 'torch_loaded':'torch' in sys.modules,
            'model_forwards':0,'optimizer_steps':0}


def metadata():
    manifest, old, cg = [read_json(ROOT/p) for p in SOURCES]
    if sha256(ROOT/SOURCES[0]) != old['manifest_sha256'] or sha256(ROOT/SOURCES[1]) != cg['old_split_sha256']:
        raise ValueError('Existing metadata identity mismatch')
    if sha256(ROOT/SOURCES[0]) != cg['manifest_sha256']:
        raise ValueError('Controlled pool manifest mismatch')
    return {x['path']:x for x in manifest['files']},old,cg


def pick_backgrounds(rows, old, cg):
    paths = [p for key in ('N','Vnew','Tfinal','excluded_train') for p in old[key]]
    paths += [p for key in ('N_gen','S_bg','Q_cal') for p in cg[key]]
    excluded = {rows[p]['rgb_content_sha256'] for p in paths}
    groups = {}
    for p in old['unused_normal_train']:
        if not p.startswith('display/train/good/'):
            raise ValueError('Unexpected normal candidate path')
        group=rows[p]['rgb_content_sha256']
        if group not in excluded:
            groups.setdefault(group,[]).append(p)
    names=sorted(groups)
    order=np.random.Generator(np.random.PCG64(20261016)).permutation(len(names))
    if len(names)<2:
        raise ValueError('Insufficient unexposed normal groups')
    return [min(groups[names[i]]) for i in order[:2]],len(groups),excluded


def source_record(relative, rows):
    return {'path':relative, **rows[relative]}


def prepare_inputs():
    if (OUT/'completion.json').exists():
        return verify_inputs()
    allowed = {'source_guard.json','mura_source_audit.json','mura_candidate_review.png','background_review.png',
               'preparation_stopped_before_outputs.txt'}
    if not OUT.is_dir() or {p.name for p in OUT.iterdir()}-allowed:
        raise RuntimeError('Unknown or incomplete preparation outputs: preserve and investigate')
    marker=OUT/'preparation.incomplete'
    with marker.open('x',encoding='utf-8') as f:
        f.write('Day1 CPU input transaction; do not automatically resume or overwrite.\n')
    rows,old,cg=metadata(); data=Path(old['root'])
    background_paths,count,excluded=pick_backgrounds(rows,old,cg)
    backgrounds={p:read_rgb(data/p,rows[p]) for p in background_paths}
    candidates=[]; refs={}; masks={}
    audit=read_json(OUT/'mura_source_audit.json')
    eligible=[x for x in audit if x['eligible']]
    eligible.sort(key=lambda x:(x['group'],x['path']))
    chosen_set={f'display/test/mura/{n}.png' for n in CANDIDATES}
    ordered=[x for x in eligible if x['path'] in chosen_set]
    if [x['path'] for x in ordered] != [f'display/test/mura/{n}.png' for n in CANDIDATES]:
        raise ValueError('Candidate order differs from reviewed RGB group order')
    for n in CANDIDATES:
        p=f'display/test/mura/{n}.png';mp=f'display/ground_truth/mura/{n}_mask.png'
        rgb=read_rgb(data/p,rows[p]);mask=read_mask(data/mp,rows[mp])
        prepared=prepare_reference(rgb,mask,{'context_factor':4,'model_side':512})
        folder=OUT/'references'/n;folder.mkdir(parents=True)
        Image.fromarray(prepared['rgb']).save(folder/'rgb512.png')
        Image.fromarray(prepared['mask'].astype('uint8')*255).save(folder/'mask512.png')
        candidates.append({'id':n,'image':source_record(p,rows),'mask':source_record(mp,rows),
            'review_reason':REASONS[n], 'decoded_RGB_raw_bytes_sha256':__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
            **{k:v for k,v in prepared.items() if k not in ('rgb','mask')}})
        refs[n]=rgb;masks[n]=mask
    if len({x['image']['rgb_content_sha256'] for x in candidates})!=5:
        raise ValueError('Duplicate candidate RGB identity')
    actual=CANDIDATES[:3]; requests=[]; summaries=[]
    overview=Image.new('RGB',(4*300,6*324),'#242424');draw=ImageDraw.Draw(overview)
    for ri,n in enumerate(actual):
        for bi,p in enumerate(background_paths):
            rgb=backgrounds[p]
            for vi,variant in enumerate(VARIANTS):
                req={'variant':variant,'model_side':512}
                result=build_geometry(masks[n],rgb.shape,req);g=result['metadata']
                folder=OUT/'geometry'/f'{n}_b{bi}_{variant}';folder.mkdir(parents=True)
                with (folder/'support.npz').open('xb') as f:
                    np.savez_compressed(f,original=result['roi'],local512=result['local_roi'])
                box=g['crop_xyxy_exclusive']
                crop=np.array(Image.fromarray(rgb).crop(box).resize((512,512),Image.Resampling.BILINEAR))
                Image.fromarray(crop).save(folder/'background512.png')
                # Display annotations only; these pixels never enter model inputs.
                preview=rgb.copy();contours,_=cv2.findContours(result['roi'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview,contours,-1,(255,100,0),2)
                cv2.rectangle(preview,tuple(box[:2]),(box[2]-1,box[3]-1),(30,210,255),2)
                x,y=vi*300,(ri*2+bi)*324
                overview.paste(Image.fromarray(preview).resize((300,300),Image.Resampling.BILINEAR),(x,y+24))
                draw.text((x+4,y+4),f'{n} / {Path(p).stem} / {variant}',fill='white')
                rel=str(folder.relative_to(OUT)).replace('\\','/')
                for seed in (20261016,20261017):
                    requests.append({'id':f'r{len(requests):03d}', 'reference_id':n,
                        'background_path':p,'variant':variant,'seed':seed,'geometry':g,'arrays':rel+'/support.npz',
                        'background512':rel+'/background512.png','input_status':'frozen_CPU_geometry_domain_blocked'})
                summaries.append(g)
    overview.save(OUT/'CPU_position_overview.png')
    write_json(OUT/'requests.json',{'schema':'reference-requests-v1','requests':requests,'count':len(requests),
        'distinct_geometries':24,'intended_neural_conditions':['reference_off','reference_on'],
        'neural_generations_completed':0,'not_authorized_for_generation':True})
    guard=read_json(OUT/'source_guard.json')
    if any(sha256(ROOT/k)!=v for k,v in guard.items()):
        raise ValueError('Old protected source changed')
    protocol={'schema':'reference-day1-input-v1','date':'2026-10-16','status':'CPU_PREPARED_DOMAIN_BLOCKED',
        'ready_for_day2':False,'generation_authorized':False,
        'blocker':{'finding':'Fixed normal draw mixes a dark screen and a bright screen; references are dark-context pale patches.',
            'paths':background_paths,'cause':'Normal RGB group sampling excludes old roles but has no display-background domain constraint.',
            'action':'Preserve fixed draw; do not replace images or begin generation. Review a separate same-domain selection rule.'},
        'normal_pool':{'qualified_RGB_groups':count,'rng':'PCG64','seed':20261016,
            'exclusion_rule':'old N/Vnew/Tfinal/excluded_train and controlled N_gen/S_bg/Q_cal, including matching saved RGB group IDs',
            'excluded_RGB_count':len(excluded),'backgrounds':[source_record(p,rows) for p in background_paths]},
        'references':candidates,'actual_reference_ids':actual,
        'selection':{'order':'decoded RGB hash then path','review':'first five reviewed crop-eligible, dark blue-gray diffuse pale-patch candidates',
            'reviewed_prefix_ids':['019','024','014','013','037','020','035','023','021'],
            'rejected_visual':{'014':'purple elongated feature','013':'purple elongated feature',
                               '037':'gray-background dark spot','035':'mixed field with small purple feature'},
            'eligible_crop_count':len(eligible),'total_mura':50,
            'physical_source_groups_known':False,'near_duplicate_or_augmented_relationships':'not established; similar shapes may be related',
            'use':'development only; not independent real validation'},
        'processing':{'RGB':'uint8 original RGB; RGBA requires all alpha255; no tone/contrast normalization',
            'mask':'binary0/255; nearest for discrete resampling','image_resize':'Pillow bilinear with half-pixel centers',
            'target_resize':'OpenCV INTER_NEAREST_EXACT; whole-image warp INTER_NEAREST',
            'coordinates':'original x,y pixel centers; clockwise positive angle; bbox right/bottom exclusive',
            'reference_crop':'4*bbox max dimension, floor(center-(side-1)/2); no clipping/padding',
            'source_mask_semantic':True,'target_support_is_semantic_GT':False,
            'RGB_group_identity':'Immutable old manifest group IDs for exclusion; source bytes verified. Old group digest serialization is unspecified; new SHA256 is raw uint8 HWC RGB bytes.'},
        'geometry_counts':{'requests':48,'reference_ids':3,'backgrounds':2,'variants':4,'seeds':2,'distinct_geometry_arrays':24},
        'geometry_max_roundtrip_error_px':max(x['roundtrip_max_error_px'] for x in summaries),
        'quality_observations_predeclared':['no discernible defect','opposite reference polarity','unexpected structure','visible seam'],
        'resources':resources(),'source_metadata':{p:sha256(ROOT/p) for p in SOURCES},
        'implementation':{n:sha256(Path(__file__).parent/n) for n in CORE},
        'decisions':['No git repository; independent package in user workspace; no git workflow.',
            'No process ledger/software suite per explicit user scope; scientific protocol holds material decisions.',
            'RGB groups are exact-content groups, not confirmed physical capture groups.',
            'Freeze CPU evidence only; domain assumption failed, so do not promote these inputs to model generation.',
            'First prepare stopped before output because new raw-RGB digest was wrongly equated to old unspecified group digest. Exact source bytes matched. Corrected this assumption; retained transaction marker; no generated outputs existed.']}
    if protocol['resources']['torch_loaded']:
        raise RuntimeError('Torch unexpectedly imported')
    write_json(OUT/'input_protocol.json',protocol)
    artifacts={str(p.relative_to(OUT)).replace('\\','/'):sha256(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and p!=marker}
    write_json(OUT/'completion.json',{'schema':'CPU-preparation-completion-v1','artifacts':artifacts,
        'status':protocol['status'],'scope':'CPU preparation only; visual domain gate NOT passed'})
    marker.unlink()
    return verify_inputs()


def verify_inputs():
    if (OUT/'preparation.incomplete').exists():
        raise RuntimeError('Incomplete transaction: preserve and investigate')
    completion=read_json(OUT/'completion.json');protocol=read_json(OUT/'input_protocol.json')
    for p,h in completion['artifacts'].items():
        if sha256(OUT/p)!=h:
            raise ValueError('Saved artifact identity changed: '+p)
    for p,h in protocol['source_metadata'].items():
        if sha256(ROOT/p)!=h:
            raise ValueError('Source metadata changed')
    for p,h in protocol['implementation'].items():
        if sha256(Path(__file__).parent/p)!=h:
            raise ValueError('CPU implementation changed')
    for p,h in read_json(OUT/'source_guard.json').items():
        if sha256(ROOT/p)!=h:
            raise ValueError('Old protected file changed')
    rows,old,cg=metadata();data=Path(old['root'])
    backgrounds,count,excluded=pick_backgrounds(rows,old,cg)
    if backgrounds != protocol['blocker']['paths'] or count!=protocol['normal_pool']['qualified_RGB_groups']:
        raise ValueError('Normal selection changed')
    # Only selected references and two selected backgrounds may be decoded.
    for p in backgrounds:
        read_rgb(data/p,rows[p])
        if rows[p]['rgb_content_sha256'] in excluded:
            raise ValueError('Excluded background')
    masks={}
    for row in protocol['references']:
        p=row['image']['path'];mp=row['mask']['path'];rgb=read_rgb(data/p,rows[p]);m=read_mask(data/mp,rows[mp])
        prepare_reference(rgb,m,{'context_factor':4,'model_side':512});masks[row['id']]=m
    requests=read_json(OUT/'requests.json')['requests']
    if len(requests)!=48 or len({x['id'] for x in requests})!=48:
        raise ValueError('Wrong request denominator')
    for request in requests:
        p=request['background_path'];result=build_geometry(masks[request['reference_id']],tuple(rows[p]['size'][::-1]),
            {'variant':request['variant'],'model_side':512})
        if result['metadata']!=request['geometry']:
            raise ValueError('Geometry metadata no longer reproducible')
        with np.load(OUT/request['arrays'],allow_pickle=False) as saved:
            if not np.array_equal(saved['original'],result['roi']) or not np.array_equal(saved['local512'],result['local_roi']):
                raise ValueError('Geometry array mismatch')
    if 'torch' in sys.modules:
        raise RuntimeError('Torch unexpectedly loaded')
    return {'status':protocol['status'],'requests_verified':48,'geometry_arrays_verified':24,
        'candidate_references':5,'actual_references':3,'backgrounds':2,
        'ready_for_day2':False,'domain_gate_passed':False,'new_model_forwards':0,
        'optimizer_steps':0,'protected_files_verified':len(read_json(OUT/'source_guard.json')),
        'protocol_sha256':sha256(OUT/'input_protocol.json')}
