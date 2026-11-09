"""Independent v3 CPU scope preparation; original normal scope is fixed to two cached images."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from .reference import read_mask,sha256,crop_box
from .coordinates import build_geometry,VARIANTS

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'results/reference_generation/v1'
V2=ROOT/'results/reference_generation/v2_domain_matched'
OUT=ROOT/'results/reference_generation/v3_local_context'
CORE=('local_preparation_v3.py','local_review_v3.py','local_verification_v3.py','local_cli_v3.py')
REPORT='第1天v3局部背景适用性结果.md'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def old_identities():
    if sha256(V2/'completion.json')!='652323555f19533e68d5bdb6908c6138015a37ddaf0d21eaedb9eaa6622b069f':
        raise ValueError('Frozen v2 completion changed')
    complete=load(V2/'completion.json');scope2=load(V2/'scope_protocol.json')
    # Necessary metadata/code hashes only: do not decode or read all64 normal source files.
    guard={p:h for p,h in scope2['protected'].items() if Path(p).suffix in ('.py','.json','.md','.txt')}
    guard.update({'reference_generation/'+p:h for p,h in scope2['implementation'].items()})
    for p,h in complete['artifacts'].items():
        if Path(p).suffix in ('.json','.md') or p in ('visual/b0_original.png','visual/b1_original.png'):
            guard['results/reference_generation/v2_domain_matched/'+p]=h
    guard['results/reference_generation/v2_domain_matched/completion.json']=sha256(V2/'completion.json')
    for p,h in guard.items():
        if sha256(ROOT/p)!=h:raise ValueError('Protected identity changed: '+p)
    rows={x['path']:x for x in load(ROOT/'results/weak_localization/dataset_manifest.json')['files']}
    old=load(ROOT/'data/weak_localization/split.json');cg=load(ROOT/'data/controlled_generation/split.json')
    excluded={rows[p]['rgb_content_sha256'] for k in ('N','Vnew','Tfinal','excluded_train') for p in old[k]}
    excluded|={rows[p]['rgb_content_sha256'] for k in ('N_gen','S_bg','Q_cal') for p in cg[k]}
    audit=load(V2/'domain_audit.json');selected=audit['selected']
    if [x['path'] for x in selected]!=['display/train/good/1949.png','display/train/good/1540.png']:
        raise ValueError('v2 fixed draw changed')
    for p in selected:
        if rows[p['path']]['rgb_content_sha256'] in excluded:raise ValueError('Frozen-role/RGB overlap')
        if sha256(Path(old['root'])/p['path'])!=rows[p['path']]['sha256']:raise ValueError('Selected source bytes changed')
    return guard,rows,Path(old['root']),selected


def geometries():
    _,rows,data,_=old_identities();result={}
    for n in ('019','024','020'):
        mp=f'display/ground_truth/mura/{n}_mask.png';mask=read_mask(data/mp,rows[mp])
        for variant in VARIANTS:
            g=build_geometry(mask,(1024,1024),{'variant':variant,'model_side':512})
            bbox=g['metadata']['original_bbox_xyxy_exclusive']
            side=math.ceil(2*max(bbox[2]-bbox[0],bbox[3]-bbox[1]))
            guard_box=crop_box(g['metadata']['requested_center_xy'],side,(1024,1024))
            l,t,r,b=guard_box;gm=np.zeros((1024,1024),bool);gm[t:b,l:r]=True
            if np.any(g['roi']&~gm):raise ValueError('Guard does not contain R')
            ml,mt,mr,mb=g['metadata']['crop_xyxy_exclusive']
            if not(ml<=l<r<=mr and mt<=t<b<=mb):raise ValueError('Guard outside model crop')
            g['guard']=gm;g['metadata']={**g['metadata'],'guard_xyxy_exclusive':guard_box,
                'guard_side_original_px':side,'guard_pixels':int(gm.sum())}
            result[n+'_'+variant]=g
    return result


def read_backgrounds():
    images=[]
    for bi in range(2):
        name=f'visual/b{bi}_original.png';p=V2/name
        if sha256(p)!=load(V2/'completion.json')['artifacts'][name]:raise ValueError('Sealed selected background changed')
        with Image.open(p) as im:
            if im.mode!='RGB' or im.size!=(1024,1024):raise ValueError('Unexpected cached background dimensions/mode')
            images.append(np.array(im))
    return images


def prepare_local_scope_v3():
    if (OUT/'completion.json').exists():
        from .local_verification_v3 import verify_local_inputs_v3
        return verify_local_inputs_v3()
    if OUT.exists():raise RuntimeError('Existing partial v3: preserve and investigate, do not restart')
    protected,_,_,selected=old_identities();geo=geometries()
    if 'torch' in sys.modules:raise RuntimeError('Torch unexpectedly loaded')
    protocol={'schema':'local-scope-v3','date':'2026-10-16','protected':protected,
        'implementation':{p:sha256(Path(__file__).parent/p) for p in CORE},
        'selected':selected,'candidate_reference_ids':['019','024','020','023','021'],
        'reference_ids':['019','024','020'],'generated_seeds':[20261016,20261017],
        'geometries':{k:g['metadata'] for k,g in geo.items()},
        'rules':{'guard_side':'ceil(2*max(R bbox width,height))','dilation_square':17,
            'structure_rectangles':'integer x0,y0,x1,y1; right/bottom exclusive; conservative complete group bbox',
            'uncertain_is_not_clean':True,'required_zero_overlap':True,'all_cases_required':24,
            'model_crop':'original4x rule, unchanged; outer structures may remain',
            'display':'original uint8 RGB[0,255], no per-image enhancement; native1024 originals plus local1:1',
            'numeric':'v2 reference anchors unchanged; guard minus dilated R context >=1024; linear quantiles',
            'review_masks_are_semantic_GT':False},
        'anchors':load(V2/'reference_anchors.json'),
        'decisions':['Independent new acceptance unit; v2 failure stays unchanged.',
            'No process ledger/software suite/Git per user. New standalone CPU package modules only.',
            'Only two v2 sealed RGB images decoded; no resampling, no other normal source pixel access.',
            'Source groups and background structure provenance remain unknown; development observation only.']}
    OUT.mkdir();save(OUT/'local_protocol.json',protocol)
    (OUT/'preparation.incomplete').write_text('CPU v3 scope awaiting explicit one-time structure review.\n',encoding='utf-8')
    # All region geometry/rules are now on disk before any selected background inspection.
    backgrounds=read_backgrounds();visual=OUT/'visual';visual.mkdir()
    for bi,rgb in enumerate(backgrounds):
        # Grid is a display overlay, never a model input.
        grid=Image.fromarray(rgb.copy());dr=ImageDraw.Draw(grid)
        for x in range(0,1024,128):
            dr.line((x,0,x,1023),fill=(85,95,110),width=1);dr.text((x+2,2),str(x),fill='white')
        for y in range(0,1024,128):
            dr.line((0,y,1023,y),fill=(85,95,110),width=1);dr.text((2,y+2),str(y),fill='white')
        grid.save(visual/f'b{bi}_coordinate_review.png')
        page=Image.new('RGB',(4*300,3*324),'#252525');dr=ImageDraw.Draw(page)
        for ri,n in enumerate(protocol['reference_ids']):
            for vi,variant in enumerate(VARIANTS):
                g=geo[n+'_'+variant];view=rgb.copy();box=g['metadata']['crop_xyxy_exclusive']
                contours,_=cv2.findContours(g['roi'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(view,contours,-1,(255,110,0),2)
                l,t,r,b=g['metadata']['guard_xyxy_exclusive'];cv2.rectangle(view,(l,t),(r-1,b-1),(60,255,100),2)
                crop=Image.fromarray(view).crop(box)
                # Native local crop saved as well as thumbnail; no image normalization.
                crop.save(visual/f'b{bi}_{n}_{variant}_native.png')
                page.paste(crop.resize((300,300),Image.Resampling.BILINEAR),(vi*300,ri*324+24))
                dr.text((vi*300+4,ri*324+4),n+' / '+variant+' : orangeR greenGuard',fill='white')
        page.save(visual/f'b{bi}_all_positions.png')
    snapshot={'schema':'v3-review-inputs','status':'AWAITING_STRUCTURE_REVIEW',
        'artifacts':{str(p.relative_to(OUT)).replace('\\','/'):sha256(p) for p in sorted(OUT.rglob('*'))
                     if p.is_file() and p.name!='preparation.incomplete'}}
    save(OUT/'review_preparation.json',snapshot)
    return {'status':snapshot['status'],'backgrounds':2,'frozen_positions':24,'new_model_forwards':0,'ready_for_day2':False}
