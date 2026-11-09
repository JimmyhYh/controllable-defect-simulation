"""Independent metadata preparation and verification, no rendering entry."""
import argparse
import hashlib
import json
from pathlib import Path
from .inputs import ROOT, inspect_environment
from .protocol import DEFAULTS,contract,validate_request

OUT=ROOT/'results/defect_candidate_tool/v1/preparation_v1'
SOURCES=['controlled_generation/geometry.py','controlled_generation/render.py','controlled_generation/cli.py',
         'controlled_generation/data.py','data/controlled_generation/split.json','data/weak_localization/split.json',
         'results/weak_localization/dataset_manifest.json','README.md']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def identity():
    paths=[ROOT/p for p in SOURCES]+sorted(Path(__file__).parent.glob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def scalar_checks():
    r=dict(DEFAULTS,class_id=1,shape='ellipse',polarity='dark',center_xy=[512,400],semiaxes_px=[60,30],angle_deg=25)
    validate_request(r,800,1200)
    invalid=[dict(r,amplitude=0),dict(r,amplitude=float('nan')),dict(r,class_id=2),dict(r,center_xy=[0,0]),dict(r,base_seed=True),dict(r,unexpected=1)]
    for x in invalid:
        try: validate_request(x,800,1200)
        except ValueError: pass
        else: raise RuntimeError('Invalid scalar request accepted')
    for shape in ('irregular_block','line','polyline'):
        x=dict(r,shape=shape)
        if shape in ('line','polyline'):
            x.pop('semiaxes_px');x['class_id']=2;x['width_px']=6
            if shape=='line': x['length_px']=120
            else:
                x.pop('center_xy');x.pop('angle_deg');x['points_xy']=[[200,200],[260,210],[320,240]]
        validate_request(x,800,1200)
    return {'valid_native_non_square_cases':4,'rejected_invalid_scalar_cases':len(invalid),'images_read':0,'rasterization':False}

def backgrounds():
    split=read(ROOT/'data/controlled_generation/split.json')
    old=read(ROOT/'data/weak_localization/split.json')
    manifest=read(ROOT/'results/weak_localization/dataset_manifest.json')
    if digest(ROOT/'results/weak_localization/dataset_manifest.json')!=split['manifest_sha256'] or digest(ROOT/'data/weak_localization/split.json')!=split['old_split_sha256']:
        raise RuntimeError('Frozen metadata identity mismatch')
    rows={x['path']:x for x in manifest['files']}
    excluded=set(split['S_bg']+split['Q_cal']+old['N']+old['Vnew']+old['Tfinal']+old['excluded_train'])
    excluded_rgb={rows[p].get('rgb_content_sha256') for p in excluded if p in rows}-{None}
    selected=[];rejected=[]
    for p in split['N_gen']:
        row=rows[p]; reasons=[]
        if p in excluded or row.get('rgb_content_sha256') in excluded_rgb: reasons.append('frozen role or known RGB duplicate')
        if row.get('mode') not in ('RGB','L'): reasons.append('mode rejected: '+str(row.get('mode')))
        if row['sha256']!=split['source_sha256'][p]: raise RuntimeError('Manifest/source hash mismatch')
        item={'path':str(Path(split['root'])/p),'relative_path':p,'source_sha256_from_metadata':row['sha256'],
              'rgb_sha256_from_metadata':row.get('rgb_content_sha256'),'size_from_metadata':row['size'],'mode_from_metadata':row['mode'],
              'image_bytes_read':False,'reasons':reasons}
        if reasons: rejected.append(item)
        elif len(selected)<2: selected.append(item)
    return {'candidate_pool':'original N_gen80 order only; no range expansion','pool_size':len(split['N_gen']),
            'selected':selected,'rejected':rejected,'proposed_first_two_for_explanation':rejected[:2] if not selected else [],
            'source':'https://github.com/SijieLuo/DLAS-Net','use':'existing public AnoLCD, personal noncommercial only; no separate license inferred',
            'source_evidence':'existing project README and frozen split/manifest; no new web or image reads',
            'admitted':len(selected)==2,'known_capture_groups':False,'stage2_rehash_and_decode_required':True}

def verify():
    complete=read(OUT/'complete.json')
    for name,h in complete['files'].items():
        if digest(OUT/name)!=h: raise RuntimeError('Preparation file changed: '+name)
    if {p.name for p in OUT.iterdir()}!=set(complete['files'])|{'complete.json'}: raise RuntimeError('Unknown preparation contents')
    if read(OUT/'identity.json')!=identity(): raise RuntimeError('Source/implementation changed')
    return {'verification':'passed','status':read(OUT/'protocol.json')['status'],'new_image_reads':0,'new_generation':0}

def prepare():
    if OUT.exists():
        if (OUT/'complete.json').exists(): return verify()
        raise RuntimeError('Unknown/partial preparation preserved; do not retry')
    before=identity();env=inspect_environment();bg=backgrounds();scalars=scalar_checks();cfg=contract()
    max_pixels=cfg['resources']['max_native_pixels'];res=cfg['resources']
    required=res['fixed_overhead_bytes']+res['estimated_peak_bytes_per_pixel']*max_pixels+res['reserve_bytes']
    resources_ok=env['RAM_available_bytes']>=required and env['disk_free_bytes']>=res['disk_min_free_bytes']
    deps_ok=all(x['import_ok'] for x in env['packages'].values())
    cfg.update(status='ready_for_stage2' if resources_ok and deps_ok and env['tk']['import_ok'] and bg['admitted'] else 'prepared_with_blockers',
               background_admitted=bg['admitted'],resource_admitted=resources_ok,dependency_admitted=deps_ok,
               tk_import_admitted=env['tk']['import_ok'],required_available_RAM_bytes_at_max_pixels=required)
    if identity()!=before: raise RuntimeError('Source changed during preparation')
    OUT.mkdir(parents=True,exist_ok=False)
    for name,x in [('identity.json',before),('protocol.json',cfg),('environment.json',env),('backgrounds.json',bg),('scalar_verification.json',scalars)]: save(OUT/name,x)
    save(OUT/'complete.json',{'files':{p.name:digest(p) for p in sorted(OUT.iterdir())}})
    return verify()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify-preparation'])
    args=parser.parse_args();print(json.dumps(prepare() if args.command=='prepare' else verify(),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
