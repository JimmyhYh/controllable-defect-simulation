"""Frozen metadata scope and explicit dark-context domain criteria; CPU only."""
import json
from pathlib import Path

import numpy as np

from .reference import sha256

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'results/reference_generation/v1'
OUT=ROOT/'results/reference_generation/v2_domain_matched'
META=('results/weak_localization/dataset_manifest.json','data/weak_localization/split.json',
      'data/controlled_generation/split.json')
NEW_CORE=('domain_selection_v2.py','domain_preparation_v2.py','domain_verification_v2.py','domain_cli_v2.py')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def scope_from_metadata():
    manifest,old,cg=[load(ROOT/p) for p in META]
    v1=load(V1/'input_protocol.json')
    for p,h in v1['source_metadata'].items():
        if sha256(ROOT/p)!=h: raise ValueError('Old metadata identity changed')
    rows={x['path']:x for x in manifest['files']}
    excluded_paths=[p for k in ('N','Vnew','Tfinal','excluded_train') for p in old[k]]
    excluded_paths += [p for k in ('N_gen','S_bg','Q_cal') for p in cg[k]]
    excluded={rows[p]['rgb_content_sha256'] for p in excluded_paths}
    groups={}
    for p in old['unused_normal_train']:
        if not p.startswith('display/train/good/'): raise ValueError('Unexpected normal source role')
        g=rows[p]['rgb_content_sha256']
        if g not in excluded: groups.setdefault(g,[]).append(p)
    if len(groups)!=1693: raise ValueError('Normal group population differs from frozen expectation1693')
    keys=sorted(groups)
    order=np.random.Generator(np.random.PCG64(20261016)).permutation(len(keys))[:64]
    selected=[{'group_id':keys[i],'path':min(groups[keys[i]]),
               'file_sha256':rows[min(groups[keys[i]])]['sha256']} for i in order]
    return rows,old,v1,selected


def freeze_scope_v2():
    # This function does not decode, open or hash any dataset image.
    rows,old,v1,selected=scope_from_metadata()
    if sha256(V1/'input_protocol.json')!='938eca3d7b4b95ccac8f0dcba989af34c2c8ec73fd110f718663002a64c48a45':
        raise ValueError('Previously frozen v1 protocol identity differs')
    if OUT.exists(): raise RuntimeError('v2 directory already exists; preserve and verify, never overwrite')
    guard=load(V1/'source_guard.json')
    guard.update({'reference_generation/'+p:h for p,h in v1['implementation'].items()})
    for name in ('cli.py','README.md'):
        guard['reference_generation/'+name]=sha256(ROOT/'reference_generation'/name)
    for name in ('input_protocol.json','requests.json','completion.json','source_guard.json',
                 '第1天CPU输入准备与域条件核验.md'):
        guard['results/reference_generation/v1/'+name]=sha256(V1/name)
    for name,h in load(V1/'completion.json')['artifacts'].items():
        guard['results/reference_generation/v1/'+name]=h
    for p,h in guard.items():
        if sha256(ROOT/p)!=h: raise ValueError('Protected identity mismatch before scope freeze')
    criteria={'Y_coefficients':[.2126,.7152,.0722],'quantile_method':'linear',
        'global_center_fraction':[.1,.9],'global_median_Y_max':.25,'global_q90_Y_max':.35,
        'context_dilate_square':17,'minimum_context_pixels':1024,
        'median_Y_margin':.06,'median_RGB_margin':.06,'Y_range_floor':.15,'Y_range_margin':.03,
        'reference_ids':['019','024','020'],'candidate_reference_ids':['019','024','020','023','021'],
        'geometry_variants':['base','move','scale','rotate'],'generated_image_seeds':[20261016,20261017],
        'target_shape':[1024,1024],'model_side':512,'scope_count':64,'draw_seed':20261018,
        'no_replacement_after_visual':True,'thresholds_are':'predeclared engineering screen, not industry standard'}
    scope={'schema':'domain-scope-v2','date':'2026-10-16','dataset_root':old['root'],
        'metadata':{p:sha256(ROOT/p) for p in META},'protected':guard,
        'v1_protocol_sha256':sha256(V1/'input_protocol.json'),
        'implementation':{p:sha256(Path(__file__).parent/p) for p in NEW_CORE},
        'criteria':criteria,'scope_seed':20261016,'qualified_group_count':1693,
        'scope_order':'old RGB group sort -> PCG64 permutation -> first64; representative=min path',
        'scope':selected,'candidates':v1['references'],'actual_reference_ids':v1['actual_reference_ids'],
        'physical_source_groups_known':False,
        'decisions':['No Git workflow/process ledger/software suite per user; independent CPU modules.',
            'Frozen references and geometry reused without modifying v1.',
            'Old immutable RGB group IDs plus exact source bytes used; their old serialization remains unspecified.',
            'Visual review is a separate explicit finalize action after full64 screening and fixed draw.']}
    OUT.mkdir()
    save(OUT/'scope_protocol.json',scope)
    (OUT/'preparation.incomplete').write_text('CPU v2 transaction; preserve on failure.\n',encoding='utf-8')
    return scope


def statistics(rgb,region):
    values=rgb[region].astype(np.float64)/255
    if len(values)<1024: raise ValueError('Insufficient context pixels; do not relax or replace')
    y=values@np.array([.2126,.7152,.0722])
    q=np.quantile(y,[.1,.5,.9],method='linear')
    result={'pixels':len(values),'median_Y':float(q[1]),'q90_Y':float(q[2]),
            'q90_minus_q10_Y':float(q[2]-q[0]),'median_RGB':np.median(values,axis=0).tolist()}
    if not np.isfinite([result['median_Y'],result['q90_Y'],result['q90_minus_q10_Y'],*result['median_RGB']]).all():
        raise ValueError('Nonfinite domain statistics')
    return result


def rejection_reasons(global_stats,local_stats,anchors):
    reasons=[]
    if global_stats['median_Y']>.25: reasons.append('global_median_Y')
    if global_stats['q90_Y']>.35: reasons.append('global_q90_Y')
    for key,stats in local_stats.items():
        if not anchors['median_Y_interval'][0]<=stats['median_Y']<=anchors['median_Y_interval'][1]:
            reasons.append(key+':median_Y')
        if np.any(np.asarray(stats['median_RGB'])<anchors['median_RGB_lo']) or np.any(
                np.asarray(stats['median_RGB'])>anchors['median_RGB_hi']): reasons.append(key+':median_RGB')
        if stats['q90_minus_q10_Y']>anchors['Y_range_max']: reasons.append(key+':Y_range')
    return reasons


def classify_domain_v2(rgb,reference_contexts,geometry):
    if rgb.shape!=(1024,1024,3) or rgb.dtype!=np.uint8: raise ValueError('Unexpected candidate dimensions/type')
    # Integer pixel centers in [102.4,921.6): indices103..921 inclusive.
    global_region=np.zeros((1024,1024),bool);global_region[103:922,103:922]=True
    gs=statistics(rgb,global_region)
    local={key:statistics(rgb,region) for key,region in geometry.items()}
    reasons=rejection_reasons(gs,local,reference_contexts)
    return {'global':gs,'local':local,'eligible':not reasons,'rejection_reasons':reasons}


def draw_backgrounds_v2(eligible_groups):
    if len(eligible_groups)<2:return []
    ordered=sorted(eligible_groups,key=lambda x:x['group_id'])
    order=np.random.Generator(np.random.PCG64(20261018)).permutation(len(ordered))
    return [ordered[i] for i in order[:2]]
