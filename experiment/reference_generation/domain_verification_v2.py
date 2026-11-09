"""Strict saved-input verification; no source RGB decoding or model imports."""
import sys
from pathlib import Path

import numpy as np

from .reference import read_mask,sha256
from .coordinates import build_geometry,VARIANTS
from .domain_selection_v2 import (ROOT,V1,OUT,load,scope_from_metadata,rejection_reasons,draw_backgrounds_v2)


def verify_domain_inputs_v2():
    if (OUT/'preparation.incomplete').exists():raise RuntimeError('Incomplete v2 transaction; preserve and investigate')
    complete=load(OUT/'completion.json');scope=load(OUT/'scope_protocol.json')
    expected=set(complete['artifacts'])|{'completion.json'}
    actual={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Strict inventory: unexpected or missing files')
    expected_dirs={str(Path(p).parent).replace('\\','/') for p in expected}
    expected_dirs.discard('.')
    # Include all ancestor directories of nested artifacts.
    for p in expected:
        expected_dirs.update(str(a).replace('\\','/') for a in Path(p).parents if str(a)!='.')
    actual_dirs={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_dir()}
    if actual_dirs!=expected_dirs:raise ValueError('Strict directory inventory differs')
    for p,h in complete['artifacts'].items():
        if sha256(OUT/p)!=h:raise ValueError('Artifact hash mismatch: '+p)
    for p,h in scope['protected'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Protected old identity changed: '+p)
    for p,h in scope['implementation'].items():
        if sha256(Path(__file__).parent/p)!=h:raise ValueError('v2 implementation changed')
    rows,old,v1,selected_scope=scope_from_metadata()
    if selected_scope!=scope['scope']:raise ValueError('Fixed64 scope no longer matches metadata permutation')
    data=Path(old['root'])
    for item in scope['scope']:
        if sha256(data/item['path'])!=item['file_sha256']:raise ValueError('Normal source bytes changed')
    for ref in scope['candidates']:
        for key in ('image','mask'):
            record=ref[key]
            if sha256(data/record['path'])!=record['sha256']:raise ValueError('Reference source bytes changed')
    audit=load(OUT/'domain_audit.json');anchors=load(OUT/'reference_anchors.json')
    if audit['scope_sha256']!=sha256(OUT/'scope_protocol.json') or len(audit['rows'])!=64 or audit['classified']!=64:
        raise ValueError('Incorrect audit scope/denominator')
    local_keys={n+'_'+v for n in scope['actual_reference_ids'] for v in VARIANTS}
    for item,saved in zip(scope['scope'],audit['rows']):
        if any(saved[k]!=v for k,v in item.items()) or set(saved['local'])!=local_keys:
            raise ValueError('Audit order or region set changed')
        reasons=rejection_reasons(saved['global'],saved['local'],anchors)
        if reasons!=saved['rejection_reasons'] or saved['eligible']!=(not reasons):raise ValueError('Saved classification inconsistent')
    eligible=[x for x in audit['rows'] if x['eligible']]
    if len(eligible)!=audit['eligible']:raise ValueError('Eligibility denominator differs')
    draw=draw_backgrounds_v2(eligible)
    if [{'group_id':x['group_id'],'path':x['path']} for x in draw]!=audit['selected']:
        raise ValueError('Fixed drawing differs')
    protocol=load(OUT/'input_protocol.json')
    if protocol['scope_sha256']!=sha256(OUT/'scope_protocol.json') or protocol['status']!=complete['status']:
        raise ValueError('Input protocol identity mismatch')
    if protocol['torch_loaded'] or protocol['model_forwards'] or protocol['optimizer_steps'] or 'torch' in sys.modules:
        raise ValueError('Unexpected model action')
    if protocol['ready_for_day2']:
        decision=load(OUT/'visual_decision.json')
        if decision['decision']!='passed' or protocol['status']!='CPU_DOMAIN_READY':raise ValueError('Missing visual pass')
        requests=load(OUT/'requests.json')['requests'];k=0;array_paths=set()
        for n in scope['actual_reference_ids']:
            mp=f'display/ground_truth/mura/{n}_mask.png';mask=read_mask(data/mp,rows[mp])
            for bg in draw:
                for variant in VARIANTS:
                    g=build_geometry(mask,(1024,1024),{'variant':variant,'model_side':512})
                    for seed in (20261016,20261017):
                        req=requests[k]
                        if (req['id'],req['reference_id'],req['background_path'],req['variant'],req['seed'])!=(
                                f'v2_r{k:03d}',n,bg['path'],variant,seed):raise ValueError('Request order/factor mismatch')
                        if req['geometry']!=g['metadata']:raise ValueError('CPU geometry differs')
                        with np.load(OUT/req['arrays'],allow_pickle=False) as arrays:
                            if not np.array_equal(arrays['original'],g['roi']) or not np.array_equal(arrays['local512'],g['local_roi']):
                                raise ValueError('Geometry array mismatch')
                        array_paths.add(req['arrays']);k+=1
        if len(requests)!=48 or len(array_paths)!=24:raise ValueError('Wrong request/geometry denominator')
    else:
        if protocol['request_count']!=0 or (OUT/'requests.json').exists():raise ValueError('Blocked state contains promoted requests')
    return {'status':protocol['status'],'ready_for_day2':protocol['ready_for_day2'],
        'fixed_scope':64,'classified':audit['classified'],'eligible':len(eligible),
        'selected':audit['selected'],'requests_verified':protocol['request_count'],
        'geometry_arrays_verified':protocol['distinct_geometries'],
        'new_source_RGB_decodes':0,'new_model_forwards':0,'optimizer_steps':0,
        'strict_inventory':True,'protected_file_count':len(scope['protected'])}
