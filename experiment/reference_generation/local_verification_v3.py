"""Read-only v3 file inventory, identity, geometry and saved-statistics verification."""
from pathlib import Path
import sys

import cv2
import numpy as np

from .reference import sha256
from .local_preparation_v3 import ROOT,OUT,load,old_identities,geometries
from .local_review_v3 import rectangle_mask,numeric_issues


def verify_local_inputs_v3():
    if (OUT/'preparation.incomplete').exists():raise RuntimeError('Incomplete v3 transaction; preserve and investigate')
    completion=load(OUT/'completion.json');protocol=load(OUT/'local_protocol.json')
    expected=set(completion['artifacts'])|{'completion.json'}
    actual={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Strict v3 file inventory differs')
    dirs={str(a).replace('\\','/') for p in expected for a in Path(p).parents if str(a)!='.'}
    if dirs!={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_dir()}:
        raise ValueError('Strict v3 directory inventory differs')
    for p,h in completion['artifacts'].items():
        if sha256(OUT/p)!=h:raise ValueError('v3 artifact changed: '+p)
    for p,h in protocol['protected'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Protected old file changed')
    for p,h in protocol['implementation'].items():
        if sha256(Path(__file__).parent/p)!=h:raise ValueError('v3 core changed')
    protected,_,_,selected=old_identities()
    if protected!=protocol['protected'] or selected!=protocol['selected']:raise ValueError('Old identity/draw mismatch')
    review=load(OUT/'structure_review.json');audit=load(OUT/'local_audit.json');outcome=load(OUT/'input_protocol.json')
    if audit['review_sha256']!=sha256(OUT/'structure_review.json') or outcome['local_protocol_sha256']!=sha256(OUT/'local_protocol.json'):
        raise ValueError('Review/protocol identity mismatch')
    geo=geometries();k=0;passed=0;max_s=0;max_u=0;remote_count=0
    for bi in range(2):
        s=rectangle_mask(review['structure_rectangles'][f'b{bi}']);u=rectangle_mask(review['uncertain_rectangles'][f'b{bi}'])
        sp=cv2.dilate(s.astype('uint8'),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        up=cv2.dilate(u.astype('uint8'),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        for n in protocol['reference_ids']:
            for variant in ('base','move','scale','rotate'):
                row=audit['records'][k];g=geo[n+'_'+variant]
                if (row['background'],row['reference_id'],row['variant'])!=(f'b{bi}',n,variant):raise ValueError('Audit order changed')
                if g['metadata']!=protocol['geometries'][n+'_'+variant]:raise ValueError('Frozen geometry differs')
                sc=int((sp&g['guard']).sum());uc=int((up&g['guard']).sum());max_s=max(max_s,sc);max_u=max(max_u,uc)
                l,t,r,b=g['metadata']['crop_xyxy_exclusive'];model=np.zeros((1024,1024),bool);model[t:b,l:r]=True
                remote=int((sp&model&~g['guard']).sum());remote_count+=remote>0
                ru=int((up&model&~g['guard']).sum())
                dr=cv2.dilate(g['roi'].astype('uint8'),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
                pixels=int((g['guard']&~dr).sum());stats=row['guard_context_statistics']
                if stats['pixels']!=pixels or pixels<1024:raise ValueError('Wrong context area')
                issues=numeric_issues(stats,protocol['anchors']);p=not(sc or uc or issues)
                if (row['structure_guard_overlap_pixels'],row['uncertain_guard_overlap_pixels'],
                    row['structure_in_model_outside_guard_pixels'],row['uncertain_in_model_outside_guard_pixels'],
                    row['numeric_rejection_reasons'],row['passed'])!=(sc,uc,remote,ru,issues,p):
                    raise ValueError('Audit intersection/classification differs')
                passed+=p;k+=1
    if len(audit['records'])!=24 or audit['passed']!=passed or audit['position_count']!=24:
        raise ValueError('Wrong all-position denominator')
    if (audit['max_structure_guard_overlap_pixels'],audit['max_uncertain_guard_overlap_pixels'],
        audit['positions_with_remote_structure_in_model'])!=(max_s,max_u,remote_count):raise ValueError('Wrong audit summary')
    status='CPU_LOCAL_READY' if passed==24 else ('LOCAL_REVIEW_UNCERTAIN' if max_u else (
        'LOCAL_STRUCTURE_BLOCKED' if max_s else 'LOCAL_NUMERIC_BLOCKED'))
    if outcome['status']!=status or completion['status']!=status or outcome['ready_for_day2']!=(passed==24):
        raise ValueError('Incorrect readiness/status')
    if outcome['model_forwards'] or outcome['optimizer_steps'] or outcome['torch_loaded'] or 'torch' in sys.modules:
        raise ValueError('Unexpected model action')
    array_paths=set()
    if passed==24:
        requests=load(OUT/'requests.json')['requests'];index=0
        for n in protocol['reference_ids']:
            for item in selected:
                for variant in ('base','move','scale','rotate'):
                    g=geo[n+'_'+variant]
                    for seed in (20261016,20261017):
                        req=requests[index]
                        if (req['id'],req['reference_id'],req['background_path'],req['variant'],req['seed'])!=(
                            f'v3_r{index:03d}',n,item['path'],variant,seed):raise ValueError('Wrong request ordering')
                        if req['geometry']!=g['metadata']:raise ValueError('Request geometry mismatch')
                        with np.load(OUT/req['arrays'],allow_pickle=False) as arr:
                            if not(np.array_equal(arr['original'],g['roi']) and np.array_equal(arr['local512'],g['local_roi'])
                                   and np.array_equal(arr['guard'],g['guard'])):raise ValueError('Wrong saved geometry')
                        array_paths.add(req['arrays']);index+=1
        if len(requests)!=48 or len(array_paths)!=24 or outcome['request_count']!=48 or outcome['distinct_geometries']!=24:
            raise ValueError('Wrong promoted request denominator')
    elif (OUT/'requests.json').exists() or outcome['request_count'] or outcome['distinct_geometries']:
        raise ValueError('Blocked set was promoted')
    return {'status':status,'ready_for_day2':passed==24,'positions_passed':passed,'positions_evaluated':24,
        'max_structure_guard_overlap_pixels':max_s,'max_uncertain_guard_overlap_pixels':max_u,
        'positions_with_remote_structure_in_model':remote_count,'requests_verified':outcome['request_count'],
        'geometry_arrays_verified':len(array_paths),'new_normal_RGB_decodes':0,'new_model_forwards':0,
        'optimizer_steps':0,'strict_inventory':True,'protected_nonimage_metadata_and_selected_input_files':len(protected)}
