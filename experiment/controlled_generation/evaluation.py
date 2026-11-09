"""Existing frozen AnoLCD PatchCore response only; not a new real evaluation."""
import gc
import time
from pathlib import Path

import numpy as np

from .data import OUT, ROOT, code_identity, digest, load_split, normal_rgb, read_json, save_json


def predict_trial():
    import torch
    from localization_calibration import load_model, predict_map
    from .cli import audit
    audit()  # Fresh identities and residuals; never trust a stale audit flag.
    split=load_split();trial=read_json(OUT/'trial_results.json')
    if not read_json(OUT/'artifact_audit.json')['same_input_rerender_equal']:
        raise ValueError('Generation artifact audit has not passed')
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen loader requires CUDA; generation remains available without prediction')
    frozen_path=ROOT/'results/weak_localization/frozen_config.json'
    frozen=read_json(frozen_path)
    if digest(ROOT/'localization_calibration.py')!=frozen['signature']['implementation_sha256']:
        raise ValueError('Frozen model loader/scorer implementation changed')
    core=('data.py','geometry.py','render.py','__init__.py')
    if any(trial['signature']['implementation'].get(name)!=sha for name,sha in code_identity().items()
           if Path(name).name in core):
        raise ValueError('Generator implementation differs from saved trial')
    files=frozen['signature']['models']['PatchCore']
    for path,sha in files.items():
        if digest(path)!=sha:
            raise ValueError('Frozen PatchCore bank identity differs')
    backbone=ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth'
    protocol=read_json(ROOT/'results/weak_localization/protocol.json')
    if digest(backbone)!=protocol['environment']['files'][str(backbone)]['sha256']:
        raise ValueError('Frozen ImageNet backbone changed')
    bank=read_json(ROOT/'results/weak_localization/patchcore_bank.json')
    if not bank['N_only'] or bank['normal_images']!=80:
        raise ValueError('Old bank is not the original normal-only N80 bank')
    calibration=read_json(ROOT/'results/weak_localization/calibration_PatchCore.json')
    threshold=frozen['thresholds']['PatchCore']
    if threshold!=calibration['threshold']:
        raise ValueError('Frozen threshold differs from original calibration result')
    signature={'frozen_config_sha256':digest(frozen_path),'trial_sha256':digest(OUT/'trial_results.json'),
               'backbone_sha256':digest(backbone),'bank_files':files,
               'implementation':{name:sha for name,sha in code_identity().items() if Path(name).name in core},
               'threshold':threshold,'input':'RGB256/255, frozen ImageNet normalization; float simulated input',
               'note':'Original N80/Vnew threshold reused for development demonstration; no new calibration'}
    folder=OUT/'detector_response';folder.mkdir(exist_ok=True)
    path=folder/'response.json'
    if path.exists():
        old=read_json(path)
        prior={**old['signature'],'implementation':{name:sha for name,sha in old['signature']['implementation'].items()
                                                   if Path(name).name in core}}
        if prior!=signature or digest(folder/'heatmaps.npz')!=old['heatmaps_sha256']:
            raise ValueError('Existing detector response differs; preserve it')
        print('PREDICT_CACHED',old['generated_alarm_count'],old['generated_images']);return old
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model=load_model('PatchCore');cache={};arrays={};records=[]
    for record in trial['records']:
        case=record['request']; source=case['background_path']
        if source not in cache:
            score,heatmap=predict_map('PatchCore',model,normal_rgb(split,source))
            cache[source]={'score':score,'alarm':score>threshold}
            arrays['background_'+str(case['background_index'])]=heatmap
        with np.load(OUT/'trial'/case['id']/'arrays.npz',allow_pickle=False) as saved:
            # Label-free model input: no requested or effective mask passed to PatchCore.
            score,heatmap=predict_map('PatchCore',model,saved['image'])
        if not np.isfinite(score) or heatmap.shape!=(256,256) or not np.isfinite(heatmap).all():
            raise ValueError('Nonfinite or wrong-shaped detector response')
        arrays[case['id']]=heatmap
        normal=cache[source]
        records.append({'id':case['id'],'background_path':source,'kind':case['spec']['kind'],
                        'background_score':normal['score'],'background_alarm':normal['alarm'],
                        'generated_score':score,'generated_alarm':score>threshold,'score_change':score-normal['score']})
    torch.cuda.synchronize()
    with (folder/'heatmaps.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
    old_scale_full=read_json(ROOT/'results/weak_localization/real_evaluation/display_scale.json')
    if old_scale_full['signature']['frozen_config_sha256']!=digest(frozen_path):
        raise ValueError('Normal-derived visualization scale has different model identity')
    old_scale={key:old_scale_full[key] for key in ('rule','PatchCore_heatmap_vmin','PatchCore_heatmap_vmax','N_heatmaps_sha256')}
    result={'signature':signature,'records':records,'normal_backgrounds':len(cache),
            'normal_background_alarm_count':sum(row['alarm'] for row in cache.values()),
            'generated_images':len(records),'generated_alarm_count':sum(row['generated_alarm'] for row in records),
            'counts_by_type':{kind:{'alarm':sum(row['generated_alarm'] for row in records if row['kind']==kind),
                                   'images':sum(row['kind']==kind for row in records)} for kind in ('mura_like','line_like')},
            'source_display_scale':old_scale,'display_scale_source':'previous normal N80-derived visualization scale, not current response',
            'heatmaps_sha256':digest(folder/'heatmaps.npz'),'load_and_infer_seconds':time.perf_counter()-started,
            'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'gpu':torch.cuda.get_device_name(0),'inferred_images':len(cache)+len(records),
            'training_started':False,'bank_rebuilt':False,'threshold_changed':False,'real_test_scored':False,
            'scientific_scope':'paired synthetic development response; alarms/24 are not real recall; backgrounds n=4, not24'}
    save_json(path,result)
    del model;gc.collect();torch.cuda.empty_cache()
    print('PREDICT', {key:result[key] for key in ('normal_background_alarm_count','normal_backgrounds','generated_alarm_count',
                                                'generated_images','counts_by_type','load_and_infer_seconds')},flush=True)
    return result
