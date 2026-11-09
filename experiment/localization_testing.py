"""Frozen full Tfinal inference, disk-backed outputs and label-free single-image demo."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from localization_data import ROOT, OUT, SPLIT, digest

TEST_OUT = OUT/'real_evaluation'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def verified_scope():
    from localization_resources import frozen_resources
    from localization_calibration import verify_models
    cfg, split, identity = frozen_resources()
    files, _ = verify_models(identity, cfg)
    frozen_path = OUT/'frozen_config.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['signature']['resources'] != identity or frozen['signature']['models'] != files:
        raise ValueError('Frozen models or resources differ')
    if frozen['signature']['implementation_sha256'] != digest(ROOT/'localization_calibration.py'):
        raise ValueError('Frozen calibration implementation differs')
    for path, sha in frozen['calibration_artifacts'].items():
        if digest(path) != sha:
            raise ValueError('Calibration artifact changed')
    for condition, sha in frozen['training_record_sha256'].items():
        if digest(OUT/f'training_{condition}_20261020.json') != sha:
            raise ValueError('Final training record changed')
    signature = {'frozen_config_sha256': digest(frozen_path), 'split_sha256': digest(SPLIT),
                 'inference_implementation_sha256': digest(__file__), 'Tfinal_paths': split['Tfinal'],
                 'input': 'RGB whole-image bilinear256 float32/255; fixed per-model normalization',
                 'labels': 'good=0, other author types=1; original0/255 masks nearest256; good all-zero',
                 'heatmaps': 'float32 at256; continuous map bilinear if needed; no normalization'}
    return split, frozen, signature


def rgb256(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((256,256), Image.Resampling.BILINEAR), dtype=np.float32)/np.float32(255)


def masks_and_paths(split, signature):
    record = TEST_OUT/'labels.json'
    path = TEST_OUT/'masks.npy'
    if record.exists():
        prior = json.loads(record.read_text(encoding='utf-8'))
        if prior['signature'] != signature or prior['masks_sha256'] != digest(path):
            raise ValueError('Saved label identities changed')
        return prior
    manifest = json.loads((OUT/'dataset_manifest.json').read_text(encoding='utf-8'))
    if digest(OUT/'dataset_manifest.json') != split['manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    rows = {r['path']: r for r in manifest['files']}
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    masks = np.lib.format.open_memmap(path, mode='w+', dtype='uint8', shape=(1100,256,256))
    labels=[]; types=[]; original_positive=[]; mapped_positive=[]; mask_paths=[]; dropped=[]
    for i, relative in enumerate(split['Tfinal']):
        kind=Path(relative).parent.name
        labels.append(int(kind != 'good')); types.append(kind)
        original_positive.append(0); mapped_positive.append(0); mask_paths.append(None)
        masks[i]=0
        if kind != 'good':
            mask_relative=f'display/ground_truth/{kind}/{Path(relative).stem}_mask.png'
            source=Path(split['root'])/mask_relative
            if mask_relative not in rows or digest(source) != rows[mask_relative]['sha256']:
                raise ValueError('Original mask identity differs')
            with Image.open(source) as image:
                raw=np.asarray(image.convert('L'))
                if not set(np.unique(raw)).issubset({0,255}):
                    raise ValueError('Unexpected mask labels')
                original_positive[-1]=int((raw>0).sum())
                masks[i]=np.asarray(image.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
            mapped_positive[-1]=int(masks[i].sum()); mask_paths[-1]=mask_relative
            if original_positive[-1] > 0 and mapped_positive[-1] == 0:
                dropped.append(relative)
    masks.flush(); del masks; gc.collect()
    value={'signature':signature,'paths':split['Tfinal'],'image_labels':labels,'types':types,'mask_paths':mask_paths,
           'original_positive_pixels_per_image':original_positive,'positive_pixels_per_image':mapped_positive,
           'positive_pixels':sum(mapped_positive),'negative_pixels':1100*256*256-sum(mapped_positive),
           'total_pixels':1100*256*256,'normal_images':labels.count(0),'anomaly_images':labels.count(1),
           'positive_mask_lost_at256':dropped,'masks_sha256':digest(path)}
    atomic_json(record,value)
    return value


def score_all():
    import torch
    from localization_calibration import load_model, predict_map
    from localization_train import available_ram, process_ram
    split, frozen, signature=verified_scope()
    if len(split['Tfinal'])!=1100 or len(set(split['Tfinal']))!=1100:
        raise ValueError('Full Tfinal scope differs')
    TEST_OUT.mkdir(parents=True,exist_ok=True)
    labels=masks_and_paths(split,signature)
    for name in frozen['models']:
        folder=TEST_OUT/name;folder.mkdir(exist_ok=True)
        score_path=folder/'scores.npy';map_path=folder/'heatmaps.npy';record=folder/'inference.json'
        header=folder/'identity.json'
        if record.exists():
            old=json.loads(record.read_text(encoding='utf-8'))
            if old['signature']!=signature or any(digest(folder/n)!=h for n,h in old['files'].items()):
                raise ValueError('Completed predictions changed')
            print(json.dumps({'inference_cached':name}),flush=True)
            continue
        if header.exists():
            if json.loads(header.read_text(encoding='utf-8')) != signature:
                raise ValueError('Partial prediction scope differs')
            scores=np.load(score_path,mmap_mode='r+')
            maps=np.load(map_path,mmap_mode='r+')
        else:
            if score_path.exists() or map_path.exists():
                raise ValueError('Unidentified partial predictions')
            scores=np.lib.format.open_memmap(score_path,mode='w+',dtype='float64',shape=(1100,))
            maps=np.lib.format.open_memmap(map_path,mode='w+',dtype='float32',shape=(1100,256,256))
            scores[:]=np.nan;scores.flush();maps.flush()
            atomic_json(header,signature)
        missing=np.flatnonzero(np.isnan(scores));before=available_ram();started=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if len(missing):
            model=load_model(name)
            for i in missing:
                relative=split['Tfinal'][i];source=Path(split['root'])/relative
                if digest(source)!=split['source_sha256'][relative]:
                    raise ValueError('Test image identity differs')
                score,heatmap=predict_map(name,model,rgb256(source))
                if heatmap.shape!=(256,256):
                    heatmap=np.asarray(Image.fromarray(heatmap).resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
                if not np.isfinite(score) or not np.isfinite(heatmap).all():
                    raise ValueError('Nonfinite test prediction')
                maps[i]=heatmap;maps.flush()
                scores[i]=score;scores.flush()
                if (i+1)%100==0 or i+1==1100:
                    print(json.dumps({'model':name,'Tfinal_scored':int(np.isfinite(scores).sum()),'of':1100}),flush=True)
            torch.cuda.synchronize()
            if name=='PatchCore' and not (TEST_OUT/'display_scale.json').exists():
                n_path=TEST_OUT/'N_PatchCore_heatmaps.npy'
                n_maps=np.lib.format.open_memmap(n_path,mode='w+',dtype='float32',shape=(80,256,256))
                for i,relative in enumerate(split['N']):
                    _,n_maps[i]=predict_map(name,model,rgb256(Path(split['root'])/relative))
                n_maps.flush()
                upper=float(np.quantile(n_maps.reshape(-1),.995,method='linear'))
                del n_maps
                atomic_json(TEST_OUT/'display_scale.json',{'signature':signature,'rule':'N80 all heatmap pixels q0.995 linear',
                            'PatchCore_heatmap_vmin':0,'PatchCore_heatmap_vmax':upper,'N_paths':split['N'],
                            'N_heatmaps_sha256':digest(n_path),'DRAEM_range':[0,1],'scoring_changed':False})
            del model;gc.collect()
        scores.flush();maps.flush();del scores,maps;gc.collect()
        value={'signature':signature,'model':name,'images':1100,'newly_scored_this_process':len(missing),
               'this_process_load_read_infer_save_seconds':time.perf_counter()-started,'available_RAM_before_GiB':before,
               'process_ram':process_ram(),'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
               'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
               'files':{'scores.npy':digest(score_path),'heatmaps.npy':digest(map_path)},'training_started':False}
        atomic_json(record,value);torch.cuda.empty_cache()
    print(json.dumps({'full_inference_complete':True,'models':frozen['models'],'normal':labels['normal_images'],
                      'anomaly':labels['anomaly_images'],'positive_pixels':labels['positive_pixels'],
                      'positive_masks_lost_at256':len(labels['positive_mask_lost_at256'])}),flush=True)


def evaluate(stage='all'):
    if stage=='score':
        score_all()
    elif stage=='metrics':
        from localization_metrics import compute_metrics
        compute_metrics()
    else:
        # Separate processes release CUDA/model runtime before exact large-pixel sorting.
        subprocess.run([sys.executable,str(ROOT/'localization.py'),'evaluate','--stage','score'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'localization.py'),'evaluate','--stage','metrics'],cwd=ROOT,check=True)


def predict_single(image_path, name='B20261020', output_dir=None):
    split,frozen,signature=verified_scope()
    if name not in frozen['models']:raise ValueError('Model outside frozen list')
    source=Path(image_path).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    rgb=rgb256(source);cached=False
    try:
        relative=source.relative_to(Path(split['root']).resolve()).as_posix()
    except ValueError:
        relative=None
    folder=TEST_OUT/name
    if relative in split['Tfinal'] and (folder/'inference.json').exists():
        record=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
        if record['signature']!=signature or digest(source)!=split['source_sha256'][relative]:raise ValueError('Demo cache identity differs')
        for n,h in record['files'].items():
            if digest(folder/n)!=h:raise ValueError('Demo cache file differs')
        i=split['Tfinal'].index(relative)
        score=float(np.load(folder/'scores.npy',mmap_mode='r')[i])
        heatmap=np.load(folder/'heatmaps.npy',mmap_mode='r')[i].copy();cached=True
    else:
        from localization_calibration import load_model,predict_map
        model=load_model(name);score,heatmap=predict_map(name,model,rgb);del model
    if name=='PatchCore':
        scale=json.loads((TEST_OUT/'display_scale.json').read_text(encoding='utf-8'))['PatchCore_heatmap_vmax']
    else:scale=1.0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output_dir) if output_dir else OUT/'single_demo'
    output.mkdir(parents=True,exist_ok=True)
    colored=plt.get_cmap('magma')(np.clip(heatmap/scale,0,1))[...,:3]
    fig,axes=plt.subplots(1,3,figsize=(10,3.6))
    for ax,im,title in zip(axes,[rgb,heatmap,.6*rgb+.4*colored],['Input RGB','Anomaly map','Overlay']):
        if title=='Anomaly map':ax.imshow(im,cmap='magma',vmin=0,vmax=scale)
        else:ax.imshow(im)
        ax.set_title(title);ax.axis('off')
    tau=frozen['thresholds'][name]
    fig.suptitle(f'{name} | score={score:.8g} | threshold={tau:.8g} | {"ANOMALY" if score>tau else "NORMAL"}')
    fig.tight_layout()
    target=output/'preview.png'
    try:fig.savefig(target,dpi=180)
    except OSError:
        target=output/f'preview_{time.time_ns()}.png';fig.savefig(target,dpi=180)
    plt.close(fig)
    np.savez(output/'prediction.npz',score=score,heatmap=heatmap)
    result={'image':str(source),'image_sha256':digest(source),'model':name,'score':score,'threshold':tau,
            'alarm':bool(score>tau),'used_saved_test_prediction':cached,'label_used':False,'preview':str(target),
            'heatmap_range':[0,scale],'frozen_config_sha256':signature['frozen_config_sha256']}
    atomic_json(output/'prediction.json',result)
    print(json.dumps(result,ensure_ascii=True),flush=True)
