"""Read-only final-probe BN diagnosis; no optimizer, gradients or buffer updates.

Alternative normalization is diagnostic, not a replacement for frozen evaluation.
"""
import hashlib
import sys
import time
import types

import numpy as np

from .data import ROOT, digest, save_json
from .preparation import MODEL_FOLDER, PROBE, verify_probe
from .training import completed_status


def state_hash(model):
    sha=hashlib.sha256()
    for name,value in model.state_dict().items():
        sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def diagnose():
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import average_precision_score
    cfg,metadata,(images,targets,labels)=verify_probe()
    result=completed_status()
    if result.get('steps')!=1000:raise ValueError('Diagnostic requires completed fixed1000-step probe')
    weight_sha={name:digest(MODEL_FOLDER/name) for name in ('reconstruction.pth','segmentation.pth')}
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'vendor/DRAEM'))
    from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork
    rec=ReconstructiveSubNetwork(3,3).cuda();seg=DiscriminativeSubNetwork(6,2).cuda()
    rec.load_state_dict(torch.load(MODEL_FOLDER/'reconstruction.pth',map_location='cpu',weights_only=True))
    seg.load_state_dict(torch.load(MODEL_FOLDER/'segmentation.pth',map_location='cpu',weights_only=True))
    rec.eval();seg.eval();initial={'rec':state_hash(rec),'seg':state_hash(seg)}
    bns={name:[layer for layer in network.modules() if isinstance(layer,torch.nn.BatchNorm2d)]
         for name,network in [('rec',rec),('seg',seg)]}
    original={id(layer):layer.forward for group in bns.values() for layer in group}
    def batch_statistics_without_update(layer,x):
        # No running buffers supplied: current-input statistics, fixed learned affine.
        # inference_mode covers this entire call; no parameter/buffer update occurs.
        return functional.batch_norm(x,None,None,layer.weight,layer.bias,True,0.0,layer.eps)
    modes=[('original_eval',[]),('rec_batch_statistics',['rec']),('seg_batch_statistics',['seg']),
           ('both_batch_statistics',['rec','seg'])]
    summaries={};maps_by_mode={};start=time.perf_counter()
    for name,changed in modes:
        for group in bns.values():
            for layer in group:layer.forward=original[id(layer)]
        for network in changed:
            for layer in bns[network]:layer.forward=types.MethodType(batch_statistics_without_update,layer)
        maps=np.empty((8,256,256),np.float32);mse=[];ranges=[]
        with torch.inference_mode():
            for i in range(8):
                x=torch.from_numpy(np.ascontiguousarray(images[i].transpose(2,0,1))).unsqueeze(0).cuda()
                clean=torch.from_numpy(np.ascontiguousarray(targets[i].transpose(2,0,1))).unsqueeze(0).cuda()
                reconstructed=rec(x)
                probability=torch.softmax(seg(torch.cat((reconstructed,x),dim=1)),dim=1)[0,1]
                maps[i]=probability.cpu().numpy()
                mse.append(float(functional.mse_loss(reconstructed,clean).cpu()))
                ranges.append([float(reconstructed.min().cpu()),float(reconstructed.max().cpu())])
        if not np.isfinite(maps).all():raise ValueError('Nonfinite diagnostic inference')
        per_image=[]
        for i,row in enumerate(metadata['rows']):
            mask=labels[i];scores=maps[i]
            per_image.append({'index':i,'kind':row['kind'],'positive_pixels':int(mask.sum()),
                'pixel_AP':float(average_precision_score(mask.reshape(-1),scores.reshape(-1))) if mask.any() else None,
                'foreground_mean':float(scores[mask].mean()) if mask.any() else None,
                'background_mean':float(scores[~mask].mean()),'reconstruction_MSE':mse[i],
                'reconstruction_range':ranges[i]})
        summaries[name]={'global_training_pool_pixel_AP':float(average_precision_score(labels.reshape(-1),maps.reshape(-1))),
                         'mean_reconstruction_MSE':float(np.mean(mse)),'per_image':per_image,
                         'normalization_changed_for_diagnostic_only':changed}
        maps_by_mode[name]=maps
        print('BN_DIAGNOSTIC',name,summaries[name]['global_training_pool_pixel_AP'],summaries[name]['mean_reconstruction_MSE'],flush=True)
    for group in bns.values():
        for layer in group:layer.forward=original[id(layer)]
    final={'rec':state_hash(rec),'seg':state_hash(seg)}
    if final!=initial:raise RuntimeError('Diagnostic unexpectedly altered parameters or buffers')
    if any(digest(MODEL_FOLDER/name)!=sha for name,sha in weight_sha.items()):raise RuntimeError('Weight file changed')
    saved_maps=np.load(PROBE/'final_train_heatmaps.npy')
    max_error=float(np.max(np.abs(saved_maps-maps_by_mode['original_eval'])))
    if max_error>1e-6:raise ValueError('Original evaluation does not reproduce saved maps within tolerance')
    folder=PROBE/'diagnosis';folder.mkdir(exist_ok=True)
    with (folder/'normalization_maps.npz').open('wb') as stream:np.savez_compressed(stream,**maps_by_mode)
    report={'probe_result_sha256':digest(PROBE/'training_result.json'),'diagnostic_code_sha256':digest(__file__),
            'weights_sha256':weight_sha,'modes':summaries,'parameters_and_BN_buffers_unchanged':initial==final,
            'original_map_max_abs_reproduction_error':max_error,'samples':8,'unique_backgrounds':4,
            'positive_pixels':int(labels.sum()),'negative_pixels':int(labels.size-labels.sum()),
            'train_side_prevalence':float(labels.mean()),'diagnostic_elapsed_seconds':time.perf_counter()-start,
            'optimizer_updates':0,'backward_calls':0,'running_statistics_updates':0,
            'held_out_scoring':False,'original_gate_changed':False,
            'scope':'same trained weights and8 training-side inputs; altered normalization only to isolate failure, not a new passing result',
            'official_BN_reference':'https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html',
            'maps_sha256':digest(folder/'normalization_maps.npz')}
    save_json(folder/'normalization_diagnosis.json',report)
    return report


if __name__=='__main__':diagnose()
