"""New AnoLCD PatchCore bank; held-out evaluation is not implemented here."""
import json
import random
import time
import numpy as np
import torch
from torchvision.models import wide_resnet50_2
from detector import UnicodePathFaissNN
import patchcore.patchcore
import patchcore.sampler
from localization_data import ROOT, OUT, save, normal_rgb, digest
from localization_resources import frozen_resources, MODEL_ROOT
from localization_train import process_ram, available_ram

def fit_patchcore():
    config, split, identity = frozen_resources()
    target = MODEL_ROOT/'patchcore'
    record = OUT/'patchcore_bank.json'
    if record.exists():
        prior = json.loads(record.read_text(encoding='utf-8'))
        if prior['identity'] != identity or any(digest(target/n) != h for n,h in prior['files'].items()):
            raise ValueError('Saved bank identity differs')
        print(json.dumps({'bank_cached':True, 'vectors':prior['vectors']}), flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise ValueError('Partial bank exists; inspect before rebuilding')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    backbone=wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth', map_location='cpu', weights_only=True))
    backbone.name='wideresnet50'
    model=patchcore.patchcore.PatchCore(device)
    model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],device=device,
               input_shape=(3,256,256),pretrain_embed_dimension=1024,target_embed_dimension=1024,
               patchsize=3,anomaly_score_num_nn=1,
               featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.01,device),
               nn_method=UnicodePathFaissNN(on_gpu=False,num_workers=4))
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    def batches():
        for path in split['N']:
            rgb=normal_rgb(split,path)
            tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))
            yield ((tensor-mean)/std).unsqueeze(0)
    before=available_ram()
    model.fit(batches())
    torch.cuda.synchronize()
    target.mkdir(parents=True,exist_ok=True)
    model.save_to_path(str(target))
    nn=model.anomaly_scorer.nn_method.search_index
    result={'identity':identity,'vectors':nn.ntotal,'dimensions':nn.d,'normal_images':80,
            'elapsed_load_fit_save_seconds':time.perf_counter()-started,
            'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'process_ram':process_ram(),'RAM_available_before_fit_GiB':before,
            'N_only':True,'Vnew_Tfinal_read':False,'scoring':False,'threshold_calibrated':False,
            'seed':42,'backbone':'WideResNet50-2 ImageNetV1','input':256,'coreset':.01,
            'score':'squaredL2 nearest-neighbor maximum patch; no score computed on held-out data',
            'initialization':'author load placeholder-forward/BN behavior retained',
            'files':{p.name:digest(p) for p in target.iterdir() if p.is_file()}}
    save(record,result)
    print(json.dumps(result,ensure_ascii=True),flush=True)
