"""Frozen first-round models and Vnew-only calibration; never trains or opens Tfinal."""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import wide_resnet50_2

from detector import UnicodePathFaissNN
import patchcore.patchcore
import patchcore.sampler
from localization_data import ROOT, OUT, digest, save
from localization_resources import frozen_resources, MODEL_ROOT
from localization_train import ReconstructiveSubNetwork, DiscriminativeSubNetwork, available_ram, process_ram

MODELS = ('PatchCore', 'A20261020', 'B20261020')


def verify_models(identity, cfg):
    bank = json.loads((OUT/'patchcore_bank.json').read_text(encoding='utf-8'))
    if bank['identity'] != identity or not bank['N_only'] or bank['normal_images'] != 80:
        raise ValueError('PatchCore bank scope differs')
    files = {'PatchCore': {str(MODEL_ROOT/'patchcore'/n): h for n, h in bank['files'].items()}}
    records = {}
    for condition in ('A', 'B'):
        record = json.loads((OUT/f'training_{condition}_20261020.json').read_text(encoding='utf-8'))
        if (record['identity'] != identity or record['condition'] != condition or record['seed'] != 20261020
                or record['steps'] != cfg['steps'] or not record['final_weights_finite'] or record['held_out_scoring']
                or record['implementation_sha256'] != digest(ROOT/'localization_training.py')):
            raise ValueError('Formal training identity or completion differs: '+condition)
        if [p['step'] for p in record['loss_points']] != list(range(1000, 10001, 1000)):
            raise ValueError('Incomplete fixed-budget loss evidence')
        records[condition] = record
        files[condition+'20261020'] = {str(MODEL_ROOT/f'draem_{condition}_20261020'/n): h
                                     for n, h in record['weight_files'].items()}
        if set(record['weight_files']) != {'reconstruction.pth', 'segmentation.pth'}:
            raise ValueError('Unexpected final weight set')
    a, b = records['A'], records['B']
    for key in ('initial_first_conv_sha256', 'component_trace_sha256', 'active_anomalies'):
        if a[key] != b[key]:
            raise ValueError('A/B pairing differs: '+key)
    if a['replaced_anomalies'] != 0 or b['replaced_anomalies'] != b['active_anomalies']//2:
        raise ValueError('Effective anomaly replacement count differs')
    for model_files in files.values():
        for path, expected in model_files.items():
            if digest(path) != expected:
                raise ValueError('Final weight or bank hash differs: '+path)
    return files, records


def load_model(name):
    """Load final weights/bank only; identical architecture and placeholder BN initialization."""
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if name == 'PatchCore':
        device = torch.device('cuda')
        backbone = wide_resnet50_2(weights=None)
        backbone.load_state_dict(torch.load(ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth',
                                            weights_only=True, map_location='cpu'))
        backbone.name = 'wideresnet50'
        model = patchcore.patchcore.PatchCore(device)
        model.load(backbone=backbone, layers_to_extract_from=['layer2', 'layer3'], device=device,
                   input_shape=(3, 256, 256), pretrain_embed_dimension=1024, target_embed_dimension=1024,
                   patchsize=3, anomaly_score_num_nn=1,
                   featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.01, device),
                   nn_method=UnicodePathFaissNN(on_gpu=False, num_workers=4))
        model.anomaly_scorer.load(str(MODEL_ROOT/'patchcore'))
        model.forward_modules.eval()
        return model
    condition = name[0]
    folder = MODEL_ROOT/f'draem_{condition}_20261020'
    rec = ReconstructiveSubNetwork(3, 3).cuda()
    seg = DiscriminativeSubNetwork(6, 2).cuda()
    rec.load_state_dict(torch.load(folder/'reconstruction.pth', weights_only=True, map_location='cpu'))
    seg.load_state_dict(torch.load(folder/'segmentation.pth', weights_only=True, map_location='cpu'))
    rec.eval(); seg.eval()
    return rec, seg


def predict_map(name, model, rgb):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    with torch.inference_mode():
        if name == 'PatchCore':
            mean = torch.tensor([.485, .456, .406])[None, :, None, None]
            std = torch.tensor([.229, .224, .225])[None, :, None, None]
            scores, maps = model.predict((tensor-mean)/std)
            return float(scores[0]), np.asarray(maps[0], dtype=np.float32)
        x = tensor.cuda()
        rec, seg = model
        probability = torch.softmax(seg(torch.cat((rec(x), x), 1)), dim=1)[0, 1]
        return float(probability.max().cpu()), probability.cpu().numpy()


def atomic_scores(path, paths, scores, identity):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, paths=np.asarray(paths), scores=scores,
                 identity_json=np.asarray(json.dumps(identity, sort_keys=True)))
    temporary.replace(path)


def calibrate():
    cfg, split, identity = frozen_resources()
    model_files, training = verify_models(identity, cfg)
    paths = split['Vnew']
    if len(paths) != 479 or len(set(paths)) != 479 or set(paths) & set(split['N']):
        raise ValueError('Vnew scope differs')
    for relative in paths:
        if '/train/good/' not in relative or digest(Path(split['root'])/relative) != split['source_sha256'][relative]:
            raise ValueError('Vnew original image identity differs')
    signature = {'resources': identity, 'models': model_files, 'Vnew_paths': paths,
                 'implementation_sha256': digest(__file__), 'q': .95, 'method': 'linear', 'alarm': 'score>threshold'}
    frozen = OUT/'frozen_config.json'
    if frozen.exists():
        prior = json.loads(frozen.read_text(encoding='utf-8'))
        if prior['signature'] != signature:
            raise ValueError('Frozen calibration scope changed')
        for path, expected in prior['calibration_artifacts'].items():
            if digest(path) != expected:
                raise ValueError('Calibration cache identity differs')
        print(json.dumps({'calibration_cached': True, 'thresholds': prior['thresholds']}), flush=True)
        return
    thresholds = {}; summaries = {}; artifacts = {}
    for name in MODELS:
        score_path = OUT/f'Vnew_scores_{name}.npz'
        summary_path = OUT/f'calibration_{name}.json'
        score_identity = {'signature': signature, 'model': name}
        scores = np.full(len(paths), np.nan, dtype=np.float64)
        if score_path.exists():
            with np.load(score_path, allow_pickle=False) as stored:
                if stored['paths'].tolist() != paths or json.loads(str(stored['identity_json'])) != score_identity:
                    raise ValueError('Vnew score cache differs: '+name)
                scores = stored['scores'].copy()
            if scores.shape != (479,) or np.isinf(scores).any():
                raise ValueError('Invalid Vnew score cache')
        if summary_path.exists():
            result = json.loads(summary_path.read_text(encoding='utf-8'))
            if result['identity'] != score_identity or not np.isfinite(scores).all() or digest(score_path) != result['scores_sha256']:
                raise ValueError('Incomplete or changed calibration summary')
        else:
            missing = np.flatnonzero(np.isnan(scores))
            start = time.perf_counter()
            before = available_ram()
            torch.cuda.reset_peak_memory_stats()
            if len(missing):
                model = load_model(name)
                for i in missing:
                    with Image.open(Path(split['root'])/paths[i]) as image:
                        rgb = np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)/np.float32(255)
                    score, heatmap = predict_map(name, model, rgb)
                    if not np.isfinite(score) or heatmap.shape != (256, 256) or not np.isfinite(heatmap).all():
                        raise ValueError('Invalid calibration prediction')
                    scores[i] = score
                    atomic_scores(score_path, paths, scores, score_identity)
                    if (i+1)%100 == 0 or i+1 == len(paths):
                        print(json.dumps({'model': name, 'Vnew_scored': int(np.isfinite(scores).sum()), 'of': 479}), flush=True)
                torch.cuda.synchronize()
                del model; gc.collect()
            tau = float(np.quantile(scores, .95, method='linear'))
            fp = int((scores > tau).sum())
            result = {'identity': score_identity, 'model': name, 'threshold': tau, 'normal_calibration_images': 479,
                      'calibration_FP': fp, 'calibration_FPR': fp/479, 'independent_validation': False,
                      'score_min_median_max': [float(scores.min()), float(np.median(scores)), float(scores.max())],
                      'scores_sha256': digest(score_path), 'newly_scored_this_process': len(missing),
                      'this_process_load_read_infer_cache_seconds': time.perf_counter()-start,
                      'available_RAM_before_GiB': before, 'process_ram': process_ram(),
                      'gpu_peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                      'gpu_peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
                      'Tfinal_read': False, 'training_started': False}
            save(summary_path, result)
            torch.cuda.empty_cache()
        thresholds[name] = result['threshold']; summaries[name] = result
        artifacts[str(score_path)] = digest(score_path); artifacts[str(summary_path)] = digest(summary_path)
    pair = {'steps_each': 10000, 'seed': 20261020,
            'component_trace_sha256': training['A']['component_trace_sha256'],
            'initial_first_conv_sha256': training['A']['initial_first_conv_sha256'],
            'active_anomalies_each': training['A']['active_anomalies'],
            'B_weak_replacements': training['B']['replaced_anomalies'],
            'B_actual_replacement_fraction': training['B']['replaced_anomalies']/training['B']['active_anomalies'],
            'pairing_verification': 'final rolling component digest includes shared background, texture, mask and sampling metadata'}
    save(frozen, {'signature': signature, 'models': list(MODELS), 'thresholds': thresholds,
                  'calibration_artifacts': artifacts, 'paired_training': pair,
                  'training_record_sha256': {c: digest(OUT/f'training_{c}_20261020.json') for c in ('A', 'B')},
                  'input': 'whole RGB bilinear256, float32/255; PatchCore ImageNet normalization; DRAEM no extra normalization',
                  'scoring': {'PatchCore': 'raw squaredL2 nearest-neighbor maximum patch distance',
                              'DRAEM': 'maximum anomaly softmax probability from concat(reconstruction,input), eval mode'},
                  'threshold_rule': 'Vnew479 q0.95 linear; strict score>threshold; no cross-image normalization',
                  'calibration_only': True, 'Tfinal_scored': False, 'no_model_selection': True})
    print(json.dumps({'calibration_complete': True, 'thresholds': thresholds,
                      'calibration_FP_over_479': {n: r['calibration_FP'] for n, r in summaries.items()},
                      'Tfinal_scored': False}), flush=True)
