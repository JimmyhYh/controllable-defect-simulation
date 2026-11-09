"""Read-only fixed8/checkpoint diagnosis: no optimizer, backward or train call."""
import gc
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from .data import ROOT, OUT, digest, read_json, resources, save_json
from . import preparation as original
from .batchstats_preparation import CONFIG, DATA, INPUTS, MODEL_FOLDER, PROBE, verify_probe
from .batchstats_training import (available_ram, completed_status, configure_stateless_bn,
                                 identity, parameters_hash, resume_preflight, verify_stateless_bn)

FOLDER = OUT / 'probe_learning_diagnosis'
STEPS = (200, 400, 600, 800, 1000)


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def state_hash(model):
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def provenance():
    paths = [CONFIG, DATA, INPUTS, original.CONFIG, original.PROBE/'training_result.json',
             PROBE/'training_result.json', PROBE/'final_train_heatmaps.npy', MODEL_FOLDER/'latest.json',
             original.PROBE/'diagnosis/normalization_diagnosis.json',
             original.PROBE/'diagnosis/normalization_maps.npz',
             ROOT/'data/controlled_generation/split.json',
             OUT/'training_preparation/g1_sampling.json', OUT/'training_preparation/N_gen_sampling_audit.json',
             OUT/'training_preparation/common_training_config.json']
    for folder in (MODEL_FOLDER, original.MODEL_FOLDER):
        paths += [folder/'reconstruction.pth', folder/'segmentation.pth']
    paths += [MODEL_FOLDER/'checkpoints'/f'step_{step:06d}.pt' for step in STEPS]
    paths += [Path(__file__).with_name(name) for name in
              ('data.py', 'geometry.py', 'render.py', 'sampling.py', 'preparation.py', 'training.py',
               'batchstats_preparation.py', 'batchstats_training.py')]
    paths += [ROOT/'vendor/DRAEM'/name for name in ('model_unet.py', 'loss.py')]
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def diagnose():
    cfg, meta, (images, targets, labels) = verify_probe()
    finished = completed_status()
    signature = identity(cfg, meta)
    header = resume_preflight(signature)
    if finished.get('steps') != 1000 or header is None or header['step'] != 1000:
        raise ValueError('All five committed checkpoints and completed fixed1000 result are required')
    old = read_json(original.PROBE/'training_result.json')
    bn = read_json(original.PROBE/'diagnosis/normalization_diagnosis.json')
    if (bn['probe_result_sha256'] != digest(original.PROBE/'training_result.json')
            or bn['weights_sha256'] != old['weight_files']
            or bn['maps_sha256'] != digest(original.PROBE/'diagnosis/normalization_maps.npz')):
        raise ValueError('Original saved BN diagnosis provenance differs')
    source = provenance()
    if (FOLDER/'diagnostic_results.json').exists():
        saved = read_json(FOLDER/'diagnostic_results.json')
        if saved['source_sha256'] != source or saved['diagnostic_code_sha256'] != digest(__file__):
            raise ValueError('Existing diagnostic identity differs; preserve instead of overwriting')
        for name, sha in saved['artifacts_sha256'].items():
            if sha_file(FOLDER/name) != sha:
                raise ValueError('Saved diagnostic artifact differs')
        print('LEARNING_DIAGNOSIS_CACHED', [r['step'] for r in saved['checkpoints']], flush=True)
        return saved
    if FOLDER.exists() and any(FOLDER.iterdir()):
        raise ValueError('Partial diagnostic output exists; preserve and diagnose before rerunning')
    before_resources = resources()
    if available_ram() < 3.0:
        raise RuntimeError('Read-only checkpoint diagnosis requires at least3.0GiB free RAM; no inference started')
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import average_precision_score
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen FP32/CUDA diagnosis needs CUDA; no automatic CPU fallback')
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    if free_gpu < 2*2**30:
        raise RuntimeError('Need at least2GiB free GPU memory before diagnosis; no inference started')
    sys.path.insert(0, str(ROOT/'vendor/DRAEM'))
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
    from loss import SSIM, FocalLoss
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    rec = ReconstructiveSubNetwork(3, 3).cuda()
    seg = DiscriminativeSubNetwork(6, 2).cuda()
    bn_counts = {'rec': configure_stateless_bn(torch, rec), 'seg': configure_stateless_bn(torch, seg)}
    rec.eval().requires_grad_(False)
    seg.eval().requires_grad_(False)
    ssim, focal = SSIM(), FocalLoss()
    rows = []
    reconstructions = np.empty((5, 8, 256, 256, 3), np.float32)
    maps = np.empty((5, 8, 256, 256), np.float32)
    reconstruction_residuals = np.empty_like(maps)
    start = time.perf_counter()
    minimum_ram = available_ram()
    for j, step in enumerate(STEPS):
        path = MODEL_FOLDER/'checkpoints'/f'step_{step:06d}.pt'
        if sha_file(path) != source[str(path.relative_to(ROOT))]:
            raise ValueError('Checkpoint changed before loading')
        # Locally user-produced checkpoint, fixed path/source fingerprint; mmap avoids loading optimizer tensors.
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if (state['identity'] != signature or state['step'] != step
                or len(state['losses']) != step or len(state['gradient_maxima']) != step
                or state['initial_rec_parameters_sha256'] != cfg['expected_initial_parameters_sha256']['rec']
                or state['initial_seg_parameters_sha256'] != cfg['expected_initial_parameters_sha256']['seg']):
            raise ValueError('Checkpoint internal step, identity or initial parameter hashes differ')
        rec.load_state_dict(state['rec'], strict=True)
        seg.load_state_dict(state['seg'], strict=True)
        del state
        gc.collect()
        verify_stateless_bn(torch, rec)
        verify_stateless_bn(torch, seg)
        initial = {'rec': state_hash(rec), 'seg': state_hash(seg)}
        samples = []
        with torch.inference_mode():
            for i, sample in enumerate(meta['rows']):
                x = torch.from_numpy(np.ascontiguousarray(images[i].transpose(2, 0, 1))).unsqueeze(0).cuda()
                clean = torch.from_numpy(np.ascontiguousarray(targets[i].transpose(2, 0, 1))).unsqueeze(0).cuda()
                mask = torch.from_numpy(labels[i].astype(np.float32)[None, None]).cuda()
                reconstructed = rec(x)
                probabilities = torch.softmax(seg(torch.cat((reconstructed, x), dim=1)), dim=1)
                components = {'MSE': float(functional.mse_loss(reconstructed, clean).cpu()),
                              'SSIM_loss': float(ssim(reconstructed, clean).cpu()),
                              'Focal': float(focal(probabilities, mask).cpu())}
                prediction = reconstructed[0].permute(1, 2, 0).cpu().numpy()
                heat = probabilities[0, 1].cpu().numpy()
                if not (np.isfinite(prediction).all() and np.isfinite(heat).all()
                        and np.isfinite(list(components.values())).all()):
                    raise ValueError('Nonfinite diagnostic output')
                reconstructions[j, i] = prediction
                maps[j, i] = heat
                residual = np.max(np.abs(prediction-targets[i]), axis=-1)
                reconstruction_residuals[j, i] = residual
                m = labels[i]
                prediction_mean = prediction.mean(axis=(0, 1))
                target_mean = targets[i].mean(axis=(0, 1))
                samples.append({'index': i, 'background': sample['background'], 'kind': sample['kind'],
                    'positive_pixels': int(m.sum()), 'negative_pixels': int(m.size-m.sum()),
                    'losses': components, 'total_loss': sum(components.values()),
                    'reconstruction_range': [float(prediction.min()), float(prediction.max())],
                    'reconstruction_channel_mean_RGB': prediction_mean.tolist(),
                    'target_channel_mean_RGB': target_mean.tolist(),
                    'channel_mean_error_RGB': (prediction_mean-target_mean).tolist(),
                    'reconstruction_channel_fraction_outside_0_1': float(((prediction < 0) | (prediction > 1)).mean()),
                    'author_SSIM_dynamic_range': (255 if prediction.max() > 128 else 1)-( -1 if prediction.min() < -.5 else 0),
                    'within_image_pixel_AP': float(average_precision_score(m.ravel(), heat.ravel())) if m.any() else None,
                    'positive_mean_score': float(heat[m].mean()) if m.any() else None,
                    'negative_mean_score': float(heat[~m].mean()),
                    'negative_q95_score': float(np.quantile(heat[~m], .95)),
                    'max_score': float(heat.max())})
        final = {'rec': state_hash(rec), 'seg': state_hash(seg)}
        if initial != final or any(p.grad is not None for net in (rec, seg) for p in net.parameters()):
            raise RuntimeError('Read-only diagnosis altered network state or created gradients')
        aggregate = {name: float(np.mean([s['losses'][name] for s in samples]))
                     for name in ('MSE', 'SSIM_loss', 'Focal')}
        ap = float(average_precision_score(labels.ravel().astype(np.uint8), maps[j].ravel()))
        row = {'step': step, 'global_training_pool_pixel_AP': ap, 'mean_losses': aggregate,
               'mean_total_loss': sum(aggregate.values()), 'samples': samples,
               'state_sha256_before': initial, 'state_sha256_after': final,
               'parameters_and_buffers_unchanged': True, 'gradients_absent': True}
        rows.append(row)
        torch.cuda.synchronize()
        minimum_ram = min(minimum_ram, available_ram())
        print('CHECKPOINT_DIAGNOSIS', {'step': step, 'AP': ap, 'mean_losses': aggregate}, flush=True)
    saved_maps = np.load(PROBE/'final_train_heatmaps.npy', allow_pickle=False)
    map_error = float(np.max(np.abs(saved_maps-maps[-1])))
    mse_error = float(np.max(np.abs(np.array([s['losses']['MSE'] for s in rows[-1]['samples']])
                                    -np.array(finished['reconstruction_mse_by_sample']))))
    final_parameters = {'rec': parameters_hash(rec), 'seg': parameters_hash(seg)}
    if (map_error > 1e-6 or mse_error > 1e-6
            or abs(rows[-1]['global_training_pool_pixel_AP']-finished['training_pool_pixel_AP']) > 1e-8
            or final_parameters != {'rec': finished['final_reconstruction_parameters_sha256'],
                                    'seg': finished['final_segmentation_parameters_sha256']}):
        raise ValueError('Final checkpoint does not reproduce frozen result within tolerance')
    if provenance() != source:
        raise RuntimeError('Protected source content changed during diagnosis')
    FOLDER.mkdir(parents=True)
    arrays_path = FOLDER/'diagnostic_arrays.npz'
    with arrays_path.open('wb') as stream:
        np.savez_compressed(stream, steps=np.array(STEPS), inputs=images, targets=targets, labels=labels,
                            reconstructions=reconstructions, reconstruction_residuals=reconstruction_residuals,
                            heatmaps=maps)
    result = {'schema': 'controlled-fixed8-readonly-checkpoint-diagnosis-v1', 'source_sha256': source,
        'diagnostic_code_sha256': digest(__file__), 'steps': list(STEPS), 'checkpoints': rows,
        'images_per_checkpoint': 8, 'unique_backgrounds': 4, 'total_pixels_per_checkpoint': int(labels.size),
        'positive_pixels_per_checkpoint': int(labels.sum()), 'negative_pixels_per_checkpoint': int(labels.size-labels.sum()),
        'different_checkpoints_not_independent_samples': True, 'normalization': cfg['normalization'],
        'BN_modules': bn_counts, 'gate_evaluation_step': 1000, 'gate_passed_unchanged': finished['gate_passed'],
        'original_AP': old['training_pool_pixel_AP'],
        'reused_original_weight_BN_diagnostic_AP': bn['modes']['both_batch_statistics']['global_training_pool_pixel_AP'],
        'final_map_max_abs_reproduction_error': map_error, 'final_MSE_max_abs_reproduction_error': mse_error,
        'final_parameter_hashes_match': True, 'all_protected_sources_unchanged': True,
        'optimizer_creations': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'BN_buffer_updates': 0,
        'model_sample_forward_pairs': 40, 'held_out_inference': False, 'real_evaluation': False,
        'scientific_scope': 'train-side fitting diagnosis only; inference loss is not per-training-step log or gradient attribution',
        'resources_before': before_resources, 'GPU': torch.cuda.get_device_name(0),
        'free_GPU_before_MiB': free_gpu/2**20, 'GPU_total_MiB': total_gpu/2**20,
        'GPU_peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
        'GPU_peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
        'minimum_sampled_available_RAM_GiB': minimum_ram, 'elapsed_seconds': time.perf_counter()-start,
        'artifacts_sha256': {'diagnostic_arrays.npz': sha_file(arrays_path)}}
    save_json(FOLDER/'diagnostic_results.json', result)
    print('LEARNING_DIAGNOSIS_FINISHED', {'steps': list(STEPS), 'updates': 0, 'original_gate_changed': False}, flush=True)
    return result


if __name__ == '__main__':
    diagnose()
