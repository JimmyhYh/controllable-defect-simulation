"""Train-side resource probe only; does not create a production checkpoint."""
import ctypes
import gc
import hashlib
import sys
import time
import numpy as np
import torch
from localization_data import ROOT
from localization_synthesis import PairStream, blend
sys.path.insert(0, str(ROOT / 'vendor/DRAEM'))
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]

def available_ram():
    m = Memory(); m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('RAM query failed')
    return m.available / 2**30

def process_ram():
    class Info(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in ['peak','resident','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile']]
    info = Info(); info.cb = ctypes.sizeof(info)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info), ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise OSError('Process memory query failed')
    return {'resident_GiB':info.resident/2**30, 'peak_working_set_GiB':info.peak/2**30}

def weights_init(module):
    name = module.__class__.__name__
    if 'Conv' in name:
        module.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in name:
        module.weight.data.normal_(1.0, 0.02); module.bias.data.fill_(0)

def run_probe(split, weak, condition, precision='fp32'):
    torch.set_num_threads(4)
    torch.manual_seed(20261020); torch.cuda.manual_seed_all(20261020)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    rec = ReconstructiveSubNetwork(3, 3).cuda()
    seg = DiscriminativeSubNetwork(6, 2).cuda()
    rec.apply(weights_init); seg.apply(weights_init)
    params = list(rec.parameters()) + list(seg.parameters())
    initial = rec.encoder.block1[0].weight.detach().clone()
    initial_hash = hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0)
    mse = torch.nn.MSELoss(); ssim = SSIM(); focal = FocalLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=precision != 'fp32')
    stream = PairStream(split, 20261020 + 100000, first_eight=True)
    times = []; losses = []; count = 0; replaced = 0
    min_ram = available_ram()
    started = time.perf_counter()
    for step in range(40):
        torch.cuda.synchronize(); begin = time.perf_counter()
        sample = stream.next()
        strength = weak if condition == 'B' and sample['replace'] else sample['lambda_A']
        aug = blend(sample, strength)
        x = torch.from_numpy(np.ascontiguousarray(aug.transpose(2,0,1))).unsqueeze(0).cuda()
        clean = torch.from_numpy(np.ascontiguousarray(sample['background'].transpose(2,0,1))).unsqueeze(0).cuda()
        mask = torch.from_numpy(sample['mask'].astype(np.float32)[None,None]).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=precision != 'fp32'):
            reconstructed = rec(x)
            probabilities = torch.softmax(seg(torch.cat((reconstructed, x), 1)), dim=1)
            # Author losses remain float32 for AMP numerical stability.
            loss = mse(reconstructed.float(), clean) + ssim(reconstructed.float(), clean) + focal(probabilities.float(), mask)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite probe loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('Nonfinite gradients')
        scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        if step >= 10:
            times.append(time.perf_counter() - begin)
        losses.append(float(loss.detach().cpu()))
        count += int(sample['active']); replaced += int(sample['replace'])
        min_ram = min(min_ram, available_ram())
    result = {'condition': condition, 'precision': precision, 'warmup_steps':10, 'timed_steps':30,
              'step_times_seconds': times, 'median_step_seconds': float(np.median(times)),
              'losses': losses, 'weights_changed': bool(not torch.equal(initial, rec.encoder.block1[0].weight)),
              'initial_first_conv_sha256': initial_hash, 'effective_anomalies': count,
              'eligible_replacements': replaced, 'elapsed_seconds': time.perf_counter()-started,
              'gpu_peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
              'gpu_peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
              'minimum_available_ram_GiB': min_ram, 'parameters': sum(p.numel() for p in params),
              'process_ram':process_ram(), 'checkpoint_saved': False, 'background_pool': split['N'][:8]}
    if not result['weights_changed']:
        raise RuntimeError('No weight update')
    del rec, seg, optimizer, params, scaler, stream, initial, x, clean, mask, reconstructed, probabilities, loss
    gc.collect(); torch.cuda.empty_cache()
    return result
