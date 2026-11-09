"""Explicit E-only single-model inference; imports are lazy, no training."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import real_scoring as r


def preflight():
    r.c.require(Path(sys.executable).resolve() == (r.c.ENV / 'Scripts/python.exe').resolve(), 'Use .venv-pretrained-seg')
    command = [str(r.c.CPU / 'Scripts/python.exe'), '-B', '-X', 'utf8', '-m', 'pretrained_synthetic_pairing.real_scoring', 'verify']
    checked = subprocess.run(command, cwd=r.c.ROOT, capture_output=True, text=True, encoding='utf-8', timeout=360, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    r.c.require(checked.returncode == 0, 'CPU E identity admission failed: ' + checked.stderr[-1800:])
    r.c.require(json.loads(checked.stdout)['E_preparation_verified'], 'Unexpected CPU admission')
    from .resources import environment_metadata
    r.c.require(environment_metadata() == r.t.load(r.PREP / 'environment.json'), 'Runtime environment differs')
    return r.t.load(r.PREP / 'protocol.json'), r.t.load(r.PREP / 'requests.json'), r.t.load(r.PREP / 'models.json')


def run(group, seed):
    p, req, models = preflight(); entry = r.a.find_model(models, group, seed); root, runtime = r.paths(group, seed); bound = r.binding(p, entry)
    if root.exists():
        r.guard(cpu=True, cache=root)
        from .real_cache import verify_results
        return verify_results(group, seed, p, req, models)
    r.c.require(r.a.new_run_allowed(root.exists(), runtime.exists()), 'Unknown/partial E output/runtime; preserve')
    snapshot = r.t.resources(model=True)
    r.guard(cpu=False, destinations=(root, runtime), raw_e=True, cache=root, temporary=runtime / 'tmp')
    r.predecessor_ready(group, seed, p, req, models)
    root.mkdir(parents=True, exist_ok=False); runtime.mkdir(parents=True, exist_ok=False)
    r.t.save(root / 'transaction.json', {'binding': bound, 'resources': snapshot, 'intended_images': 1004, 'intended_windows': 4270})
    r.t.save(runtime / 'owner.json', bound); rows = []; samples = []; current_index = None
    try:
        tmp = runtime / 'tmp'; tmp.mkdir(exist_ok=False)
        os.environ.update({'TEMP': str(tmp), 'TMP': str(tmp), 'TORCH_HOME': str(tmp/'torch'), 'HF_HOME': str(tmp/'hf'), 'XDG_CACHE_HOME': str(tmp/'xdg'),
                           'TORCHINDUCTOR_CACHE_DIR': str(tmp/'inductor'), 'TRITON_CACHE_DIR': str(tmp/'triton'), 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        import tempfile
        tempfile.tempdir = str(tmp); started = time.monotonic()
        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import load_file
        from pretrained_segmentation.model import build_model, state_hash, buffers_hash
        from .calibration_runtime import predict_native, append_intention
        model, initial = build_model(seed, {'compatibility': p['model_compatibility']})
        r.c.require(initial == entry['initialization'], 'Initial constructor/BN differs')
        model.load_state_dict(load_file(entry['final_weight_path'], device='cpu'), strict=True); model.to('cuda').eval()
        before = state_hash(model.state_dict()); buffers = buffers_hash(model)
        r.c.require(before == entry['expected_state_hash'] and buffers == entry['expected_buffers_hash'], 'Final model/BN differs')
        torch.cuda.reset_peak_memory_stats(); (root / 'probabilities').mkdir(exist_ok=False)
        for request in req:
            current_index = request['index']; snap = r.c.resource_snapshot(); snap['image_index'] = current_index
            r.c.require(snap['available_RAM_GiB'] >= 2 and snap['E_free_GiB'] >= 1, 'Per-image resource stop'); samples.append(snap)
            append_intention(root, {'event': 'decode_start', 'index': current_index, 'id': request['id']})
            raw = (r.a.SPLIT.parent / request['image']).read_bytes()
            r.c.require(hashlib.sha256(raw).hexdigest() == request['image_sha256'], 'Actual E RGB hash differs')
            with Image.open(io.BytesIO(raw)) as im:
                r.c.require(im.mode in ('RGB', 'L') and im.size == (request['width'], request['height']), 'Actual E RGB mode/shape differs')
                rgb = np.asarray(im.convert('RGB')).copy()
            def before_window(i, y):
                append_intention(root, {'event': 'window_start', 'index': current_index, 'id': request['id'], 'window_index': i, 'y': y})
            prob, count = predict_native(model, rgb, request['ys'], before_window)
            r.c.require(count == request['window_count'] and list(prob.shape) == request['native_shape'], 'E actual window count/shape differs')
            name = r.probability_path(request)
            with (root / name).open('xb') as stream:
                np.save(stream, prob, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            rows.append({'index': current_index, 'id': request['id'], 'source_image_sha256': request['image_sha256'], 'probability': name,
                         'probability_sha256': r.t.sha(root / name), 'score': float(prob.max()), 'window_forwards': count})
            print(json.dumps({'model': f'{group}_{seed}', 'image': current_index+1, 'of': 1004, 'RAM_GiB': snap['available_RAM_GiB']}), flush=True)
            del raw, rgb, prob
        torch.cuda.synchronize(); after = state_hash(model.state_dict()); after_buffers = buffers_hash(model)
        r.c.require(before == after and buffers == after_buffers, 'E inference changed model/BN')
        scores = {'binding': bound, 'items': rows, 'images_scored': 1004, 'raw_E_image_decodes': 1004, 'window_forwards': sum(x['window_forwards'] for x in rows),
                  'native_pixels': 146411528, 'constructor_forwards': 0, 'state_before': before, 'state_after': after, 'buffers_before': buffers, 'buffers_after': after_buffers,
                  'original_threshold': entry['calibration']['threshold'], 'GT_read_by_predictor': False, 'metrics_computed': False, 'training_or_parameter_updates': 0,
                  'score_rule': r.a.SCORE_RULE, 'elapsed_seconds': time.monotonic()-started, 'resource_samples': samples,
                  'sampled_min_RAM_GiB': min(x['available_RAM_GiB'] for x in samples), 'peak_cuda_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                  'peak_cuda_reserved_MiB': torch.cuda.max_memory_reserved()/2**20}
        r.t.save(root / 'scores.json', scores)
        r.t.save(root / 'E1004评分结果.md', f'# {group}_{seed} E1004评分\n\n最终10000步；1004图／4270窗、原生float32 softmax拼回max；仅保存概率／分数及原tau来源，未读GT或计算真实指标。E历史已见，仅探索性，不称新未见验证或屏幕效果。完整缓存只核验，未知事务保留停止。\n')
        r.t.save(root / 'runtime_manifest.json', {'binding': bound, 'manifest': r.a.runtime_manifest(runtime)})
        r.t.seal(root, r.run_files(req), bound)
    except BaseException as exc:
        if not (root / 'failure.json').exists(): r.t.save(root / 'failure.json', {'error': repr(exc), 'current_request_index': current_index, 'completed_images': len(rows), 'rule': 'uncertain/partial preserve; no retry or completion'})
        raise
    from .real_cache import verify_results
    return verify_results(group, seed, p, req, models)
