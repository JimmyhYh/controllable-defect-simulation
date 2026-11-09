"""CPU-only identity/config preparation for the independent LR-decay probe."""
import hashlib
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

from .data import OUT, ROOT, digest, read_json, resources, save_json
from . import batchstats_preparation as baseline
from .preparation import locked_json

PROBE = OUT / 'learnability_probe_lr_decay'
CONFIG = PROBE / 'config.json'
DATA = baseline.DATA
INPUTS = baseline.INPUTS
MODEL_FOLDER = ROOT / 'models/controlled_generation/learnability_probe_lr_decay_20261029'
TRAINER = Path(__file__).with_name('probe_lr_decay_training.py')
BASELINE_RESULT = baseline.PROBE / 'training_result.json'
SCHEDULE = {'base_learning_rate': 1e-4, 'milestones_completed_steps': [800], 'gamma': .2,
            'after_milestone_learning_rate': 2e-5,
            'step_numbering': '1-based optimizer updates: 1..800=1e-4;801..1000=2e-5',
            'boundary': 'after update800, before checkpoint800; all optimizer groups',
            'final_next_learning_rate': None}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_baseline():
    cfg, meta, arrays = baseline.verify_probe()
    from .batchstats_training import identity
    result = read_json(BASELINE_RESULT)
    if result['identity'] != identity(cfg, meta) or result['steps'] != 1000 or result['seed'] != 20261029:
        raise ValueError('Completed constant-LR control identity differs')
    for name, sha in result['weight_files'].items():
        if sha_file(baseline.MODEL_FOLDER/name) != sha:
            raise ValueError('Control weight file changed')
    for name, sha in result['evaluation_artifact_sha256'].items():
        if sha_file(baseline.PROBE/name) != sha:
            raise ValueError('Control saved evaluation changed')
    if (cfg['learning_rate_schedule'] != 'constant for1000-step diagnostic only'
            or cfg['learning_rate'] != 1e-4 or cfg['normalization']['track_running_stats'] is not False):
        raise ValueError('Wrong reference experiment')
    return cfg, meta, arrays, result


def proposed_config():
    cfg, meta, arrays, result = source_baseline()
    cfg = dict(cfg)
    cfg.update(schema='controlled-eight-sample-learnability-lr-decay-v1',
        only_changed_factor='learning_rate_schedule relative to completed stateless-BN constant-LR control',
        learning_rate_schedule=SCHEDULE,
        initialization='from scratch seed20261029; only own identity-matched checkpoints may resume',
        source_baseline_config_sha256=digest(baseline.CONFIG),
        source_baseline_result_sha256=digest(BASELINE_RESULT),
        source_baseline_preparation_sha256=digest(baseline.__file__),
        source_baseline_training_sha256=digest(Path(baseline.__file__).with_name('batchstats_training.py')),
        source_training_pool_pixel_AP=result['training_pool_pixel_AP'], source_gate_passed=result['gate_passed'],
        minimum_available_disk_before_start_GiB=8.0, recommended_free_GPU_GiB=4.0,
        model_output=str(MODEL_FOLDER), result_output=str(PROBE),
        training_implementation_sha256=digest(TRAINER), preparation_implementation_sha256=digest(__file__),
        formal_G0_G1_normalization='pending actual fixed1000 gate; formal common config unchanged',
        no_baseline_checkpoint_initialization=True)
    return cfg, meta, arrays


def verify_lr_decay_probe():
    cfg = read_json(CONFIG)
    expected, meta, arrays = proposed_config()
    if cfg != expected:
        raise ValueError('Independent LR probe data/code/config identity differs; preserve instead of overwriting')
    from .probe_lr_decay_training import learning_rate_for_step, schedule_state, validate_schedule_record
    for step, value in ((1, 1e-4), (800, 1e-4), (801, 2e-5), (1000, 2e-5)):
        if learning_rate_for_step(step) != value:
            raise ValueError('Frozen LR boundary differs')
    for completed in (0, 600, 800, 1000):
        validate_schedule_record(schedule_state(completed), completed)
    print('LR_DECAY_INPUT_AUDIT', {'images': 8, 'backgrounds': 4, 'positive_pixels': int(arrays[2].sum()),
                                  'optimizer_creations': 0, 'model_forwards': 0}, flush=True)
    return cfg, meta, arrays


def preparation_resources():
    result = resources()
    result['packages'].update({name: importlib.metadata.version(name) for name in ('torch','torchvision','scikit-learn')})
    result['disk_free_GiB'] = shutil.disk_usage(ROOT).free/2**30
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],
                         capture_output=True, text=True, timeout=10,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    result['GPU_readonly_query'] = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
    result['optimizer_creations'] = result['model_forwards'] = 0
    return result


def prepare_lr_decay_probe():
    cfg, _, arrays = proposed_config()
    locked_json(CONFIG, cfg)
    verify_lr_decay_probe()
    mask = arrays[2]
    locked_json(PROBE/'input_audit.json', {'config_sha256':digest(CONFIG), 'inputs_sha256':digest(DATA),
        'baseline_result_sha256':digest(BASELINE_RESULT), 'images':8, 'backgrounds':4,
        'total_pixels':int(mask.size), 'positive_pixels':int(mask.sum()), 'negative_pixels':int(mask.size-mask.sum()),
        'source_labels_and_normal_regeneration_verified':True, 'model_forwards':0, 'optimizer_creations':0,
        'manual_training_pending':True, 'scope':'same frozen8 N_gen train-side samples only'})
    resource_path = PROBE/'preparation_resources.json'
    if not resource_path.exists():
        save_json(resource_path, preparation_resources())
    print('LR_DECAY_PREPARED; training not started; control and original inputs preserved', flush=True)
    return cfg
