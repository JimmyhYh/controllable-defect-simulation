"""Freeze a separate stateless-BN probe; CPU input checks only, no training."""
from pathlib import Path

import numpy as np

from .data import OUT, ROOT, SPLIT, digest, load_split, normal_rgb, read_json, resources, save_json
from . import preparation as original
from .render import render
from .sampling import AUDIT_PATH, POLICY_PATH, generator_identity, validate_policy

PROBE = OUT / 'learnability_probe_batchstats'
CONFIG = PROBE / 'config.json'
DATA = original.DATA
INPUTS = original.INPUTS
MODEL_FOLDER = ROOT / 'models/controlled_generation/learnability_probe_batchstats_20261029'
TRAINER = Path(__file__).with_name('batchstats_training.py')
OLD_RESULT = original.PROBE / 'training_result.json'


def source_inputs():
    """Read original frozen materials without rewriting any original artifact."""
    cfg = read_json(original.CONFIG)
    meta = read_json(INPUTS)
    split = load_split()
    bindings = [(original.CONFIG, meta['config_sha256']), (DATA, meta['arrays_sha256']),
                (SPLIT, cfg['split_sha256']), (POLICY_PATH, cfg['sampling_policy_sha256']),
                (AUDIT_PATH, cfg['sampling_audit_sha256']), (original.COMMON, cfg['common_config_sha256']),
                (original.__file__, cfg['preparation_implementation_sha256']),
                (Path(original.__file__).with_name('training.py'), cfg['training_implementation_sha256'])]
    if any(digest(path) != sha for path, sha in bindings):
        raise ValueError('Original probe data/code/config identity differs; preserve and diagnose')
    validate_policy(read_json(POLICY_PATH))
    if not read_json(AUDIT_PATH)['ready_for_probe_preparation']:
        raise ValueError('G1 sampling audit is not ready')
    if (cfg['normal_paths'] != split['N_gen'][:4] or cfg['generator_identity'] != generator_identity()
            or cfg['author'] != original.pinned_author()):
        raise ValueError('Original train-side scope or pinned source changed')
    old = read_json(OLD_RESULT)
    if (old['identity']['config_sha256'] != digest(original.CONFIG)
            or old['identity']['inputs_sha256'] != digest(DATA)
            or old['identity']['inputs_metadata_sha256'] != digest(INPUTS)
            or old['identity']['implementation_sha256'] != cfg['training_implementation_sha256']
            or old['steps'] != 1000 or old['seed'] != cfg['seed']):
        raise ValueError('Original completed probe provenance differs')
    for name, sha in old['weight_files'].items():
        if digest(original.MODEL_FOLDER / name) != sha:
            raise ValueError('Original weights changed')
    for name, sha in old['evaluation_artifact_sha256'].items():
        if digest(original.PROBE / name) != sha:
            raise ValueError('Original evaluation artifact changed')
    with np.load(DATA, allow_pickle=False) as saved:
        x, clean, mask = saved['x'], saved['clean'], saved['mask']
    if (x.shape != (8, 256, 256, 3) or clean.shape != x.shape or mask.shape != (8, 256, 256)
            or x.dtype != np.float32 or clean.dtype != np.float32 or mask.dtype != np.bool_):
        raise ValueError('Incorrect frozen input shape/dtype')
    if (not np.isfinite(x).all() or not np.isfinite(clean).all()
            or x.min() < 0 or x.max() > 1 or clean.min() < 0 or clean.max() > 1):
        raise ValueError('Invalid float RGB')
    if not np.array_equal(mask, np.max(np.abs(x-clean), axis=-1) > np.float32(1/255)):
        raise ValueError('Labels differ from actual float residual')
    for i, relative in enumerate(cfg['normal_paths']):
        rgb = normal_rgb(split, relative)
        generated = render(rgb, cfg['anomaly_specs'][i], cfg['outline_seeds'][i])
        if (not np.array_equal(x[2*i], rgb) or not np.array_equal(clean[2*i], rgb)
                or not np.array_equal(clean[2*i+1], rgb)
                or not np.array_equal(x[2*i+1], generated['image']) or not generated['stats']['valid']):
            raise ValueError('Frozen samples differ from N_gen-only regeneration')
    return cfg, meta, (x, clean, mask), old


def proposed_config():
    old_cfg, meta, arrays, old = source_inputs()
    cfg = dict(old_cfg)
    cfg.update(schema='controlled-eight-sample-learnability-stateless-bn-v1',
               normalization={'module': 'author BatchNorm2d, same eps/affine/placement',
                              'track_running_stats': False, 'running_buffers': None,
                              'train_and_eval': 'current single-input batch/spatial statistics, batch_size1',
                              'scope': 'both reconstructive and discriminative networks; no calibration or held-out mixing'},
               only_changed_factor='BN running-statistics policy; data/initial parameters/loss/order/budget unchanged',
               initialization='from scratch with seed20261029; no original trained weights/checkpoint loaded',
               eval='all8 once at final1000, eval/inference_mode, stateless BN, exact global pixelAP',
               expected_initial_parameters_sha256={
                   'rec': old['initial_reconstruction_parameters_sha256'],
                   'seg': old['initial_segmentation_parameters_sha256']},
               source_config_sha256=digest(original.CONFIG), source_result_sha256=digest(OLD_RESULT),
               source_inputs_metadata_sha256=digest(INPUTS), source_inputs_sha256=digest(DATA),
               source_training_implementation_sha256=old_cfg['training_implementation_sha256'],
               source_preparation_implementation_sha256=old_cfg['preparation_implementation_sha256'],
               source_training_pool_pixel_AP=old['training_pool_pixel_AP'], source_gate_passed=old['gate_passed'],
               inputs_file=str(DATA), model_output=str(MODEL_FOLDER),
               training_implementation_sha256=digest(TRAINER),
               preparation_implementation_sha256=digest(__file__),
               formal_G0_G1_normalization='pending this separate probe; old common config remains unchanged')
    return cfg, meta, arrays


def verify_probe():
    cfg = read_json(CONFIG)
    expected, meta, arrays = proposed_config()
    if cfg != expected:
        raise ValueError('Independent probe protocol/source identity differs; do not overwrite')
    labels = arrays[2]
    print('BATCHSTATS_INPUT_AUDIT', {'images': 8, 'backgrounds': 4, 'positive_pixels': int(labels.sum()),
                                  'optimizer_steps': 0, 'model_inferences': 0}, flush=True)
    return cfg, meta, arrays


def prepare_probe():
    cfg, _, arrays = proposed_config()
    original.locked_json(CONFIG, cfg)
    verify_probe()
    labels = arrays[2]
    original.locked_json(PROBE / 'input_audit.json', {
        'config_sha256': digest(CONFIG), 'inputs_sha256': digest(DATA), 'samples': 8,
        'unique_backgrounds': 4, 'total_pixels': int(labels.size), 'positive_pixels': int(labels.sum()),
        'negative_pixels': int(labels.size-labels.sum()), 'all_float_labels_match_actual_residual': True,
        'N_gen_only_regeneration_matches': True, 'optimizer_steps': 0, 'model_inferences': 0,
        'original_results_preserved': True, 'manual_training_pending': True,
        'scope': 'same frozen first4 N_gen; no S_bg/Q_cal/real evaluation inputs or scores'})
    path = PROBE / 'preparation_resources.json'
    if not path.exists():
        save_json(path, resources())
    print('BATCHSTATS_PREPARED; training not started; old probe preserved', flush=True)
    return cfg
