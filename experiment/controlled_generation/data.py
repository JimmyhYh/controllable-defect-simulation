"""Normal-only development pools; old test content is never opened here."""
import ctypes
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/controlled_generation'
SPLIT = ROOT / 'data/controlled_generation/split.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def code_identity():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in ('total', 'available', 'page_total', 'page_available',
                                               'virtual_total', 'virtual_available', 'extended')]
    record = {'python': platform.python_version(), 'system': platform.platform(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('numpy', 'Pillow', 'opencv-python', 'scipy', 'matplotlib', 'imgaug')},
              'generation_device': 'CPU', 'training_started': False}
    if platform.system() == 'Windows':
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError('Unable to query actual Windows memory')
        record.update(total_RAM_GiB=memory.total / 2**30, available_RAM_GiB=memory.available / 2**30)
    return record


def prepare():
    old_path = ROOT / 'data/weak_localization/split.json'
    manifest_path = ROOT / 'results/weak_localization/dataset_manifest.json'
    old = read_json(old_path)
    if digest(manifest_path) != old['manifest_sha256']:
        raise ValueError('Old manifest identity changed')
    manifest = read_json(manifest_path)
    rows = {row['path']: row for row in manifest['files']}
    # Metadata exclusion only: no decoding or prediction of old test/validation images.
    excluded = {rows[p]['rgb_content_sha256'] for field in ('N', 'Vnew', 'Tfinal') for p in old[field]}
    excluded |= {rows[p]['rgb_content_sha256'] for p in old['excluded_train']}
    groups = {}
    for relative in old['unused_normal_train']:
        if '/train/good/' not in relative:
            raise ValueError('Candidate is not author normal training data')
        group = rows[relative]['rgb_content_sha256']
        if group not in excluded:
            groups.setdefault(group, []).append(relative)
    if len(groups) < 140:
        raise ValueError('Fewer than 140 qualified normal RGB groups')
    names = sorted(groups)
    order = np.random.Generator(np.random.PCG64(20261014)).permutation(len(names))
    representatives = [min(groups[names[i]]) for i in order[:140]]
    chosen = dict(N_gen=representatives[:80], S_bg=representatives[80:100], Q_cal=representatives[100:140])
    split = {'schema': 'controlled-normal-pools-v1', 'seed': 20261014, 'rng': 'PCG64',
             'root': old['root'], 'old_split_sha256': digest(old_path),
             'manifest_sha256': digest(manifest_path), 'qualified_RGB_groups': len(groups),
             'input_candidate_files': len(old['unused_normal_train']),
             'group_rule': 'sorted decoded RGB SHA256 -> PCG64 permutation -> lexicographic representative',
             'physical_capture_groups_known': False, 'independent_real_validation': 'pending new data audit',
             **chosen, 'counts': {key: len(value) for key, value in chosen.items()},
             'source_sha256': {p: rows[p]['sha256'] for p in representatives},
             'rgb_sha256': {p: rows[p]['rgb_content_sha256'] for p in representatives}}
    if SPLIT.exists() and read_json(SPLIT) != split:
        raise ValueError('New split already exists with different identity; preserve it')
    save_json(SPLIT, split)
    protocol = {'schema': 'controlled-generation-trial-v1', 'split_sha256': digest(SPLIT),
                'working_size': [256, 256], 'coordinates': 'x,y in 256 grid; integer coordinates are pixel centers',
                'normal_input': 'RGB whole image bilinear256 float32 /255; no normalization',
                'effective_mask': 'max RGB absolute float residual > float32(1/255)',
                'saved_png_effective_mask': 'max uint8 RGB absolute residual >1 code value',
                'validity': {'minimum_effective_coverage': .90, 'outside_ROI_max_change': 0.0},
                'renderer': 'signed additive scalar brightness field, finite support, clipped [0,1]',
                'mask_sources': 'analytic geometry / OpenCV polygon; AnoStyler msp_dot for blob outline',
                'anostyler': read_json(ROOT / 'vendor/AnoStyler/provenance.json'),
                'no_model_training': True, 'no_true_test_scoring': True,
                'trial_only': True, 'formal_G0_G1_training_protocol': 'not implemented or frozen by this trial'}
    path = OUT / 'generation_protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Trial protocol already differs; do not overwrite')
    save_json(path, protocol)
    save_json(OUT / 'resources.json', resources())
    return split


def load_split():
    split = read_json(SPLIT)
    protocol = read_json(OUT / 'generation_protocol.json')
    if digest(SPLIT) != protocol['split_sha256']:
        raise ValueError('Controlled split identity differs')
    if digest(ROOT / 'data/weak_localization/split.json') != split['old_split_sha256']:
        raise ValueError('Old split has changed')
    if digest(ROOT / 'results/weak_localization/dataset_manifest.json') != split['manifest_sha256']:
        raise ValueError('Source manifest has changed')
    for name, record in protocol['anostyler']['files'].items():
        if digest(ROOT / 'vendor/AnoStyler' / name) != record['sha256']:
            raise ValueError('Pinned upstream mask source or license has changed')
    return split


def normal_rgb(split, relative):
    if relative not in split['N_gen'] + split['S_bg']:
        raise ValueError('Generation permits N_gen / S_bg normals only; Q_cal and real tests are excluded')
    source = Path(split['root']) / relative
    if digest(source) != split['source_sha256'][relative]:
        raise ValueError('Background source identity differs')
    with Image.open(source) as image:
        # The pinned original manifest owns its decoded-RGB grouping convention.
        # Verify the complete source file instead of assuming that convention is
        # SHA256(raw RGB bytes) without the original audit's header/metadata.
        return np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255)
