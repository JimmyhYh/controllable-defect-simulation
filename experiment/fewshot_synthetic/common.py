"""Independent scientific identities and CPU-only safety boundaries for day 2."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from . import data as d

ROOT, DATA = d.ROOT, d.DATA
INPUT = d.OUT
PREP = ROOT / 'results/fewshot_synthetic/ksdd2_v1/preparation'
MODELS = ROOT / 'models/fewshot_synthetic/ksdd2_v1'
RESULTS = ROOT / 'results/fewshot_synthetic/ksdd2_v1'
SEEDS = (2026101903, 2026101904, 2026101905)
PROBE_SEED = 2026101906
SOURCE_NAMES = ('common.py', 'sampling.py', 'model.py', 'training.py', 'metrics.py', 'inference.py')
require, sha, read_json, save = d.require, d.sha, d.read_json, d.save


def identities():
    return {name: sha(Path(__file__).parent / name) for name in SOURCE_NAMES}


def environment():
    return {'python': sys.version, 'executable': sys.executable,
            'packages': {n: importlib.metadata.version(n) for n in
                         ('numpy', 'Pillow', 'scipy', 'opencv-python', 'torch', 'scikit-learn')}}


def source_metadata():
    """Do not call day-1 verify-data: it hashes Q/E/U image files too."""
    c = read_json(INPUT / 'completion.json')
    p = read_json(INPUT / 'preparation_complete.json')
    require(c['input_ready'] is True and c['decision'] == 'pass', 'Day-1 input not ready')
    require(sha(INPUT / 'preparation_complete.json') == c['preparation_sha256'], 'Day-1 preparation differs')
    require(sha(Path(__file__).parent / 'verification.py') == c['verification_sha256'], 'Day-1 verifier differs')
    require(set(c['files']) == {'visual_review.json', '第1天输入准备与适用性核验.md'}, 'Day-1 final files differ')
    for name, expected in c['files'].items():
        require(sha(INPUT / name) == expected, 'Day-1 final evidence differs: ' + name)
    for name, expected in p['files'].items():
        path = DATA / name[5:] if name.startswith('DATA/') else INPUT / name
        require(sha(path) == expected, 'Day-1 scientific metadata differs: ' + name)
    for name, expected in p['implementation'].items():
        require(sha(Path(__file__).parent / name) == expected, 'Frozen day-1 source differs')
    protocol = read_json(INPUT / 'input_protocol.json')
    require(sha(d.AUDIT) == d.AUDIT_HASH and sha(d.ARCHIVE) == d.ARCHIVE_HASH, 'Archive/audit identity differs')
    for name, expected in protocol['source_identity'].items():
        require(sha(ROOT / name) == expected, 'Old bound source differs: ' + name)
    split = read_json(DATA / 'split.json')
    counts = dict(zip(('R_real5','Q_normal200','N_train1885','U_anomaly241','E_official1004'), (5,200,1885,241,1004)))
    require({k: len(split[k]) for k in counts} == counts, 'Roles differ')
    require([r['id'] for r in split['R_real5']] == ['11670','11884','12070','12245','12303'], 'R5 differs')
    review = read_json(INPUT / 'visual_review.json')
    require(review['decision'] == 'pass' and set(review['R5']) == set(protocol['visual_scope']['R5'])
            and set(review['N16']) == set(protocol['visual_scope']['N16']), 'Visual evidence scope differs')
    return split, {'completion': sha(INPUT / 'completion.json'), 'split': sha(DATA / 'split.json'),
                   'protocol': sha(INPUT / 'input_protocol.json'), 'old_sources': protocol['source_identity']}


def guard(split, cpu=True):
    """Process-local audit gate; this is not an OS sandbox."""
    forbidden = ('torch', 'torchvision', 'diffusers', 'transformers')
    if cpu:
        require(not any(n.split('.')[0] in forbidden for n in sys.modules), 'CPU preparation must not load model packages')
    allowed = {(DATA / r[k]).resolve() for role in ('R_real5','N_train1885') for r in split[role] for k in ('image','mask')}
    def audit(event, args):
        if cpu and event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('Model import forbidden during CPU preparation')
        if event == 'socket.connect' or (cpu and event in ('subprocess.Popen', 'pickle.find_class')):
            raise RuntimeError('No network/subprocess/pickle in CPU preparation')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1] or '', args[2] or 0
            writing = any(c in str(mode) for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            if writing:
                require(path.is_relative_to(PREP if cpu else RESULTS) or (not cpu and path.is_relative_to(MODELS)),
                        'Write outside independent destinations: ' + str(path))
            elif path.is_relative_to(ROOT / 'data'):
                require(path in allowed or path == DATA / 'split.json', 'Q/E/U or old dataset read forbidden')
            elif path.is_relative_to(ROOT / 'results'):
                require(path == d.AUDIT or path.is_relative_to(RESULTS), 'Old result read forbidden')
    sys.addaudithook(audit)


def resource(min_ram=2, min_disk=80):
    ram, disk = d.available_ram(), shutil.disk_usage(ROOT).free / 2**30
    require(ram >= min_ram, f'Only {ram:.3f}GiB RAM; need {min_ram}; no generation/update started')
    require(disk >= min_disk, f'Only {disk:.3f}GiB disk; need {min_disk}')
    return {'available_RAM_GiB': ram, 'free_disk_GiB': disk}


def lr_at(update, probe=False):
    total, a, b = (1000, 800, 900) if probe else (10000, 8000, 9000)
    require(type(update) is int and 1 <= update <= total, 'Update out of range')
    return 1e-4 if update <= a else (2e-5 if update <= b else 4e-6)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_manifest(root, completion='complete.json'):
    root = Path(root)
    c = read_json(root / completion)
    expected_files = set(c['files']) | {completion}
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    expected_dirs = {str(Path(n).parent).replace('\\','/') for n in c['files'] if '/' in n}
    expected_dirs |= {str(Path(n).parent).replace('\\','/') for n in list(expected_dirs) if '/' in n}
    actual_dirs = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir()}
    require(actual_files == expected_files and actual_dirs == expected_dirs, 'Unknown/missing files/directories; preserve transaction')
    for name, expected in c['files'].items():
        require(sha(root / name) == expected, 'Scientific file differs: ' + name)
    return c


def seal(root, extra):
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    save(root / 'complete.json', {**extra, 'files': files})
