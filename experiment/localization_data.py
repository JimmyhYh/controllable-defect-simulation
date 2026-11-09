"""Public AnoLCD identities and train-only preprocessing; no held-out scoring."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/weak_localization'
SPLIT = ROOT / 'data/weak_localization/split.json'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def prepare(cap, persist=True):
    manifest_path = OUT / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if digest(manifest_path) != protocol['new_real_dataset']['manifest_sha256']:
        raise ValueError('Manifest identity mismatch')
    excluded = set(manifest['resource_decision']['exclude_training_test_RGB_overlap'])
    rows = {r['path']: r for r in manifest['files']}
    groups = {}
    for row in rows.values():
        if '/train/' in row['path'] and row['path'] not in excluded:
            groups.setdefault(row['rgb_content_sha256'], []).append(row['path'])
    names = sorted(groups)
    order = np.random.default_rng(20261012).permutation(len(names))
    nval = max(1, int(len(names) * 0.2))
    vg = [names[i] for i in order[:nval]]
    ng = [names[i] for i in order[nval:]]
    val = [p for g in vg for p in sorted(groups[g])]
    selected = []
    selected_groups = []
    for g in ng:
        paths = sorted(groups[g])
        if len(selected) + len(paths) > cap:
            break
        selected.extend(paths)
        selected_groups.append(g)
    if len(val) < 20 or len(selected) < 8:
        raise ValueError('Insufficient normal data for protocol')
    test = sorted(p for p in rows if '/test/' in p)
    if set(selected) & set(val) or set(selected_groups) & set(vg):
        raise ValueError('Normal split overlaps')
    if set(selected_groups) & {rows[p]['rgb_content_sha256'] for p in test}:
        raise ValueError('Training/test RGB overlap')
    value = {'seed': 20261012, 'rng': 'PCG64', 'group_rule': 'decoded RGB SHA256, sorted IDs',
             'physical_capture_groups_known': False, 'root': manifest['summary']['extract_root'],
             'manifest_sha256': digest(manifest_path), 'N_cap': cap, 'N': selected, 'Vnew': val,
             'Tfinal': test, 'unused_normal_train': [p for g in ng for p in sorted(groups[g]) if p not in selected],
             'N_groups': selected_groups, 'Vnew_groups': vg, 'excluded_train': sorted(excluded),
             'counts': {'N': len(selected), 'Vnew': len(val), 'Tfinal': len(test)},
             'source_sha256': {p: rows[p]['sha256'] for p in selected + val + test}}
    if not persist:
        return value
    if SPLIT.exists():
        old = json.loads(SPLIT.read_text(encoding='utf-8'))
        if old != value:
            raise ValueError('Existing split differs; do not overwrite')
    else:
        save(SPLIT, value)
    return value

def normal_rgb(split, relative):
    if relative not in split['N']:
        raise ValueError('Only N images are allowed in the day2 generation/probe')
    path = Path(split['root']) / relative
    if digest(path) != split['source_sha256'][relative]:
        raise ValueError('Normal image identity differs')
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255)
