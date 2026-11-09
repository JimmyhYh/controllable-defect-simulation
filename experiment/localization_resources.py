"""Frozen first-round resources. Opens N/DTD only, never calibration/test images."""
import json
from pathlib import Path
from localization_data import ROOT, OUT, SPLIT, digest, save

MODEL_ROOT = ROOT / 'models/weak_localization'

def frozen_resources():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    config_path = OUT/'first_round_config.json'
    if digest(config_path) != protocol['active_execution_config']['sha256']:
        raise ValueError('Active execution config changed')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    split = json.loads(SPLIT.read_text(encoding='utf-8'))
    if digest(SPLIT) != config['split_sha256'] or digest(OUT/'training_config.json') != config['source_day2_config_sha256']:
        raise ValueError('Frozen split or original config changed')
    if config['paired_seeds'] != [20261020] or config['steps'] != 10000 or len(split['N']) != 80:
        raise ValueError('Unexpected execution scope')
    if config['precision'] != 'fp32' or config['input_size'] != 256 or config['batch_size'] != 1:
        raise ValueError('Resource settings differ')
    for name, expected in config['probe_source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen probe/generator source differs: '+name)
    for entry in protocol['source']['draem']['files']:
        if digest(ROOT/'vendor/DRAEM'/entry['name']) != entry['sha256']:
            raise ValueError('Author DRAEM source differs')
    for name in ['data_loader.py','perlin.py','LICENSE']:
        path = ROOT/'vendor/DRAEM'/name
        expected = protocol['environment']['files'][str(path)]['sha256']
        if digest(path) != expected:
            raise ValueError('Original generator source differs')
    for name, entry in protocol['source']['patchcore_files'].items():
        if digest(ROOT/'vendor/patchcore-inspection'/name) != entry['sha256']:
            raise ValueError('PatchCore author source differs')
    weight = ROOT/'models/checkpoints/wide_resnet50_2-95faca4d.pth'
    if digest(weight) != protocol['environment']['files'][str(weight)]['sha256']:
        raise ValueError('Backbone weight differs')
    for path in split['N']:
        if not path.startswith('display/train/good/') or digest(Path(split['root'])/path) != split['source_sha256'][path]:
            raise ValueError('N source identity differs')
    names = sorted((ROOT/'data/dtd/labels/train1.txt').read_text().splitlines())
    import hashlib
    content = hashlib.sha256(''.join(n+' '+digest(ROOT/'data/dtd/images'/n)+'\n' for n in names).encode()).hexdigest()
    if content != protocol['dtd']['pool_content_sha256'] or len(names) != 1880:
        raise ValueError('DTD train1 content differs')
    return config, split, {'first_round_config_sha256':digest(config_path), 'split_sha256':digest(SPLIT),
                          'DTD_pool_content_sha256':content, 'N_paths':split['N']}
