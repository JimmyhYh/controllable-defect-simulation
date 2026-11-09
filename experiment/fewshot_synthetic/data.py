"""Day-1 CPU preparation. Test RGB is decoded for identity only, never displayed."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / 'KolektorSDD2.zip'
AUDIT = ROOT / 'results/fewshot_synthetic/ksdd2_input_audit_v1/archive_verification.json'
DATA = ROOT / 'data/fewshot_synthetic/ksdd2_v1'
OUT = ROOT / 'results/fewshot_synthetic/ksdd2_v1/input'
ARCHIVE_HASH = 'edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8'
AUDIT_HASH = '7595d43a39ce58a6d73fb45c2c8ea2e3b17a4be8de7cc8f633917086c299f4a0'
_GUARDED = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def guard():
    global _GUARDED
    forbidden = ('torch', 'torchvision', 'diffusers', 'transformers')
    require(not any(n.split('.')[0] in forbidden for n in sys.modules), 'Model package already loaded')
    if _GUARDED:
        return

    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('Day-1 model import forbidden')
        if event in ('socket.connect', 'subprocess.Popen', 'pickle.find_class'):
            raise RuntimeError('No network, subprocess or pickle execution in day-1 entry')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1] or ''
            flags = args[2] or 0
            writing = any(x in str(mode) for x in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing:
                require(path.is_relative_to(OUT) or path.is_relative_to(DATA), 'Write outside independent day-1 destinations')
            elif path.is_relative_to(ROOT / 'data'):
                require(path.is_relative_to(DATA), 'Old dataset read forbidden')
            elif path.is_relative_to(ROOT / 'results'):
                require(path == AUDIT or path.is_relative_to(OUT), 'Old results read forbidden')
    sys.addaudithook(audit)
    _GUARDED = True


def available_ram():
    class M(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ('total', 'avail', 'tp', 'ap', 'tv', 'av', 'ex')]
    m = M()
    m.length = ctypes.sizeof(m)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)), 'RAM query failed')
    return m.avail / 2**30


def core_identity():
    return {name: sha(Path(__file__).parent / name) for name in ('__init__.py', 'data.py')}


def upstream():
    require(sha(AUDIT) == AUDIT_HASH, 'Sealed audit identity differs')
    require(sha(ARCHIVE) == ARCHIVE_HASH, 'Archive identity differs')
    a = read_json(AUDIT)
    for name, expected in a['source_identity_unchanged'].items():
        require(sha(ROOT / name) == expected, 'Old source differs: ' + name)
    return a


def window_starts(height):
    require(height >= 256, 'Height cannot be silently resized')
    return sorted(set(list(range(0, height - 256 + 1, 128)) + [height - 256]))


def rgb_identity(path, width, height):
    # Deliberately return only identity, shape and mode; no image, pixels, or appearance statistics.
    from PIL import Image
    require(not str(path).endswith('_GT.png'), 'Identity decoder must not open masks')
    with Image.open(path) as im:
        require(im.size == (width, height) and im.mode in ('RGB', 'L'), 'Unexpected RGB dimensions/mode')
        source_mode = im.mode
        rgb = im.convert('RGB')
        h = hashlib.sha256(struct.pack('<II', width, height))
        h.update(rgb.tobytes())
    return {'rgb_sha256': h.hexdigest(), 'source_mode': source_mode, 'width': width, 'height': height}


def prepare_data():
    guard()
    if (OUT / 'preparation_complete.json').exists():
        return verify_data()
    require(not OUT.exists() and not DATA.exists(), 'Unknown/incomplete destinations: preserve and investigate, do not rerun')
    ram = available_ram()
    require(ram >= 2.0, f'Only {ram:.3f} GiB free RAM; need 2 GiB; no input work started')
    a = upstream()
    import numpy as np
    from PIL import Image
    OUT.mkdir(parents=True)
    DATA.mkdir(parents=True)
    save(OUT / 'preparation_transaction.json', {'archive_sha256': ARCHIVE_HASH, 'implementation': core_identity(),
                                              'rule': 'Without preparation_complete.json this is an incomplete transaction'})
    samples = sorted(a['canonical_samples'], key=lambda x: (x['part'], x['id']))
    members = {x['path']: x for x in a['members']}
    require(len(samples) == 3335, 'Unexpected canonical population')
    file_hashes = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for row in samples:
            for key in ('image', 'mask'):
                name = row[key]
                target = DATA / name
                require(target.resolve().is_relative_to(DATA), 'Unsafe export path')
                target.parent.mkdir(exist_ok=True)
                h = hashlib.sha256()
                with z.open(name) as src, target.open('xb') as dst:
                    while block := src.read(1024 * 1024):
                        h.update(block)
                        dst.write(block)
                require(h.hexdigest() == members[name]['sha256'], 'Export differs: ' + name)
                file_hashes[name] = h.hexdigest()
    identities = {}
    groups = {}
    for row in samples:
        info = rgb_identity(DATA / row['image'], row['width'], row['height'])
        identities[row['image']] = info
        groups.setdefault(info['rgb_sha256'], []).append(row['image'])
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    save(OUT / 'rgb_identity.json', {'images': identities, 'duplicate_groups': duplicates,
                                   'test_use': 'identity only; no appearance statistics or display',
                                   'test_GT_decoded': False, 'physical_groups': 'unknown'})
    if duplicates:
        save(OUT / 'input_failure.json', {'reason': 'Canonical exact RGB duplicates; stop before sampling', 'groups': duplicates})
        raise RuntimeError('RGB duplicates found; inputs preserved, no sampling or substitution')
    train = sorted([x for x in samples if x['part'] == 'train'], key=lambda x: x['id'])
    pos = [x for x in train if x['training_label'] == 'anomaly']
    neg = [x for x in train if x['training_label'] == 'normal']
    require(len(pos) == 246 and len(neg) == 2085, 'Training population differs')
    real = sorted([pos[int(i)] for i in np.random.Generator(np.random.PCG64(2026101901)).choice(246, 5, replace=False)], key=lambda x: x['id'])
    q = sorted([neg[int(i)] for i in np.random.Generator(np.random.PCG64(2026101902)).choice(2085, 200, replace=False)], key=lambda x: x['id'])
    real_ids = {x['id'] for x in real}
    q_ids = {x['id'] for x in q}
    normal = [x for x in neg if x['id'] not in q_ids]
    unused = [x for x in pos if x['id'] not in real_ids]
    test = [x for x in samples if x['part'] == 'test']
    review_normal = sorted([normal[int(i)] for i in np.random.Generator(np.random.PCG64(2026101907)).choice(1885, 16, replace=False)], key=lambda x: x['id'])
    split = {'R_real5': real, 'Q_normal200': q, 'N_train1885': normal, 'U_anomaly241': unused,
             'E_official1004': test, 'visual_N16_ids': [x['id'] for x in review_normal],
             'seeds': {'R': 2026101901, 'Q': 2026101902, 'visual_N': 2026101907}}
    save(DATA / 'split.json', split)
    save(OUT / 'export_inventory.json', {'files': file_hashes, 'copy_files_excluded_from_export': a['copied_files'],
                                        'pickles_exported_or_executed': False})
    scope = {x['id']: x for x in real + review_normal}
    review_info = []
    previews = OUT / 'previews'
    previews.mkdir()
    normal_canvases = []
    for index, row in enumerate(real + review_normal):
        require(row['part'] == 'train' and row['id'] in scope, 'Visual scope violation')
        with Image.open(DATA / row['image']) as im:
            image = im.convert('RGB')
            require(image.size == (row['width'], row['height']), 'Review image size differs')
        with Image.open(DATA / row['mask']) as im:
            require(im.mode in ('L', '1'), 'Unexpected review mask mode')
            mask = im.convert('L')
        arr = np.array(mask)
        require(set(np.unique(arr).tolist()) <= {0, 255}, 'Nonbinary training review GT')
        require(mask.size == image.size and 2 <= image.width <= 256, 'Invalid working dimensions')
        starts = window_starts(image.height)
        windows = [{'y': y, 'positive_pixels': int(np.count_nonzero(arr[y:y + 256]))} for y in starts]
        is_real = row['id'] in real_ids
        require(bool(np.any(arr)) == is_real, 'Review GT / assigned class differ')
        if is_real:
            require(any(w['positive_pixels'] for w in windows), 'No nonempty training window')
        review_info.append({'id': row['id'], 'role': 'R_real5' if is_real else 'N_review16',
                            'width': image.width, 'height': image.height, 'windows': windows,
                            'positive_pixels': int(np.count_nonzero(arr)), 'valid_width': image.width,
                            'padding': 'right reflection to256; excluded from loss/labels'})
        if is_real:
            canvas = Image.new('RGB', (image.width * 3 + 80, image.height + 60), 'white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), f"R {row['id']} | RGB original | GT binary | GT overlay", fill='black')
            overlay = np.array(image).copy()
            overlay[arr > 0] = (255, 0, 0)
            for col, panel in enumerate((image, mask.convert('RGB'), Image.fromarray(overlay))):
                canvas.paste(panel, (10 + col * (image.width + 20), 40))
            canvas.save(previews / f"R_{row['id']}.png")
        else:
            normal_canvases.append((row['id'], image.copy()))
    for page in range(4):
        canvas = Image.new('RGB', (550, 1440), 'white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for cell, (sample_id, image) in enumerate(normal_canvases[page * 4:page * 4 + 4]):
            x, y = 10 + (cell % 2) * 275, 10 + (cell // 2) * 710
            draw.text((x, y), f'N {sample_id} | empty GT', fill='black')
            canvas.paste(image, (x, y + 25))
        canvas.save(previews / f'N_page{page + 1}.png')
    save(OUT / 'review_geometry.json', {'samples': review_info, 'visual_gate': 'pending human/assistant inspection, not inferred from masks'})
    protocol = {'schema': 'fewshot-ksdd2-day1-v1', 'archive_sha256': ARCHIVE_HASH,
                'audit_sha256': AUDIT_HASH, 'implementation': core_identity(),
                'source_identity': a['source_identity_unchanged'], 'role_counts': {k: len(split[k]) for k in
                ('R_real5', 'Q_normal200', 'N_train1885', 'U_anomaly241', 'E_official1004')},
                'visual_scope': {'R5': sorted(real_ids), 'N16': split['visual_N16_ids']},
                'RGB_identity_includes_dimensions': True, 'test_GT_decoded': False,
                'E_native_pixels': sum(x['width'] * x['height'] for x in test),
                'physical_groups': 'unknown; no evidence of separate objects/batches',
                'input_ready': False, 'readiness_authority': 'completion.json after explicit visual review',
                'resource': {'available_RAM_at_start_GiB': ram, 'python': sys.version,
                             'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'Pillow')}},
                'model_import_or_forward_or_training': False}
    save(OUT / 'input_protocol.json', protocol)
    current = {str(p.relative_to(OUT)).replace('\\', '/'): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    current['DATA/split.json'] = sha(DATA / 'split.json')
    require(all(sha(ROOT / name) == expected for name, expected in a['source_identity_unchanged'].items()), 'Old source changed')
    save(OUT / 'preparation_complete.json', {'files': current, 'implementation': core_identity(), 'awaiting_visual_review': True})
    return verify_data()


def verify_data():
    guard()
    upstream()
    require((OUT / 'preparation_complete.json').is_file(), 'No complete preparation; preserve incomplete transaction')
    manifest = read_json(OUT / 'preparation_complete.json')
    require(manifest['implementation'] == core_identity(), 'CPU preparation implementation differs')
    for name, expected in manifest['files'].items():
        path = DATA / name[5:] if name.startswith('DATA/') else OUT / name
        require(sha(path) == expected, 'Prepared artifact differs: ' + name)
    inv = read_json(OUT / 'export_inventory.json')['files']
    actual = {str(p.relative_to(DATA)).replace('\\', '/') for p in DATA.rglob('*') if p.is_file()}
    require(actual == set(inv) | {'split.json'}, 'Unknown or missing exported data')
    for name, expected in inv.items():
        require(sha(DATA / name) == expected, 'Exported bytes differ: ' + name)
    fixed = set(manifest['files']) - {'DATA/split.json'}
    actual_out = {str(p.relative_to(OUT)).replace('\\', '/') for p in OUT.rglob('*') if p.is_file()}
    allowed = fixed | {'preparation_complete.json', 'visual_review.json', 'completion.json', '第1天输入准备与适用性核验.md'}
    require(actual_out <= allowed, 'Unknown input outputs; preserve and investigate')
    ready = False
    if (OUT / 'completion.json').exists():
        c = read_json(OUT / 'completion.json')
        require(c['preparation_sha256'] == sha(OUT / 'preparation_complete.json'), 'Final/preparation binding differs')
        for name, expected in c['files'].items():
            require(sha(OUT / name) == expected, 'Final artifact differs')
        ready = c['input_ready']
    return {'preparation_verified': True, 'input_ready': ready,
            'visual_gate': 'recorded' if (OUT / 'completion.json').exists() else 'pending',
            'roles': read_json(OUT / 'input_protocol.json')['role_counts'],
            'R5': read_json(OUT / 'input_protocol.json')['visual_scope']['R5'],
            'RGB_exact_duplicate_groups': 0, 'test_GT_decoded': False,
            'new_model_operations': 0, 'cache_reuse': True}


def finalize_review(review):
    verify_data()
    require(not any((OUT / n).exists() for n in ('visual_review.json', 'completion.json', '第1天输入准备与适用性核验.md')),
            'Review already exists or incomplete; preserve, do not replace')
    scope = read_json(OUT / 'input_protocol.json')['visual_scope']
    require(set(review['R5']) == set(scope['R5']) and set(review['N16']) == set(scope['N16']), 'Review must cover all predetermined images')
    require(review['decision'] in ('pass', 'fail', 'uncertain'), 'Invalid decision')
    require(all(isinstance(v, str) and v.strip() for v in list(review['R5'].values()) + list(review['N16'].values())), 'Missing per-image observations')
    save(OUT / 'visual_review.json', review)
    ready = review['decision'] == 'pass'
    report = '# 第1天输入准备与适用性核验\n\n'
    report += '实际状态：' + ('input_ready，可进入第2天准备；尚无模型／合成／训练。' if ready else '输入适用性未通过，停止；不替补样本。') + '\n\n'
    report += '规范3335对均已导出并核对原始hash；全部RGB仅用于身份核验，精确重复0组。原包及copy不动，作者pickle未执行。\n\n'
    report += '固定R5：' + '、'.join(scope['R5']) + '。Q200／N1885／U241／E1004已固定；物理组独立性未知。test未显示或作外观统计，GT未解码；110／894仍是官方数量。\n\n'
    report += review['summary'] + '\n\n'
    for role in ('R5', 'N16'):
        report += '## ' + role + '固定视觉审阅\n\n'
        for sample_id, observation in review[role].items():
            report += '- ' + sample_id + '：' + observation + '\n'
        report += '\n'
    report += '视觉采用未拉伸原始RGB，GT与红色覆盖仅辅助对照；不把清晰标注或输入通过当生成拟真／训练收益。工作窗不缩放、右侧反射补齐，有效域外不计标签或损失；实际几何见review_geometry.json。\n\n'
    report += '可复制核验命令见fewshot_synthetic/README.md；prepare-data完整缓存只核验，verify-data不解码RGB或掩码。原始准备调用实际解码3335张RGB用于身份，另21张训练图与21张GT用于预定总览；没有合成、Torch、模型前向或训练。\n'
    (OUT / '第1天输入准备与适用性核验.md').write_text(report, encoding='utf-8')
    save(OUT / 'completion.json', {'input_ready': ready, 'decision': review['decision'],
         'preparation_sha256': sha(OUT / 'preparation_complete.json'),
         'files': {n: sha(OUT / n) for n in ('visual_review.json', '第1天输入准备与适用性核验.md')}})
    return verify_data()
