"""Sealed stage-A inputs. Only 1949 and the three preselected Mura references.

Old code is imported only for pure coordinate/reference functions, never an old
preparation or verification entry point that would touch the second background.
"""
import ctypes
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from reference_generation.coordinates import build_geometry
from reference_generation.reference import crop_box, prepare_reference, read_mask, read_rgb, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/reference_interface_demo/v1'
OLD = ROOT / 'results/reference_generation'
IDS = ('019', '024', '020')
SEEDS = (20261016, 20261017)
BG = 'display/train/good/1949.png'
STATUS = 'CPU_INTERFACE_INPUTS_READY'
CORE = ('__init__.py', 'cli.py', 'preparation.py')
REPORT = '阶段A_CPU输入准备结果.md'
ARTIFACT_NAMES = {
    'source_guard.json', 'scope_protocol.json', 'background_original.npy',
    'background512.png', 'CPU_position_overview.png', 'requests.json',
    'input_protocol.json', REPORT,
} | {f'references/{n}_{suffix}' for n in IDS for suffix in ('source.npz', 'rgb512.png', 'mask512.png')} | {
    f'geometry/{n}_move.npz' for n in IDS
}
PINS = {
    'v1/completion.json': 'c063166ea0abb66168a15057b18f934144fae6cf1f525cc49987089e32e894b7',
    'v2_domain_matched/completion.json': '652323555f19533e68d5bdb6908c6138015a37ddaf0d21eaedb9eaa6622b069f',
    'v3_local_context/completion.json': '9e761426780d6c30741f20daa5298c7e80b1d3583fb7335f41cba282472650b2',
    'v3_local_context/local_protocol.json': '8333496875da12a3037ddf30a69f8fb9144e2571593daa09954a2129d607ae2e',
}
_AUDIT_INSTALLED = False


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def install_scope_guard(data_root):
    """Deny other source reads, old image reads, old writes, network and Torch."""
    global _AUDIT_INSTALLED
    if _AUDIT_INSTALLED:
        return
    data_root = data_root.resolve()
    allowed = {(data_root / BG).resolve(),
               (OLD / 'v2_domain_matched/visual/b0_original.png').resolve()}
    for n in IDS:
        allowed.add((data_root / f'display/test/mura/{n}.png').resolve())
        allowed.add((data_root / f'display/ground_truth/mura/{n}_mask.png').resolve())
    old_root = OLD.resolve()
    workspace = ROOT.resolve()
    output = OUT.resolve()

    def guard(event, args):
        if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
            raise RuntimeError('Stage A forbids Torch imports')
        if event.startswith('socket.'):
            raise RuntimeError('Stage A forbids network access')
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0]).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (1 | 2 | 64 | 512 | 1024))
        if writing and path.is_relative_to(workspace) and not path.is_relative_to(output):
            raise RuntimeError('Stage A cannot write outside its independent result directory: ' + str(path))
        if path.is_relative_to(data_root) and path not in allowed:
            raise RuntimeError('Out-of-scope dataset file access: ' + str(path))
        if path.is_relative_to(old_root) and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy', '.npz') and path not in allowed:
            raise RuntimeError('Out-of-scope old image/array access: ' + str(path))

    sys.addaudithook(guard)
    _AUDIT_INSTALLED = True


def frozen_sources():
    old_split = load(ROOT / 'data/weak_localization/split.json')
    data_root = Path(old_split['root'])
    install_scope_guard(data_root)
    require('torch' not in sys.modules, 'Torch already loaded')
    for name, digest in PINS.items():
        require(sha256(OLD / name) == digest, 'Frozen anchor changed: ' + name)
    v3 = load(OLD / 'v3_local_context/local_protocol.json')
    protected = {p: h for p, h in v3['protected'].items()
                 if Path(p).suffix.lower() in ('.py', '.json', '.md', '.txt')}
    protected.update({'reference_generation/' + p: h for p, h in v3['implementation'].items()})
    # Validate sealed metadata only: the old inventories include 1540 and other
    # images, whose bytes must not be opened by this stage.
    for version in ('v1', 'v2_domain_matched', 'v3_local_context'):
        completion = load(OLD / version / 'completion.json')
        protected[f'results/reference_generation/{version}/completion.json'] = PINS[version + '/completion.json']
        for name, digest in completion['artifacts'].items():
            if Path(name).suffix.lower() in ('.py', '.json', '.md', '.txt'):
                protected[f'results/reference_generation/{version}/{name}'] = digest
    for name, digest in protected.items():
        require(sha256(ROOT / name) == digest, 'Old metadata/code changed: ' + name)
    manifest_path = ROOT / 'results/weak_localization/dataset_manifest.json'
    cg = load(ROOT / 'data/controlled_generation/split.json')
    require(sha256(manifest_path) == old_split['manifest_sha256'] == cg['manifest_sha256'], 'Manifest chain differs')
    require(sha256(ROOT / 'data/weak_localization/split.json') == cg['old_split_sha256'], 'Split chain differs')
    rows = {r['path']: r for r in load(manifest_path)['files']}
    exclusions = {rows[p]['rgb_content_sha256'] for k in ('N', 'Vnew', 'Tfinal', 'excluded_train') for p in old_split[k]}
    exclusions |= {rows[p]['rgb_content_sha256'] for k in ('N_gen', 'S_bg', 'Q_cal') for p in cg[k]}
    require(BG in old_split['unused_normal_train'], '1949 not in reserved unused metadata')
    require(rows[BG]['rgb_content_sha256'] not in exclusions, 'Frozen-role/RGB exclusion failed')
    require(v3['selected'][0]['path'] == BG and v3['selected'][0]['group_id'] == rows[BG]['rgb_content_sha256'], 'Background identity differs')
    require(v3['reference_ids'] == list(IDS) and v3['generated_seeds'] == list(SEEDS), 'Reference/seed protocol differs')
    audit = load(OLD / 'v3_local_context/local_audit.json')
    passed = [r for r in audit['records'] if r['passed']]
    require(audit['position_count'] == len(audit['records']) == 24 and audit['passed'] == len(passed) == 3, 'Original 3/24 conclusion differs')
    require([(r['background'], r['reference_id'], r['variant']) for r in passed] == [('b0', n, 'move') for n in IDS], 'Passing records differ')
    old_outcome = load(OLD / 'v3_local_context/input_protocol.json')
    require(old_outcome['status'] == 'LOCAL_REVIEW_UNCERTAIN' and old_outcome['ready_for_day2'] is False, 'Original failure changed')
    refs1 = {r['id']: r for r in load(OLD / 'v1/input_protocol.json')['references']}
    sources = {BG: rows[BG]}
    for n in IDS:
        for key, name in (('image', f'display/test/mura/{n}.png'), ('mask', f'display/ground_truth/mura/{n}_mask.png')):
            require(refs1[n][key] == {'path': name, **rows[name]}, 'v1 reference record differs')
            sources[name] = rows[name]
    require(len({rows[f'display/test/mura/{n}.png']['rgb_content_sha256'] for n in IDS}) == 3, 'Reference RGB duplicate')
    for name, record in sources.items():
        require(sha256(data_root / name) == record['sha256'], 'Allowed source identity differs: ' + name)
    sealed_bg = OLD / 'v2_domain_matched/visual/b0_original.png'
    bg_hash = load(OLD / 'v2_domain_matched/completion.json')['artifacts']['visual/b0_original.png']
    require(sha256(sealed_bg) == bg_hash, 'Sealed 1949 cache changed')
    guard = {'protected_text_code_metadata': protected, 'source_root': str(data_root), 'sources': sources,
             'sealed_1949_cache_sha256': bg_hash, 'background_RGB_excluded': False,
             'exclusion_RGB_group_count': len(exclusions), 'v3_original_positions_passed': 3,
             'v3_original_positions_evaluated': 24, 'v3_ready_for_day2': False,
             'selected_passing_records': passed, 'old_RGB_hash_convention': 'opaque saved group IDs, used unchanged for exclusions'}
    return guard, v3, refs1, data_root


def inherited_geometry(mask, frozen):
    g = build_geometry(mask, (1024, 1024), {'variant': 'move', 'model_side': 512})
    b = g['metadata']['original_bbox_xyxy_exclusive']
    side = math.ceil(2 * max(b[2] - b[0], b[3] - b[1]))
    box = crop_box(g['metadata']['requested_center_xy'], side, (1024, 1024))
    guard = np.zeros((1024, 1024), bool)
    l, t, r, b = box
    guard[t:b, l:r] = True
    metadata = {**g['metadata'], 'guard_xyxy_exclusive': box,
                'guard_side_original_px': side, 'guard_pixels': int(guard.sum())}
    require(metadata == frozen, 'Original/512 transform or guard differs from frozen v3')
    require(not np.any(g['roi'] & ~guard), 'Support escapes guard')
    return {'original': g['roi'], 'local512': g['local_roi'], 'guard': guard}, metadata


def expected_requests(v3):
    result = []
    for n in IDS:
        for seed in SEEDS:
            result.append({'id': f'if_v1_r{len(result):03d}', 'reference_id': n, 'background_path': BG,
                           'variant': 'move', 'seed': seed, 'geometry_source_key': n + '_move',
                           'geometry': v3['geometries'][n + '_move'],
                           'arrays': f'geometry/{n}_move.npz', 'reference_rgb512': f'references/{n}_rgb512.png',
                           'reference_mask512': f'references/{n}_mask512.png', 'background512': 'background512.png',
                           'conditions_planned': ['reference_off', 'reference_on'], 'generation_authorized': False})
    return result


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong) for k in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    m = Memory()
    m.length = ctypes.sizeof(m)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)), 'Resource query failed')
    return {'python': platform.python_version(), 'device_used': 'CPU',
            'available_RAM_GiB': m.available / 2**30, 'total_RAM_GiB': m.total / 2**30,
            'free_disk_GiB': shutil.disk_usage(ROOT).free / 2**30,
            'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'Pillow', 'opencv-python')},
            'measurement': 'snapshot, not peak; GPU/model resource feasibility not evaluated'}


def draw_overview(bg, ref_rgbs, geometries, v3):
    page = Image.new('RGB', (1280, 1040), '#242424')
    draw = ImageDraw.Draw(page)
    headers = ('Reference RGB (native contrast)', '1949 full frame: display only', 'Model context (native contrast)', 'R orange / guard green')
    for i, title in enumerate(headers):
        draw.text((i * 320 + 5, 6), title, fill='white')
    for i, n in enumerate(IDS):
        g = geometries[n]
        meta = v3['geometries'][n + '_move']
        l, t, r, b = meta['crop_xyxy_exclusive']
        crop = bg[t:b, l:r]
        full = bg.copy()
        cv2.rectangle(full, (l, t), (r - 1, b - 1), (40, 210, 255), 2)
        overlay = crop.copy()
        contours, _ = cv2.findContours(g['local512'].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 110, 0), 2)
        gl, gt, gr, gb = meta['guard_xyxy_exclusive']
        cv2.rectangle(overlay, (gl - l, gt - t), (gr - l - 1, gb - t - 1), (60, 255, 100), 2)
        y = 45 + i * 330
        for j, rgb in enumerate((ref_rgbs[n], full, crop, overlay)):
            page.paste(Image.fromarray(rgb).resize((308, 308), Image.Resampling.BILINEAR), (j * 320 + 5, y))
        draw.text((5, y - 17), f'ref {n} / move / seeds {SEEDS[0]},{SEEDS[1]}', fill='white')
    draw.text((5, 1030 - 12), 'INPUT OVERVIEW ONLY | 1 selected background; 3 masks; 6 requests; 0 neural outputs | no normalization', fill='white')
    page.save(OUT / 'CPU_position_overview.png')


def verify_inputs():
    require(not (OUT / 'preparation.incomplete').exists(), 'Incomplete transaction: preserve and investigate')
    require((OUT / 'completion.json').is_file(), 'No complete stage-A cache')
    completion = load(OUT / 'completion.json')
    require(set(completion['artifacts']) == ARTIFACT_NAMES, 'Unknown/missing artifact in sealed inventory')
    require(set(completion['implementation']) == set(CORE), 'Stage-A implementation inventory differs')
    expected = set(completion['artifacts']) | {'completion.json'}
    actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    require(actual == expected, 'Strict output file inventory differs')
    expected_dirs = {a.as_posix() for name in expected for a in Path(name).parents if str(a) != '.'}
    require({p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_dir()} == expected_dirs, 'Strict directory inventory differs')
    for name, digest in completion['artifacts'].items():
        require(sha256(OUT / name) == digest, 'New artifact changed: ' + name)
    for name, digest in completion['implementation'].items():
        require(sha256(Path(__file__).parent / name) == digest, 'Frozen stage-A implementation differs: ' + name)
    guard, v3, refs1, _ = frozen_sources()
    require(load(OUT / 'source_guard.json') == guard, 'Source guard differs')
    reqs = load(OUT / 'requests.json')
    require(reqs['requests'] == expected_requests(v3), 'Six-request order/input identity differs')
    require(reqs['count'] == 6 and reqs['distinct_geometries'] == 3 and reqs['planned_neural_outputs'] == 12, 'Wrong denominators')
    with (OUT / 'background_original.npy').open('rb') as f:
        bg = np.load(f, allow_pickle=False)
    require(bg.dtype == np.uint8 and bg.shape == (1024, 1024, 3), 'Wrong saved background')
    validation = load(OUT / 'input_protocol.json')
    require(hashlib.sha256(bg.tobytes()).hexdigest() == validation['background_raw_RGB_sha256'], 'Background RGB identity differs')
    for n in IDS:
        with np.load(OUT / f'references/{n}_source.npz', allow_pickle=False) as z:
            require(set(z.files) == {'original_mask', 'native_rgb', 'native_mask'}, 'Reference archive keys differ')
            mask, native, native_mask = z['original_mask'], z['native_rgb'], z['native_mask']
        require(mask.dtype == bool and mask.shape == (1024, 1024), 'Wrong original reference mask')
        require(int(mask.sum()) == refs1[n]['mask']['positive_pixels'], 'Mask count differs')
        l, t, r, b = refs1[n]['crop_xyxy_exclusive']
        require(np.array_equal(native_mask, mask[t:b, l:r]), 'Native reference mask differs')
        require(native.dtype == np.uint8 and native.shape == (b - t, r - l, 3), 'Native RGB crop differs')
        with Image.open(OUT / f'references/{n}_rgb512.png') as im:
            require(im.mode == 'RGB' and np.array_equal(np.array(im), np.array(Image.fromarray(native).resize((512, 512), Image.Resampling.BILINEAR))), 'Reference model RGB differs')
        with Image.open(OUT / f'references/{n}_mask512.png') as im:
            expected_mask = np.array(Image.fromarray(native_mask.astype('uint8') * 255).resize((512, 512), Image.Resampling.NEAREST))
            require(im.mode == 'L' and np.array_equal(np.array(im), expected_mask), 'Reference model mask differs')
        inherited, _ = inherited_geometry(mask, v3['geometries'][n + '_move'])
        with np.load(OUT / f'geometry/{n}_move.npz', allow_pickle=False) as z:
            require(set(z.files) == set(inherited) and all(np.array_equal(z[k], arr) for k, arr in inherited.items()), 'Frozen geometry array differs')
        l, t, r, b = v3['geometries'][n + '_move']['crop_xyxy_exclusive']
        with Image.open(OUT / 'background512.png') as im:
            require(im.mode == 'RGB' and np.array_equal(np.array(im), bg[t:b, l:r]), 'Common background crop differs')
    require(validation['status'] == completion['status'] == STATUS, 'Incorrect stage-A status')
    require(validation['generation_authorized'] is False and validation['original_v3_ready_for_day2'] is False, 'Unauthorized promotion')
    require(validation['model_forwards'] == validation['optimizer_steps'] == 0 and not validation['torch_loaded'] and 'torch' not in sys.modules, 'Unexpected model use')
    return {'status': STATUS, 'requests_verified': 6, 'geometries_verified': 3, 'backgrounds': 1,
            'planned_neural_outputs': 12, 'neural_outputs_completed': 0, 'original_v3_passed': '3/24',
            'original_v3_ready_for_day2': False, 'generation_authorized': False, 'strict_inventory': True,
            'protected_text_code_metadata_verified': len(guard['protected_text_code_metadata']),
            'new_source_image_decodes': 0, 'new_model_forwards': 0, 'optimizer_steps': 0,
            'torch_loaded': 'torch' in sys.modules, 'output': str(OUT)}


def prepare_inputs():
    if (OUT / 'completion.json').exists():
        return verify_inputs()
    require(not OUT.exists(), 'Unknown/partial output directory: preserve and investigate, do not restart')
    guard, v3, refs1, data_root = frozen_sources()
    # Freeze scope and implementation before any allowed image decode.
    OUT.mkdir(parents=True)
    (OUT / 'preparation.incomplete').write_text('Stage A CPU input transaction; no automatic restart.\n', encoding='utf-8')
    implementation = {name: sha256(Path(__file__).parent / name) for name in CORE}
    save(OUT / 'source_guard.json', guard)
    save(OUT / 'scope_protocol.json', {
        'schema': 'minimal-reference-interface-scope-v1', 'background_path': BG,
        'reference_ids': list(IDS), 'seeds': list(SEEDS), 'variant': 'move',
        'ordered_geometry_sources': [n + '_move' for n in IDS],
        'geometry_metadata': {n: v3['geometries'][n + '_move'] for n in IDS},
        'requests': 6, 'planned_conditions': ['reference_off', 'reference_on'], 'planned_neural_outputs': 12,
        'scope': 'selected-condition interface and background compositing only; not original48 acceptance',
        'selection': 'post-v3 development selection, no sampling or new structural review',
        'original_result': '3/24; LOCAL_REVIEW_UNCERTAIN; ready_for_day2=false',
        'implementation': implementation, 'image_policy': 'original contrast; no normalization/gamma/enhancement',
        'numerical_conventions': 'original integer pixel centers and existing local512 transforms; xyxy right/bottom exclusive',
        'generation_authorized': False, 'training_authorized': False})
    bg = read_rgb(data_root / BG, guard['sources'][BG])
    with Image.open(OLD / 'v2_domain_matched/visual/b0_original.png') as im:
        require(im.mode == 'RGB' and np.array_equal(bg, np.array(im)), 'Raw1949 and sealed1949 cache differ')
    with (OUT / 'background_original.npy').open('xb') as f:
        np.save(f, bg, allow_pickle=False)
    (OUT / 'references').mkdir()
    (OUT / 'geometry').mkdir()
    geometries, rgbs = {}, {}
    crop_box_shared = None
    refs_summary = {}
    for n in IDS:
        source = refs1[n]
        rgb = read_rgb(data_root / source['image']['path'], source['image'])
        mask = read_mask(data_root / source['mask']['path'], source['mask'])
        require(hashlib.sha256(rgb.tobytes()).hexdigest() == source['decoded_RGB_raw_bytes_sha256'], 'Original reference RGB differs')
        ref = prepare_reference(rgb, mask, {'context_factor': 4, 'model_side': 512})
        for key in ('bbox_xyxy_exclusive', 'crop_xyxy_exclusive', 'source_positive_pixels', 'model_positive_pixels'):
            require(ref[key] == source[key], 'Original reference crop differs: ' + key)
        l, t, r, b = ref['crop_xyxy_exclusive']
        with (OUT / f'references/{n}_source.npz').open('xb') as f:
            np.savez_compressed(f, original_mask=mask, native_rgb=rgb[t:b, l:r], native_mask=mask[t:b, l:r])
        Image.fromarray(ref['rgb']).save(OUT / f'references/{n}_rgb512.png')
        Image.fromarray(ref['mask'].astype('uint8') * 255).save(OUT / f'references/{n}_mask512.png')
        arrays, meta = inherited_geometry(mask, v3['geometries'][n + '_move'])
        with (OUT / f'geometry/{n}_move.npz').open('xb') as f:
            np.savez_compressed(f, **arrays)
        box = meta['crop_xyxy_exclusive']
        if crop_box_shared is None:
            crop_box_shared = box
        require(box == crop_box_shared, 'Unexpected different shared model context; stop, do not adapt')
        geometries[n], rgbs[n] = arrays, ref['rgb']
        refs_summary[n] = {k: ref[k] for k in ('bbox_xyxy_exclusive', 'crop_xyxy_exclusive', 'source_positive_pixels', 'model_positive_pixels')}
    l, t, r, b = crop_box_shared
    require((r - l, b - t) == (512, 512), 'Unexpected inherited context dimensions')
    Image.fromarray(bg[t:b, l:r]).save(OUT / 'background512.png')
    draw_overview(bg, rgbs, geometries, v3)
    save(OUT / 'requests.json', {'schema': 'minimal-reference-interface-requests-v1', 'requests': expected_requests(v3),
                               'count': 6, 'distinct_geometries': 3, 'planned_neural_outputs': 12, 'neural_outputs_completed': 0})
    stats = resources()
    protocol = {'schema': 'minimal-reference-interface-inputs-v1', 'status': STATUS,
                'date': '2026-10-16', 'request_count': 6, 'distinct_geometries': 3, 'background_count': 1,
                'reference_ids': list(IDS), 'seeds': list(SEEDS), 'planned_neural_outputs': 12,
                'neural_outputs_completed': 0, 'generation_authorized': False, 'training_authorized': False,
                'original_v3_positions_passed': 3, 'original_v3_positions_evaluated': 24, 'original_v3_ready_for_day2': False,
                'background_raw_RGB_sha256': hashlib.sha256(bg.tobytes()).hexdigest(),
                'reference_crops': refs_summary, 'resources_snapshot': stats,
                'execution': {'distinct_original_backgrounds_decoded': 1, 'normal_decode_calls': 2,
                              'normal_decode_detail': 'original1949 RGBA plus sealed1949 RGB, verified equal',
                              'original_reference_RGB_decodes': 3, 'original_reference_mask_decodes': 3,
                              'other_normal_image_reads': 0, 'new_structure_reviews': 0, 'sampling_calls': 0},
                'torch_loaded': 'torch' in sys.modules, 'model_forwards': 0, 'optimizer_steps': 0,
                'limitations': ['post-selection development inputs, no generalization evidence',
                               'different reference masks confound cross-reference appearance comparisons',
                               'R and C_guard are control/review regions, not generated semantic GT',
                               'no model, compositing or appearance quality has been verified yet']}
    save(OUT / 'input_protocol.json', protocol)
    report = f'''# 阶段A：独立最小参考条件接口Demo CPU输入准备

日期：2026年10月16日。状态：{STATUS}。仅CPU输入就绪，不授权模型生成。

背景1949原move位置，参考019／024／020，种子20261016／20261017：6请求、3参考几何、1背景；未来参考关闭／开启计划12份神经输出，当前完成0。
原v3完整结论仍为3/24通过、LOCAL_REVIEW_UNCERTAIN、ready_for_day2=false；本分支不能替代原48请求验收。没有重抽背景、修改结构框或重新审阅结构。

原图中心[665.6,512]；共同模型上下文[410,256,922,768]，512×512，不缩区；2倍保护区[564,410,768,614]。精确变换与掩码分别在scope_protocol.json及geometry中，继承v3原离散几何。三个参考原图支持域像素数分别为5364／5098／5548，形状不同，跨参考输出差异不能单归因于风格。

source_guard.json绑定v1／v2／v3冻结元信息、源码、原始来源文件hash与保存的RGB排除组身份。仅解码一个正常背景：1949原图及其v2封存副本各一次，两者RGB一致；参考原图3次、原掩码3次。未打开1540或其他正常图／旧评价图像，S_bg／Q_cal仅在既有划分元信息中用于排除，没有访问其图像或分数。图像保留原始对比度，无增强／gamma／归一化。

CPU_position_overview.png仅为输入位置总览，包含三个参考、1949整图上下文位置、原模型图块和R／保护区显示叠加。叠加不进入模型输入；不是合成结果、缺陷真值或新的结构判定。1949整图原有结构未删除。

实际资源快照：可用RAM {stats['available_RAM_GiB']:.3f}GiB／总{stats['total_RAM_GiB']:.3f}GiB；磁盘空闲{stats['free_disk_GiB']:.3f}GiB；Python {stats['python']}。CPU执行，非峰值；扩散显存及运行时间仍未实测。

没有Torch、模型前向、复合／合成、下载、训练或参数更新；没有准备模型权重。所有未来训练仍由用户手动启动。背景复合能力、参考条件响应及真实外观均尚未验证。

已实现：独立CLI的prepare-inputs／verify-inputs。完整同身份缓存仅核验复用，不重新解码原图或重建结果；未知文件、锁、部分事务保留并停止查因，不自动修复／重生成。只读核验会重建坐标数组以核对保存几何，不重新审阅背景。
'''
    (OUT / REPORT).write_text(report, encoding='utf-8')
    # Recheck sources/old text identities after preparation before sealing.
    require(frozen_sources()[0] == guard, 'Source identity changed during preparation')
    artifacts = {p.relative_to(OUT).as_posix(): sha256(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and p.name != 'preparation.incomplete'}
    require(set(artifacts) == ARTIFACT_NAMES, 'Unknown/missing file during preparation; preserve and investigate')
    save(OUT / 'completion.json', {'schema': 'minimal-reference-interface-completion-v1', 'status': STATUS,
                                   'implementation': implementation, 'artifacts': artifacts})
    (OUT / 'preparation.incomplete').unlink()
    result = verify_inputs()
    result['execution'] = protocol['execution']
    result['new_source_image_decodes'] = 8
    return result
