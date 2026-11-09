"""Standard-library-only stage-B metadata and local-file checks.

No imports of stage-A modules, Torch, model packages or image libraries. There
is deliberately no downloader, installer, model constructor or inference path.
"""
import ctypes
import hashlib
import importlib.metadata
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from .resource_spec import (A_COMPLETION_SHA256, BASE_FILES, BASE_REPO, BASE_REV,
                            EVIDENCE, IP_FILES, IP_REPO, IP_REV, PACKAGE_PINS,
                            PAIRED_PROTOCOL)

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'results/reference_interface_demo/v1'
OUT = ROOT / 'results/reference_interface_demo/resources_v1'
MODELS = ROOT / 'models/reference_interface_demo/v1'
ENV = ROOT / '.venv-reference-interface'
CORE = ('resources.py', 'resources_cli.py', 'resource_spec.py')
REPORT = '阶段B_静态资源与成对协议准备.md'
ARTIFACTS = {'stage_a_binding.json', 'resource_manifest.json', 'paired_inference_protocol.json',
             'dependency_protocol.json', 'official_sources.json', 'resource_snapshot.json',
             'requirements-reference-interface.txt', REPORT}
STATUS = 'STATIC_PROTOCOL_PREPARED_RESOURCES_PENDING'
_GUARDED = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def guard_static_scope():
    global _GUARDED
    if _GUARDED:
        return
    # source_guard is sealed by stage-A completion before it is trusted below.
    data_root = Path(load(A / 'source_guard.json')['source_root']).resolve()
    workspace, output = ROOT.resolve(), OUT.resolve()
    forbidden_imports = ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'PIL', 'cv2')

    def guard(event, args):
        if event == 'import' and args[0].split('.')[0] in forbidden_imports:
            raise RuntimeError('Stage B forbids model/image package import: ' + args[0])
        if event.startswith('socket.'):
            raise RuntimeError('Resource CLI is strictly offline')
        if event != 'open' or not isinstance(args[0], str):
            return
        path = Path(args[0]).resolve()
        mode = args[1]
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        if writing and path.is_relative_to(workspace) and not path.is_relative_to(output):
            raise RuntimeError('No write outside stage-B result directory: ' + str(path))
        if path.is_relative_to(data_root) or (path.is_relative_to(workspace) and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy', '.npz')):
            raise RuntimeError('Stage B does not open source images or image/array caches: ' + str(path))

    sys.addaudithook(guard)
    _GUARDED = True
    require(not any(k in sys.modules for k in forbidden_imports), 'Forbidden package already imported')


def stage_a_binding():
    guard_static_scope()
    require(digest(A / 'completion.json') == A_COMPLETION_SHA256, 'Stage-A completion identity differs')
    complete = load(A / 'completion.json')
    expected = set(complete['artifacts']) | {'completion.json'}
    require({p.relative_to(A).as_posix() for p in A.rglob('*') if p.is_file()} == expected, 'Stage-A inventory changed')
    expected_dirs = {a.as_posix() for name in expected for a in Path(name).parents if str(a) != '.'}
    require({p.relative_to(A).as_posix() for p in A.rglob('*') if p.is_dir()} == expected_dirs, 'Stage-A directory inventory changed')
    metadata_hashes = {}
    for name, value in complete['artifacts'].items():
        if Path(name).suffix in ('.json', '.md', '.txt'):
            require(digest(A / name) == value, 'Stage-A text metadata changed: ' + name)
            metadata_hashes[name] = value
    for name, value in complete['implementation'].items():
        require(digest(Path(__file__).parent / name) == value, 'Frozen stage-A implementation changed: ' + name)
    source = load(A / 'source_guard.json')
    for name, value in source['protected_text_code_metadata'].items():
        require(digest(ROOT / name) == value, 'Old text/code metadata changed: ' + name)
    requests = load(A / 'requests.json')
    expected_order = [(n, seed) for n in ('019', '024', '020') for seed in (20261016, 20261017)]
    require(len(requests['requests']) == requests['count'] == 6 and requests['distinct_geometries'] == 3, 'Stage-A count differs')
    require([(r['reference_id'], r['seed']) for r in requests['requests']] == expected_order, 'Request sequence differs')
    require(all(r['background_path'] == 'display/train/good/1949.png' and r['variant'] == 'move' for r in requests['requests']), 'Input scope differs')
    require(source['v3_original_positions_passed'] == 3 and source['v3_original_positions_evaluated'] == 24 and source['v3_ready_for_day2'] is False, 'Original v3 failure changed')
    return {'schema': 'reference-interface-stage-a-binding-v1', 'completion_sha256': A_COMPLETION_SHA256,
            'text_metadata_sha256': metadata_hashes, 'stage_a_implementation': complete['implementation'],
            'old_text_code_metadata_verified': len(source['protected_text_code_metadata']),
            'binary_artifact_hashes_inherited_not_reread': {p: h for p, h in complete['artifacts'].items() if Path(p).suffix not in ('.json', '.md', '.txt')},
            'source_file_identities_inherited': source['sources'],
            'request_order': [{'id': r['id'], 'reference_id': r['reference_id'], 'seed': r['seed'],
                              'geometry_source_key': r['geometry_source_key']} for r in requests['requests']],
            'source_image_bytes_reverified_this_stage': False, 'image_or_array_bytes_opened': 0,
            'original_v3': {'passed': 3, 'evaluated': 24, 'ready_for_day2': False},
            'future_forward_prerequisite': 'reverify all stage-A binary hashes before any explicit model loading/forward',
            'generation_authorized': False}


def manifest():
    rows = []
    for prefix, repo, rev, files in (('base', BASE_REPO, BASE_REV, BASE_FILES), ('ip_adapter', IP_REPO, IP_REV, IP_FILES)):
        for name, size, blob, sha in files:
            rows.append({'local_path': prefix + '/' + name, 'repo_id': repo, 'revision': rev,
                         'repository_path': name, 'bytes': size, 'git_blob_sha1': blob,
                         'LFS_content_sha256': sha, 'resolve_url': f'https://huggingface.co/{repo}/resolve/{rev}/{name}'})
    return {'schema': 'reference-interface-resource-manifest-v1', 'model_root': str(MODELS),
            'files': rows, 'count': len(rows), 'total_expected_bytes': sum(r['bytes'] for r in rows),
            'weight_files': sum(r['LFS_content_sha256'] is not None for r in rows),
            'source': 'public Hugging Face repository API blobs metadata; weight contents not requested',
            'metadata_urls': [f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true' for repo, rev in ((BASE_REPO, BASE_REV), (IP_REPO, IP_REV))],
            'base_is_non_Runway_affiliated_mirror': True,
            'licenses': {'base': 'CreativeML OpenRAIL-M, retain model-card restrictions',
                         'adapter_repository': 'Apache-2.0', 'vision_origin': 'LAION ViT-H original model card MIT',
                         'diffusers_code': 'Apache-2.0'},
            'scope': 'personal noncommercial public-resource research; not a company deployment approval',
            'load_contract': {'base': 'local base path, variant=fp16, use_safetensors=True, low_cpu_mem_usage=True',
                              'vision': 'explicit local ip_adapter/models/image_encoder path, use_safetensors=True, torch_dtype=float16',
                              'adapter': 'local ip_adapter path, subfolder=models, weight_name=ip-adapter-plus_sd15.safetensors, image_encoder_folder=None',
                              'remote_code': False, 'local_files_only': True},
            'verified': 'file listings/config/API static only; weight/header/parameter compatibility and runtime pending'}


def requirements_text():
    return ('# Proposed isolated environment only; not installed by this module.\n'
            '--extra-index-url https://download.pytorch.org/whl/cu128\n'
            + ''.join(f'{p}=={v}\n' for p, v in PACKAGE_PINS.items()))


def dependencies():
    return {'schema': 'reference-interface-dependencies-v1', 'environment_path': str(ENV),
            'python': '3.12.7', 'include_system_site_packages': False, 'core_exact_pins': PACKAGE_PINS,
            'metadata_compatibility': ['diffusers requires hub>=0.34.0 and safetensors>=0.3.1',
                                       'transformers requires hub>=0.34.0,<1 and tokenizers>=0.22.0,<=0.23.0',
                                       'accelerate requires torch>=2.0 and safetensors>=0.4.3'],
            'status': 'proposed core constraints only, not installed/resolver-tested/runtime-tested',
            'transitive_dependencies': 'resolver output and installed distribution identities must be frozen in a separate acquisition result before model use',
            'no_old_environment_modification': True, 'no_auto_installer': True}


def hardware_snapshot():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (x, ctypes.c_ulonglong) for x in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)), 'RAM query failed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    require(len(gpu) == 1, 'Expected one device; preserve and audit configuration')
    name, total, free, driver = [x.strip() for x in gpu[0].split(',')]
    installed = {}
    for p in PACKAGE_PINS:
        try:
            installed[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            installed[p] = None
    ram = memory.available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    gates = PAIRED_PROTOCOL['start_gates']
    return {'python': platform.python_version(), 'total_RAM_GiB': memory.total / 2**30,
            'free_RAM_GiB': ram, 'free_disk_GiB': disk,
            'GPU': {'name': name, 'total_MiB': int(total), 'free_MiB': int(free), 'driver': driver},
            'conservative_gate_pass': {'RAM': ram >= gates['free_RAM_GiB_min'],
                                       'GPU': int(free) >= gates['free_GPU_MiB_min'], 'disk': disk >= gates['free_disk_GiB_min']},
            'old_environment_metadata_only': installed, 'isolated_environment_exists': ENV.exists(),
            'model_root_exists': MODELS.exists(), 'snapshot_not_peak': True,
            'Torch_loaded': 'torch' in sys.modules, 'model_forwards': 0, 'optimizer_steps': 0}


def check_static_protocol():
    p = PAIRED_PROTOCOL
    require(p['reference_scales'] == {'reference_off': 0.0, 'reference_on': 0.5}, 'Unexpected reference contrast')
    require(p['num_inference_steps'] == 30 and p['strength'] == 1.0 and p['eta'] == 0.0, 'Unexpected sampling setup')
    require(p['height'] == p['width'] == 512 and p['padding_mask_crop'] is None, 'Unexpected crop')
    require(not p['generation_authorized'] and not p['training_authorized'] and not p['inference_implementation_exists'], 'Premature authorization')
    require(p['runtime']['safety_checker_enabled'] and not p['runtime']['attention_slicing'] and not p['runtime']['CPU_offload'], 'Wrong common runtime')
    require(len(manifest()['files']) == 20 and manifest()['weight_files'] == 6, 'Wrong resource denominator')


def verify_resources():
    guard_static_scope()
    require(not (OUT / 'preparation.incomplete').exists(), 'Incomplete static transaction: preserve and investigate')
    require((OUT / 'completion.json').is_file(), 'No complete stage-B resource protocol')
    complete = load(OUT / 'completion.json')
    require(set(complete['artifacts']) == ARTIFACTS and set(complete['implementation']) == set(CORE), 'Resource inventory schema differs')
    require({p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()} == ARTIFACTS | {'completion.json'}, 'Unknown/missing result file')
    require(not any(p.is_dir() for p in OUT.rglob('*')), 'Unexpected result subdirectory')
    for name, value in complete['implementation'].items():
        require(digest(Path(__file__).parent / name) == value, 'Frozen stage-B implementation changed: ' + name)
    for name, value in complete['artifacts'].items():
        require(digest(OUT / name) == value, 'Frozen resource artifact changed: ' + name)
    require(load(OUT / 'stage_a_binding.json') == stage_a_binding(), 'Upstream identity differs')
    require(load(OUT / 'resource_manifest.json') == manifest(), 'Resource manifest differs')
    require(load(OUT / 'paired_inference_protocol.json') == PAIRED_PROTOCOL, 'Paired protocol differs')
    require(load(OUT / 'dependency_protocol.json') == dependencies(), 'Dependency protocol differs')
    require(load(OUT / 'official_sources.json')['evidence'] == EVIDENCE, 'Official evidence record differs')
    require((OUT / 'requirements-reference-interface.txt').read_text(encoding='utf-8') == requirements_text(), 'Requirements differ')
    check_static_protocol()
    require(complete['status'] == STATUS and complete['ready_for_stage_C'] is False, 'Premature readiness')
    return {'status': STATUS, 'resource_protocol_verified': True, 'stage_A_requests': 6,
            'stage_A_geometries': 3, 'backgrounds': 1, 'planned_outputs': 12, 'actual_outputs': 0,
            'reference_scales': PAIRED_PROTOCOL['reference_scales'], 'original_v3_passed': '3/24',
            'original_v3_ready': False, 'stage_B_image_or_array_bytes_read': 0, 'model_forwards': 0,
            'torch_loaded': 'torch' in sys.modules, 'ready_for_stage_C': False,
            'cache_reuse': 'static identity verification only, no network, snapshot refresh, image bytes or models',
            'output': str(OUT)}


def safe_header(path):
    """Read only safetensors JSON header, never tensor data or Torch state."""
    with path.open('rb') as f:
        prefix = f.read(8)
        require(len(prefix) == 8, 'Truncated safetensors file')
        length = struct.unpack('<Q', prefix)[0]
        require(2 <= length <= 16 * 1024 * 1024, 'Invalid/oversized safetensors header')
        header = json.loads(f.read(length))
    payload_size = path.stat().st_size - 8 - length
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    for key, value in tensors.items():
        start, end = value['data_offsets']
        require(0 <= start <= end <= payload_size and all(isinstance(x, int) and x >= 0 for x in value['shape']), 'Invalid tensor header: ' + key)
    return tensors


def check_selected_shapes(local_path, header):
    if local_path == 'base/unet/diffusion_pytorch_model.fp16.safetensors':
        require(header['conv_in.weight']['shape'] == [320, 9, 3, 3], 'Expected nine-channel SD1.5 inpainting UNet')
    elif local_path == 'ip_adapter/models/image_encoder/model.safetensors':
        require(header['vision_model.embeddings.patch_embedding.weight']['shape'] == [1280, 3, 14, 14], 'Expected ViT-H/14 encoder')
    elif local_path == 'ip_adapter/models/ip-adapter-plus_sd15.safetensors':
        require(header['image_proj.proj_in.weight']['shape'][1] == 1280, 'Plus feature dimension differs')
        require(header['image_proj.proj_out.weight']['shape'][0] == 768, 'Adapter output must match SD1.5 attention')
        kv = [v for k, v in header.items() if k.startswith('ip_adapter.') and k.endswith(('to_k_ip.weight', 'to_v_ip.weight'))]
        require(kv and all(v['shape'][1] == 768 for v in kv), 'Adapter attention dimensions differ')


def inspect_local_resources():
    verify_resources()
    spec = manifest()
    expected = {r['local_path'] for r in spec['files']}
    if MODELS.exists():
        actual = {p.relative_to(MODELS).as_posix() for p in MODELS.rglob('*') if p.is_file()}
        allowed_dirs = {a.as_posix() for name in expected for a in Path(name).parents if str(a) != '.'}
        actual_dirs = {p.relative_to(MODELS).as_posix() for p in MODELS.rglob('*') if p.is_dir()}
        require(actual <= expected and actual_dirs <= allowed_dirs, 'Unknown/partial model files or directories: preserve and investigate')
    found, missing = [], []
    for row in spec['files']:
        path = MODELS / row['local_path']
        if not path.is_file():
            missing.append(row['local_path'])
            continue
        require(path.stat().st_size == row['bytes'], 'Partial/wrong file size: ' + row['local_path'])
        h = digest(path)
        if row['LFS_content_sha256']:
            require(h == row['LFS_content_sha256'], 'Wrong weight hash: ' + row['local_path'])
            header = safe_header(path)
            check_selected_shapes(row['local_path'], header)
            summary = {'tensor_header_entries': len(header), 'tensor_data_loaded': False}
        else:
            data = path.read_bytes()
            git = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            require(git == row['git_blob_sha1'], 'Wrong pinned small-file content: ' + row['local_path'])
            summary = {}
        found.append({'path': row['local_path'], 'sha256': h, **summary})
    cfg = ENV / 'pyvenv.cfg'
    installed = {}
    env_ok = False
    if cfg.exists():
        settings = dict(line.split(' = ', 1) for line in cfg.read_text(encoding='utf-8').splitlines() if ' = ' in line)
        require(settings.get('include-system-site-packages') == 'false', 'Isolated environment includes old site packages')
        require(settings.get('version') == '3.12.7', 'Unexpected isolated Python version')
        installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions(path=[str(ENV / 'Lib/site-packages')])}
        env_ok = (ENV / 'Scripts/python.exe').is_file() and all(installed.get(p.lower().replace('_', '-')) == v for p, v in PACKAGE_PINS.items())
    hardware = hardware_snapshot()
    return {'resource_files_verified': len(found), 'resource_files_expected': len(spec['files']),
            'verified_files': found, 'missing_files': missing, 'isolated_environment_core_metadata_match': env_ok,
            'environment_core_versions': {p: installed.get(p.lower().replace('_', '-')) for p in PACKAGE_PINS},
            'transitive_environment_identity_frozen': False, 'runtime_compatibility_verified': False,
            'hardware_snapshot': hardware, 'no_image_decode_model_load_or_network': True,
            'static_files_env_gates_ok': not missing and env_ok and all(hardware['conservative_gate_pass'].values()),
            'forward_authorized': False, 'inference_implementation_exists': False}


def prepare_resources():
    guard_static_scope()
    if (OUT / 'completion.json').exists():
        return verify_resources()
    require(not OUT.exists(), 'Existing unknown/partial stage-B directory: preserve and investigate')
    binding = stage_a_binding()
    check_static_protocol()
    hardware = hardware_snapshot()
    OUT.mkdir()
    (OUT / 'preparation.incomplete').write_text('Static resource transaction. Preserve incomplete state; no automatic restart.\n', encoding='utf-8')
    implementation = {name: digest(Path(__file__).parent / name) for name in CORE}
    save(OUT / 'stage_a_binding.json', binding)
    save(OUT / 'resource_manifest.json', manifest())
    save(OUT / 'paired_inference_protocol.json', PAIRED_PROTOCOL)
    save(OUT / 'dependency_protocol.json', dependencies())
    save(OUT / 'official_sources.json', {'date': '2026-10-16', 'evidence': EVIDENCE,
         'network_issue': 'Sandbox urllib failed at socket connect with WinError10013; same metadata-only request succeeded with approved network access. No repository/model changes made.',
         'weights_downloaded': 0, 'packages_downloaded_or_installed': 0})
    save(OUT / 'resource_snapshot.json', hardware)
    (OUT / 'requirements-reference-interface.txt').write_text(requirements_text(), encoding='utf-8')
    size = manifest()['total_expected_bytes']
    report = f'''# 阶段B：静态资源与成对推理协议准备

日期：2026年10月16日。状态：{STATUS}。阶段B静态准备完成，不代表资源已取得或模型可运行；ready_for_stage_C=false。

## 固定输入与隔离

绑定阶段A completion的SHA256：{A_COMPLETION_SHA256}。仅复核封存文字元信息、三份阶段A源码及64份旧文字／源码／元信息；继承已核验的来源与二进制hash，本轮不打开任何图像或数组字节，不重新审阅结构。仍为1949原move、019／024／020、两种子，6请求／3几何／1背景。原v3仍3/24、ready=false，不替代原48请求。

阶段A代码／21文件及旧成果不变。阶段B采用resources.py、resources_cli.py、resource_spec.py独立实现与本目录独立协议，没有改旧CLI。基础模型、编码器、适配器未加载，未导入Torch、下载权重、安装依赖、前向、合成或训练。

## 候选与官方依据

基础模型固定为[{BASE_REPO}](https://huggingface.co/{BASE_REPO}/blob/{BASE_REV}/README.md)，revision={BASE_REV}。该仓库明确是非Runway关联镜像，不能称原厂官方发布；模型卡标示CreativeML OpenRAIL-M及研究用途。[模型卡所链许可](https://huggingface.co/spaces/CompVis/stable-diffusion-license)规定使用限制及再分发时的许可／声明保留；镜像清单未列单独LICENSE文件。本项目继续仅个人非商业公开研究并保留这些限制，不将镜像声明当作原厂权重同一性证明。

参考条件固定为[h94/IP-Adapter](https://huggingface.co/{IP_REPO}/blob/{IP_REV}/README.md)的models/ip-adapter-plus_sd15.safetensors及models/image_encoder，revision={IP_REV}。作者模型卡说明Plus使用ViT-H局部特征；不是SDXL、FaceID或bigG。适配器库标示Apache-2.0，原[LAION ViT-H模型卡](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/README.md)标示MIT；本轮绑定的是h94提供的转换文件，未另换LAION权重。

静态配置确认补绘UNet输入9通道／交叉注意力768，图像编码器224输入／patch14／hidden1280。[Diffusers v0.35.1参考加载源码](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/loaders/ip_adapter.py)与[补绘API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/inpaint)提供对应入口，形成可审查的接线依据；本地权重header、完整参数和运行兼容性仍待核验。它们不是弱Mura有效性证据。

## 已冻结的成对设置

同一StableDiffusionInpaintPipeline和已加载适配器，参考关闭0、开启0.5；不卸载适配器。[官方注意力实现](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/models/attention_processor.py)的零scale分支跳过IP贡献，未来仍须核对实际处理器和全部scale。

512×512、batch1、FP16、DDIM30步、strength1.0、CFG7.5、eta0；精确beta／调度设置见paired_inference_protocol.json。提示固定为“a dark display surface with a subtle diffuse brightness non-uniformity”，负提示固定为“text, symbols, objects, watermark, sharp edges, high contrast”。这是生成前的工程选择，不是按样例调出的最优参数。

参考保留阶段A原始512上下文，不涂黑语义掩码；按[固定版本CLIPImageProcessor](https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/models/clip/image_processing_clip.py)标准224缩放／中心裁剪／均值方差归一化。该预处理不改变封存原图，不做自适应对比度增强；224编码可能丢失弱Mura，作用未知。Plus取倒数第二层特征；条件／无条件特征同一份缓存供开关条件使用。原参考掩码用于既定几何，不额外增加IP注意力区域掩码。

每种子未来在CPU产生一份[1,4,64,64]float32初始噪声，保存hash后转FP16，成对共享，传入latents时不自行乘scheduler sigma。每请求的masked-image VAE采样用单独同seed CPU生成器，沿用作者sample行为而非改用mode，预先保存一份潜变量并通过masked_image_latents共享。重置调度与条件生成器，eta0；只在同设备／软件条件下约束比较，不承诺跨版本逐位一致。以上本轮均只写协议，没有产生Torch噪声或潜变量。

仅使用SDPA math，保持IPAdapterAttnProcessor2_0；不启用可能替换处理器的attention slicing、xformers、CPU offload或VAE tiling。关闭TF32／benchmark、启用确定性要求，保留安全检查器；不支持就停止，不自动切换。原始模型输出与R内受限复合未来分别保存，域外零变化来自工程复合，不能算模型学会保留背景。

## 尚缺资源与实际设备

resource_manifest.json列出20个确切文件，其中6个权重，总计{size}字节（{size / 2**30:.3f}GiB，含小配置／分词器／模型卡，不含Python软件包）。ViT-H下载文件自身约2.35GiB；模型卡泛称适配器轻量不包括它。仅选择base中fp16 safetensors，以及h94的一个Plus文件和一个编码器safetensors，不下载整个模型仓库或.bin重复版本。文件大小、LFS SHA256／小文件Git blob SHA1及固定revision链接均已保存。

预期本地根：{MODELS}，分别放base／ip_adapter子目录；尚未创建或取得。本次独立环境{ENV}也未创建。requirements-reference-interface.txt仅为经PyPI元信息审查的核心候选pin，Python3.12.7、Torch2.8.0+cu128等；没有执行完整依赖解析、安装、二进制导入或运行，不把候选依赖称为可用环境。后续另存完整传递依赖与实际身份，旧环境不修改。

实际快照：{hardware['GPU']['name']}，总显存{hardware['GPU']['total_MiB']}MiB、空闲{hardware['GPU']['free_MiB']}MiB，驱动{hardware['GPU']['driver']}；RAM可用{hardware['free_RAM_GiB']:.3f}／总{hardware['total_RAM_GiB']:.3f}GiB，磁盘空闲{hardware['free_disk_GiB']:.3f}GiB。快照非峰值。保守启动门槛为可用RAM≥6GiB、显存≥7168MiB、磁盘≥20GiB，当前分别{hardware['conservative_gate_pass']}。这些是工程门槛而非实测最低需求；约3.8GiB的FP16权重驻留估算不含激活／临时副本／CUDA及加载峰值，8GB运行可行性仍未知，不能套用旧DRAEM耗时。

## 已实现命令与停止规则

独立resources_cli提供prepare-resources、verify-resources、inspect-local-resources；没有下载、安装、模型加载或生成命令。完整同身份prepare只核验缓存，不刷新快照或网络；inspect只读当前指定目录、文件hash与safetensors JSON header及目标环境dist-info。若权重存在，完整hash会流式读取整个权重文件字节，header解析只读取其中JSON元信息；不导入Torch、反序列化张量或实例化模型。header只核对必要接口尺寸，不代替完整模型加载兼容性检验。本次指定模型目录不存在，没有读取任何本地权重字节。

未知文件、partial／锁、身份／API不符、资源不足、OOM、安全拦截或不明确前向必须保留并查因，不换seed、挑结果、放宽规则或自动重试。当前缺环境和权重，且前向实现尚不存在，不进入阶段C。全部未来训练继续由用户手动启动。

元信息访问曾在sandbox socket连接处报WinError10013，经systematic-debugging定位访问层，获准的同类公开元信息请求成功；没有将该错误当成模型下架，也没有改项目数据或安全检查器。
'''
    (OUT / REPORT).write_text(report, encoding='utf-8')
    require(stage_a_binding() == binding, 'Upstream changed during preparation')
    artifacts = {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != 'preparation.incomplete'}
    require(set(artifacts) == ARTIFACTS, 'Unexpected static output; preserve and investigate')
    save(OUT / 'completion.json', {'schema': 'reference-interface-static-resources-completion-v1',
                                  'status': STATUS, 'ready_for_stage_C': False,
                                  'implementation': implementation, 'artifacts': artifacts})
    (OUT / 'preparation.incomplete').unlink()
    return verify_resources()
