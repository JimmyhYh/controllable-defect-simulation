"""Pinned public-resource acquisition. Standard library only; no model imports.

A failed/partial transaction is retained and cannot be restarted by this CLI.
The frozen stage-B verifier runs in a separate offline process.
"""
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'results/reference_interface_demo/resources_v1'
OUT = ROOT / 'results/reference_interface_demo/acquisition_v1'
ENV = ROOT / '.venv-reference-interface'
MODELS = ROOT / 'models/reference_interface_demo/v1'
OLD_ENV = ROOT / '.venv'
OLD_PY = OLD_ENV / 'Scripts/python.exe'
NEW_PY = ENV / 'Scripts/python.exe'
CORE = ('acquisition.py', 'acquisition_cli.py')
B_COMPLETION = '41ac6dad88a9e37888153c9052cd32553e5e4463e1ff7c96884c34fd893266ee'
REPORT = '阶段B后续_资源取得与身份核验.md'
ARTIFACTS = {'acquisition_protocol.json', 'downloaded_files.json', 'safetensors_headers.json',
             'pip_install_report.json', 'environment_identity.json', 'local_inspection.json', REPORT}
_GUARDED = False


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def progress(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), flush=True)


def scope_guard():
    global _GUARDED
    if _GUARDED:
        return
    source_root = Path(load(ROOT / 'results/reference_interface_demo/v1/source_guard.json')['source_root']).resolve()
    allowed_write = [p.resolve() for p in (OUT, ENV, MODELS)]
    forbidden = ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'PIL', 'cv2')

    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in forbidden:
            raise RuntimeError('Acquisition forbids model/image package imports: ' + args[0])
        if event != 'open' or not isinstance(args[0], str):
            return
        path = Path(args[0]).resolve()
        mode = args[1]
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        if path.is_relative_to(source_root):
            raise RuntimeError('No source dataset access during acquisition')
        if path.is_relative_to(ROOT) and not path.is_relative_to(ENV) and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy', '.npz'):
            raise RuntimeError('No project image/array reads during acquisition')
        if writing and path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in allowed_write):
            raise RuntimeError('Write outside independent acquisition/model/environment directories')

    sys.addaudithook(audit)
    _GUARDED = True
    require(not any(x in sys.modules for x in forbidden), 'Forbidden package already loaded')


def child_env():
    cleaned = {k: v for k, v in os.environ.items() if not k.upper().startswith(('PIP_', 'PYTHON'))}
    return {**cleaned, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1',
            'PIP_CONFIG_FILE': os.devnull, 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}


def frozen_inspect(command):
    result = subprocess.run([str(OLD_PY), '-B', '-X', 'utf8', '-m', 'reference_interface_demo.resources_cli', command],
                            cwd=ROOT, env=child_env(), text=True, encoding='utf-8', capture_output=True)
    require(result.returncode == 0, 'Frozen static verifier failed: ' + result.stderr[-3000:])
    return json.loads(result.stdout)


def upstream():
    require(sha(B / 'completion.json') == B_COMPLETION, 'Frozen stage-B completion differs')
    result = frozen_inspect('verify-resources')
    require(result['resource_protocol_verified'] and not result['torch_loaded'], 'Upstream static identity invalid')
    spec = load(B / 'resource_manifest.json')
    require(spec['count'] == 20 and Path(spec['model_root']).resolve() == MODELS.resolve(), 'Unexpected manifest/root')
    for row in spec['files']:
        relative = Path(row['local_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path')
        require((MODELS / relative).resolve().is_relative_to(MODELS.resolve()), 'Manifest escapes model root')
        url = urllib.parse.urlsplit(row['resolve_url'])
        require(url.scheme == 'https' and url.netloc == 'huggingface.co' and len(row['revision']) == 40, 'Unexpected pinned resource origin')
    return spec


def old_environment_identity():
    files = [OLD_ENV / 'pyvenv.cfg', OLD_PY]
    for d in importlib.metadata.distributions(path=[str(OLD_ENV / 'Lib/site-packages')]):
        for name in ('METADATA', 'RECORD'):
            p = Path(d._path) / name
            if p.is_file():
                files.append(p)
    return {p.relative_to(OLD_ENV).as_posix(): sha(p) for p in sorted(set(files))}


def download_one(row):
    target = MODELS / row['local_path']
    part = target.with_name(target.name + '.download.part')
    require(not target.exists() and not part.exists(), 'Existing resource or partial download: preserve and investigate')
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(row['resolve_url'], headers={
        'User-Agent': 'reference-interface-acquisition/1', 'Cache-Control': 'no-cache'})
    started = time.monotonic()
    digest = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(row['bytes']).encode() + b'\0')
    count, next_report = 0, 64 * 1024 * 1024
    progress('download_start', file=row['local_path'], expected_bytes=row['bytes'])
    # One logical file request (normal HTTPS redirects allowed); no retry,
    # range resume or cached-file replacement.
    with urllib.request.urlopen(request, timeout=120) as response:
        host = urllib.parse.urlsplit(response.url).hostname
        require(response.status == 200 and host and (host == 'huggingface.co' or host.endswith(('.hf.co', '.huggingface.co'))), 'Unexpected download response/origin')
        content_length = response.headers.get('Content-Length')
        if content_length:
            require(int(content_length) == row['bytes'], 'Unexpected HTTP length; preserve without retry')
        with part.open('xb') as f:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                count += len(block)
                require(count <= row['bytes'], 'Resource exceeds frozen size')
                f.write(block)
                digest.update(block)
                blob.update(block)
                if count >= next_report:
                    progress('download_progress', file=row['local_path'], bytes=count,
                             total=row['bytes'], elapsed_seconds=round(time.monotonic() - started, 1))
                    next_report += 64 * 1024 * 1024
    require(count == row['bytes'], 'Incomplete download: preserve partial file and investigate')
    if row['LFS_content_sha256']:
        require(digest.hexdigest() == row['LFS_content_sha256'], 'Downloaded weight hash differs; preserve partial file')
    else:
        require(blob.hexdigest() == row['git_blob_sha1'], 'Downloaded small-file Git blob differs; preserve partial file')
    require(not target.exists(), 'Target appeared during download')
    part.rename(target)
    record = {'path': row['local_path'], 'bytes': count, 'sha256': digest.hexdigest(),
              'git_blob_sha1': None if row['LFS_content_sha256'] else blob.hexdigest(),
              'source_revision': row['revision'], 'final_response_host': host,
              'elapsed_seconds': time.monotonic() - started, 'download_attempts': 1}
    progress('download_verified', file=row['local_path'], bytes=count, seconds=round(record['elapsed_seconds'], 1))
    return record


PIP_BOOTSTRAP = r'''
import sys,runpy,sysconfig,os,importlib.metadata
from pathlib import Path
expected=Path(sys.argv[1]).resolve()
sys.argv.pop(1)
if Path(sys.prefix).resolve()!=expected or not Path(sys.executable).resolve().is_relative_to(expected):
    raise RuntimeError('pip interpreter/prefix is not the independent environment')
if any(not Path(sysconfig.get_path(k)).resolve().is_relative_to(expected) for k in ('purelib','platlib','scripts','data')):
    raise RuntimeError('pip installation scheme escapes the independent environment')
if os.environ.get('PIP_CONFIG_FILE')!=os.devnull or importlib.metadata.version('pip')!='24.2':
    raise RuntimeError('Unexpected pip bootstrap version/configuration')
def guard(event,args):
    if event=='import' and args[0].split('.')[0] in ('torch','torchvision','diffusers','transformers','accelerate','PIL','cv2'):
        raise RuntimeError('Package acquisition must not import model/image packages: '+args[0])
sys.addaudithook(guard)
if any(k in sys.modules for k in ('torch','torchvision','diffusers','transformers','accelerate','PIL','cv2')):
    raise RuntimeError('Forbidden package imported during interpreter startup')
from pip._internal.configuration import Configuration
config=Configuration(isolated=True)
config.load()
if list(config.items()):
    raise RuntimeError('pip configuration is not empty; stop before installation')
sys.argv=['pip']+sys.argv[1:]
runpy.run_module('pip',run_name='__main__')
'''


def pip_command(arguments, capture=False):
    command = [str(NEW_PY), '-I', '-B', '-X', 'utf8', '-c', PIP_BOOTSTRAP, str(ENV), *arguments]
    result = subprocess.run(command, cwd=ROOT, env=child_env(), text=True, encoding='utf-8',
                            capture_output=capture)
    require(result.returncode == 0, 'Isolated pip failed; preserve environment/report, do not rerun install automatically')
    return result.stdout if capture else None


def environment_identity():
    # -S avoids .pth/sitecustomize; explicitly scan only the new distribution path.
    script = 'import sys,json;print(json.dumps({"version":sys.version,"version_triplet":list(sys.version_info[:3]),"executable":sys.executable,"base_executable":sys._base_executable,"torch_loaded":"torch" in sys.modules}))'
    identity = json.loads(subprocess.check_output([str(NEW_PY), '-I', '-S', '-B', '-c', script], cwd=ROOT, env=child_env(), text=True))
    require(identity['version_triplet'] == [3, 12, 7] and not identity['torch_loaded'], 'Unexpected isolated interpreter')
    site = ENV / 'Lib/site-packages'
    pths = {p.name: p.read_text(encoding='utf-8') for p in site.glob('*.pth')}
    require(all('torch' not in text.lower() and 'diffusers' not in text.lower() and 'transformers' not in text.lower() for text in pths.values()), 'Potential model import in .pth; stop before another interpreter startup')
    distributions = []
    for d in importlib.metadata.distributions(path=[str(site)]):
        info = Path(d._path)
        distributions.append({'name': d.metadata['Name'], 'version': d.version, 'requires_dist': list(d.requires or []),
                              'metadata_sha256': sha(info / 'METADATA'),
                              'record_sha256': sha(info / 'RECORD') if (info / 'RECORD').exists() else None})
    distributions.sort(key=lambda x: x['name'].lower())
    versions = {d['name'].lower().replace('_', '-'): d['version'] for d in distributions}
    pins = load(B / 'dependency_protocol.json')['core_exact_pins']
    require(all(versions.get(p.lower().replace('_', '-')) == v for p, v in pins.items()), 'Installed core pin mismatch; preserve and investigate')
    identity.update({'pyvenv_cfg': (ENV / 'pyvenv.cfg').read_text(encoding='utf-8'),
                     'pyvenv_cfg_sha256': sha(ENV / 'pyvenv.cfg'), 'launcher_sha256': sha(NEW_PY),
                     'base_python_sha256': sha(Path(identity['base_executable'])),
                     'pth_files': pths, 'all_distributions': distributions,
                     'pip_check': pip_command(['--isolated', '--disable-pip-version-check', 'check'], capture=True).strip(),
                     'method': 'stdlib metadata/RECORD identities, no model/package import or runtime compatibility proof'})
    return identity


def header_summaries(spec):
    result = []
    for row in spec['files']:
        if not row['LFS_content_sha256']:
            continue
        path = MODELS / row['local_path']
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            require(2 <= length <= 16 * 1024 * 1024, 'Unexpected safetensors header length')
            raw = f.read(length)
        header = json.loads(raw)
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        selected = {k: v for k, v in tensors.items() if k in (
            'conv_in.weight', 'vision_model.embeddings.patch_embedding.weight',
            'vision_model.embeddings.position_embedding.weight', 'image_proj.proj_in.weight',
            'image_proj.proj_out.weight', 'image_proj.latents') or k.endswith(('to_k_ip.weight', 'to_v_ip.weight'))}
        result.append({'path': row['local_path'], 'header_bytes': length, 'header_sha256': hashlib.sha256(raw).hexdigest(),
                       'tensor_entries': len(tensors), 'dtype_counts': {k: sum(v['dtype'] == k for v in tensors.values()) for k in sorted({v['dtype'] for v in tensors.values()})},
                       'necessary_tensor_metadata': selected, 'tensor_data_deserialized': False})
    return {'weights': result, 'count': len(result), 'note': 'Full file hash reads all bytes; only header JSON is parsed, no tensor deserialization.'}


def verify_acquisition():
    scope_guard()
    require(not (OUT / 'acquisition.incomplete').exists(), 'Incomplete acquisition: preserve and diagnose; do not restart')
    complete = load(OUT / 'completion.json')
    require(set(complete['artifacts']) == ARTIFACTS and set(complete['implementation']) == set(CORE), 'Unexpected acquisition inventory')
    require({p.name for p in OUT.iterdir() if p.is_file()} == ARTIFACTS | {'completion.json'} and not any(p.is_dir() for p in OUT.iterdir()), 'Unknown acquisition files/directories')
    for name, value in complete['artifacts'].items():
        require(sha(OUT / name) == value, 'Acquisition artifact changed: ' + name)
    for name, value in complete['implementation'].items():
        require(sha(Path(__file__).parent / name) == value, 'Acquisition implementation changed: ' + name)
    upstream()
    before = load(OUT / 'acquisition_protocol.json')
    require(old_environment_identity() == before['old_environment_metadata_identity'], 'Old environment metadata changed')
    inspection = frozen_inspect('inspect-local-resources')
    require(inspection['resource_files_verified'] == 20 and inspection['isolated_environment_core_metadata_match'], 'Acquired resource identity incomplete')
    current_env = environment_identity()
    require(current_env == load(OUT / 'environment_identity.json'), 'Installed environment identity differs')
    return {'status': 'RESOURCES_ACQUIRED_STATIC_IDENTITY_VERIFIED', 'resource_files': 20,
            'weights': 6, 'installed_distributions': len(current_env['all_distributions']),
            'torch_imported': False, 'model_forwards': 0, 'training_steps': 0,
            'image_array_reads': 0, 'actual_generated_outputs': 0, 'runtime_compatibility_verified': False,
            'current_resource_gates': inspection['hardware_snapshot']['conservative_gate_pass'],
            'forward_authorized': False, 'original_v3_passed': '3/24', 'output': str(OUT)}


def acquire():
    scope_guard()
    if (OUT / 'completion.json').exists():
        return verify_acquisition()
    require(not any(p.exists() for p in (OUT, ENV, MODELS)), 'Existing unknown/partial acquisition, environment or model directory: preserve and investigate')
    spec = upstream()
    old_identity = old_environment_identity()
    implementation = {n: sha(Path(__file__).parent / n) for n in CORE}
    OUT.mkdir()
    (OUT / 'acquisition.incomplete').write_text('Authorized acquisition only. Partial state preserved; no automatic restart/redownload/reinstall.\n', encoding='utf-8')
    save(OUT / 'acquisition_protocol.json', {'stage_B_completion_sha256': B_COMPLETION,
         'resource_manifest_sha256': sha(B / 'resource_manifest.json'),
         'requirements_sha256': sha(B / 'requirements-reference-interface.txt'),
         'old_environment_metadata_identity': old_identity,
         'model_files': 20, 'expected_model_bytes': spec['total_expected_bytes'],
         'model_root': str(MODELS), 'new_environment': str(ENV), 'retry_count': 0,
         'installation_isolation': {'PIP_CONFIG_FILE': os.devnull, 'inherited_pip_python_options_removed': True,
                                    'pip_version': '24.2', 'explicit_prefix': str(ENV),
                                    'prefix_scheme_and_empty_configuration_checked_before_pip': True},
         'generation_authorized': False, 'training_authorized': False, 'implementation': implementation})
    progress('environment_create', path=str(ENV))
    subprocess.run([str(Path(sys._base_executable)), '-I', '-S', '-B', '-m', 'venv', str(ENV)], cwd=ROOT, env=child_env(), check=True)
    pip_version = pip_command(['--isolated', '--disable-pip-version-check', '--version'], capture=True).strip()
    progress('environment_destination_verified', pip=pip_version, prefix=str(ENV))
    MODELS.mkdir(parents=True)
    records = [download_one(row) for row in spec['files']]
    save(OUT / 'downloaded_files.json', {'files': records, 'count': len(records), 'total_bytes': sum(r['bytes'] for r in records)})
    progress('isolated_wheel_install_start', core_pin_count=10)
    pip_command(['--isolated', '--disable-pip-version-check', '--no-cache-dir', '--retries', '0', '--timeout', '120',
                 'install', '--prefix', str(ENV), '--only-binary=:all:', '--no-compile', '--index-url', 'https://pypi.org/simple',
                 '--report', str(OUT / 'pip_install_report.json'), '-r', str(B / 'requirements-reference-interface.txt')])
    identity = environment_identity()
    save(OUT / 'environment_identity.json', identity)
    progress('static_local_resource_inspection')
    inspection = frozen_inspect('inspect-local-resources')
    require(inspection['resource_files_verified'] == 20 and inspection['isolated_environment_core_metadata_match'], 'Resource inspection incomplete')
    save(OUT / 'local_inspection.json', inspection)
    save(OUT / 'safetensors_headers.json', header_summaries(spec))
    require(old_environment_identity() == old_identity, 'Old environment metadata changed')
    upstream()
    resource = inspection['hardware_snapshot']
    report = f'''# 阶段B后续：资源取得与静态身份核验

20文件／6权重已按冻结revision、精确大小和内容hash取得；合计{spec['total_expected_bytes']}字节（{spec['total_expected_bytes']/2**30:.3f}GiB），未下载全仓库或.bin重复权重。基础模型仍是已注明的非原厂镜像。

新环境{ENV}已建立，Python3.12.7、10个核心pin一致，共{len(identity['all_distributions'])}个实际分发包。pip_install_report.json保存实际取得的wheel来源与归档hash，environment_identity.json保存完整传递依赖、METADATA／RECORD、解释器／launcher／base Python及pyvenv.cfg身份；pip check结果：{identity['pip_check']}。仅安装wheel，无源码构建；没有导入Torch、图像／模型包或加载模型。旧.venv元信息／RECORD及解释器身份前后核验相等；没有全量重hash旧包二进制。

执行前独立审阅发现--isolated仍可能加载系统／venv pip配置，已核对本机捆绑pip24.2的配置源码并在新入口修正：子进程移除继承PIP／PYTHON选项、PIP_CONFIG_FILE=os.devnull、显式新env --prefix，并在pip操作前检查解释器prefix、安装scheme及配置为空。未修改机器pip.ini或旧环境；这不是通用系统级沙箱。

已有inspect-local-resources实际核验20/20文件与核心环境元信息；safetensors_headers.json保存6权重必要JSON header尺寸、dtype与header hash，不反序列化张量。全文件hash会读取权重全字节；本轮没有读取任何背景／参考图像或数组字节。A／B封存源码、21／9产物与原v3的3/24失败保持不变。

实际资源快照：RAM可用{resource['free_RAM_GiB']:.3f}GiB／总{resource['total_RAM_GiB']:.3f}GiB；空闲显存{resource['GPU']['free_MiB']}／{resource['GPU']['total_MiB']}MiB；磁盘空闲{resource['free_disk_GiB']:.3f}GiB。快照非峰值，保守工程门槛当前为{resource['conservative_gate_pass']}，不是实测最低需求。若RAM不足6GiB，未来加载前需释放内存，不自动缩配置或CPU卸载。

状态仅为资源取得与静态身份核验完成，不代表运行兼容、确定性内核、峰值资源、参考响应或背景复合已验证。6请求计划12份输出，实际0；推理入口尚未实现，本次不进入阶段C。所有训练仍由用户手动启动。

已实现acquisition_cli的acquire／verify-acquisition；完整同身份结果仅静态核验复用。未知／不完整事务停止保留，不自动重下载／重装、换pin或改旧环境。资源核验可能需数分钟（全量hash），具体取决于磁盘；本轮没有模型运行耗时估计。
'''
    (OUT / REPORT).write_text(report, encoding='utf-8')
    artifacts = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'acquisition.incomplete'}
    require(set(artifacts) == ARTIFACTS, 'Unknown acquisition artifacts; preserve and diagnose')
    save(OUT / 'completion.json', {'status': 'RESOURCES_ACQUIRED_STATIC_IDENTITY_VERIFIED', 'implementation': implementation, 'artifacts': artifacts,
                                  'model_forward_authorized': False})
    (OUT / 'acquisition.incomplete').unlink()
    return {'status': 'RESOURCES_ACQUIRED_STATIC_IDENTITY_VERIFIED', 'resource_files': 20, 'weights': 6,
            'installed_distributions': len(identity['all_distributions']), 'model_forwards': 0,
            'torch_imported': False, 'resource_gates': resource['conservative_gate_pass'], 'output': str(OUT)}
