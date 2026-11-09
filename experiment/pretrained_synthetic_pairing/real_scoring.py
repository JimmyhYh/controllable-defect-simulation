"""Independent E1004 metadata preparation and explicit single-model CLI."""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
from pathlib import Path
import sys
from . import calibration as a

t, c = a.t, a.c
PREP = c.RESULTS / "real_scoring_preparation_v1"
CAL = c.RESULTS / "calibration_verifications_v1/all_six_v1"
CAL_SHA = "52d2feed695c0fac15528e3e155c6d6863cdda82bb484ae1b5e6f0e0d7ed30f6"
CODE = ("real_scoring.py", "real_runtime.py", "real_cache.py", "README_real_scoring.md")
REPORT = "第5阶段E1004真实评分准备与运行.md"
FILES = {"transaction.json", "protocol.json", "requests.json", "metric_sources.json", "models.json", "sources.json", "environment.json", "scalar_verification.json", REPORT}
_GUARDED = False


def paths(group, seed):
    t.paths(group, seed)
    return c.RESULTS / "real_evaluation_v1" / f"{group}_{seed}", c.LARGE / "runtime_real_evaluation_v1" / f"{group}_{seed}"


def implementation():
    return {n: t.sha(c.PACKAGE / n) for n in CODE}


def requests():
    c.require(t.sha(a.SPLIT) == a.SPLIT_SHA and t.sha(a.INVENTORY) == a.INVENTORY_SHA, "E metadata anchors differ")
    split = t.load(a.SPLIT); inv = t.load(a.INVENTORY)["files"]; rows = split["E_official1004"]
    c.require(len(rows) == len({r['id'] for r in rows}) == 1004, "E1004 original count differs")
    others = {r['id'] for k in ('R_real5', 'N_train1885', 'Q_normal200', 'U_anomaly241') for r in split[k]}
    c.require(not others.intersection(r['id'] for r in rows), "E role overlap")
    req, gt = [], []
    for i, row in enumerate(rows):
        c.require(row['part'] == 'test' and row['training_label'] is None and row['image'] == f"test/{row['id']}.png" and row['mask'] == f"test/{row['id']}_GT.png", "E source metadata differs")
        c.require(row['image_sha256'] == inv[row['image']], "E image hash metadata differs")
        req.append({'index': i, 'id': row['id'], 'image': row['image'], 'image_sha256': row['image_sha256'], 'height': row['height'], 'width': row['width'], **a.geometry(row['height'], row['width'])})
        gt.append({'index': i, 'id': row['id'], 'mask': row['mask'], 'mask_sha256': inv[row['mask']], 'native_shape': [row['height'], row['width']]})
    c.require(sum(r['window_count'] for r in req) == 4270 and sum(r['native_pixels'] for r in req) == 146411528, "E planned native denominators differ")
    return req, gt


def scalar_verification():
    current = paths('B', c.SEEDS[0])[0]; previous = paths('A', c.SEEDS[0])[0] / 'probabilities/metadata_only_fake.npy'
    cases = [probability_read_allowed(previous, None, True, True), probability_read_allowed(previous, current, False, True),
             probability_read_allowed(previous, current, False, False), probability_read_allowed(previous, current, True, True)]
    c.require(cases == [False, True, False, False], 'Future predecessor hash-only scope differs')
    return {'fictitious_only': True, 'inherited_window_FP32_transaction_scalars': a.scalar_verification(),
            'future_predecessor_hash_only_and_metadata_no_probability_read': cases, 'E_labels_or_real_metrics_computed': False}


def probability_read_allowed(path, cache, cpu, hashing):
    if cache is None: return False
    if path.is_relative_to(cache / 'probabilities'): return True
    if cpu or not hashing: return False
    previous = []
    for group, seed in t.ORDER:
        root, _ = paths(group, seed)
        if root == cache: return any(path.is_relative_to(p / 'probabilities') for p in previous)
        previous.append(root)
    return False


def guard(cpu=True, destinations=(), raw_e=False, cache=None, temporary=None):
    global _GUARDED
    c.require(not _GUARDED, "One explicit operation per process")
    destinations = tuple(c.no_redirect(p) for p in destinations)
    cache = c.no_redirect(cache) if cache is not None else None
    temporary = c.no_redirect(temporary) if temporary is not None else None
    ancestors = {p for d in destinations for p in d.parents if p != Path(d.anchor)}
    allowed = {c.no_redirect(a.SPLIT.parent / r['image']) for r in requests()[0]} if raw_e else set()
    blocked = c.BLOCKED - ({'numpy', 'pickle'} if cache is not None else set())
    c.require(not cpu or not any(n.split('.')[0] in blocked or a.model_module(n) for n in sys.modules), "Forbidden CPU model/data import")
    def audit(event, args):
        if event == 'import' and (args[0] == 'controlled_generation.render' or cpu and (args[0].split('.')[0] in blocked or a.model_module(args[0]))):
            raise RuntimeError('CPU/model/renderer import outside operation')
        if event == 'socket.connect' or event.startswith('pickle.'):
            raise RuntimeError('Network/pickle execution forbidden')
        if event in {'os.link', 'os.symlink'}: raise RuntimeError('Links forbidden')
        if event in {'os.remove', 'os.rename', 'os.rmdir'}:
            affected = args[:2] if event == 'os.rename' else args[:1]
            c.require(not cpu and temporary is not None and all(Path(x).absolute().is_relative_to(temporary) and Path(x).absolute() != temporary for x in affected), 'Delete/move outside own temporary scope')
        if event == 'os.mkdir':
            p = Path(args[0]).absolute()
            c.require(p in ancestors or any(p == d or p.is_relative_to(d) for d in destinations), 'Directory outside new transaction')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(), '\\\\.\\nul'}: return
            p = Path(name).absolute(); mode, flags = args[1] or '', args[2] or 0
            write = any(x in str(mode) for x in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if write: c.require(any(p.is_relative_to(d) for d in destinations), 'Write outside new independent output')
            elif p.is_relative_to(c.ROOT / 'data'): c.require(p == a.SPLIT or p in allowed, 'Q/E/U RGB/GT access forbidden except explicit E RGB runtime')
            elif p.suffix.lower() == '.npy':
                c.require(probability_read_allowed(p, cache, cpu, p in t._HASH_SCOPE), 'Q/E probability bytes forbidden outside own cache or explicit predecessor hash')
            elif p.suffix.lower() in c.OPAQUE_SUFFIXES:
                hashed = p in t._HASH_SCOPE or p in c._READ_SCOPES
                model = not cpu and (p == c.GENERIC or any(p == t.paths(g, s)[1] / 'final_weights.safetensors' for g, s in t.ORDER))
                c.require(hashed or model, 'Unbound tensor/array/image payload')
    sys.dont_write_bytecode = True; sys.addaudithook(audit); _GUARDED = True


def upstream():
    # The existing preparation hashes training-side caches, never Q probabilities.
    a.verify_preparation()
    sources = dict(t.load(a.PREP / 'sources.json'))
    c.require(t.sha(CAL / 'complete.json') == CAL_SHA, 'Six-calibration anchor differs')
    total = t.load(CAL / 'verification.json')
    cal_identity = {'calibration_preparation_complete_sha256': t.sha(a.PREP / 'complete.json'), 'training_all_six_complete_sha256': a.ASSEMBLY_SHA,
                    'model_order': total['model_order'], 'implementation': a.implementation(), 'plan_contract': c.plan_contract()}
    t.verify_seal(CAL, {'sources.json', 'verification.json', '第5阶段六模型正常校准最终核验.md'}, cal_identity)
    c.require(total['calibration_assembly_verified'] and total['models'] == len(total['entries']) == 6 and total['unique_normal_images'] == 200, 'Six-calibration reference incomplete')
    # Do NOT re-open Q probability bytes. Exact namespaces + anchored full per-model
    # verification evidence bind them; all remaining source files are stream-hashed.
    skipped = 0
    for path, digest in t.load(CAL / 'sources.json').items():
        if Path(path).suffix.lower() == '.npy': skipped += 1; continue
        c.require(t.sha(path) == digest, 'Calibration metadata/weight source differs: ' + path)
        sources[path] = digest
    c.require(skipped == 1200, 'Six Q probability reference count differs')
    models = t.load(a.PREP / 'models.json')
    c.require([(m['group'], m['seed']) for m in models] == t.ORDER and total['model_order'] == [f'{g}_{s}' for g, s in t.ORDER], 'Model order differs')
    qreq = t.load(a.PREP / 'requests.json')
    for model, row in zip(models, total['entries']):
        group, seed = model['group'], model['seed']; root, runtime = a.paths(group, seed); bound = a.binding(t.load(a.PREP / 'protocol.json'), model)
        c.require(row['group'] == group and row['seed'] == seed and row['binding'] == bound, 'Calibration model binding differs')
        complete = t.load(root / 'complete.json')
        c.require(t.sha(root / 'complete.json') == row['calibration_complete_sha256'] and complete['identity'] == bound and set(complete['files']) == a.run_files(qreq), 'Calibration closure identity differs')
        c.verify_closed_tree(root, a.run_files(qreq) | {'complete.json'})
        a.verify_runtime(root, runtime, bound)
        threshold = t.load(root / 'threshold.json')
        c.require(t.sha(root / 'threshold.json') == row['threshold_sha256'] and t.sha(root / 'scores.json') == row['scores_sha256'], 'Original threshold/score source hash differs')
        c.require(threshold['binding'] == bound and threshold['threshold'] == row['threshold'] and threshold['rule'] == a.THRESHOLD_RULE and threshold['score_source_sha256'] == row['scores_sha256'], 'Original tau provenance differs')
        c.require(type(threshold['threshold']) is float and math.isfinite(threshold['threshold']), 'Invalid original tau')
        proof = Path(row['verification_root']); pv = t.load(proof / 'verification.json'); pc = t.load(proof / 'complete.json')
        t.verify_seal(proof, set(pc['files']))
        c.require(t.sha(proof / 'complete.json') == row['verification_complete_sha256'] and pv['calibration_verified'] and pv['identity']['binding'] == bound and pv['saved_threshold'] == threshold['threshold'] and pv['all_200_probability_shape_dtype_finite_range_and_native_max_verified_by_CLI'], 'Saved full calibration validation missing')
        model['calibration'] = {'root': str(root), 'complete_sha256': row['calibration_complete_sha256'], 'threshold_path': str(root / 'threshold.json'), 'threshold_sha256': row['threshold_sha256'], 'threshold': threshold['threshold'], 'verification_complete_sha256': row['verification_complete_sha256']}
    for folder in (a.PREP, CAL):
        for file in folder.iterdir():
            if file.is_file(): sources[str(file)] = t.sha(file)
    for name, digest in a.implementation().items():
        sources[str(c.PACKAGE / name)] = digest
    sources[str(a.INVENTORY)] = t.sha(a.INVENTORY)
    return sources, models


def protocol(models):
    return {'schema': 'paired-pretrained-E1004-scoring-v1', 'implementation': implementation(), 'plan_contract': c.plan_contract(),
            'calibration_all_six_complete_sha256': CAL_SHA, 'training_all_six_complete_sha256': a.ASSEMBLY_SHA,
            'split_sha256': a.SPLIT_SHA, 'inventory_sha256': a.INVENTORY_SHA, 'model_order': [f"{m['group']}_{m['seed']}" for m in models],
            'model_compatibility': t.load(a.PREP / 'protocol.json')['model_compatibility'], 'score_rule': a.SCORE_RULE, 'threshold_rule': a.THRESHOLD_RULE,
            'model_constructor': 'same generic constructor then strict own final10000 load; no optimizer or constructor forward',
            'images_per_model': 1004, 'windows_per_model': 4270, 'native_pixels_per_model_metadata': 146411528, 'unique_E_images': 1004,
            'planned_model_image_scores': 6024, 'planned_total_windows': 25620,
            'frozen_native_predictor': 'pretrained_synthetic_pairing.calibration_runtime.predict_native',
            'probability_source_scope': 'Q caches: closed metadata and anchored full prior validation only; Q/E probability payloads not opened now',
            'resources': t.load(a.PREP / 'protocol.json')['resources'], 'no_threshold_recalculation': True,
            'transaction': 'new process/model; predecessor complete incl runtime; per-window durable intention; unknown/partial preserve/stop; no repeat',
            'meaning': 'E historically seen; exploratory only; scoring without GT/metrics; not new unseen validation or screen benefit'}


def verify_preparation():
    t.resources(); t.verify_seal(PREP, FILES, {'implementation': implementation(), 'calibration_all_six_complete_sha256': CAL_SHA, 'plan_contract': c.plan_contract()})
    p = t.load(PREP / 'protocol.json'); models = t.load(PREP / 'models.json'); req, gt = requests()
    c.require(p == protocol(models) and req == t.load(PREP / 'requests.json') and gt == t.load(PREP / 'metric_sources.json') and t.load(PREP / 'scalar_verification.json') == scalar_verification(), 'Frozen E preparation differs')
    c.require([(m['group'], m['seed']) for m in models] == t.ORDER and t.sha(CAL / 'complete.json') == CAL_SHA, 'E model/calibration reference differs')
    for path, digest in t.load(PREP / 'sources.json').items(): c.require(t.sha(path) == digest, 'E upstream source changed: ' + path)
    return p, req, models


def prepare():
    c.require(Path(sys.executable).resolve() == (c.CPU / 'Scripts/python.exe').resolve(), 'Use CPU .venv')
    guard(cpu=True, destinations=(PREP,)); snapshot = t.resources()
    if PREP.exists(): verify_preparation(); return {'E_preparation_ready': True, 'reused': True, 'E_scoring_executed': False}
    sources, models = upstream(); req, gt = requests()
    from .resources import environment_metadata
    env = environment_metadata(); c.require(env == t.load(a.PREP / 'environment.json'), 'Frozen environment differs')
    for g, s in t.ORDER:
        for path in paths(g, s): c.require(not path.exists(), 'Unknown future E result/runtime')
    for name in CODE:
        if name.endswith('.py'): ast.parse((c.PACKAGE / name).read_text(encoding='utf-8'), filename=name)
    p = protocol(models)
    PREP.mkdir(parents=True, exist_ok=False)
    values = {'transaction.json': {'scope': 'metadata/hash/scalars only; zero raw RGB/GT/probability/model operations', 'resources': snapshot},
              'protocol.json': p, 'requests.json': req, 'metric_sources.json': gt, 'models.json': models, 'sources.json': sources,
              'environment.json': env, 'scalar_verification.json': scalar_verification(), REPORT: report(snapshot)}
    for name, value in values.items(): t.save(PREP / name, value)
    t.seal(PREP, FILES, {'implementation': implementation(), 'calibration_all_six_complete_sha256': CAL_SHA, 'plan_contract': c.plan_contract()})
    verify_preparation()
    return {'E_preparation_ready': True, 'models': 6, 'E_metadata': 1004, 'planned_windows_per_model': 4270, 'native_pixels_per_model_metadata': 146411528, 'RGB_GT_probability_payload_reads': 0, 'model_operations': 0, 'E_scoring_executed': False}


def binding(p, model):
    return {'preparation_complete_sha256': t.sha(PREP / 'complete.json'), 'protocol_sha256': t.sha(PREP / 'protocol.json'),
            'requests_sha256': t.sha(PREP / 'requests.json'), 'metric_sources_sha256': t.sha(PREP / 'metric_sources.json'),
            'models_sha256': t.sha(PREP / 'models.json'), 'implementation': p['implementation'], 'calibration_all_six_complete_sha256': CAL_SHA, 'model': model}


def run_files(req):
    return {'transaction.json', 'window_journal.jsonl', 'runtime_manifest.json', 'scores.json', 'E1004评分结果.md'} | {probability_path(r) for r in req}


def probability_path(request):
    return f"probabilities/{request['index']:04d}_{request['id']}.npy"


def verify_saved_metadata(group, seed, p, req, models):
    root, runtime = paths(group, seed); model = a.find_model(models, group, seed); bound = binding(p, model)
    t.verify_seal(root, run_files(req), bound); a.verify_runtime(root, runtime, bound)
    scores = t.load(root / 'scores.json'); transaction = t.load(root / 'transaction.json')
    c.require(scores['binding'] == transaction['binding'] == bound and transaction['intended_images'] == 1004 and transaction['intended_windows'] == 4270, 'E transaction differs')
    c.require(scores['images_scored'] == scores['raw_E_image_decodes'] == len(scores['items']) == 1004 and scores['window_forwards'] == 4270 and scores['native_pixels'] == 146411528 and scores['constructor_forwards'] == 0, 'E actual counts differ')
    c.require(scores['state_before'] == scores['state_after'] == model['expected_state_hash'] and scores['buffers_before'] == scores['buffers_after'] == model['expected_buffers_hash'], 'E state/BN differs')
    c.require(scores['GT_read_by_predictor'] is False and scores['metrics_computed'] is False and scores['training_or_parameter_updates'] == 0 and scores['score_rule'] == a.SCORE_RULE and scores['original_threshold'] == model['calibration']['threshold'], 'E scoring-only rule differs')
    expected = []
    for r in req:
        expected.append({'event': 'decode_start', 'index': r['index'], 'id': r['id']})
        expected.extend({'event': 'window_start', 'index': r['index'], 'id': r['id'], 'window_index': i, 'y': y} for i, y in enumerate(r['ys']))
    actual = [json.loads(line) for line in (root / 'window_journal.jsonl').read_text(encoding='utf-8').splitlines()]
    c.require(actual == expected, 'E decode/window intentions differ')
    for row, r in zip(scores['items'], req):
        c.require(row['index'] == r['index'] and row['id'] == r['id'] and row['source_image_sha256'] == r['image_sha256'] and row['probability'] == probability_path(r) and row['window_forwards'] == r['window_count'], 'E row order/source differs')
    return scores


def predecessor_ready(group, seed, p, req, models):
    for g, s in t.ORDER[:t.ORDER.index((group, seed))]: verify_saved_metadata(g, s, p, req, models)


def report(snapshot):
    fence = chr(96)*3
    return f'''# 第5阶段独立E1004真实评分准备与运行

本次仅CPU来源／环境元信息、文件hash、整数窗口与虚构标量准备。六模型最终10000步／60保存点和全部正常校准引用／逐组完整证据／原tau来源闭合。Q概率只核固定命名空间、完成身份与已保存全量核验证据，不重新打开1200份概率字节；不读取Q/E/U原图或GT、E概率，不导入Torch／SMP／timm／模型或创建tensor。不开发指标，不做前向／训练／生成／校准／实际AP／AUROC／阈值计数，不改冻结源码／全局路径／权重／阈值／缓存。按用户要求不建过程表、软件测试、提交或工作树。

唯一独立静态审阅发现1项Important：初稿将未来runtime的前组概率hash误当跨组载荷读取拦截，后组会在准入时停止；根因为当前缓存读取与前组只hash校验共用过窄白名单。封存前同轮修正为仅future非CPU模式、固定原序前组目录且处于显式hash scope时放行；当前CPU准备／Q概率／非hash跨组读取仍禁止。虚构路径四态[拒绝、允许、拒绝、拒绝]及AST校验通过，没有打开概率或运行模型，无延期Minor。实际CUDA／第三方导入、退出runtime清单稳定性、耗时和资源仍待人工运行，不以静态审阅替代实测。

严格固定A01/B01/A02/B02/A03/B03顺序，各自最终权重及原Q tau。E1004原序与规范GT路径／已有hash分开保存到requests.json与metric_sources.json；GT未打开，实际异常／正常／阳性像素分母本阶段未知。每模型预定1004图／4270窗／146,411,528原生像素；6024模型图像评分／25620窗仍同一1004唯一图，均非本次实际推理数。

复用封存predict_native：ImageNet归一化、原生256窗stride128／尾窗去重，右reflect补边剔除，y升序FP32等权原始异常softmax拼回max，无平滑／拉伸／池归一化。模型同一单头／encoder固定BN／decoder GN8／FP32／batch1／workers0，加载自身最终权重；无optimizer、构造前向或GT输入，核参数／BN前后不变。E只保存概率与分数，不用GT、不算阈值TP/FN/FP/TN或AP，不重算Q或改原tau。

单模型新进程、每图／窗一次。逐窗前fsync意向，独立runtime owner／全文件hash／目录清单，当前缓存及前组准入共用严格元信息核验；完整同身份缓存只读合法性核验，不新增前向。未知／部分输出、锁、空目录或不明确前向均保留停止，不补推、不换分母成功交集、不自动清理。第三方退出若改变临时文件也按严格规则停止查因，当前E运行仍未实测。

CPU准备RAM>=2GiB、D/E各>=2GiB；实际模型启动RAM>=4.5GiB／空闲显存>=6GiB／E>=20GiB，逐图RAM>=2GiB／E>=1GiB。快照RAM{snapshot['available_RAM_GiB']:.3f}GiB、E{snapshot['E_free_GiB']:.3f}GiB、空闲显存{snapshot['GPU']['free_MiB']}MiB仅元信息。先导E评分约172.44秒只是历史参考，逐窗日志／新包装的实际时间资源与CUDA交互未实测，不保证耗时；不自动资源前向试跑或降低门槛。

独立准备{PREP.as_posix()}；未来结果results/real_evaluation_v1/{{group}}_{{seed}}，runtime_real_evaluation_v1/{{group}}_{{seed}}。

只读准备核验：

{fence}powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring verify
{fence}

首组A01人工评分（已实现，本次未执行）：

{fence}powershell
& '.\\.venv-pretrained-seg\\Scripts\\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring run --group A --seed 2026102301
{fence}

完整结果存在后仅CPU real_scoring verify-results --group A --seed 2026102301读取该组保存概率核验。real_cache.iter_metric_inputs只提供未来授权CPU指标所需的概率／分数和独立GT元信息，不解码GT或计算指标，本次未调用。E历史已见，未来只作探索性评价，不称新未见验证、稳定收益或屏幕效果。所有训练仍由用户手动启动。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'verify', 'run', 'verify-results'))
    parser.add_argument('--group', choices=('A', 'B')); parser.add_argument('--seed', type=int, choices=c.SEEDS); args = parser.parse_args()
    if args.command in {'prepare', 'verify'}:
        c.require(args.group is None and args.seed is None, 'No model arguments in metadata operation')
        c.require(Path(sys.executable).resolve() == (c.CPU / 'Scripts/python.exe').resolve(), 'Use CPU .venv')
        if args.command == 'prepare': result = prepare()
        else:
            guard(cpu=True); verify_preparation(); result = {'E_preparation_verified': True, 'RGB_GT_probability_payload_reads': 0, 'model_operations': 0, 'E_scoring_executed': False}
    else:
        c.require(args.group is not None and args.seed is not None, 'Explicit single model required')
        if args.command == 'run':
            from .real_runtime import run
            result = run(args.group, args.seed)
        else:
            c.require(Path(sys.executable).resolve() == (c.CPU / 'Scripts/python.exe').resolve(), 'Use CPU .venv')
            root, _ = paths(args.group, args.seed); guard(cpu=True, cache=root); p, req, models = verify_preparation()
            from .real_cache import verify_results
            result = verify_results(args.group, args.seed, p, req, models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.modules.setdefault('pretrained_synthetic_pairing.real_scoring', sys.modules[__name__]); main()
