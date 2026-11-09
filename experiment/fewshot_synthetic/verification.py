"""Strict final evidence/namespace checks; frozen CPU preparation remains unchanged."""
from . import data as d

FINAL_NAMES = {'visual_review.json', '第1天输入准备与适用性核验.md'}


def exact_directories(actual, expected):
    d.require(actual == expected, 'Unknown/missing directory or directory lock: preserve and investigate')


def validate_review(review, scope):
    d.require(isinstance(review, dict) and review.get('decision') in ('pass', 'fail', 'uncertain'), 'Invalid visual decision')
    d.require(isinstance(review.get('summary'), str) and bool(review['summary'].strip()), 'Nonempty summary required before saving')
    for role in ('R5', 'N16'):
        d.require(isinstance(review.get(role), dict) and set(review[role]) == set(scope[role]), 'Incomplete predetermined visual scope')
        d.require(all(isinstance(v, str) and bool(v.strip()) for v in review[role].values()), 'Missing per-image visual observation')


def validate_completion(completion, review, scope):
    validate_review(review, scope)
    d.require(isinstance(completion.get('files'), dict) and set(completion['files']) == FINAL_NAMES,
              'Final evidence must contain exactly visual review and report')
    d.require(type(completion.get('input_ready')) is bool and completion.get('decision') == review['decision'], 'Final decision type/value mismatch')
    d.require(completion['input_ready'] == (review['decision'] == 'pass'), 'Final readiness differs from visual decision')
    d.require(completion.get('verification_sha256') == d.sha(__file__), 'Strict verification implementation differs or missing')


def verify_data():
    state = d.verify_data()
    exact_directories({p.relative_to(d.DATA).as_posix() for p in d.DATA.rglob('*') if p.is_dir()}, {'train', 'test'})
    exact_directories({p.relative_to(d.OUT).as_posix() for p in d.OUT.rglob('*') if p.is_dir()}, {'previews'})
    scope = d.read_json(d.OUT / 'input_protocol.json')['visual_scope']
    cpath = d.OUT / 'completion.json'
    if cpath.exists():
        completion = d.read_json(cpath)
        review = d.read_json(d.OUT / 'visual_review.json')
        validate_completion(completion, review, scope)
    else:
        d.require(not any((d.OUT / n).exists() for n in FINAL_NAMES), 'Partial visual transaction; preserve and investigate')
    state['strict_visual_and_directory_gate_verified'] = True
    return state


def record_visual_review(review):
    verify_data()
    scope = d.read_json(d.OUT / 'input_protocol.json')['visual_scope']
    validate_review(review, scope)
    d.require(not any((d.OUT / n).exists() for n in FINAL_NAMES | {'completion.json'}), 'Existing final review: no overwrite')
    # Preparation's finalizer is deliberately not used: its implementation is frozen with the prepared inputs.
    d.save(d.OUT / 'visual_review.json', review)
    ready = review['decision'] == 'pass'
    report = '# 第1天输入准备与适用性核验\n\n'
    report += '状态：' + ('input_ready，可进入第2天准备。' if ready else '未通过输入关口，保存并停止。') + '\n\n'
    report += '规范3335对已导出并逐文件核对原始hash；全部3335规范RGB仅用于带尺寸身份核验，精确重复0组。原zip与copy不动，作者pickle未执行。\n\n'
    report += '固定R5：' + '、'.join(scope['R5']) + '；Q200、N1885、U241、E1004已固定。test RGB身份解码不显示、不统计外观，test GT未解码；110／894仍为官方资料。物理工件／采集组独立性未知。\n\n'
    report += review['summary'] + '\n\n'
    for role in ('R5', 'N16'):
        report += '## ' + role + '全部预定样例审阅\n\n'
        for sample_id, observation in review[role].items():
            report += '- ' + sample_id + '：' + observation + '\n'
        report += '\n'
    report += '总览使用原始RGB，无对比度拉伸；GT及红色覆盖辅助对照。窗口y起点及非空像素统计见review_geometry.json，每张R至少有非空256窗口；原图宽227—232，右侧反射补齐及有效域排除是后续训练协议，本日未合成或执行训练补边。\n\n'
    report += 'prepared input_protocol中input_ready=false保留准备时待审阅含义，最终权威为completion。完整缓存verify-data及prepare-data经CLI只核验hash，无新导出／RGB解码／GT解码。实际首次3335张身份解码，另5＋16训练RGB及GT用于总览；未看Q／E／U外观，无Torch、模型、合成、参数更新或训练。\n\n'
    report += '独立审阅修正：新增verification.py严格绑定视觉文件集合／全部R5和N16观察、decision与布尔input_ready，并核对DATA仅train/test、OUT仅previews目录。原data.py及其准备身份不变，CLI改走严格入口。没有软件测试套件、过程表或提交流程。\n\n'
    report += '可运行命令见fewshot_synthetic/README.md；资源快照与环境见input_protocol.json。只有第1天输入适用性通过，不代表合成真实、模型可学或检测提升；第2天尚未执行，全部训练由用户手动启动。\n'
    (d.OUT / '第1天输入准备与适用性核验.md').write_text(report, encoding='utf-8')
    completion = {'input_ready': ready, 'decision': review['decision'],
                  'preparation_sha256': d.sha(d.OUT / 'preparation_complete.json'),
                  'verification_sha256': d.sha(__file__),
                  'files': {n: d.sha(d.OUT / n) for n in sorted(FINAL_NAMES)}}
    d.save(d.OUT / 'completion.json', completion)
    return verify_data()
