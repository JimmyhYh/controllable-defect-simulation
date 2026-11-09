"""Independent paired training preparation. Preparation is strictly stdlib-only."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from . import common as c

INPUT = c.RESULTS / "inputs_v1"
PREP = c.RESULTS / "training_preparation_v1"
FORWARD = c.RESULTS / "forward_compatibility_v1"
INPUT_SHA = "f0da7192dac3f5c6a2db0ffd1d037cc844377a5e0945debe3684634623deb417"
PROOF = c.BOOT / "inputs_verification_v1"
PROOF_SHA = "74154d433acc3f9c3d4e022448e8edab5834c61cb5208ae54c526e667e7d38aa"
NATIVE = c.BASE / "formal_preparation_v1"
CODE = ("training.py", "training_data.py", "training_runtime.py", "README_training.md")
PREP_FILES = {"transaction.json", "protocol.json", "sources.json", "requests_identity.json", "schedule_verification.json", "result.json", "第3阶段训练包装准备与运行边界.md"}
CHECK_FILES = {"model.safetensors", "optimizer.pt", "rng.json", "history.json", "checkpoint.json"}
ORDER = [(g,s) for s in c.SEEDS for g in "AB"]
_GUARDED = False
_HASH_SCOPE = set()


def sha(path):
    path = c.no_redirect(path)
    _HASH_SCOPE.add(path)
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4*1024*1024), b""): h.update(block)
        return h.hexdigest()
    finally:
        _HASH_SCOPE.discard(path)


def load(path):
    return c.load(path)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = c.no_redirect(path)
    raw = value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with path.open("xb") as f:
        f.write(raw.encode("utf-8")); f.flush(); os.fsync(f.fileno())


def seal(folder, names, identity):
    c.verify_closed_tree(folder,set(names))
    save(Path(folder)/"complete.json",{"identity":identity,"files":{n:sha(Path(folder)/n) for n in sorted(names)}})


def verify_seal(folder, names, identity=None):
    folder = c.no_redirect(folder)
    complete = load(folder/"complete.json")
    c.require(set(complete["files"]) == set(names), "Incomplete/unknown exact completion file list")
    c.verify_closed_tree(folder,set(names)|{"complete.json"})
    for name, expected in complete["files"].items():
        c.require(sha(folder/str(c.safe_relative(name))) == expected, "Sealed hash differs: "+str(folder/name))
    if identity is not None: c.require(complete["identity"] == identity, "Scientific binding differs")
    return complete


def implementation():
    return {n:sha(c.PACKAGE/n) for n in CODE}


def lr_at(update):
    c.require(type(update) is int and 1 <= update <= 10000,"Invalid update for scalar LR")
    return [1e-5,1e-4] if update <= 8000 else [2e-6,2e-5] if update <= 9000 else [4e-7,4e-6]


def next_lr(done):
    c.require(type(done) is int and 0 <= done <= 10000,"Invalid completed update cursor")
    return lr_at(done+1) if done < 10000 else None


def schedule_verification():
    updates = [1,8000,8001,9000,9001,10000]
    expected = [[1e-5,1e-4],[1e-5,1e-4],[2e-6,2e-5],[2e-6,2e-5],[4e-7,4e-6],[4e-7,4e-6]]
    c.require([lr_at(s) for s in updates] == expected,"Update LR boundaries differ")
    completed = [0,999,1000,7999,8000,8999,9000,9999,10000]
    expected_next = [[1e-5,1e-4]]*4 + [[2e-6,2e-5]]*2 + [[4e-7,4e-6]]*2 + [None]
    c.require([next_lr(s) for s in completed] == expected_next,"Resume LR boundaries differ")
    full = [lr_at(s) for s in range(1,10001)]
    c.require(Counter(tuple(x) for x in full) == Counter({(1e-5,1e-4):8000,(2e-6,2e-5):1000,(4e-7,4e-6):1000}),"Full scalar LR schedule differs")
    # Hypothetical cursor/RNG-free exposure prefixes; no optimizer or tensor is created.
    return {"update_boundaries":dict(zip(map(str,updates),expected)),"resume_completed_boundaries":dict(zip(map(str,completed),expected_next)),
            "all_10000_scalar_rates_sha256":stable_hash(full),"optimizer_instantiated":False,"GPU_resume_measured":False}


def paths(group, seed):
    c.require(group in "AB" and len(group)==1 and type(seed) is int and seed in c.SEEDS,"Invalid paired model identity")
    key=f"{group}_{seed}"
    return c.RESULTS/"training_v1"/key, c.MODELS/key, c.LARGE/"runtime_v1"/key


def guard(cpu=True, destinations=(), temporary=None):
    global _GUARDED
    c.require(not _GUARDED,"One explicit operation per process; start a new process")
    blocked = c.BLOCKED | {"diffusers","transformers"}
    if cpu: c.require(not any(n.split(".")[0] in blocked for n in sys.modules),"CPU preparation already contains data/model package")
    sys.dont_write_bytecode=True
    destinations=tuple(c.no_redirect(p) for p in destinations)
    temporary=c.no_redirect(temporary) if temporary is not None else None
    ancestors={p for d in destinations for p in d.parents if p != Path(d.anchor)}
    native_paths=set(load(INPUT/"source_refs.json"))
    def audit(event,args):
        if event=="import" and (args[0]=="controlled_generation.render" or args[0].startswith("pretrained_segmentation.model") and cpu or cpu and args[0].split(".")[0] in blocked):
            raise RuntimeError("Unauthorized data/model/renderer import")
        if event=="socket.connect" or cpu and event.startswith("pickle."): raise RuntimeError("Network/pickle execution forbidden")
        if event in {"os.link","os.symlink"}: raise RuntimeError("Links forbidden")
        if event in {"os.remove","os.rename","os.rmdir"}:
            affected=args[:2] if event=='os.rename' else args[:1]
            c.require(not cpu and temporary is not None and all(Path(x).absolute().is_relative_to(temporary)
                      and Path(x).absolute() not in {temporary,temporary/'owner.json'} for x in affected),"Delete/move forbidden outside own disposable runtime cache")
        if event=="os.mkdir":
            path=Path(args[0]).absolute()
            c.require(path in ancestors or any(path==d or path.is_relative_to(d) for d in destinations),"Directory outside this independent transaction")
        if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            if name.lower() in {os.devnull.lower(),"\\\\.\\nul"}: return
            path=Path(name).absolute(); mode,flags=args[1] or "",args[2] or 0
            writing=any(ch in str(mode) for ch in "wax+") or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing:
                c.require(any(path.is_relative_to(d) for d in destinations),"Write outside exact new operation scope")
            elif path.suffix.lower() in c.OPAQUE_SUFFIXES:
                allowed=path in _HASH_SCOPE or path in c._READ_SCOPES
                if not cpu:
                    allowed = allowed or path==c.GENERIC or str(path) in native_paths and path.suffix==".npz" or any(path.is_relative_to(d) for d in destinations)
                c.require(allowed,"Raw RGB/GT/probability/unbound tensor payload forbidden")
            elif path.is_relative_to(c.ROOT/"data"):
                c.require(path==c.ROOT/"data/fewshot_synthetic/ksdd2_v1/split.json","Q/E/U/raw data access forbidden")
    sys.addaudithook(audit); _GUARDED=True


def admitted(s,required,model=False,full_round=False):
    return (s['available_RAM_GiB'] >= (4.5 if model else 2) and s['D_free_GiB'] >= 2
            and s['E_free_GiB'] >= (required if full_round else 20 if model else 2)
            and (not model or s['GPU']['free_MiB'] >= 6144))


def resource_scalar_verification():
    required=load(c.OUT/'resource_assessment.json')['disk_forecast']['required_free_before_round_bytes']/2**30
    s={'available_RAM_GiB':7,'D_free_GiB':100,'E_free_GiB':120,'GPU':{'free_MiB':7000}}
    values={'fake_E120_readonly':admitted(s,required),'fake_E120_later_train_or_restore':admitted(s,required,model=True),
            'fake_E120_initial_full_round':admitted(s,required,full_round=True)}
    c.require(values=={'fake_E120_readonly':True,'fake_E120_later_train_or_restore':True,'fake_E120_initial_full_round':False},'Hypothetical resource policy differs')
    return {'hypothetical_only':True,'conditions':s,'initial_full_round_required_GiB':required,'admission':values,'no_resource_training_probe':True}


def resources(model=False, full_round=False):
    s=c.resource_snapshot()
    required=load(c.OUT/"resource_assessment.json")["disk_forecast"]["required_free_before_round_bytes"]/2**30
    c.require(admitted(s,required,model,full_round),"RAM/VRAM/disk admission not met")
    return s


def verify_input_sources():
    """Independent stdlib identity verification, without frozen inputs' initial-space admission."""
    c.require(Path(sys.executable).resolve()==(c.CPU/'Scripts/python.exe').resolve(),"Use CPU verifier interpreter")
    guard(cpu=True); resources()
    # Reuse only frozen identity/metadata helpers in this clean CPU process, never its run/verify/resource entry.
    from . import inputs as i
    i.bind_stage1()
    items,selected,refs=i.source_metadata()
    c.require(sha(INPUT/'complete.json')==INPUT_SHA,"Frozen inputs completion anchor differs")
    verify_seal(INPUT,i.OUTPUT_FILES,{'implementation':i.code_identity(),'stage1_complete_sha256':i.STAGE1_SHA,'plan_contract':c.plan_contract()})
    c.require(load(INPUT/'protocol.json')==i.preparation_protocol() and load(INPUT/'native_inputs.json')==items
              and load(INPUT/'source_refs.json')==refs,"Frozen input protocol/source references differ")
    for path,expected in refs.items(): c.require(sha(path)==expected,"CPU input source hash differs")
    _,_,requests=input_metadata(); summary=load(INPUT/'summary.json'); synthetic=load(INPUT/'synthetic_verification.json')
    c.require(summary['CPU_inputs_ready'] and summary['exposures']['A']==requests['A'] and summary['exposures']['B']==requests['B']
              and summary['exposures']['shared_slots']==7500 and len(selected)==synthetic['effective']==len(synthetic['items'])==2500
              and synthetic['types']==dict.fromkeys(c.TYPES,625),"Frozen CPU input completeness differs")
    return {'CPU_inputs_ready':True,'verification_decoded_cache_payloads':False,'cache_reuse':True,
            'input_complete_sha256':INPUT_SHA,'stage1_sources':2655,'stage1_closed_roots':7,'training_source_caches':1890,
            'synthetic_sources':2500,'requests':10000,'compatibility':25,'model_or_array_imports':False,
            'identity_verifier':'independent stdlib wrapper; no inputs.verify/resources or global mutation','resource_policy':'CPU2GiB/D2GiB/E2GiB; no initial full-round budget on read-only reuse'}


def upstream():
    """A clean CPU child owns the no-model guard; never import inputs in the model process."""
    script="from pretrained_synthetic_pairing import training as t; import json; print(json.dumps(t.verify_input_sources(),ensure_ascii=False))"
    command=[str(c.CPU/"Scripts/python.exe"),"-B","-X","utf8","-c",script]
    started=time.monotonic()
    run=subprocess.run(command,cwd=c.ROOT,capture_output=True,text=True,encoding="utf-8",timeout=300,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    c.require(run.returncode==0,"Frozen input verification failed: "+run.stderr[-1600:])
    checked=json.loads(run.stdout)
    c.require(sha(INPUT/"complete.json")==INPUT_SHA and checked["CPU_inputs_ready"] and not checked["verification_decoded_cache_payloads"],"Unexpected input identity/verification mode")
    c.require(sha(PROOF/'complete.json')==PROOF_SHA,"Frozen independent input evidence identity differs")
    verify_seal(PROOF,{"verification.json","第2阶段独立只读核验结果.md"},{"stage2_complete_sha256":INPUT_SHA})
    proof=load(PROOF/"verification.json")
    c.require(proof["exit_code"]==0 and proof["frozen_output_hashes_before_after_identical"] and proof["protected_old_sources_unchanged"]==78,"Independent input evidence missing")
    sources={**load(c.OUT/"sources.json"),**load(INPUT/"source_refs.json"),**proof["protected_source_hashes"]}
    for folder in [c.OUT,INPUT,PROOF]:
        for file in folder.iterdir():
            c.require(file.is_file(),"Unknown directory in sealed input source")
            sources[str(file)]=sha(file)
    for name,expected in proof["protected_source_hashes"].items(): c.require(sha(name)==expected,"Old protected source differs")
    return sources,{"command":command,"exit_code":run.returncode,"elapsed_seconds":time.monotonic()-started,"source_verification":checked,"model_or_array_imports_in_parent":False}


def input_metadata():
    rows=load(INPUT/"requests.json"); compat=load(INPUT/"compatibility.json")
    c.require(len(rows)==10000 and [r["update"] for r in rows]==list(range(1,10001)),"Fixed request cursor differs")
    counts={g:Counter() for g in "AB"}; ids={g:Counter() for g in "AB"}; syn=[]; shared=0
    for r in rows:
        for g in "AB":
            ref=r.get("common_input",r.get(g+"_input")); counts[g][ref["kind"]]+=1
            if ref["kind"]=="real": ids[g][ref["id"]]+=1
            c.require(set(r["input_identity_"+g])=={"image","target","valid","pack"},"Array identity fields differ")
        if "common_input" in r:
            shared+=1; c.require(r["input_identity_A"]==r["input_identity_B"],"Shared CPU identities differ")
        else: syn.append(r["B_input"])
    c.require(counts["A"]==Counter(real=5000,normal=5000) and counts["B"]==Counter(real=2500,synthetic=2500,normal=5000),"Frozen exposure differs")
    c.require(shared==7500 and [x["ordinal"] for x in syn]==list(range(2500)) and Counter(x["type"] for x in syn)==Counter(dict.fromkeys(c.TYPES,625)),"Frozen material subset/order differs")
    c.require(ids["A"]==Counter(dict.fromkeys(c.R5,1000)) and ids["B"]==Counter(dict.fromkeys(c.R5,500)),"Per-R5 exposure differs")
    c.require(len(compat["inputs"])==25 and compat["valid_pixels"]==1460224 and compat["positive_pixels"]==26759,"Compatibility metadata differs")
    return rows,compat,{"requests_sha256":sha(INPUT/"requests.json"),"compatibility_sha256":sha(INPUT/"compatibility.json"),"input_complete_sha256":INPUT_SHA,
                        "A":dict(counts["A"]),"B":dict(counts["B"]),"shared_slots":shared,"each_R_by_group":{g:dict(ids[g]) for g in "AB"},"four_types":dict(Counter(x["type"] for x in syn)),
                        "compatibility_count":25,"compatibility_valid_pixels":1460224,"compatibility_positive_pixels":26759}


def protocol():
    old=load(NATIVE/"protocol.json")
    return {"schema":"pretrained-synthetic-paired-training-v1","implementation":implementation(),"plan_contract":c.plan_contract(),
            "inputs_complete_sha256":INPUT_SHA,"inputs_implementation":load(INPUT/"protocol.json")["implementation"],"data_seed":c.DATA_SEED,
            "seeds":list(c.SEEDS),"order":[[g,s] for g,s in ORDER],"steps":10000,"checkpoint_steps":list(range(1000,10001,1000)),
            "resource_stop_interval":100,"model_function":"pretrained_segmentation.model.build_model; no old globals changed",
            "model_compatibility":old["compatibility"],"model_and_loss":old["model_and_loss"],
            "initialization":"same generic ImageNet encoder plus fresh decoder/head per seed; no probe/formal weights",
            "FP32":True,"batch":1,"workers":0,"encoder_BN":"fixed buffers; affine trainable","decoder_GN":8,
            "optimizer":{"encoder_lr":1e-5,"decoder_head_lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0},
            "loss":"valid-only unweighted CE + Dice; smooth1; empty normal target retained","schedule":schedule_verification(),
            "data_adapter":"independent training_data in model interpreter; hash every float32 image/target/valid against CPU freeze before use",
            "CPU_guard_isolation":"frozen inputs identity/metadata helpers only in clean CPU child; independent verifier bypasses initial-space admission, not identities; no model-process inputs import or path monkeypatch",
            "resource_policy":{"initial_preparation_and_first_fresh_train":"full-round forecast","read_only_identity_verification":"RAM2/D2/E2GiB","other_train_or_explicit_restore":"RAM4.5/VRAM6/E20GiB","no_frozen_resource_entry_or_monkeypatch":True},
            "resource_scalar_verification":resource_scalar_verification(),
            "forward_gate":{"seed":c.SEEDS[0],"inputs":25,"each_once":True,"eval_inference_mode":True,"optimizer":False,"AP":False,"unchanged_parameters_and_BN":True},
            "restore":"explicit latest own complete checkpoint only; own bound optimizer weights_only=True; no implicit restart; Ctrl+C no extra save",
            "interpretation":"training composition changes R5, area and background exposure; E historically seen; exploratory only",
            "manual_training_only":True,"GPU_restore_measured":False,"runtime_adapter_numerically_measured":False}


def identity():
    return {"implementation":implementation(),"input_complete_sha256":INPUT_SHA,"plan_contract":c.plan_contract()}


def prepare():
    if PREP.exists(): return verify()
    started=time.monotonic()
    c.require(Path(sys.executable).resolve()==(c.CPU/"Scripts/python.exe").resolve(),"Use CPU .venv for preparation")
    guard(cpu=True,destinations=(PREP,)); snap=resources(full_round=True)
    sources,check=upstream(); rows,compat,requests=input_metadata()
    for n in CODE:
        if n.endswith('.py'): ast.parse((c.PACKAGE/n).read_text(encoding='utf-8'),filename=n)
    for group,seed in ORDER:
        run,model,runtime=paths(group,seed)
        c.require(not run.exists() and not model.exists() and not runtime.exists(),"Unknown pre-existing future run target")
    c.require(not FORWARD.exists() and not (c.LARGE/"runtime_v1/forward").exists(),"Unknown pre-existing compatibility target")
    PREP.mkdir(parents=True,exist_ok=False)
    try:
        save(PREP/"transaction.json",{"manual_scope":"stage3 preparation only","resources":snap})
        save(PREP/"protocol.json",protocol()); save(PREP/"sources.json",sources)
        save(PREP/"requests_identity.json",requests); save(PREP/"schedule_verification.json",schedule_verification())
        result={"preparation_ready":True,"inputs_and_source_hashes_verified":True,"source_entries":len(sources),"upstream_verification":check,
                "CPU_scalar_schedule_verified":True,"compatibility_input_metadata":25,"model_forward_calls":0,"training_updates":0,
                "NPZ_RGB_GT_probability_decodes":0,"model_or_array_packages_imported":False,"future_GPU_restore_measured":False,
                "runtime_adapter_numerically_measured":False,"resources":snap,"cache_reuse":False}
        result["elapsed_seconds"]=time.monotonic()-started
        save(PREP/"result.json",result); save(PREP/"第3阶段训练包装准备与运行边界.md",report(result))
        seal(PREP,PREP_FILES,identity())
        return result
    except Exception as error:
        if not (PREP/"complete.json").exists() and not (PREP/"failure.json").exists(): save(PREP/"failure.json",{"error":repr(error),"rule":"preserve; no auto retry or altered inputs"})
        raise


def verify():
    if not _GUARDED: guard(cpu=True)
    resources()
    verify_seal(PREP,PREP_FILES,identity())
    sources,check=upstream()
    c.require(load(PREP/"sources.json")==sources and load(PREP/"protocol.json")==protocol(),"Frozen training/source protocol differs")
    _,_,requests=input_metadata()
    c.require(load(PREP/"requests_identity.json")==requests and load(PREP/"schedule_verification.json")==schedule_verification(),"Frozen scalar/input metadata differs")
    result=load(PREP/"result.json")
    c.require(result["preparation_ready"] and result["training_updates"]==result["model_forward_calls"]==result["NPZ_RGB_GT_probability_decodes"]==0,"Preparation scope differs")
    result["cache_reuse"]=True; result["current_readonly_upstream_verification"]=check
    return result


def runtime_preflight():
    """Fresh stdlib CPU verifier; never install CPU model-blocking hooks in model process."""
    command=[str(c.CPU/"Scripts/python.exe"),"-B","-X","utf8","-m","pretrained_synthetic_pairing.training","verify"]
    run=subprocess.run(command,cwd=c.ROOT,capture_output=True,text=True,encoding="utf-8",timeout=360,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    c.require(run.returncode==0,"Independent preparation verification failed: "+run.stderr[-1600:])
    from .resources import environment_metadata
    c.require(Path(sys.executable).resolve()==(c.ENV/"Scripts/python.exe").resolve() and environment_metadata()==load(c.OUT/"environment_metadata.json"),"Model interpreter/dependency identity differs")
    return load(PREP/"protocol.json")


def binding(group,seed):
    run,model,runtime=paths(group,seed)
    return {"preparation_complete_sha256":sha(PREP/"complete.json"),"protocol_sha256":sha(PREP/"protocol.json"),"input_complete_sha256":INPUT_SHA,
            "forward_complete_sha256":sha(FORWARD/"complete.json"),
            "requests_sha256":sha(INPUT/"requests.json"),"group":group,"seed":seed,"steps":10000,"implementation":implementation(),"run":str(run),"model":str(model)}


def check_forward():
    from .training_runtime import check_forward as call
    return call()


def train(group,seed,resume_from=None):
    from .training_runtime import train as call
    return call(group,seed,resume_from)


def verify_run(group,seed):
    from .training_runtime import verify_run as call
    return call(group,seed)


def report(r):
    return f'''# 第3阶段：独立训练包装准备及运行边界

本阶段仅CPU元信息／hash／冻结请求／虚构标量核验。独立进程重新核第1／2阶段、2655来源／7闭合目录、输入NPZ逐文件身份及独立核验证据，来源清单{r['source_entries']}项；本次所有NPZ／原RGB／原GT／概率解码0，无Torch／SMP／timm／NumPy／模型导入、tensor、参数加载、模型／优化器实例或前向／训练／合成／校准／指标。

保留原inputs永久禁止模型导入及CPU解释器要求。新独立CPU子进程仅复用冻结身份／元信息帮助函数，并独立核相同封存、来源、闭合目录和请求；不调用旧inputs verify/resources，不修改全局路径。这样初次全轮135.662GiB预算不会错误套到后续恢复或只读复用；只读RAM/D/E各2GiB，首组从头及首次准备另核全轮预算，模型后续组／恢复保留4.5GiB RAM、6GiB显存及20GiB E门槛。training_data为独立模型环境CPU数组适配：只读固定请求和原NPZ，同字节hash后allow_pickle=False；真实uint8/255再ImageNet归一化；合成直接float32减均值／除标准差，不二次除255、量化、翻转或补边。每次使用前逐项核image／target／valid及联合hash。源码已静态核对，本阶段未数值运行；未来25项只读check_forward及逐请求hash不符即停止，不放宽hash或换样。

共用数据seed2026102300、10000请求、2500旧合成，A R5000/N5000、B R2500/S2500/N5000，共享7500槽。三对初始化2026102301/02/03从同通用编码器和新解码器／头开始；旧probe／正式权重不作初始化。原模型／GN8／固定BN affine微调／CE+Dice／Adam两组LR／FP32／batch1/workers0保持。LR第1—8000次为1e-5/1e-4，8001—9000为2e-6/2e-5，9001—10000为4e-7/4e-6；全部标量边界及完成0/999/1000/7999/8000/8999/9000/9999/10000后的下一LR已核，终点无下一更新。

人工训练与显式恢复包装已实现但未运行。必须先独立核完整25项兼容证据，才允许用户手动启动首组；本报告不提供训练启动命令。每组自己的模型／optimizer受限读取／Python+NumPy+Torch CPU/CUDA RNG／请求游标／完整LR及损失前缀／曝光像素计数／参数与BN身份／资源／来源绑定；每1000步封存，RAM<2GiB在100步已完成边界另存完整停止点。恢复仅手动指定自己的最新完整保存点；完整结果只核验复用，未知／部分事务保留，Ctrl+C不额外保存。GPU恢复、模型前向、适配和实际新训练耗时均未实测。

CPU准备启动RAM{r['resources']['available_RAM_GiB']:.3f}GiB、E剩余{r['resources']['E_free_GiB']:.3f}GiB，仅快照。模型准入RAM4.5GiB／显存6GiB／E20GiB，首组还核全轮空间估算，CPU2GiB；不降低规格。先导每10000步约11.32分钟仅历史参考，新B的NPZ读取／逐输入hash及检查点开销未知，不承诺六组时长。

独立只读审阅发现1项Important：冻结inputs verify把135.662GiB初始预算用于全部核验，未来会错误阻止空间充足的后续组／恢复／只读复用。已在本阶段首次封存前修正为独立同等身份核验及分操作资源条件，未改冻结inputs；虚构E120GiB仅校验准入标量：只读／后续模型允许、初次全轮预算拒绝，没有实际低资源或训练试跑。无遗留Minor。跨环境数组字节、25项GPU前向、Adam/RNG恢复及资源峰值均待实测；10000步恰逢低RAM会保留终点检查点并停止，不能常规恢复冒充最终封存；依赖临时目录按owner绑定、只允许自身可丢弃缓存变动，不作为科学完整产物。后两项保持严格停止／隔离语义，若触发需另行诊断。

实际只可运行prepare／verify准备入口。下一项需另行授权25项check_forward，不自动运行、不训练。E已见只探索性，三初始化同数据不等于独立数据集，真实曝光／像素／背景组成混杂保留；全部训练由用户手动启动。
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","verify","check-forward","train","verify-run"])
    parser.add_argument("--group",choices=["A","B"]); parser.add_argument("--seed",type=int,choices=list(c.SEEDS))
    parser.add_argument("--resume-from")
    args=parser.parse_args()
    if args.command in {"prepare","verify","check-forward"}: c.require(args.group is None and args.seed is None and args.resume_from is None,"No extra runtime arguments for this operation")
    else: c.require(args.group is not None and args.seed is not None,"Explicit group and seed required")
    c.require(args.command=="train" or args.resume_from is None,"Restore only in manual train operation")
    if args.command=="prepare": result=prepare()
    elif args.command=="verify": result=verify()
    elif args.command=="check-forward": result=check_forward()
    elif args.command=="train": result=train(args.group,args.seed,args.resume_from)
    else:
        verify()
        result=verify_run(args.group,args.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    sys.modules.setdefault("pretrained_synthetic_pairing.training",sys.modules[__name__])
    main()
