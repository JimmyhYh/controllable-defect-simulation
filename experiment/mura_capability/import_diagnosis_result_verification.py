"""CPU-only historical diagnostic verification. Stage A prepares; run is stage B."""
from __future__ import annotations
import argparse
import ast
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
PROJECT=ROOT.parent.parent
BASE=ROOT/"results/mura_capability/week1_v1"
SOURCE=BASE/"import_diagnosis_launch_v1"
LEGACY=BASE/"import_diagnosis_v1"
DAY1=BASE/"day1_admission_v1"
OLD=ROOT/"results/mura_response/semu_applicability_v1"
OUT=BASE/"import_diagnosis_result_verification_v1"
SPEC=PROJECT/"计划/Mura第2天A独立只读结果核验身份适配方案.md"
SOURCE_COMPLETE="942429386ad21600c67f0c970c94c1dbd48318023376d9b7c2e5a9df5474887c"
LAUNCH_CODE="af4566d5100c698b59ccaff885b8454b76b80f7b35ac17a5ebff37bbace4a905"
LEGACY_CODE="32e6a3972fba7c5b16a0495277d337873797783601b8fb05946269d15cbe59ed"
INIT_CODE="1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa"
DAY1_COMPLETE="be272e26ff988e192d2d834bdb40b9ed2a9b6b2bd4c1e99e9f114c7730e2da1e"
SCHEMA="mura-launch-handshake-v1"
SOURCE_FILES=set(("sources.json protocol.json preparation_complete.json started.json popen_observation.json child_boot.json child_identity.json parent_identity.json decision.json import_permission.json child_result.json stdout.txt stderr.txt result.json verification.json complete.json first_event.json").split())
PREP={"sources.json","protocol.json","preparation_complete.json"}
ALL=PREP|{"run_started.json","verification.json","complete.json"}
HEAVY={"numpy","torch","torchvision","faiss","scipy","PIL","timm","patchcore","safetensors"}
ENV_KEYS=("TEMP","TMP","TMPDIR")
MODEL_ASSETS={str((ROOT/p).resolve()) for p in (
 "models/checkpoints/wide_resnet50_2-95faca4d.pth",
 "models/weak_localization/patchcore/nnscorer_search_index.faiss",
 "models/weak_localization/patchcore/patchcore_params.pkl",
 "models/controlled_generation/formal_draem/G1_20261030/reconstruction.pth",
 "models/controlled_generation/formal_draem/G1_20261030/segmentation.pth")}
PAYLOAD={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".npy",".npz",".pt",".pth",".pkl",".pickle",".faiss",".safetensors"}
MODE=None

def require(ok,message):
    if not ok:raise RuntimeError(message+"; preserve state; no retry or substitution")

def all_true(checks):
    return bool(checks) and all(v is True for v in checks.values())

def obj(x):return x if isinstance(x,dict) else {}

def eq(a,b):return None if a is None or b is None else a==b

def integer(x):return type(x) is int

def positive(x):return integer(x) and x>0

def finite(x):return type(x) in (int,float) and math.isfinite(x)

def hex64(x):return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)

def environment():return {k:os.environ.get(k) for k in ENV_KEYS}

def norm(p):return os.path.normcase(str(Path(p).resolve()))

def no_redirect(path):
    p=Path(path).absolute()
    for q in (p,*p.parents):
        require(not q.is_symlink() and not q.is_junction(),"Redirected path: "+str(q))
    return p

def digest(path):
    with no_redirect(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def pairs(items):
    result={}
    for k,v in items:
        require(k not in result,"Duplicate JSON key: "+k);result[k]=v
    return result

def reject_constant(value):raise ValueError("Non-finite JSON constant: "+value)

def parsed_float(value):
    number=float(value)
    require(math.isfinite(number),"Non-finite JSON number")
    return number

def parse(text):
    return json.loads(text,object_pairs_hook=pairs,parse_constant=reject_constant,parse_float=parsed_float)

def read(path):
    value=parse(no_redirect(path).read_text(encoding="utf-8"))
    require(isinstance(value,dict),"Expected JSON object: "+str(path))
    return value

def write(path,value):
    no_redirect(path)
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")

def inventory(path,names):
    folder=no_redirect(path)
    require(folder.is_dir(),"Missing directory")
    require({p.name for p in folder.iterdir()}==set(names),"Exact inventory differs: "+str(path))
    require(all(no_redirect(p).is_file() for p in folder.iterdir()),"Unexpected non-file member")

def resources():
    class Memory(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ("total","available","page","page_available","virtual","virtual_available","extended")]
    m=Memory();m.length=ctypes.sizeof(m)
    f=ctypes.WinDLL("kernel32",use_last_error=True).GlobalMemoryStatusEx
    f.argtypes=[ctypes.POINTER(Memory)];f.restype=ctypes.c_int
    require(f(ctypes.byref(m)),"RAM query failed")
    value={"RAM_GiB":m.available/2**30,"D_free_GiB":shutil.disk_usage(ROOT).free/2**30}
    require(value["RAM_GiB"]>=2 and value["D_free_GiB"]>=40,"Resource admission failed")
    return value

def directory_stats(path):
    rows=[]
    for p in sorted(no_redirect(path).rglob("*")):
        no_redirect(p);s=p.stat()
        rows.append([p.relative_to(path).as_posix(),"D" if p.is_dir() else "F",s.st_size,s.st_mtime_ns])
    return {"members":len(rows),"stat_sha256":hashlib.sha256(json.dumps(rows).encode()).hexdigest()}

def safe_hash_source(path):
    p=no_redirect(path).resolve()
    require(any(p.is_relative_to(q) for q in (ROOT,PROJECT/"计划",PROJECT/"报告",Path(sys.base_prefix))),"Source path outside fixed roots")
    require(p.suffix.lower() not in PAYLOAD or str(p) in MODEL_ASSETS,"Source payload access refused")
    return digest(p)

def metadata_anchor(path,expected,hashes):
    require(hex64(expected) and safe_hash_source(path)==expected,"Source identity differs: "+str(path))
    hashes[str(Path(path))]=expected

def read_sources():
    """Stage A metadata/opaque-hash admission, not historical result acceptance."""
    resources();inventory(SOURCE,SOURCE_FILES)
    hashes={}
    metadata_anchor(SOURCE/"complete.json",SOURCE_COMPLETE,hashes)
    complete=read(SOURCE/"complete.json")
    require(set(complete["outputs"])==SOURCE_FILES-{"complete.json"},"Original complete outputs differ")
    for n,h in complete["outputs"].items():metadata_anchor(SOURCE/n,h,hashes)
    source=read(SOURCE/"sources.json")
    metadata_anchor(ROOT/"mura_capability/import_diagnosis_launch.py",LAUNCH_CODE,hashes)
    metadata_anchor(ROOT/"mura_capability/import_diagnosis.py",LEGACY_CODE,hashes)
    metadata_anchor(ROOT/"mura_capability/__init__.py",INIT_CODE,hashes)
    for n,h in read(SOURCE/"preparation_complete.json")["outputs"].items():
        require(n in ("sources.json","protocol.json"),"Preparation source traversal")
        metadata_anchor(SOURCE/n,h,hashes)
    inventory(LEGACY,source["legacy_diagnostic_sha256"])
    for n,h in source["legacy_diagnostic_sha256"].items():metadata_anchor(LEGACY/n,h,hashes)
    metadata_anchor(DAY1/"complete.json",DAY1_COMPLETE,hashes)
    dc=read(DAY1/"complete.json");inventory(DAY1,set(dc["outputs"])|{"complete.json"})
    for n,h in dc["outputs"].items():
        require(Path(n).name==n,"Day1 source traversal");metadata_anchor(DAY1/n,h,hashes)
    day=read(DAY1/"sources.json")
    for n,h in day["protected_files_sha256"].items():metadata_anchor(Path(n),h,hashes)
    require(len(day["protected_files_sha256"])==695,"Protected source count differs")
    stats={}
    for n,wanted in day["protected_directory_stats"].items():
        p=(OLD/n).resolve();require(p.is_relative_to(OLD.resolve()),"Directory outside protected root")
        actual=directory_stats(p);require(actual==wanted,"Protected directory changed: "+n);stats[n]=actual
    require(len(stats)==9,"Protected directory count differs")
    for n,h in source["source_chain_identity"].items():metadata_anchor(Path(n),h,hashes)
    for n,h in source["old_failed_metadata_sha256"].items():
        require(Path(n).name==n,"Old failure traversal");metadata_anchor(OLD/"fit_scores_v1/PatchCore"/n,h,hashes)
    failure=OLD/"fit_scores_v1/PatchCore"
    require(not (failure/"complete.json").exists() and not list((failure/"records").iterdir()) and not list((failure/"heatmaps").iterdir()),"Old failure changed")
    for name,wanted in source["normal_workpoints"].items():
        require(name in ("PatchCore","G1"),"Unexpected model role")
        folder=OLD/"normal_scoring_v1"/name
        metadata_anchor(folder/"complete.json",wanted["complete_sha256"],hashes)
        require(read(folder/"q_result.json")["threshold"]==wanted["threshold"],"Original tau changed")
        hashes[str(folder/"q_result.json")]=digest(folder/"q_result.json")
    report=PROJECT/"报告/Mura能力说明第1天资源与真实材料准入.md"
    metadata_anchor(report,source["day1_independent_report_sha256"],hashes)
    metadata_anchor(PROJECT/"报告/Mura第2天A一次启动握手与导入诊断结果.md",
        "b1fa0dc7b09376cd6ffb60e0c073604a71280ece9b5e63018796ba45b1fb8dc5",hashes)
    interp=source["interpreter_identity"]
    for pk,hk in (("executable","executable_sha256"),("base_executable","base_executable_sha256"),("launcher","launcher_sha256")):
        metadata_anchor(Path(interp[pk]),interp[hk],hashes)
    metadata_anchor(ROOT/".venv/pyvenv.cfg",interp["pyvenv_cfg_sha256"],hashes)
    require(norm(sys.executable)==interp["executable"] and norm(sys.prefix)==interp["prefix"] and norm(sys.base_prefix)==interp["base_prefix"] and norm(sys._base_executable)==interp["base_executable"] and sys.version.split()[0]==interp["python_version"],"Interpreter runtime identity differs")
    packages={}
    for n,h in source["package_metadata"].items():
        try:v=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:v=None
        require(v==h,"Dependency metadata changed: "+n);packages[n]=v
    old_spec=Path(source["spec_path"]);text=old_spec.read_text(encoding="utf-8")
    require(hashlib.sha256(text.split("## 二、",1)[1].split("## 五、",1)[0].encode()).hexdigest()==source["spec_rules_sha256"],"Original technical rules changed")
    core=SPEC.read_text(encoding="utf-8").split("## 二、",1)[1].split("## 五、",1)[0]
    return {"original_complete_sha256":SOURCE_COMPLETE,"files_sha256":hashes,
      "directory_signatures":stats,"original_sources":source,
      "implementation":{str(Path(__file__).resolve()):digest(__file__)},
      "plan_rules_sha256":hashlib.sha256(core.encode()).hexdigest(),"package_metadata":packages,
      "protected_file_count":695,"protected_directory_count":9,
      "admission_only_not_stage_B":True}

PARENT_CHILD_KEYS=["HELLO_checks_match_saved","HELLO_matches_saved_child","implementation_match","interpreter_base_binary_hash_is_current_observation_not_historical_certificate","interpreter_base_executable","interpreter_base_executable_sha256","interpreter_base_prefix","interpreter_executable","interpreter_executable_sha256","interpreter_launcher","interpreter_launcher_sha256","interpreter_prefix","interpreter_python_version","interpreter_pyvenv_cfg_sha256","launch_file_unchanged","protocol_file_match","saved_child_checks_all_true","self_kernel_alive","self_kernel_image_allowed","self_kernel_pid_match","source_file_match","started_file_match","temporary_environment_match","token_match"]
CHILD_KEYS=["challenge_body","challenge_controller_pid","challenge_deadline","challenge_kind","challenge_protocol","challenge_schema","challenge_session","implementation_match","interpreter_base_binary_hash_is_current_observation_not_historical_certificate","interpreter_base_executable","interpreter_base_executable_sha256","interpreter_base_prefix","interpreter_executable","interpreter_executable_sha256","interpreter_launcher","interpreter_launcher_sha256","interpreter_prefix","interpreter_python_version","interpreter_pyvenv_cfg_sha256","protocol_file_match","self_kernel_alive","self_kernel_image_allowed","self_kernel_pid_match","source_file_match","started_file_match","temporary_environment_match","token_match"]
COMMON_KEYS=["HELLO_checks_match_saved","HELLO_matches_saved_child","child_alive","child_creation_known","child_kernel_image_allowed","child_pid_positive","child_ppid_positive","controller_pid_match","controller_pid_positive","implementation_match","interpreter_base_binary_hash_is_current_observation_not_historical_certificate","interpreter_base_executable","interpreter_base_executable_sha256","interpreter_base_prefix","interpreter_executable","interpreter_executable_sha256","interpreter_launcher","interpreter_launcher_sha256","interpreter_prefix","interpreter_python_version","interpreter_pyvenv_cfg_sha256","launch_file_unchanged","launch_record_unchanged","popen_alive","popen_creation_known","popen_handle_pid_match","popen_image_allowed","popen_pid_positive","protocol_file_match","reported_child_pid_matches_handle","same_child_creation_time","same_child_image_hash","same_child_image_path","saved_child_checks_all_true","self_kernel_alive","self_kernel_image_allowed","self_kernel_pid_match","source_file_match","started_file_match","temporary_environment_match","token_match"]
PERMISSION_KEYS=["body","deadline","decision.json_hash","entry_deadline_live","kind","launch_still_identical","parent_decision_GO","parent_identity.json_hash","parent_identity_passed","protocol","schema","session"]
MESSAGE_KEYS=["body","deadline","kind","protocol","schema","session"]

def saved_true(value,keys):
    return isinstance(value,dict) and set(value)==set(keys) and all_true(value)

def compare_recorded_environment(bundle,current):
    b=bundle;p=b["parent_identity.json"]["observations"]
    reference=b["started.json"]["environment"]
    values={"started":reference,"boot":b["child_boot.json"]["environment"],
      "identity_boot":b["child_identity.json"]["boot"]["environment"],
      "parent_launch":p["launch"]["environment"],"parent_boot":p["boot"]["environment"],
      "event":b["first_event.json"]["temporary_environment"]}
    required={}
    for name,value in values.items():
        required[name+"_schema"]=isinstance(value,dict) and set(value)==set(ENV_KEYS) and isinstance(value["TEMP"],str) and isinstance(value["TMP"],str) and value["TMPDIR"] is None
        required[name+"_same_run"]=value==reference
    required["recorded_child_environment_unchanged"]=b["child_result.json"]["environment_unchanged"] is True
    return {"required_checks":required,"passed":all_true(required),
      "recorded_run":reference,
      "preparation":b["protocol.json"]["current_environment"],
      "legacy":b["protocol.json"]["legacy_environment"],
      "current":current,"current_differences":{k:current.get(k)!=reference.get(k) for k in ENV_KEYS},
      "current_environment_is_not_a_historical_identity_gate":True}

def kernel_valid(k,interp):
    return (isinstance(k,dict) and positive(k.get("pid")) and positive(k.get("created_100ns"))
      and k.get("alive") is True and k.get("error")==[]
      and (k.get("image_path"),k.get("image_sha256")) in {
        (interp["executable"],interp["executable_sha256"]),
        (interp["base_executable"],interp["base_executable_sha256"])})

def compare_recorded_identity(bundle,sources):
    b=bundle;o=b["parent_identity.json"]["observations"];boot=b["child_boot.json"]
    launch=b["started.json"];p=o["popen"];child=o["parent_view_child"];self_k=boot["kernel"]
    interp=sources["interpreter_identity"]
    recorded=b["parent_identity.json"]["evaluation"]
    common={
      "controller_pid":positive(launch["controller_pid"]) and o["controller"]["pid"]==launch["controller_pid"]==b["popen_observation.json"]["controller"]["pid"],
      "controller_ppid":positive(launch["controller_ppid"]) and o["controller"]["ppid"]==launch["controller_ppid"]==b["popen_observation.json"]["controller"]["ppid"],
      "popen_pid":positive(p["pid"]) and positive(p["returned_pid"]) and p["pid"]==p["returned_pid"],
      "popen_copy":p==b["popen_observation.json"]["popen"],
      "child_pid":positive(boot["pid"]) and boot["pid"]==self_k["pid"]==child["pid"],
      "child_ppid":positive(boot["ppid"]),
      "initial_popen":kernel_valid(p,interp),"initial_child":kernel_valid(child,interp),"initial_self":kernel_valid(self_k,interp),
      "same_child_created":positive(child["created_100ns"]) and child["created_100ns"]==self_k["created_100ns"],
      "same_child_image":child["image_path"]==self_k["image_path"] and child["image_sha256"]==self_k["image_sha256"],
      "runtime_interpreter":boot["interpreter"]==interp,
      "runtime_implementation":boot["implementation"]==sources["implementation"],
      "boot_copies":boot==o["boot"]==b["child_identity.json"]["boot"],
      "launch_copy":launch==o["launch"],"startup_errors_empty":boot.get("startup_errors")=={},
      "recorded_child_checks":saved_true(b["child_identity.json"]["checks"],CHILD_KEYS),
      "recorded_parent_checks":saved_true(o["child_checks"],PARENT_CHILD_KEYS),
      "recorded_common_checks":saved_true(recorded["common_checks"],COMMON_KEYS),
      "launch_unchanged_recorded":o["launch_record_unchanged"] is True}
    direct={"same_pid":boot["pid"]==p["pid"],"ppid_controller":boot["ppid"]==launch["controller_pid"],
      "same_created":child["created_100ns"]==p["created_100ns"]}
    hop={"different_pid":boot["pid"]!=p["pid"],"ppid_popen":boot["ppid"]==p["pid"],
      "bound_launcher":p["image_path"]==interp["executable"] and p["image_sha256"]==interp["executable_sha256"]==interp["launcher_sha256"],
      "bound_base":child["image_path"]==interp["base_executable"] and child["image_sha256"]==interp["base_executable_sha256"],
      "creation_order":positive(child["created_100ns"]) and positive(p["created_100ns"]) and child["created_100ns"]>=p["created_100ns"]}
    relation="DIRECT" if all_true(common) and all_true(direct) else (
      "LAUNCHER_ONE_HOP" if all_true(common) and all_true(hop) else "REJECT")
    return {"required_checks":common,"DIRECT":direct,"LAUNCHER_ONE_HOP":hop,"relation":relation,
      "passed":relation!="REJECT","matches_recorded":recorded["relation"]==relation and recorded["identity_passed"] is (relation!="REJECT")}

def message_checks(message,kind,launch,hashes):
    return {"schema":message.get("schema")==SCHEMA,"kind":message.get("kind")==kind,
      "session":hex64(launch["token_sha256"]) and message.get("session")==launch["token_sha256"],
      "protocol":message.get("protocol_sha256")==hashes["protocol.json"],
      "body":isinstance(message.get("body"),dict)}

def verify_recorded_transcript(bundle,sources):
    """Pure dictionary comparisons. Only run/verify call this on the real bundle."""
    b=bundle;launch=b["started.json"];boot=b["child_boot.json"];ident=b["child_identity.json"]
    parent=b["parent_identity.json"];decision=b["decision.json"];permission=b["import_permission.json"]
    child=b["child_result.json"];result=b["result.json"];event=b["first_event.json"]
    hashes=b["_hashes"];hello,done=b["_messages"]
    env_result=compare_recorded_environment(b,b.get("_current_environment",{}))
    identity=compare_recorded_identity(b,sources)
    entry=child["import_entry_monotonic_ns"];start=launch["submitted_monotonic_ns"]
    deadline=launch["entry_deadline_ns"];elapsed=result["elapsed_seconds"];dt=decision["elapsed_seconds"]
    pt=b["popen_observation.json"]["elapsed_seconds"]
    checks={"recorded_environment":env_result["passed"],"recorded_identity":identity["passed"] and identity["matches_recorded"],
      "boot_source_hashes":boot["sources_sha256"]==hashes["sources.json"] and boot["protocol_sha256"]==hashes["protocol.json"] and boot["started_sha256"]==hashes["started.json"],
      "launch_source_hashes":launch["sources_sha256"]==hashes["sources.json"] and launch["protocol_sha256"]==hashes["protocol.json"],
      "hello":all_true(message_checks(hello,"HELLO",launch,hashes)),
      "done":all_true(message_checks(done,"DONE",launch,hashes)),
      "hello_boot_checks":hello["body"].get("boot")==boot and hello["body"].get("checks")==ident["checks"],
      "parent_recorded_checks_match":parent["observations"]["child_checks"]==PARENT_CHILD_CHECK_SCHEMA_VALUE(parent["observations"]["child_checks"]),
      "decision_identity":decision["identity"]==parent["evaluation"] and result["decision"]==decision,
      "permission_own":permission["own_checks"]==ident["checks"] and all_true(permission["own_checks"]),
      "permission_saved_checks":saved_true(permission["checks"],PERMISSION_KEYS) and permission["permitted"] is True and permission["before_any_dependency_import"] is True and permission["error"] is None,
      "GO":decision["action"]=="GO" and saved_true(decision["message_checks"],MESSAGE_KEYS) and type(decision["hello_count"]) is int and decision["hello_count"]==1 and decision["unexpected_queued_message"] is False and decision["all_logs_saved_before_decision"] is True,
      "entry":child["entered_imports"] is True and integer(entry) and integer(start) and integer(deadline) and 0<=entry-start<=20_000_000_000 and deadline==start+20_000_000_000 and permission["entry_deadline_ns"]==deadline,
      "ordered_times":finite(pt) and finite(dt) and finite(elapsed) and integer(entry) and integer(start) and 0<=pt<=dt<=(entry-start)/1e9<=elapsed<=120,
      "HELLO_conservative_upper_bound":finite(dt) and 0<=dt<=15 and decision["message_checks"].get("deadline") is True,
      "single_attempt":type(launch["diagnostic_child_launch_attempts"]) is int and launch["diagnostic_child_launch_attempts"]==1 and type(result["new_Popen_submissions"]) is int and result["new_Popen_submissions"]==1 and result["actual_Popen_returned"] is True,
      "normal_exit_record":result["failure"] is None and type(result["hello_count"]) is int and result["hello_count"]==1 and result["pipes_closed"] is True and result["pipe_errors"]=={"stdout":None,"stderr":None},
      "event_same":child["first_event"]==event,
      "event_expected":event["kind"]=="OUTSIDE_WRITE" and event["during_import"]=="torch" and event["details"]=={"audit_event":"open","flags":130,"mode":None,"target":r"\\.\NUL"},
      "event_metadata":event["dependency_metadata"]==sources["package_metadata"],
      "event_stack":isinstance(event["stack_root_to_leaf"],list) and bool(event["stack_root_to_leaf"]) and all(isinstance(x,dict) and set(x)=={"file","function","line"} and isinstance(x["file"],str) and isinstance(x["function"],str) and positive(x["line"]) for x in event["stack_root_to_leaf"]),
      "status":child["status"]=="FIRST_GUARD_EVENT" and child["stop_reason"]=="OUTSIDE_WRITE" and child["imports"]==["numpy"] and done["body"].get("status")==child["status"],
      "child_implementation":child["implementation"]==sources["implementation"],
      "no_forbidden_calls_requested":type(child["prohibited_model_forward_training_or_data_calls_requested"]) is int and child["prohibited_model_forward_training_or_data_calls_requested"]==0,
      "original_seal":b["complete.json"]["identity_passed"] is True and b["complete.json"]["model_ready"] is False and b["complete.json"]["status"]=="DIAGNOSTIC_TRANSCRIPT_COMPLETE",
      "original_verification":b["verification.json"]["identity_passed"] is True and b["verification.json"]["child_status"]=="FIRST_GUARD_EVENT" and b["verification.json"]["entered_imports"] is True and type(b["verification.json"]["single_new_attempt"]) is int and b["verification.json"]["single_new_attempt"]==1 and len(b["verification.json"]["consistency"])==16 and all_true(b["verification.json"]["consistency"])}
    # Exact primitive types also apply to copies; Python True must not equal PID/count 1.
    numeric_ids=[launch["controller_pid"],launch["controller_ppid"],boot["pid"],boot["ppid"],
      parent["observations"]["controller"]["pid"],parent["observations"]["controller"]["ppid"]]
    for k in (boot["kernel"],parent["observations"]["popen"],parent["observations"]["parent_view_child"],*result["exit_observations"].values()):
        numeric_ids.extend((k["pid"],k["created_100ns"]))
    checks["exact_PID_creation_types"]=all(positive(v) for v in numeric_ids)
    for label,initial in (("popen",parent["observations"]["popen"]),("python_child",parent["observations"]["parent_view_child"])):
        ex=result["exit_observations"][label]
        checks[label+"_exit"]=ex["pid"]==initial["pid"] and ex["created_100ns"]==initial["created_100ns"] and ex["alive"] is False and type(ex["exit_code"]) is int and ex["exit_code"]==0
        checks[label+"_exit_image_state"]=(
          ex["image_path"]==initial["image_path"] and ex["image_sha256"]==initial["image_sha256"] and ex["error"]==[]
          or ex["image_path"] is None and ex["image_sha256"] is None and ex["error"]==["QueryFullProcessImageNameW:31"])
    # Explicitly recompute the sixteen original consistency claims, without original code imports.
    recomputed={
      "boot_matches_child_identity":boot==ident["boot"],"boot_matches_parent":boot==parent["observations"]["boot"],
      "own_checks_match":permission["own_checks"]==ident["checks"],
      "decision_matches_result":result["decision"]==decision,"decision_matches_identity":decision["identity"]==parent["evaluation"],
      "GO_has_valid_identity":decision["action"]=="GO" and parent["evaluation"]["identity_passed"] is True and all_true(decision["message_checks"]) and decision["unexpected_queued_message"] is False and finite(dt) and dt<=20,
      "permission_derived_from_checks":permission["permitted"] is (all_true(permission["checks"]) and all_true(permission["own_checks"])),
      "entry_requires_GO_permission":decision["action"]=="GO" and permission["permitted"] is True,
      "entry_deadline":integer(entry) and integer(start) and integer(deadline) and start<=entry<=deadline,
      "no_entry_time_without_entry":child["entered_imports"] is True,
      "total_budget":finite(elapsed) and 0<=elapsed<=120,"event_content_matches":child["first_event"]==event,
      "event_presence_matches":event is not None,"guard_event_status":child["status"]=="FIRST_GUARD_EVENT" and event is not None,
      "event_requires_import_entry":child["entered_imports"] is True,
      "dependency_order_prefix":child["imports"]==["numpy"]}
    checks["original_16_recomputed"]=recomputed==b["verification.json"]["consistency"] and all_true(recomputed)
    return {"status":"INDEPENDENT_READ_ONLY_SEALED_RECORD_VERIFIED" if all_true(checks) else "REJECTED",
      "required_checks":checks,"passed":all_true(checks),"environment":env_result,"identity":identity,
      "recomputed_original_16":recomputed,"token_preimage_revalidated":False,
      "historical_kernel_queries_reexecuted":False,"original_verify_cli_status":"FAILED_ENVIRONMENT_COMPARISON_PRESERVED",
      "recorded_diagnosis_status":child["status"],"completed_imports":child["imports"],
      "original_temp_caller_confirmed":False,"model_ready":False,"new_diagnostic_launches":0}

def PARENT_CHILD_CHECK_SCHEMA_VALUE(value):
    # No empty/missing-record bypass; exact expected names defined by frozen metadata schema.
    return {k:True for k in PARENT_CHILD_KEYS} if set(value)==set(PARENT_CHILD_KEYS) else {}

def read_bundle():
    require(MODE in ("run","verify"),"Actual historical result acceptance is stage B only")
    b={n:read(SOURCE/n) for n in SOURCE_FILES if n.endswith(".json")}
    messages=[]
    for name in ("stdout.txt","stderr.txt"):
        raw=(SOURCE/name).read_bytes();require(len(raw)<=4*1024*1024,"Saved stream too large")
        if name=="stdout.txt":
            for line in raw.splitlines(keepends=True):
                require(len(line)<=65536 and line.endswith(b"\n"),"Saved stdout line malformed")
                messages.append(parse(line.decode("utf-8")))
    require(len(messages)==2 and [obj(x).get("kind") for x in messages]==["HELLO","DONE"],"Duplicate/unknown saved controls")
    b["_messages"]=messages;b["_hashes"]={n:digest(SOURCE/n) for n in SOURCE_FILES}
    b["_current_environment"]=environment()
    return b

def artificial_fixture():
    h={n:hashlib.sha256(n.encode()).hexdigest() for n in SOURCE_FILES}
    interp={"executable":"v:/venv/python.exe","executable_sha256":"1"*64,
      "base_executable":"v:/base/python.exe","base_executable_sha256":"2"*64,"launcher_sha256":"1"*64}
    sources={"interpreter_identity":interp,"implementation":{"virtual.py":"3"*64},"package_metadata":{"numpy":"virtual"}}
    e={"TEMP":"virtual-run","TMP":"virtual-run","TMPDIR":None}
    launch={"controller_pid":100,"controller_ppid":50,"token_sha256":"a"*64,
      "protocol_sha256":h["protocol.json"],"sources_sha256":h["sources.json"],
      "submitted_monotonic_ns":10_000_000_000,"entry_deadline_ns":30_000_000_000,
      "environment":e,"diagnostic_child_launch_attempts":1}
    pk={"pid":200,"returned_pid":200,"alive":True,"created_100ns":10,
      "image_path":interp["executable"],"image_sha256":"1"*64,"error":[],"exit_code":259}
    ck={"pid":300,"alive":True,"created_100ns":11,"image_path":interp["base_executable"],
      "image_sha256":"2"*64,"error":[],"exit_code":259}
    boot={"pid":300,"ppid":200,"kernel":copy.deepcopy(ck),"environment":copy.deepcopy(e),
      "interpreter":copy.deepcopy(interp),"implementation":copy.deepcopy(sources["implementation"]),
      "sources_sha256":h["sources.json"],"protocol_sha256":h["protocol.json"],
      "started_sha256":h["started.json"],"startup_errors":{}}
    ident={"boot":copy.deepcopy(boot),"checks":{k:True for k in CHILD_KEYS}}
    observed={"boot":copy.deepcopy(boot),"launch":copy.deepcopy(launch),"controller":{"pid":100,"ppid":50},
      "popen":copy.deepcopy(pk),"parent_view_child":copy.deepcopy(ck),
      "child_checks":{k:True for k in PARENT_CHILD_KEYS},"launch_record_unchanged":True}
    evaluation={"relation":"LAUNCHER_ONE_HOP","identity_passed":True,"common_checks":{k:True for k in COMMON_KEYS}}
    parent={"observations":observed,"evaluation":evaluation}
    decision={"action":"GO","identity":copy.deepcopy(evaluation),"message_checks":{k:True for k in MESSAGE_KEYS},
      "hello_count":1,"unexpected_queued_message":False,"all_logs_saved_before_decision":True,"elapsed_seconds":.3}
    permission={"own_checks":copy.deepcopy(ident["checks"]),"checks":{k:True for k in PERMISSION_KEYS},
      "permitted":True,"before_any_dependency_import":True,"error":None,"entry_deadline_ns":30_000_000_000}
    event={"kind":"OUTSIDE_WRITE","during_import":"torch","details":{"audit_event":"open","flags":130,"mode":None,"target":r"\\.\NUL"},
      "dependency_metadata":sources["package_metadata"],"temporary_environment":copy.deepcopy(e),
      "stack_root_to_leaf":[{"file":"v:/platform.py","function":"_syscmd_ver","line":10}]}
    child={"first_event":copy.deepcopy(event),"status":"FIRST_GUARD_EVENT","stop_reason":"OUTSIDE_WRITE",
      "imports":["numpy"],"implementation":sources["implementation"],"environment_unchanged":True,
      "entered_imports":True,"import_entry_monotonic_ns":10_500_000_000,"prohibited_model_forward_training_or_data_calls_requested":0}
    result={"decision":copy.deepcopy(decision),"elapsed_seconds":1.1,"new_Popen_submissions":1,
      "actual_Popen_returned":True,"failure":None,"hello_count":1,"pipes_closed":True,
      "pipe_errors":{"stdout":None,"stderr":None},
      "exit_observations":{k:dict(v,alive=False,exit_code=0,image_path=None,image_sha256=None,error=["QueryFullProcessImageNameW:31"]) for k,v in (("popen",pk),("python_child",ck))}}
    msg=lambda kind,body:{"schema":SCHEMA,"kind":kind,"session":"a"*64,"protocol_sha256":h["protocol.json"],"body":body}
    b={"started.json":launch,"child_boot.json":boot,"child_identity.json":ident,"parent_identity.json":parent,
      "decision.json":decision,"import_permission.json":permission,"child_result.json":child,"first_event.json":event,
      "result.json":result,"popen_observation.json":{"controller":{"pid":100,"ppid":50},"popen":pk,"elapsed_seconds":.1},
      "protocol.json":{"current_environment":{"TEMP":"virtual-prepare","TMP":"virtual-prepare","TMPDIR":None},"legacy_environment":{"TEMP":"virtual-legacy"}},
      "complete.json":{"identity_passed":True,"model_ready":False,"status":"DIAGNOSTIC_TRANSCRIPT_COMPLETE"},
      "verification.json":{"identity_passed":True,"child_status":"FIRST_GUARD_EVENT","entered_imports":True,
        "single_new_attempt":1,"consistency":{k:True for k in (
          "boot_matches_child_identity","boot_matches_parent","own_checks_match","decision_matches_result","decision_matches_identity",
          "GO_has_valid_identity","permission_derived_from_checks","entry_requires_GO_permission","entry_deadline","no_entry_time_without_entry",
          "total_budget","event_content_matches","event_presence_matches","guard_event_status","event_requires_import_entry","dependency_order_prefix")}},
      "_messages":[msg("HELLO",{"boot":copy.deepcopy(boot),"checks":copy.deepcopy(ident["checks"])}),msg("DONE",{"status":"FIRST_GUARD_EVENT"})],
      "_hashes":h,"_current_environment":{"TEMP":"virtual-current","TMP":"virtual-current","TMPDIR":None}}
    return b,sources

def cache_state(names):
    n=set(names)
    if not n:return "ABSENT"
    if n==PREP:return "PREPARED"
    if n==ALL:return "COMPLETE"
    return "REFUSE_PARTIAL_OR_UNKNOWN"

def cache_snapshot():
    no_redirect(OUT)
    names={p.name for p in OUT.iterdir()}
    require(names<=ALL,"Unknown independent cache member")
    require(all(no_redirect(p).is_file() for p in OUT.iterdir()),"Cache member not a regular file")
    return {n:digest(OUT/n) for n in sorted(names)}

def cache_unchanged(before,after):
    return isinstance(before,dict) and before==after

def compare_hash_inventory(names,expected,actual):
    return set(names)==set(expected)==set(actual) and all(hex64(v) and actual[k]==v for k,v in expected.items())

def static_rules():
    b,s=artificial_fixture()
    cases={}
    cases["virtual_transcript_passes"]=verify_recorded_transcript(b,s)["passed"]
    cases["current_TEMP_may_differ"]=compare_recorded_environment(b,b["_current_environment"])["passed"]
    cases["preparation_and_legacy_not_run_environment"]=compare_recorded_environment(b,{})["passed"]
    cases["LAUNCHER_ONE_HOP"]=compare_recorded_identity(b,s)["relation"]=="LAUNCHER_ONE_HOP"
    direct=copy.deepcopy(b);boot=direct["child_boot.json"];boot.update(pid=200,ppid=100,kernel=copy.deepcopy(direct["popen_observation.json"]["popen"]))
    direct["child_identity.json"]["boot"]=copy.deepcopy(boot)
    o=direct["parent_identity.json"]["observations"];o["boot"]=copy.deepcopy(boot);o["parent_view_child"]=copy.deepcopy(boot["kernel"])
    cases["DIRECT"]=compare_recorded_identity(direct,s)["relation"]=="DIRECT"
    def rejects(name,change):
        bad=copy.deepcopy(b);change(bad)
        cases[name]=verify_recorded_transcript(bad,s)["passed"] is False
    rejects("historical_environment_mismatch",lambda x:x["child_boot.json"]["environment"].update(TEMP="bad"))
    rejects("historical_environment_missing",lambda x:x["first_event.json"]["temporary_environment"].pop("TMPDIR"))
    rejects("unknown_parent_relation",lambda x:x["child_boot.json"].update(ppid=400))
    rejects("missing_initial_image",lambda x:x["child_boot.json"]["kernel"].update(image_path=None))
    rejects("unknown_exit",lambda x:x["result.json"]["exit_observations"]["python_child"].update(alive=None))
    rejects("PID_reuse_creation_mismatch",lambda x:x["result.json"]["exit_observations"]["python_child"].update(created_100ns=999))
    rejects("wrong_session",lambda x:x["_messages"][0].update(session="b"*64))
    rejects("wrong_protocol",lambda x:x["_messages"][0].update(protocol_sha256="c"*64))
    rejects("token_recorded_false",lambda x:x["child_identity.json"]["checks"].update(token_match=False))
    rejects("missing_saved_check",lambda x:x["child_identity.json"]["checks"].pop("token_match"))
    rejects("late_entry",lambda x:x["child_result.json"].update(import_entry_monotonic_ns=31_000_000_000))
    rejects("late_HELLO_bound",lambda x:x["decision.json"].update(elapsed_seconds=16))
    rejects("excess_total",lambda x:x["result.json"].update(elapsed_seconds=121))
    rejects("nonfinite_time",lambda x:x["result.json"].update(elapsed_seconds=float("nan")))
    rejects("event_copy_mismatch",lambda x:x["first_event.json"]["details"].update(flags=0))
    rejects("extra_completed_import",lambda x:x["child_result.json"].update(imports=["numpy","torch"]))
    rejects("permission_false",lambda x:x["import_permission.json"].update(permitted=False))
    rejects("exit_code_failure",lambda x:x["result.json"]["exit_observations"]["popen"].update(exit_code=1))
    rejects("bool_hello_count",lambda x:x["result.json"].update(hello_count=True))
    rejects("bool_zero_counter",lambda x:x["child_result.json"].update(prohibited_model_forward_training_or_data_calls_requested=False))
    rejects("bool_attempt_counter",lambda x:x["verification.json"].update(single_new_attempt=True))
    rejects("null_TEMP",lambda x:x["child_boot.json"]["environment"].update(TEMP=None))
    rejects("nonnull_TMPDIR",lambda x:x["child_boot.json"]["environment"].update(TMPDIR="bad"))
    rejects("bool_copied_PID",lambda x:x["parent_identity.json"]["observations"]["controller"].update(pid=True))
    cases["recorded_exit_image_error31_allowed"]=verify_recorded_transcript(b,s)["passed"]
    cases["token_original_not_revalidated"]=verify_recorded_transcript(b,s)["token_preimage_revalidated"] is False
    cases["full_cache_only"]=cache_state(ALL)=="COMPLETE"
    cases["prepared_cache"]=cache_state(PREP)=="PREPARED"
    cases["partial_cache_refused"]=cache_state(PREP|{"run_started.json"})=="REFUSE_PARTIAL_OR_UNKNOWN"
    cases["unknown_cache_refused"]=cache_state(ALL|{"unknown"})=="REFUSE_PARTIAL_OR_UNKNOWN"
    cases["cache_current_environment_not_identity"]=cache_state(ALL)=="COMPLETE" and compare_recorded_environment(b,{"TEMP":"another","TMP":"another","TMPDIR":None})["passed"]
    fake={"virtual.json":"a"*64}
    cases["complete_hash"]=compare_hash_inventory(fake,fake,fake)
    cases["changed_hash_refused"]=not compare_hash_inventory(fake,fake,{"virtual.json":"b"*64})
    cases["missing_hash_member_refused"]=not compare_hash_inventory([],fake,{})
    cases["extra_hash_member_refused"]=not compare_hash_inventory(["virtual.json","extra"],fake,fake)
    cases["cache_final_snapshot_required"]=cache_unchanged({"a":"1"},{"a":"1"}) and not cache_unchanged({"a":"1"},{"a":"2"}) and not cache_unchanged({"a":"1"},{"a":"1","unknown":"2"})
    for name,text in (("duplicate_JSON", '{"x":1,"x":2}'),("NaN_JSON",'{"x":NaN}'),("Infinity_JSON",'{"x":Infinity}'),("overflow_JSON",'{"x":1e999}')):
        try:parse(text);cases[name]=False
        except (ValueError,RuntimeError):cases[name]=True
    source=Path(__file__).read_text(encoding="utf-8");compile(source,str(__file__),"exec")
    tree=ast.parse(source)
    imported=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imported.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n,ast.ImportFrom):imported.add((n.module or "").split(".")[0])
    cases["standard_library_only"]=imported<=sys.stdlib_module_names and not (imported&HEAVY) and "subprocess" not in imported
    cases["no_process_or_env_calls"]=not any(isinstance(n,ast.Attribute) and n.attr in {
      "Popen","OpenProcess","GetProcessId","GetCurrentProcess","QueryFullProcessImageNameW","GetProcessTimes",
      "system","spawn","fork","putenv","unsetenv"} for n in ast.walk(tree))
    cases["no_heavy_imports_in_process"]=not any(n.split(".")[0] in HEAVY for n in sys.modules)
    require(all_true(cases),"Artificial static rule failed: "+str([k for k,v in cases.items() if v is not True]))
    return {"case_count":len(cases),"cases":cases,"syntax_AST_verified":True,
      "actual_historical_acceptance_executed":False,"new_diagnostic_children":0,"model_imports":0,
      "data_array_probability_reads":0}

def preparation_binding():
    source=read_sources()
    return source

def prepare():
    source=preparation_binding();static=static_rules();snapshot=resources()
    if OUT.exists():
        require(cache_state(p.name for p in OUT.iterdir())=="PREPARED","Preparation cannot claim a started/unknown cache")
        verify_preparation();return {"status":"PREPARATION_REUSED","new_files":0,"static":static,"resources":snapshot}
    no_redirect(OUT);OUT.mkdir()
    write(OUT/"sources.json",source)
    write(OUT/"protocol.json",{
      "scope":"stage A preparation; stage B actual CPU acceptance requires authorization",
      "original_directory":str(SOURCE),"original_complete_sha256":SOURCE_COMPLETE,
      "original_file_names":sorted(SOURCE_FILES),"own_preparation_files":sorted(PREP),"own_complete_files":sorted(ALL),
      "historical_environment_required_equal":True,"current_environment_is_observation_only":True,
      "preparation_environment":environment(),"token_preimage_revalidated":False,
      "no_new_diagnostic_or_original_verify":True,"static":static,"resources":snapshot,
      "HELLO_conservative_upper_bound_seconds":15,"entry_deadline_seconds":20,"total_seconds":120,
      "exit_image_exception":"QueryFullProcessImageNameW:31 allowed only after matching saved PID and creation time",
      "recomputable":"hashes, saved field relations, independent dictionary decisions",
      "historical_attestations":"token original comparison, kernel queries and I/O ordering are not reexecuted",
      "refusal_rules":["source or hash change","unknown or partial cache","false or null necessary check","unknown exit"],
      "complete_cache_before_after_hash_required":True})
    write(OUT/"preparation_complete.json",{"status":"PREPARATION_ONLY",
      "outputs":{n:digest(OUT/n) for n in ("sources.json","protocol.json")},
      "actual_historical_result_acceptance_executed":False})
    return {"status":"PREPARATION_COMPLETE","directory":str(OUT),"static":static,"resources":snapshot,
      "preparation_complete_sha256":digest(OUT/"preparation_complete.json"),"stage_B_executed":False}

def verify_preparation():
    no_redirect(OUT)
    before=cache_snapshot()
    require(set(p.name for p in OUT.iterdir()) in (PREP,ALL),"Unknown/partial independent transaction")
    for p in OUT.iterdir():require(no_redirect(p).is_file(),"Unexpected output member")
    seal=read(OUT/"preparation_complete.json")
    require(set(seal["outputs"])=={"sources.json","protocol.json"},"New preparation manifest differs")
    require(all(digest(OUT/n)==h for n,h in seal["outputs"].items()),"New preparation content changed")
    require(read(OUT/"sources.json")==preparation_binding(),"Current source/implementation identity changed")
    result={"status":"PREPARATION_VERIFIED","preparation_complete_sha256":digest(OUT/"preparation_complete.json"),
      "static":static_rules(),"resources":resources(),"stage_B_executed":False,"new_diagnostic_children":0}
    require(cache_unchanged(before,cache_snapshot()),"Preparation/cache changed during read-only admission")
    return result

def run():
    verify_preparation()
    if cache_state(p.name for p in OUT.iterdir())=="COMPLETE":return verify_cache()
    require(cache_state(p.name for p in OUT.iterdir())=="PREPARED","Partial transaction cannot resume")
    admitted_preparation=cache_snapshot()
    current=environment()
    write(OUT/"run_started.json",{"time_ns":time.time_ns(),"current_environment":current,
      "preparation_complete_sha256":digest(OUT/"preparation_complete.json"),
      "original_complete_sha256":SOURCE_COMPLETE,"resources":resources()})
    started_snapshot=cache_snapshot()
    try:
        source=read(OUT/"sources.json");bundle=read_bundle()
        result=verify_recorded_transcript(bundle,source["original_sources"])
        result["source_integrity"]=read_sources()==source
        after=cache_snapshot()
        result["admitted_preparation_unchanged"]=cache_unchanged(started_snapshot,after) and {n:after[n] for n in PREP}==admitted_preparation
        result["passed"]=result["passed"] and result["source_integrity"] and result["admitted_preparation_unchanged"]
    except Exception as e:
        result={"status":"REJECTED","passed":False,"exception":type(e).__name__+":"+str(e),
          "new_diagnostic_launches":0,"original_verify_cli_status":"FAILED_ENVIRONMENT_COMPARISON_PRESERVED"}
    write(OUT/"verification.json",result)
    require(result["passed"] is True,"Independent verification failed; no completion or retry")
    inventory(OUT,ALL-{"complete.json"})
    preseal=cache_snapshot()
    require({n:preseal[n] for n in started_snapshot}==started_snapshot,"Prepared/started bytes changed before seal")
    write(OUT/"complete.json",{"status":"INDEPENDENT_READ_ONLY_RECORD_VERIFICATION_COMPLETE",
      "original_complete_sha256":SOURCE_COMPLETE,"outputs":preseal,
      "original_verify_repaired":False,"model_ready":False})
    return result

def verify_cache():
    inventory(OUT,ALL);seal=read(OUT/"complete.json")
    before=cache_snapshot()
    require(set(seal["outputs"])==ALL-{"complete.json"} and seal["original_complete_sha256"]==SOURCE_COMPLETE,"New complete identity differs")
    require(all(digest(OUT/n)==h for n,h in seal["outputs"].items()),"New cached output changed")
    verify_preparation()
    source=read(OUT/"sources.json");b=read_bundle()
    saved=read(OUT/"verification.json")
    # Generation-time current environment is historical metadata; never rewrite it.
    b["_current_environment"]=saved["environment"]["current"]
    calculated=verify_recorded_transcript(b,source["original_sources"])
    calculated["source_integrity"]=read_sources()==source
    calculated["admitted_preparation_unchanged"]=cache_unchanged(before,cache_snapshot())
    calculated["passed"]=calculated["passed"] and calculated["source_integrity"] and calculated["admitted_preparation_unchanged"]
    require(saved==calculated and calculated["passed"] is True,"Independent cached semantic checks differ")
    require(cache_unchanged(before,cache_snapshot()),"Complete cache changed before read-only success")
    return {"status":"INDEPENDENT_COMPLETE_CACHE_VERIFIED","original_verify_cli_status":"FAILED_ENVIRONMENT_COMPARISON_PRESERVED",
      "current_observation":environment(),"generation_current_observation":saved["environment"]["current"],
      "current_environment_is_not_identity_gate":True,"new_diagnostic_launches":0}

def audit(event,args):
    if event=="import" and args:
        name=args[0]
        require(not any(name==n or name.startswith(n+".") for n in HEAVY|{"subprocess","mura_response","controlled_generation","mura_capability.import_diagnosis","mura_capability.import_diagnosis_launch"}),"Forbidden import")
    if event=="open" and args and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).absolute();mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        writing=any(c in (mode or "") for c in "wax+") or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:require(p.is_relative_to(OUT.absolute()),"Write outside new independent root")
        if p.suffix.lower() in PAYLOAD:require(str(p.resolve()) in MODEL_ASSETS and not writing,"Payload parsing/access prohibited")
    elif event in ("subprocess.Popen","os.system","os.posix_spawn","socket.connect","socket.getaddrinfo","pickle.find_class"):
        require(False,"Forbidden side effect: "+event)
    elif event in ("os.putenv","os.unsetenv"):require(False,"Environment mutation prohibited")
    elif event in ("os.mkdir","os.remove","os.rmdir","os.rename","os.link","os.symlink"):
        paths=args[:2] if event in ("os.rename","os.link","os.symlink") else args[:1]
        require(all(not isinstance(p,(str,bytes)) or Path(os.fsdecode(p)).absolute().is_relative_to(OUT.absolute()) for p in paths),"Mutation outside independent root")

def main():
    global MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("inspect","prepare","verify-preparation","run","verify"))
    args=parser.parse_args();MODE=args.command;sys.dont_write_bytecode=True
    sys.addaudithook(audit)
    require(not any(n.split(".")[0] in HEAVY for n in sys.modules),"Already imported prohibited package")
    if MODE=="inspect":
        source=read_sources()
        result={"status":"CPU_SOURCE_AND_STATIC_ADMISSION","source_file_hash_count":len(source["files_sha256"]),
          "protected_files":source["protected_file_count"],"protected_directories":source["protected_directory_count"],
          "static":static_rules(),"resources":resources(),"stage_B_executed":False,"creates_cache":False}
    elif MODE=="prepare":result=prepare()
    elif MODE=="verify-preparation":result=verify_preparation()
    elif MODE=="run":result=run()
    else:result=verify_cache()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":main()

