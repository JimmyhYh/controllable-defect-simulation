"""分类卷积BN统计来源敏感性诊断：唯一新计算模块。

边界（与方案第3、5、6节一致）：
- 导入本模块不加载Torch；Torch只在 run_diagnostic 内、事务写入之后才导入。
- prepare / verify_preparation / verify_diagnostic / summarize 均为纯CPU：不读图像、不加载权重。
- 只改变 model.discriminator.dec_head[1] 的归一化统计来源；不写任何缓冲、不训练、不反向。
- run_diagnostic 启动前必须存在第1b语义核验通过标记（frozen_v1/stage1b_verification.json）。

入口见 cls_bn_sensitivity_cli.py；使用边界见 README_cls_bn_sensitivity.md。
"""
from __future__ import annotations

import ast
import contextlib
import datetime
import math
import sys
from pathlib import Path

from . import resources as r
from . import pilot_dtype_compat as pilot

ROOT = r.ROOT
BASE = ROOT / "results/ssn_baseline/ksdd2_r5_v1"
PREP = BASE / "pilot_cls_bn_sensitivity_preparation_v1"
FROZEN = PREP / "frozen_v2"
STAGE1B = PREP / "stage1b"
RUN = BASE / "pilot_cls_bn_sensitivity_v1"
SUMMARY_DIR = BASE / "pilot_cls_bn_sensitivity_summary_v1"
BASELINE_RUN = BASE / "pilot_dtype_compat_v1/initial"
BASELINE_MODELS = ROOT / "models/ssn_baseline/ksdd2_r5_v1/pilot_dtype_compat_v1/initial"
INPUT_OUT = BASE / "input_preparation_v1"
SOURCE_PROTOCOL = BASE / "pilot_dtype_compat_preparation_v1/protocol.json"
DRAFT = PREP / "protocol_draft.json"
SELF = Path(__file__).resolve()

EXPECTED_WEIGHT = "b8db0216a40206a18c23e5b572badc0b27f23ec24e1be7f09de6150b5a3f06e3"
EXPECTED_PROTOCOL = "edf27348fe1ad487fb456e0b73f58a7e524f54a29d440dcec9265e90a62f7ddc"
EXPECTED_STATE = "1513a3df1435c7d02543c716b85ef4db0c2228b52e04e1d5dc93e852184c3337"
EXPECTED_BACKBONE = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
SCHEMA_PREP = "ssn-cls-bn-sensitivity-preparation-v1"
SCHEMA_RUN = "ssn-cls-bn-sensitivity-run-v1"
CONTROL_ATOL = 1e-6
CONTROL_RTOL = 1e-5
STAT_DELTA_MIN = 1e-6
NUMERIC_FLOOR = 1e-6
BN_CHANNELS = 128
BN_SHAPE = (1, 128, 160, 58)
EXPECTED_FORWARDS = 21


def _torch_loaded():
    return any(m == "torch" or m.startswith("torch.") for m in sys.modules)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _read(path):
    return r.load(path)


# --------------------------------------------------------------------------
# 静态自审：只解析源码，不执行、不导入
# --------------------------------------------------------------------------
def static_review(path=None):
    path = Path(path or SELF)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    add("模块顶层不导入Torch", not any(m.split(".")[0] == "torch" for m in top_imports if m), top_imports)

    forbidden_calls, buffer_writes = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"train", "step", "zero_grad", "backward"}:
                forbidden_calls.append(ast.unparse(node)[:80])
        if isinstance(node, ast.Attribute) and node.attr in {"running_mean", "running_var", "num_batches_tracked"}:
            if isinstance(node.ctx, ast.Store):
                buffer_writes.append(ast.unparse(node))
        if isinstance(node, ast.Name) and node.id == "backward":
            forbidden_calls.append("Name:backward")
    add("无 .train()／.step()／zero_grad／backward 调用", not forbidden_calls, forbidden_calls[:5])
    add("无 running_mean／running_var／num_batches_tracked 赋值（不写缓冲）", not buffer_writes, buffer_writes[:5])

    patch_fn = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_batch_stat_normalization"), None)
    add("存在 _batch_stat_normalization 作用域函数", patch_fn is not None, bool(patch_fn))
    has_finally = patch_fn is not None and any(
        isinstance(n, ast.Try) and n.finalbody for n in ast.walk(patch_fn))
    add("作用域以 finally 恢复原 forward", has_finally, has_finally)

    bn_calls = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "batch_norm"]
    ok_bn, detail_bn = False, None
    for c in bn_calls:
        if len(c.args) >= 5 and all(isinstance(c.args[i], ast.Constant) and c.args[i].value is None for i in (1, 2)):
            kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
            if kw.get("training") == "True":
                ok_bn = True
                detail_bn = {"args": [ast.unparse(a) for a in c.args], "kw": kw}
    add("原生调用形如 batch_norm(x, None, None, w, b, training=True, …)", ok_bn, detail_bn)

    run_fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "run_diagnostic"), None)
    order_ok, order_detail = False, {}
    if run_fn is not None:
        def first_line(pred):
            hits = [n.lineno for n in ast.walk(run_fn) if pred(n)]
            return min(hits) if hits else None
        order_detail = {
            "transaction": first_line(lambda n: isinstance(n, ast.Call) and "transaction.json" in ast.unparse(n)),
            "torch_environment": first_line(lambda n: isinstance(n, ast.Call) and ast.unparse(n.func).endswith("model_environment")),
            "build_model": first_line(lambda n: isinstance(n, ast.Call) and ast.unparse(n.func).endswith("build_model")),
            "state_hash": first_line(lambda n: isinstance(n, ast.Call) and ast.unparse(n.func).endswith("state_hash")),
            "intervention_scope": first_line(lambda n: isinstance(n, ast.With) and "_batch_stat_normalization" in ast.unparse(n)),
        }
        order_ok = all(v is not None for v in order_detail.values()) and (
            order_detail["transaction"] < order_detail["torch_environment"] < order_detail["build_model"]
            < order_detail["state_hash"] < order_detail["intervention_scope"])
    add("run_diagnostic 顺序：事务 < Torch环境 < 构造 < 状态hash < 干预作用域", order_ok, order_detail)

    result = {"checks": checks, "all_pass": all(c["pass"] for c in checks),
              "source": str(path), "source_sha256": r.digest(path)}
    return result


# --------------------------------------------------------------------------
# 身份核对（只读）
# --------------------------------------------------------------------------
def _identity():
    src = _read(SOURCE_PROTOCOL)
    impl = {rel: r.digest(ROOT / rel) for rel in sorted(src["bindings"]["implementation"])}
    impl[str(SELF.relative_to(ROOT)).replace("\\", "/")] = r.digest(SELF)
    return src, impl


def _hash_checks():
    src, impl = _identity()
    backbone = Path(src["checkpoint_path"])
    weight = BASELINE_MODELS / "final_weights.pt"
    baseline_identity = _read(BASELINE_RUN / "scores/identity.json")
    train_summary = _read(BASELINE_RUN / "train/summary.json")
    facts = {
        "source_protocol_sha256": r.digest(SOURCE_PROTOCOL),
        "final_weights_sha256": r.digest(weight),
        "backbone_sha256": r.digest(backbone),
        "backbone_bytes": backbone.stat().st_size,
        "state_sha256_recorded": baseline_identity["state_before_and_after"],
        "state_sha256_train_summary": train_summary["final_state_sha256"],
        "backbone_path": str(backbone),
    }
    r.require(facts["source_protocol_sha256"] == EXPECTED_PROTOCOL, "Source protocol hash changed")
    r.require(facts["final_weights_sha256"] == EXPECTED_WEIGHT == baseline_identity["final_weight_sha256"],
              "Final weight hash changed")
    r.require(facts["backbone_sha256"] == EXPECTED_BACKBONE and facts["backbone_bytes"] == 138223492,
              "Backbone hash/bytes changed")
    r.require(facts["state_sha256_recorded"] == facts["state_sha256_train_summary"] == EXPECTED_STATE,
              "Recorded state hash inconsistent")
    r.require(all(impl[k] == v for k, v in src["bindings"]["implementation"].items()),
              "Frozen implementation hash changed")
    return facts, impl, src


# --------------------------------------------------------------------------
# prepare / verify_preparation（纯CPU）
# --------------------------------------------------------------------------
def prepare():
    r.require(not _torch_loaded(), "prepare must not import Torch")
    if FROZEN.exists():
        r.verify_seal(FROZEN)
        return {"preparation": "already frozen; verified only",
                "protocol_sha256": r.digest(FROZEN / "protocol.json")}

    facts, impl, src = _hash_checks()
    static = static_review()
    r.require(static["all_pass"], "Static review failed; fix before freezing")
    draft = _read(DRAFT)

    baseline = _read(BASELINE_RUN / "scores/scores.json")
    order = [[x["role"], x["id"]] for x in baseline]
    r.require(order == [[x["role"], x["id"]] for x in src["input_ids"]],
              "Baseline order differs from frozen input protocol")

    protocol = dict(draft)
    protocol.update({
        "schema": "ssn-cls-bn-sensitivity-protocol-v1",
        "status": "FROZEN_FOR_STAGE3_ONLY; stage3 requires separate stage1b authorization",
        "frozen_utc": _now(),
        "source_bindings": {
            "source_protocol_sha256": facts["source_protocol_sha256"],
            "final_weights_sha256": facts["final_weights_sha256"],
            "backbone_sha256": facts["backbone_sha256"],
            "backbone_bytes": facts["backbone_bytes"],
            "recorded_state_sha256": facts["state_sha256_recorded"],
            "implementation_sha256": impl,
            "baseline_scores_json_sha256": r.digest(BASELINE_RUN / "scores/scores.json"),
            "baseline_identity_sha256": r.digest(BASELINE_RUN / "scores/identity.json"),
            "stage1_source_review_sha256": r.digest(PREP / "source_review.json"),
        },
        "paths": {
            "run": str(RUN), "baseline_run": str(BASELINE_RUN),
            "weights": str(BASELINE_MODELS / "final_weights.pt"),
            "backbone": facts["backbone_path"], "input_cache": str(INPUT_OUT),
            "frozen": str(FROZEN),
        },
        "numeric_contract": {
            "control_atol": CONTROL_ATOL, "control_rtol": CONTROL_RTOL,
            "stat_delta_min": STAT_DELTA_MIN, "numeric_floor": NUMERIC_FLOOR,
            "bn_channels": BN_CHANNELS, "bn_input_shape": list(BN_SHAPE),
            "statistic_positions": 160 * 58,
            "variance": "biased 1/(N*H*W); N*H*W=9280",
            "expected_forwards": EXPECTED_FORWARDS,
        },
        "static_review": {"all_pass": True, "checks": len(static["checks"]),
                          "self_sha256": static["source_sha256"]},
    })
    protocol.pop("open_before_freeze", None)

    FROZEN.mkdir(parents=True, exist_ok=False)
    r.write(FROZEN / "protocol.json", protocol)
    r.write(FROZEN / "static_checks.json", static)
    r.seal(FROZEN, SCHEMA_PREP)
    r.verify_seal(FROZEN)
    return {"preparation": "frozen", "protocol_sha256": r.digest(FROZEN / "protocol.json"),
            "static_checks": len(static["checks"]), "all_pass": True}


def verify_preparation():
    r.require(not _torch_loaded(), "verify_preparation must not import Torch")
    r.verify_seal(FROZEN)
    facts, impl, src = _hash_checks()
    protocol = _read(FROZEN / "protocol.json")
    sealed_static = _read(FROZEN / "static_checks.json")
    current = static_review()
    r.require(current["all_pass"], "Static review now fails")
    r.require(current["source_sha256"] == sealed_static["source_sha256"], "Module source changed after freeze")
    self_key = str(SELF.relative_to(ROOT)).replace("\\", "/")
    r.require(protocol["source_bindings"]["implementation_sha256"][self_key] == current["source_sha256"],
              "Self hash binding mismatch")
    r.require(protocol["numeric_contract"]["control_atol"] == CONTROL_ATOL
              and protocol["numeric_contract"]["control_rtol"] == CONTROL_RTOL, "Control tolerance changed")
    r.require(protocol["inputs"]["count"] == EXPECTED_FORWARDS and protocol["inputs"]["per_image_once"],
              "Input contract changed")
    r.require(protocol["judgement"]["gate_unchanged"] is False, "Gate flag must stay false")
    return {"preparation_verified": True, "protocol_sha256": r.digest(FROZEN / "protocol.json"),
            "static_checks": len(current["checks"]), "torch_loaded": False}


# --------------------------------------------------------------------------
# 干预作用域
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _batch_stat_normalization(module, torch, probe):
    """只换归一化统计来源；不写缓冲；finally 恢复原 forward。"""
    original_func = module.forward.__func__
    state = {"invocations": 0, "restorations": 0}

    def forward_with_current_stats(x):
        r.require(x.dim() == 4 and tuple(x.shape) == BN_SHAPE, f"Unexpected BN input shape {tuple(x.shape)}")
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        d_mean = (mean - module.running_mean).abs()
        ratio = var / module.running_var
        state["invocations"] += 1
        probe.update({
            "invocations": state["invocations"],
            "delta_mean_max": float(d_mean.max()),
            "delta_mean_median": float(d_mean.median()),
            "delta_var_ratio_max": float((ratio - 1).abs().max()),
            "argmax_channel": int(d_mean.argmax()),
            "finite": bool(torch.isfinite(mean).all() and torch.isfinite(var).all() and torch.isfinite(ratio).all()),
            "training_flag": bool(module.training),
            "track_running_stats": bool(module.track_running_stats),
        })
        return torch.nn.functional.batch_norm(
            x, None, None, module.weight, module.bias,
            training=True, momentum=0.0, eps=module.eps)

    module.forward = forward_with_current_stats
    try:
        yield state
    finally:
        if "forward" in module.__dict__:
            del module.__dict__["forward"]
        r.require(module.forward.__func__ is original_func, "Original forward was not restored")
        state["restorations"] += 1


# --------------------------------------------------------------------------
# run_diagnostic：唯一模型阶段入口
# --------------------------------------------------------------------------
def run_diagnostic():
    import numpy as np

    r.require(not _torch_loaded(), "Import Torch before transaction is forbidden")
    verify_preparation()
    r.verify_seal(STAGE1B)
    marker = STAGE1B / "verification.json"
    r.require(_read(marker).get("passed") is True, "Stage1b did not pass; stage3 must not start")
    protocol = _read(FROZEN / "protocol.json")

    if RUN.exists():
        r.verify_seal(RUN)
        ident = _read(RUN / "identity.json")
        r.require(ident["protocol_sha256"] == r.digest(FROZEN / "protocol.json"),
                  "Run belongs to another protocol")
        return {"diagnostic": "complete cache reused", "new_forwards": 0}

    snapshot = pilot.resource_admission()
    RUN.mkdir(parents=True, exist_ok=False)
    r.write(RUN / "transaction.json", {
        "protocol_sha256": r.digest(FROZEN / "protocol.json"),
        "every_selected_input_once": True, "resource": snapshot,
        "torch_imported_before_transaction": False,
        "planned_outputs": ["scores/{id}.npz", "scores.json", "control.json", "identity.json"],
    })

    records, control = [], []
    hook = None
    min_ram = snapshot["free_RAM_GiB"]
    try:
        torch = pilot.model_environment()
        from .model_dtype_compat import build_model

        model = build_model(checkpoint_path=protocol["paths"]["backbone"],
                            config=_read(SOURCE_PROTOCOL)["model_config"]).to("cuda")
        model.load_state_dict(torch.load(protocol["paths"]["weights"], map_location="cuda",
                                         weights_only=True), strict=True)
        model.eval()
        before = pilot.state_hash(model)
        r.require(before == EXPECTED_STATE, "Loaded final state differs from frozen identity")
        r.require(model.training is False, "Model must stay in eval")

        bn = model.discriminator.dec_head[1]
        r.require(type(bn).__name__ == "BatchNorm2d" and bn.num_features == BN_CHANNELS,
                  "dec_head[1] is not BatchNorm2d(128)")
        r.require(bool(bn.affine) and abs(float(bn.eps) - 1e-5) < 1e-15 and bool(bn.track_running_stats),
                  "BN attributes differ from frozen review")
        buffers = {k: getattr(bn, k).detach().clone()
                   for k in ("running_mean", "running_var", "num_batches_tracked")}

        order = [x["id"] for x in _read(BASELINE_RUN / "scores/scores.json")]
        baseline_rows = {x["id"]: x for x in _read(BASELINE_RUN / "scores/scores.json")}
        r.require(len(order) == EXPECTED_FORWARDS and len(set(order)) == EXPECTED_FORWARDS,
                  "Baseline order invalid")
        cached = pilot.cached_inputs(training=False)
        r.require(set(cached) == set(order), "Cached inputs differ from baseline ids")

        (RUN / "scores").mkdir()
        capture = {}
        hook = model.discriminator.seg.register_forward_hook(
            lambda m, a, out: capture.update(logits=out.detach()))
        probe = {}
        with torch.inference_mode(), _batch_stat_normalization(bn, torch, probe) as state:
            for index, image_id in enumerate(order):
                free = pilot.memory_RAM()
                min_ram = min(min_ram, free)
                r.require(free >= protocol["resources"]["running_RAM_GiB"],
                          "RAM stop; preserve partial transaction without automatic repeat")
                logits, cls = model(torch.from_numpy(cached[image_id]["image"]).unsqueeze(0).to("cuda"))
                feature = torch.sigmoid(capture["logits"].float()).cpu().numpy()[0, 0]
                score_new = float(torch.sigmoid(cls.float()).item())
                logit_new = float(cls.item())
                r.require(math.isfinite(logit_new) and math.isfinite(score_new), "Non-finite classification output")

                r.require(probe["invocations"] == index + 1, "Intervention invocation count mismatch")
                r.require(probe["finite"], "BN statistics contain non-finite values")
                r.require(probe["delta_mean_max"] > STAT_DELTA_MIN and probe["delta_var_ratio_max"] > STAT_DELTA_MIN,
                          "Current statistics equal running statistics; intervention likely inert")
                r.require(probe["training_flag"] is False and probe["track_running_stats"] is True,
                          "BN mode/flag changed during intervention")

                with np.load(BASELINE_RUN / f"scores/{image_id}.npz", allow_pickle=False) as z:
                    base_feature = z["feature_probability"]
                max_diff = float(np.abs(feature - base_feature).max())
                control_ok = bool(np.allclose(feature, base_feature, atol=CONTROL_ATOL, rtol=CONTROL_RTOL))
                control.append({"id": image_id, "role": baseline_rows[image_id]["role"], "index": index,
                                "max_abs_diff": max_diff, "allclose": control_ok})
                r.require(control_ok,
                          f"Segmentation control failed on {image_id} (max abs diff {max_diff:.3e})")

                np.savez_compressed(
                    RUN / "scores" / f"{image_id}.npz",
                    classification_logit_new=np.float32(logit_new),
                    feature_probability_new=feature.astype("float32"),
                    input_index=np.int32(index),
                    delta_mean_max=np.float64(probe["delta_mean_max"]),
                    delta_var_ratio_max=np.float64(probe["delta_var_ratio_max"]))

                old_logit = float(baseline_rows[image_id]["classification_logit"])
                old_score = float(baseline_rows[image_id]["classification_score"])
                records.append({
                    "id": image_id, "role": baseline_rows[image_id]["role"], "index": index,
                    "classification_logit_old": old_logit, "classification_score_old": old_score,
                    "classification_logit_new": logit_new, "classification_score_new": score_new,
                    "delta_logit": logit_new - old_logit,
                    "delta_is_exact_zero": (logit_new - old_logit) == 0.0,
                    "decision_old": old_score > 0.5, "decision_new": score_new > 0.5,
                })

        r.require(state["invocations"] == EXPECTED_FORWARDS and state["restorations"] == 1,
                  "Intervention invocation/restoration count invalid")
        after = pilot.state_hash(model)
        r.require(after == before, "Diagnostic mutated parameters/BN buffers")
        r.require(bool(torch.equal(bn.running_mean, buffers["running_mean"]))
                  and bool(torch.equal(bn.running_var, buffers["running_var"]))
                  and bool(torch.equal(bn.num_batches_tracked, buffers["num_batches_tracked"])),
                  "BN buffers changed")
        r.require(bool(bn.track_running_stats) and bn.training is False, "BN attributes changed by run")

        hook.remove()
        hook = None
        r.write(RUN / "scores.json", records)
        r.write(RUN / "control.json", {"atol": CONTROL_ATOL, "rtol": CONTROL_RTOL, "rows": control})
        r.write(RUN / "identity.json", {
            "schema": SCHEMA_RUN, "protocol_sha256": r.digest(FROZEN / "protocol.json"),
            "final_weight_sha256": r.digest(Path(protocol["paths"]["weights"])),
            "state_before_and_after": before,
            "constructor_shape_probes": 1,
            "input_forwards": len(records),
            "patch_invocations": state["invocations"],
            "patch_restorations": state["restorations"],
            "minimum_sampled_free_RAM_GiB": min_ram,
            "new_model_forwards": len(records),
            "gt_read": False, "qe_u_read": False,
            "stage1b_marker_sha256": r.digest(marker),
        })
        r.seal(RUN, SCHEMA_RUN)
        r.verify_seal(RUN)
    except BaseException as exc:
        if hook is not None:
            hook.remove()
        r.write(RUN / "failure.json", {
            "error": repr(exc), "completed_ids": [x["id"] for x in records],
            "minimum_sampled_free_RAM_GiB": min_ram,
            "rule": "retain partial forwards; no auto retry; identical condition not repeated",
        })
        raise
    return {"diagnostic": "complete", "input_forwards": len(records)}


# --------------------------------------------------------------------------
# verify_diagnostic / summarize（纯CPU）
# --------------------------------------------------------------------------
def verify_diagnostic():
    import numpy as np

    r.require(not _torch_loaded(), "verify_diagnostic must not import Torch")
    r.verify_seal(RUN)
    ident = _read(RUN / "identity.json")
    control = _read(RUN / "control.json")
    rows = _read(RUN / "scores.json")
    protocol = _read(FROZEN / "protocol.json")
    order = [x["id"] for x in _read(BASELINE_RUN / "scores/scores.json")]

    r.require(ident["protocol_sha256"] == r.digest(FROZEN / "protocol.json"), "Protocol binding changed")
    r.require(ident["state_before_and_after"] == EXPECTED_STATE, "State hash differs from frozen identity")
    r.require(ident["input_forwards"] == EXPECTED_FORWARDS
              and ident["patch_invocations"] == EXPECTED_FORWARDS
              and ident["patch_restorations"] == 1, "Forward/intervention counts invalid")
    r.require(ident["gt_read"] is False and ident["qe_u_read"] is False, "GT/Q/E/U read flag tampered")
    r.require([x["id"] for x in rows] == order, "Scored order differs from baseline")
    r.require(len(control["rows"]) == EXPECTED_FORWARDS
              and all(x["allclose"] for x in control["rows"]), "Control not satisfied for all 21")
    r.require(control["atol"] == CONTROL_ATOL and control["rtol"] == CONTROL_RTOL, "Control tolerance changed")
    r.require(all(math.isfinite(x["delta_logit"]) for x in rows), "Non-finite delta")

    for row in rows:
        with np.load(RUN / "scores" / f"{row['id']}.npz", allow_pickle=False) as z:
            r.require(z["feature_probability_new"].shape == (160, 58), "Feature map shape changed")
            r.require(abs(float(z["classification_logit_new"]) - row["classification_logit_new"]) < 1e-6,
                      "Cached logit differs from scores.json")
    r.require(protocol["judgement"]["gate_unchanged"] is False, "Gate flag tampered")
    return {"verified": True, "seal_sha256": r.digest(RUN / "complete.json"), "rows": len(rows)}


def summarize():
    r.require(not _torch_loaded(), "summarize must not import Torch")
    verify_diagnostic()
    rows = _read(RUN / "scores.json")
    baseline = {x["id"]: x for x in _read(BASELINE_RUN / "scores/scores.json")}

    flips = sum(1 for x in rows if x["decision_old"] != x["decision_new"])
    r5 = [x for x in rows if x["role"] == "R5"]
    n16 = [x for x in rows if x["role"] == "N16"]
    margin_old = (min(baseline[x["id"]]["classification_logit"] for x in r5)
                  - max(baseline[x["id"]]["classification_logit"] for x in n16))
    margin_new = min(x["classification_logit_new"] for x in r5) - max(x["classification_logit_new"] for x in n16)
    max_abs_delta = max(abs(x["delta_logit"]) for x in rows)
    fp_old = sum(1 for x in n16 if baseline[x["id"]]["classification_score"] > 0.5)
    fp_new = sum(1 for x in n16 if x["classification_score_new"] > 0.5)

    if flips > 0:
        label = "A"
    elif max_abs_delta > NUMERIC_FLOOR:
        label = "B"
    else:
        label = "C"

    summary = {
        "schema": "ssn-cls-bn-sensitivity-summary-v1",
        "run_identity_sha256": r.digest(RUN / "identity.json"),
        "run_seal_sha256": r.digest(RUN / "complete.json"),
        "row_label": label,
        "row_rule": "flips>0→A; flips=0且max|Δlogit|>1e-6→B; 否则C",
        "decision_flips": flips,
        "num_images": len(rows),
        "num_exact_zero_delta": sum(1 for x in rows if x["delta_is_exact_zero"]),
        "max_abs_delta_logit": max_abs_delta,
        "margin_old": margin_old, "margin_new": margin_new, "margin_delta": margin_new - margin_old,
        "fp_old": fp_old, "fp_new": fp_new,
        "boundary_ids": ["12245", "12025"],
        "boundary_deltas": {x["id"]: x["delta_logit"] for x in rows if x["id"] in {"12245", "12025"}},
        "gate_unchanged": False,
        "note": "训练同池描述量；不含AP／AUROC／阈值搜索；Δ含两臂核算术差异；不证明BN错误或泛化",
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    r.write(SUMMARY_DIR / "summary.json", summary)
    r.seal(SUMMARY_DIR, "ssn-cls-bn-sensitivity-summary-v1")
    r.verify_seal(SUMMARY_DIR)
    return summary
