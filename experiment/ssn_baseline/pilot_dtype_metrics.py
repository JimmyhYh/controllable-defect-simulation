"""Independent CPU exact AP on final pilot training-pool caches, no Torch."""
from __future__ import annotations
import argparse
import json
import sys
from . import resources as r
from . import pilot_dtype_preparation as p
from . import pilot_config as c


def exact_AP(scores, labels):
    import numpy as np
    s, y = np.asarray(scores).reshape(-1), np.asarray(labels).reshape(-1)
    r.require(s.size == y.size and s.size > 0 and np.isfinite(s).all() and set(np.unique(y)) <= {0, 1}, "AP cache input invalid")
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-s, kind="stable"); s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    recall = tp.astype(np.float64) / positives
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision, dtype=np.float64))


def compute(name):
    p.verify()
    from .pilot_dtype_compat import paths, verify_train, checkpoint_binding
    verify_train(name)
    run, models = paths(name)
    scores = run / "scores"; out = run / "metrics"
    r.verify_seal(scores)
    source = {"binding": checkpoint_binding(), "score_complete": r.digest(scores / "complete.json"),
              "metrics_implementation": r.digest(__file__)}
    if out.exists():
        r.verify_seal(out)
        r.require(r.load(out / "source.json") == source, "Metrics identity changed")
        return {"reused": True, "metrics": r.load(out / "metrics.json")}
    identity = r.load(scores / "identity.json")
    r.require(identity["binding"] == source["binding"] and identity["final_weight_sha256"] == r.digest(models / "final_weights.pt"), "Score weights/protocol differ")
    import numpy as np
    rows = r.load(scores / "scores.json")
    expected = p.r.load(p.OUT / "protocol.json")["input_ids"]
    r.require([{k: x[k] for k in ["role", "id"]} for x in rows] == expected, "Complete fixed 21 image denominator required")
    manifest = {x["id"]: x for x in r.load(p.data.OUT / "manifest.json")["samples"]}
    feature_scores=[]; feature_labels=[]; native_scores=[]; native_labels=[]; per_real={}
    tp=fp=0; nonconstant=True
    for row in rows:
        meta = manifest[row["id"]]
        with np.load(scores / f"{row['id']}.npz", allow_pickle=False) as pred, np.load(p.data.OUT / meta["cache"], allow_pickle=False) as gt:
            f, n = pred["feature_probability"], pred["native_probability"]
            y, yn = gt["mask_feature"], gt["mask_native"]
            r.require(f.dtype == n.dtype == np.float32 and f.shape == y.shape == (160, 58) and n.shape == yn.shape and np.isfinite(f).all() and np.isfinite(n).all(), "Score array shape/dtype/finite differs")
            r.require(np.all((f >= 0) & (f <= 1)) and np.all((n >= 0) & (n <= 1)), "Probability outside [0,1]")
            nonconstant &= bool(f.min() < f.max())
            feature_scores.append(f.reshape(-1)); feature_labels.append(y.reshape(-1))
            native_scores.append(n.reshape(-1)); native_labels.append(yn.reshape(-1))
            if row["role"] == "R5": per_real[row["id"]] = exact_AP(f,y)
        r.require(np.isfinite(row["classification_score"]) and 0 <= row["classification_score"] <= 1, "Classification probability invalid")
        predicted = row["classification_score"] > .5
        if row["role"] == "R5": tp += int(predicted)
        else: fp += int(predicted)
    fa, fy, na, ny = map(np.concatenate, [feature_scores, feature_labels, native_scores, native_labels])
    ap = exact_AP(fa,fy); native_ap = exact_AP(na,ny)
    gate = bool(ap >= .95 and all(v >= .85 for v in per_real.values()) and tp == 5 and fp <= 1 and nonconstant)
    result = {"training_pool_only": True, "independent_images": 21, "real_images": 5, "normal_images": 16,
        "feature_pixels": int(fy.size), "feature_positive": int(fy.sum()), "feature_global_pixel_AP": ap,
        "each_R5_feature_AP": per_real, "native_pixels": int(ny.size), "native_positive": int(ny.sum()),
        "native_global_pixel_AP_descriptive_only": native_ap,
        "classification_fixed_reference": {"score": .5, "strict_greater": True, "TP": tp, "FN": 5-tp, "FP": fp, "TN": 16-fp, "anomaly_denominator":5,"normal_denominator":16},
        "each_feature_map_nonconstant": nonconstant, "gate": c.GATE, "gate_passed": gate,
        "limitation": "same training pool, one initialization; not heldout validation, generalization, real improvement or screen results"}
    out.mkdir(parents=True, exist_ok=False)
    r.write(out / "source.json", source); r.write(out / "metrics.json", result); r.seal(out,"ssn-pilot-cpu-training-pool-metrics-v1")
    return {"reused": False, "metrics": result}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=["compute"]);parser.add_argument("--run-name",default="initial")
    args=parser.parse_args()
    def guard(event, arguments):
        if event=="import" and (arguments[0].split(".")[0] in {"torch","torchvision","sklearn","diffusers","transformers"} or arguments[0].startswith("ssn_baseline.model") or arguments[0].startswith("ssn_baseline.core_v1")):
            raise RuntimeError("CPU metrics forbids model imports")
        if event.startswith("pickle.") or event=="socket.connect": raise RuntimeError("No pickle/network")
    sys.addaudithook(guard)
    print(json.dumps(compute(args.run_name),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
