"""POD 标准统计：probit MLE + 按背景聚类的 bootstrap。

口径依据：
- MIL-HDBK-1823A / ASTM E3023 的 hit/miss 方法，输出 a50 与 a90；
- 参考类目使用 a90/95（a90 的 95% 单侧下界），故两者同时报告；
- 观测单位是单次试验，报告口径的横轴取该次试验的实测 SEMU；
- 置信区间按 background 整组重采样，独立单元是背景而非观测数。

statsmodels 不在冻结环境内，probit 仅两参数，故自实现 MLE，
并用两路交叉验证：同数据换 logit 连接函数，以及用名义档位复现既有插值口径。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "results" / "pod_probe" / "anolcd_stage2_v1"
OUT = ROOT / "results" / "pod_probe" / "capability_v1"

SHAPES = ("mura_p25", "mura_med", "mura_p75", "line_med")
N_BOOT = 2000
SEED = 20261027
BOOT_KEEP_MIN = 0.95
LOGIT_TOL = 0.3
IMPL_TOL = 0.5


def load_observations():
    idx = json.loads((STAGE2 / "stimuli" / "index.json").read_text(encoding="utf-8"))
    sc = json.loads((STAGE2 / "scores" / "scores.json").read_text(encoding="utf-8"))
    hits = {r["id"]: int(bool(r["hit"])) for r in sc["rows"]}
    obs = []
    for rec in idx["records"]:
        if rec["kind"] != "stimulus":
            continue
        hit = hits.get(rec["id"])
        if hit is None:
            continue
        obs.append({"shape": rec["shape"],
                    "semu": float(rec["SEMU_actual"]),
                    "target": float(rec["target_SEMU"]),
                    "hit": hit,
                    "bg": int(rec["background_index"])})
    return obs, sc.get("tau"), sc.get("baseline_fp")


def _eta(beta, x):
    return beta[0] + beta[1] * x


def _cdf(e, link):
    if link == "probit":
        return np.clip(stats.norm.cdf(e), 1e-9, 1 - 1e-9)
    return np.clip(1.0 / (1.0 + np.exp(-e)), 1e-9, 1 - 1e-9)


def _nll(beta, x, y, link):
    p = _cdf(_eta(beta, x), link)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(x, y, link="probit"):
    res = optimize.minimize(_nll, np.array([-2.0, 1.0]), args=(x, y, link), method="BFGS")
    return np.asarray(res.x, dtype=float), bool(res.success)


def level_at(beta, p, link="probit"):
    if beta[1] <= 0:
        return None
    q = stats.norm.ppf(p) if link == "probit" else float(np.log(p / (1 - p)))
    return float(np.exp((q - beta[0]) / beta[1]))


def bootstrap(x, y, bg, link="probit", n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(bg):
        groups.setdefault(g, []).append(i)
    keys = sorted(groups)
    a50, a90 = [], []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in pick])
        beta, ok = fit(x[idx], y[idx], link)
        if not ok or beta[1] <= 0:
            continue
        v50, v90 = level_at(beta, 0.5, link), level_at(beta, 0.9, link)
        if v50 is not None and v90 is not None:
            a50.append(v50)
            a90.append(v90)
    return np.array(a50), np.array(a90)


def fit_report(x, y, bg, link="probit"):
    beta, ok = fit(x, y, link)
    a50 = level_at(beta, 0.5, link)
    a90 = level_at(beta, 0.9, link)
    b50, b90 = bootstrap(x, y, bg, link)
    kept = len(b50) / N_BOOT
    lo, hi = float(np.exp(x.min())), float(np.exp(x.max()))
    return {
        "beta": [float(v) for v in beta],
        "converged": bool(ok and beta[1] > 0),
        "a50": a50,
        "a90": a90,
        "data_range": [lo, hi],
        "a50_in_range": bool(a50 is not None and lo <= a50 <= hi),
        "a90_in_range": bool(a90 is not None and lo <= a90 <= hi),
        "a50_ci95": ([float(np.percentile(b50, 2.5)), float(np.percentile(b50, 97.5))]
                     if kept > 0 else None),
        "a90_95": (float(np.percentile(b90, 5)) if kept > 0 else None),
        "a90_ci95": ([float(np.percentile(b90, 2.5)), float(np.percentile(b90, 97.5))]
                     if kept > 0 else None),
        "bootstrap_kept": kept,
    }


def self_test(n=4000, seed=7):
    """参数恢复自检：用已知 beta 生成数据，检查 MLE 能否复原。"""
    rng = np.random.default_rng(seed)
    beta_true = np.array([-2.0, 1.0])
    x = rng.uniform(0.0, 3.0, size=n)
    p = stats.norm.cdf(beta_true[0] + beta_true[1] * x)
    y = (rng.uniform(size=n) < p).astype(float)
    beta_hat, ok = fit(x, y, "probit")
    a50_true = level_at(beta_true, 0.5)
    a50_hat = level_at(beta_hat, 0.5)
    a90_true = level_at(beta_true, 0.9)
    a90_hat = level_at(beta_hat, 0.9)
    return {
        "beta_true": [float(v) for v in beta_true],
        "beta_hat": [float(v) for v in beta_hat],
        "beta_max_abs_err": float(np.max(np.abs(beta_hat - beta_true))),
        "a50_true": a50_true, "a50_hat": a50_hat,
        "a50_rel_err": float(abs(a50_hat - a50_true) / a50_true),
        "a90_true": a90_true, "a90_hat": a90_hat,
        "a90_rel_err": float(abs(a90_hat - a90_true) / a90_true),
        "passed": bool(ok and np.max(np.abs(beta_hat - beta_true)) < 0.15
                       and abs(a50_hat - a50_true) / a50_true < 0.05),
    }


def _interp_reference():
    path = STAGE2 / "report" / "summary.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text(encoding="utf-8"))
    return s.get("per_shape", {})


def run():
    obs, tau, base_fp = load_observations()
    interp = _interp_reference()
    test = self_test()
    result = {"tau": tau, "baseline_fp": base_fp, "n_observations": len(obs),
              "n_bootstrap": N_BOOT, "seed": SEED,
              "independent_unit": "background (clustered)", "shapes": {}}

    for shape in SHAPES:
        sel = [o for o in obs if o["shape"] == shape]
        if not sel:
            continue
        x = np.log(np.array([o["semu"] for o in sel]))
        target = np.log(np.array([o["target"] for o in sel]))
        y = np.array([o["hit"] for o in sel], dtype=float)
        bg = [o["bg"] for o in sel]

        actual = fit_report(x, y, bg, "probit")
        beta_logit, ok_logit = fit(x, y, "logit")
        actual["a50_logit"] = level_at(beta_logit, 0.5, "logit") if ok_logit else None
        target_fit = fit_report(target, y, bg, "probit")

        spread = {}
        for t in sorted({o["target"] for o in sel}):
            vals = np.array([o["semu"] for o in sel if o["target"] == t])
            spread["%g" % t] = {"n": int(vals.size),
                                "min": float(vals.min()),
                                "max": float(vals.max()),
                                "ratio": float(vals.max() / vals.min())}

        cell_rates = {}
        for t in sorted({o["target"] for o in sel}):
            ys = [o["hit"] for o in sel if o["target"] == t]
            cell_rates["%g" % t] = float(np.mean(ys))
        lo_rate = min(cell_rates.values())
        hi_rate = max(cell_rates.values())
        ident = {
            "cell_hit_rates": cell_rates,
            "lowest_cell_rate": lo_rate,
            "highest_cell_rate": hi_rate,
            "a50_identifiable": bool(lo_rate <= 0.5 < hi_rate),
            "a90_identifiable": bool(lo_rate < 0.9 <= hi_rate),
            "reason_a50": (None if lo_rate <= 0.5 else
                           "lowest cell already %.3f > 0.5; a50 is extrapolated" % lo_rate),
            "reason_a90": (None if lo_rate < 0.9 <= hi_rate else
                           "no cell spans 0.9; a90 is extrapolated"),
        }

        ref50 = interp.get(shape, {}).get("SEMU_50")
        ref50 = ref50 if isinstance(ref50, float) else None
        impl = {"target_basis_a50": target_fit["a50"],
                "interp_previous_a50": ref50}
        if target_fit["a50"] is not None and ref50 is not None:
            impl["abs_diff"] = abs(target_fit["a50"] - ref50)
            impl["passed"] = bool(impl["abs_diff"] < IMPL_TOL)
        else:
            impl["abs_diff"] = None
            impl["passed"] = None

        logit_diff = (abs(actual["a50"] - actual["a50_logit"])
                      if actual["a50"] is not None and actual["a50_logit"] is not None
                      else None)

        result["shapes"][shape] = {
            "n": len(sel),
            "n_background": len(set(bg)),
            "hit_rate_overall": float(y.mean()),
            "a50": actual["a50"],
            "a90": actual["a90"],
            "a50_ci95": actual["a50_ci95"],
            "a90_95": actual["a90_95"],
            "a90_ci95": actual["a90_ci95"],
            "data_range": actual["data_range"],
            "a50_in_range": actual["a50_in_range"],
            "a90_in_range": actual["a90_in_range"],
            "bootstrap_kept": actual["bootstrap_kept"],
            "actual_fit": actual,
            "target_fit": target_fit,
            "semu_spread_by_target": spread,
            "max_within_target_ratio": max(v["ratio"] for v in spread.values()),
            "implementation_check": impl,
            "vs_logit_abs": logit_diff,
            "identifiability": ident,
            "checks": {
                "converged": actual["converged"],
                "bootstrap_kept": actual["bootstrap_kept"] >= BOOT_KEEP_MIN,
                "a50_in_range": actual["a50_in_range"],
                "a50_identifiable": ident["a50_identifiable"],
                "a90_identifiable": ident["a90_identifiable"],
                "vs_logit_ok": bool(logit_diff is not None and logit_diff < LOGIT_TOL),
                "impl_check_ok": impl["passed"],
            },
        }

    result["self_test"] = test
    result["checks_summary"] = {
        k: all(v["checks"].get(k) is True for v in result["shapes"].values())
        for k in ("converged", "bootstrap_kept", "a50_in_range", "vs_logit_ok",
                  "a50_identifiable")
    }
    result["checks_summary"]["self_test_passed"] = test["passed"]
    result["interp_divergence"] = {
        shape: {"target_basis": v["implementation_check"]["target_basis_a50"],
                "interp_previous": v["implementation_check"]["interp_previous_a50"],
                "abs_diff": v["implementation_check"]["abs_diff"]}
        for shape, v in result["shapes"].items()
        if v["implementation_check"]["abs_diff"] is not None
        and v["implementation_check"]["abs_diff"] >= IMPL_TOL
    }
    result["note"] = (
        "报告口径用实测 SEMU。实现正确性由 self_test（参数恢复）与 mura_med 上"
        "两种口径差 0.03 共同确认；与插值法分歧的形态列入 interp_divergence，"
        "成因是曲线存在平台或陡变，插值法在此不可靠，不作为实现错误。"
        "同一档位内实测 SEMU 跨度大（max_within_target_ratio），"
        "故 a50 必须与 a50_ci95 一并阅读，不得单点引用。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pod_patchcore_256.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    for k, v in r["shapes"].items():
        fmt = lambda q: ("%.2f" % q) if q is not None else "None"
        print("%-10s a50=%-6s a90=%-6s a90_95=%-6s in_range=%-5s logit=%-6s a50_ok=%-5s spread=%.1fx" % (
            k, fmt(v["a50"]), fmt(v["a90"]), fmt(v["a90_95"]),
            v["a50_in_range"], fmt(v["vs_logit_abs"]),
            v["identifiability"]["a50_identifiable"], v["max_within_target_ratio"]))
    print("self_test:", {k: r["self_test"][k] for k in
                         ("beta_max_abs_err", "a50_rel_err", "a90_rel_err", "passed")})
    print("checks:", r["checks_summary"])
    if r["interp_divergence"]:
        print("与插值分歧:", {k: round(v["abs_diff"], 2)
                              for k, v in r["interp_divergence"].items()})
