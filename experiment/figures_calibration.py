"""Python figures from frozen C/H scientific arrays only; no model or image inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/calibration_stability"
OUT = ROOT.parents[1] / "报告/图示"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_saved():
    p = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    s = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    if digest(DATA / "protocol.json") != s["protocol_sha256"] or digest(DATA / "scores.npz") != s["scores_sha256"]:
        raise ValueError("Frozen input identity differs")
    if s["H_evaluated"] or s["stage"] != "C_calibration_frozen_before_H":
        raise ValueError("This figure entry handles the current C-only stage; use an explicit H-stage figure implementation later")
    with np.load(DATA / "scores.npz", allow_pickle=False) as a:
        c = a["C_scores"].copy()
        thresholds = a["thresholds"].copy()
        permutations = a["permutations"].copy()
        if a["C_paths"].tolist() != p["splits"]["C"] or c.shape != (48,) or not np.isfinite(c).all():
            raise ValueError("C identity or score shape differs")
    rng = np.random.default_rng(20261011)
    expected = np.stack([rng.permutation(48) for _ in range(500)])
    recomputed = np.column_stack([np.quantile(c[expected[:, :m]], .95, axis=1, method="linear")
                                 for m in (8, 16, 32, 48)])
    if p["sample_sizes"] != [8, 16, 32, 48] or p["repetitions"] != 500 or p["sampling_seed"] != 20261011:
        raise ValueError("Unexpected protocol")
    if not np.array_equal(permutations, expected) or not np.array_equal(thresholds, recomputed):
        raise ValueError("Saved threshold sampling differs")
    canonical = float(np.quantile(c, .95, method="linear"))
    if canonical != s["canonical_C48_threshold"] or not np.all(thresholds[:, 3] == canonical):
        raise ValueError("Full-pool threshold differs")
    for j, row in enumerate(s["threshold_statistics"]):
        q05, median, q95 = np.quantile(thresholds[:, j], [.05, .5, .95], method="linear")
        if [row["q05"], row["median"], row["q95"]] != [float(q05), float(median), float(q95)]:
            raise ValueError("Saved descriptive statistics differ")
    return p, s, thresholds, c


def load_h(p, s, thresholds):
    h = json.loads((DATA / "h_evaluation.json").read_text(encoding="utf-8"))
    for name, key in (("protocol.json", "protocol_sha256"), ("scores.npz", "C_scores_sha256"),
                      ("summary.json", "C_summary_sha256"), ("h_scores.npz", "H_scores_sha256")):
        if digest(DATA / name) != h[key]:
            raise ValueError("H/C scientific identity differs: " + name)
    if not h["H_evaluated"] or h["H_count"] != 48:
        raise ValueError("Complete H48 evaluation required")
    with np.load(DATA / "h_scores.npz", allow_pickle=False) as arrays:
        scores = arrays["H_scores"].copy()
        fp = arrays["FP_by_repetition"].copy()
        if arrays["H_paths"].tolist() != p["splits"]["H"] or scores.shape != (48,) or not np.isfinite(scores).all():
            raise ValueError("H score/path identity differs")
    actual = (scores[None, None, :] > thresholds[:, :, None]).sum(axis=2)
    if not np.array_equal(actual, fp):
        raise ValueError("H FP array differs")
    for j, row in enumerate(h["H_statistics"]):
        q05, median, q95 = np.quantile(fp[:, j], [.05, .5, .95], method="linear")
        if [row["FP_q05"], row["FP_median"], row["FP_q95"]] != [float(q05), float(median), float(q95)]:
            raise ValueError("H FP description differs")
        if row["FP_min"] != int(fp[:, j].min()) or row["FP_max"] != int(fp[:, j].max()):
            raise ValueError("H FP extrema differ")
    expected = {"canonical_C48": s["canonical_C48_threshold"], "original_V49": s["original_V_threshold"]}
    for row in h["reference_results"]:
        if row["threshold"] != expected[row["name"]] or row["FP"] != int((scores > row["threshold"]).sum()):
            raise ValueError("Reference count/threshold differs")
    return h, scores, fp


def export(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / (name+"."+extension), dpi=300)
    plt.close(fig)


def plot_h(p, s, h, c, scores, fp):
    fig = plt.figure(figsize=(183/25.4, 145/25.4))
    fig.text(.08,.96,"H48正常误报：两条固定报警线结果相同",fontsize=10,weight="bold")
    fig.text(.08,.924,"R80库固定 · H48各评分一次 · 严格score > 阈值 · 全部预定比较，未根据H选参",fontsize=6.8)
    a = fig.add_axes([.13,.56,.81,.30])
    b = fig.add_axes([.13,.19,.81,.23])
    low, high = min(float(c.min()),float(scores.min()))-.025, max(float(c.max()),float(scores.max()))+.025
    for values, color, style, label in ((c,"#65789E","-","C正常图 n=48"), (scores,"#373737","--","H正常图 n=48")):
        a.step(np.r_[low,np.sort(values),high],np.r_[0,np.arange(1,49)/48,1],where="post",
               color=color,linestyle=style,lw=1.3,label=label)
    a.axvline(s["original_V_threshold"],color="#A65342",linestyle="--",lw=.85)
    a.axvline(s["canonical_C48_threshold"],color="#377E89",linestyle=":",lw=.85)
    a.set(xlim=(low,high),ylim=(-.02,1.03),xlabel="原始整图异常分数（同一评分尺度）",ylabel="经验累积分布")
    a.legend(loc="lower right",fontsize=6.5)
    fig.text(.08,.882,"a",weight="bold",fontsize=8)
    fig.text(.13,.882,"全部C/H分数，不截取高分或低分样例",fontsize=7)
    fig.text(.13,.475,f"τC48={s['canonical_C48_threshold']:.10f}（青）与原V={s['original_V_threshold']:.10f}（红）：均4/48",fontsize=6.8)
    colors=["#99A9C7","#65789E","#374F75","#373737"]
    styles=["-","--","-.",":"]
    for j,m in enumerate(p["sample_sizes"]):
        x=np.sort(fp[:,j])
        b.step(np.r_[-.5,x,18.5],np.r_[0,np.arange(1,501)/500,1],where="post",
               color=colors[j],linestyle=styles[j],lw=1.3,label=f"m={m}")
    b.set(xlim=(-.5,18.5),ylim=(-.02,1.03),xlabel="H48误报张数FP（每次分母48；4张=8.333%）",ylabel="经验累积分布")
    b.set_xticks([0,4,8,12,16,18])
    b.legend(loc="lower right",ncol=2,fontsize=6.5)
    fig.text(.08,.449,"b",weight="bold",fontsize=8)
    fig.text(.13,.449,"全部500次误报计数，保留完整范围；500次不是独立测试集",fontsize=7)
    for ax in (a,b): ax.tick_params(labelsize=6.5)
    fig.text(.08,.079,"FP的5%—95%经验范围：m8为0—7，m16为0—7，m32为3—4，m48为4—4。",fontsize=6.8)
    fig.text(.08,.052,"每次误报分母为48，经验范围不是总体置信区间；全48恒定来自同一C阈值。",fontsize=6.8)
    fig.text(.08,.025,"未验证真实异常召回；H结果不能与不同集合E的误报率比较为算法改善，也不是屏幕效果。",fontsize=6.8)
    export(fig,"figure4_normal_holdout")


def main():
    p, s, thresholds, c = load_saved()
    h, h_scores, fp = load_h(p, s, thresholds)
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
                         "axes.unicode_minus": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(183/25.4, 145/25.4))
    fig.text(.08, .96, "固定C48池中的阈值抽样敏感性", fontsize=10, weight="bold")
    fig.text(.08, .924, "R80正常库固定 · q=0.95（linear）· PCG64种子20261011 · 每档500次嵌套无放回抽样", fontsize=6.8)
    a = fig.add_axes([.13, .56, .81, .30])
    b = fig.add_axes([.13, .19, .81, .23])
    colors = ["#99A9C7", "#65789E", "#374F75", "#373737"]
    styles = ["-", "--", "-.", ":"]
    low, high = float(thresholds.min())-.025, float(thresholds.max())+.025
    for j, m in enumerate(p["sample_sizes"]):
        x = np.sort(thresholds[:, j])
        a.step(np.r_[low, x, high], np.r_[0, np.arange(1,501)/500, 1], where="post",
               color=colors[j], linestyle=styles[j], lw=1.3, label=f"m={m}")
    original = s["original_V_threshold"]
    canonical = s["canonical_C48_threshold"]
    for ax in (a, b):
        ax.axvline(original, color="#A65342", linestyle="--", lw=.85)
        ax.axvline(canonical, color="#377E89", linestyle=":", lw=.85)
        ax.set_xlim(low, high)
        ax.set_xlabel("原始评分尺度上的校准阈值", labelpad=3)
        ax.tick_params(labelsize=6.5)
    a.set_ylim(-.02,1.03)
    a.set_ylabel("经验累积分布", labelpad=4)
    a.legend(loc="lower right", ncol=2, fontsize=6.5)
    fig.text(.08, .882, "a", weight="bold", fontsize=8)
    fig.text(.13, .882, "全部500次阈值：没有挑选代表重复", fontsize=7)
    for j, row in enumerate(s["threshold_statistics"]):
        y = 3-j
        b.hlines(y, row["q05"], row["q95"], color=colors[j], lw=3)
        b.plot(row["median"],y,"D",color=colors[j],ms=4)
        b.text(high-.01,y+.15,f"跨度 {row['q95_minus_q05']:.6f}",ha="right",fontsize=6.5)
    b.set_yticks([3,2,1,0], ["m=8", "m=16", "m=32", "m=48"])
    b.set_ylim(-.5,3.5)
    fig.text(.08, .449, "b", weight="bold", fontsize=8)
    fig.text(.13, .449, "菱形：中位数；横线：5%—95%经验范围（不是总体置信区间）", fontsize=7)
    fig.text(.13,.475,f"原V49参照：{original:.10f}（红虚线）   全C48规范阈值：{canonical:.10f}（青点线）",fontsize=6.8)
    fig.text(.08,.079,"观察池为48张正常图；500次抽样相关且共用该池，不是500个独立数据集。",fontsize=6.8)
    fig.text(.08,.052,"m=48包含全池，零波动来自构造；不能据此宣称总体稳定性或真实检测提升。",fontsize=6.8)
    fig.text(.08,.025,"H评价已完成：τC48与原V阈值均误报4/48；本图不涉及真实异常召回或屏幕效果。",fontsize=6.8)
    export(fig,"figure3_calibration_stability")
    plot_h(p,s,h,c,h_scores,fp)
    print(json.dumps({"figures": ["figure3_calibration_stability","figure4_normal_holdout"],
                      "size_mm_each": [183,145], "source": "saved C/H scientific arrays only; no inference"},ensure_ascii=True))


if __name__ == "__main__":
    main()
