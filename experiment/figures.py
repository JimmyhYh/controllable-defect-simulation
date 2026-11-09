"""Day 4 figures from saved public results; no fitting, inference or tuning."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib import colors, font_manager
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "报告" / "图示"
RESULT = ROOT / "results"
SYNTH = RESULT / "synthetic_probes"
BLUE, LIGHT, GREY, RED = "#484878", "#7884B4", "#767676", "#B64342"
CN_FONT = Path("C:/Windows/Fonts/msyh.ttc")
if not CN_FONT.is_file():
    raise RuntimeError("Chinese figure font missing: Microsoft YaHei required")
font_manager.fontManager.addfont(str(CN_FONT))
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "svg.hashsalt": "hyc-public-week2-figures", "axes.unicode_minus": False,
})


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_ax(fig, box, array, cmap=None, norm=None):
    ax = fig.add_axes(box)
    im = ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    return ax, im


def colorbar(fig, image, box, label):
    cb = fig.colorbar(image, cax=fig.add_axes(box))
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label, size=6.5, labelpad=4)


def export(fig, name):
    # No tight crop: retain declared final 183-mm width.
    fig.canvas.draw()
    for suffix in ("svg", "pdf", "png", "tiff"):
        options = {"dpi": 600 if suffix == "tiff" else 300}
        if suffix == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"{name}.{suffix}", **options)
    print(name, "size_mm", np.round(fig.get_size_inches()*25.4, 3).tolist(), flush=True)
    plt.close(fig)


def load_results():
    real = read_json(RESULT/"real_evaluation.json")
    syn = read_json(SYNTH/"evaluation.json")
    cp = ROOT/"models"/"formal_patchcore"/"configuration.json"
    config = read_json(cp)
    split = read_json(ROOT/"data"/"split.json")
    if real["configuration_sha256"] != syn["configuration_sha256"] or digest(cp) != syn["configuration_sha256"]:
        raise ValueError("Real and synthetic outputs differ from frozen configuration")
    for path, key in [
        (ROOT/"data"/"split.json", "split_sha256"),
        (ROOT/"models"/"formal_patchcore"/"nnscorer_search_index.faiss", "index_sha256"),
        (ROOT/"models"/"checkpoints"/"wide_resnet50_2-95faca4d.pth", "weights_sha256")]:
        if digest(path) != config[key]:
            raise ValueError("Frozen source changed: " + str(path))
    if real["threshold"] != syn["threshold"] or real["threshold"] != config["threshold"]:
        raise ValueError("Threshold mismatch")
    with np.load(RESULT/"real_evaluation.npz") as z:
        data = {k: z[k] for k in z.files}
    rows = real["test_predictions"]
    if data["test_image_paths"].tolist() != [r["image"] for r in rows]:
        raise ValueError("Saved E path ordering mismatch")
    scores, labels = data["test_scores"], data["test_image_labels"]
    flags = scores > config["threshold"]
    m = real["metrics"]
    counts = [int(np.sum(flags & (labels==1))), int(np.sum(~flags & (labels==1))),
              int(np.sum(flags & (labels==0))), int(np.sum(~flags & (labels==0)))]
    if counts != [m["anomaly_detected"],m["anomaly_missed"],m["normal_false_positives"],m["normal_true_negatives"]]:
        raise ValueError("Saved real classification counts mismatch")
    if not np.isclose(roc_auc_score(labels,scores),m["image_auroc"],rtol=0,atol=1e-12):
        raise ValueError("AUROC mismatch")
    if not np.isclose(average_precision_score(data["test_pixel_labels"].ravel(),data["test_maps"].ravel()),
                      m["pixel_average_precision"],rtol=0,atol=1e-12):
        raise ValueError("Global pixel AP mismatch")
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["image"])
    indices = [next(i for i in ordered if rows[i]["label"]==0),
               next(i for i in ordered if rows[i]["label"]==1)]
    missed = [i for i in ordered if rows[i]["label"]==1 and not rows[i]["predicted_anomaly"]]
    errors = [i for i in ordered if rows[i]["label"]==0 and rows[i]["predicted_anomaly"]]
    if missed:
        indices.append(missed[0])
    if errors:
        indices.append(errors[0])
    if len(indices)!=3:
        raise ValueError("This completed run must have 0 missed and at least 1 false alarm")
    category = Path(split["category_root"])
    images = []
    for i in indices:
        row = rows[i]
        if digest(category/row["image"]) != row["image_sha256"]:
            raise ValueError("Selected real image changed")
        with Image.open(category/row["image"]) as im:
            rgb = np.asarray(im.convert("RGB").resize((224,224),Image.Resampling.BILINEAR))
        if row["mask"]:
            with Image.open(category/row["mask"]) as gt:
                mask = np.asarray(gt.convert("L").resize((224,224),Image.Resampling.NEAREST))>0
        else:
            mask = np.zeros((224,224),dtype=bool)
        if not np.array_equal(mask,data["test_pixel_labels"][i]):
            raise ValueError("Selected GT differs from saved evaluation")
        images.append(rgb)
    records = syn["pairs"]
    first = min(records, key=lambda p:p["P_index"])
    k = records.index(first)
    for name, expected in syn["output_sha256"].items():
        if digest(SYNTH/name)!=expected:
            raise ValueError("Saved synthesis artifact changed")
    with np.load(SYNTH/first["array_file"]) as z:
        pair = {n:z[n] for n in z.files}
    with np.load(SYNTH/"scores_heatmaps.npz") as z:
        ss, sm = z["scores_P_A_B"], z["maps_P_A_B"]
    if not np.array_equal(ss,np.asarray([p["scores_P_A_B"] for p in records])):
        raise ValueError("Synthetic score ordering mismatch")
    if (ss>config["threshold"]).sum(axis=0).tolist()!=[
        syn["summary"]["P_false_alarms"],syn["summary"]["A_detected"],syn["summary"]["B_detected"]]:
        raise ValueError("Synthetic counts mismatch")
    upper = max(float(data["test_maps"][indices].max()),float(sm[k].max()))
    return real,syn,data,indices,images,first,pair,ss,sm[k],upper


def figure_real(real,data,indices,images,upper):
    fig = plt.figure(figsize=(183/25.4,195/25.4))
    fig.text(.055,.981,"真实工业评价：排序良好，固定阈值下正常误报较高",size=9,weight="bold",va="top")
    fig.text(.055,.953,"a",size=8,weight="bold")
    for x,title in zip((.278,.528,.778),("输入 RGB","真实像素标注","原始异常热图")):
        fig.text(x,.941,title,ha="center",size=7)
    norm = colors.Normalize(0,upper)
    last = None
    for row,(i,rgb,y) in enumerate(zip(indices,images,(.721,.501,.281))):
        pred = real["test_predictions"][i]
        title = ("正常／判对","异常／检出","正常／误报")[row]
        tone = RED if row==2 else BLUE
        name = str(Path(pred["image"]).relative_to("test")).replace("\\","/")
        fig.text(.055,y+.125,title,size=7,weight="bold",color=tone)
        fig.text(.055,y+.080,name,size=6.5)
        fig.text(.055,y+.038,f"score\n{pred['score_raw']:.5f}",size=6.5)
        for col,array in enumerate((rgb,data["test_pixel_labels"][i],data["test_maps"][i])):
            _,last = image_ax(fig,[.182+col*.25,y,.192,.18],array,
                               None if col==0 else "gray" if col==1 else "magma",
                               None if col==0 else colors.Normalize(0,1) if col==1 else norm)
        if pred["mask"] is None:
            fig.text(.528,y-.011,"正常图：全零标注",size=6,ha="center",va="top",color=GREY)
    colorbar(fig,last,[.918,.281,.013,.62],"近邻距离热图（共同原始尺度）")
    ax = fig.add_axes([.182,.092,.39,.125])
    values = [100*real["metrics"]["anomaly_recall"],0,100*real["metrics"]["normal_false_positive_rate"]]
    ax.barh([2,1,0],values,color=[BLUE,GREY,RED],height=.6)
    ax.set(yticks=[2,1,0],yticklabels=["异常检出","异常漏检","正常误报"],
           xlim=(0,112),xticks=[0,50,100],xlabel="比例（%）")
    for y,value,ratio in zip((2,1,0),values,("92/92","0/92","21/32")):
        ax.text(min(value+2,105),y,ratio,va="center",size=6.5)
    fig.text(.055,.229,"b",size=8,weight="bold")
    fig.text(.625,.211,"c",size=8,weight="bold")
    fig.text(.65,.196,f"整图 AUROC  {real['metrics']['image_auroc']:.3f}\nE：124 图 = 92 异常 + 32 正常\n\n像素 AP  {real['metrics']['pixel_average_precision']:.6f}\n全 E：6,221,824 像素\n含 32 张正常图的全零标注",
             size=7,va="top",linespacing=1.35)
    fig.text(.055,.026,f"R80 正常库；V49 固定阈值 {real['threshold']:.5f}；score > 阈值判异常。",size=6.5)
    fig.text(.055,.011,"完整图像缩放至224×224；GT最近邻缩放。仅leather实测，不能代表屏幕／Mura效果。",size=6.3)
    export(fig,"figure1_real_evaluation")


def figure_synthetic(syn,first,pair,scores,maps,upper):
    fig = plt.figure(figsize=(183/25.4,205/25.4))
    fig.text(.055,.982,"受控合成观察：降低混合强度，实际残差与检测响应下降",size=9,weight="bold",va="top")
    fig.text(.055,.959,"a",size=8,weight="bold")
    norm = colors.Normalize(0,upper)
    x = [.055,.266,.477,.688]
    w,h = .181,.162
    keys = ["background","strong","weak","mask"]
    titles = ["P 正常背景","强组  λ=0.5","弱组  λ=0.1","操作掩码 M"]
    last = None
    for j,key in enumerate(keys):
        image_ax(fig,[x[j],.771,w,h],pair[key],"gray" if j==3 else None,
                 colors.Normalize(0,1) if j==3 else None)
        fig.text(x[j]+w/2,.939,titles[j],ha="center",size=7)
        if j<3:
            fig.text(x[j]+w/2,.759,f"score {first['scores_P_A_B'][j]:.5f}",ha="center",size=6.5)
        else:
            fig.text(x[j]+w/2,.759,"操作范围 ≠ 可见缺陷 GT",ha="center",size=6)
    for j in range(3):
        ax,last = image_ax(fig,[x[j],.568,w,h],maps[j],"magma",norm)
        y_peak,x_peak = first["heatmap_peaks_P_A_B"][j]["yx"]
        ax.plot(x_peak,y_peak,"+",color="#55D7DA",ms=5,mew=.7)
    image_ax(fig,[x[3],.568,w,h],pair["texture"])
    fig.text(x[3]+w/2,.737,"固定增强素材 T",ha="center",size=7)
    colorbar(fig,last,[.905,.568,.012,.365],"原始热图；+ 为最高点")
    for j in range(3):
        fig.text(x[j]+w/2,.556,"原始热图",ha="center",size=6)
    residuals = [pair["strong"]-pair["background"],pair["weak"]-pair["background"]]
    residual_upper = max(float(np.abs(delta).mean(axis=2).max()) for delta in residuals)
    rnorm = colors.Normalize(0,residual_upper)
    rlast = None
    for j,delta in enumerate(residuals):
        _,rlast = image_ax(fig,[x[j],.367,w,h],np.abs(delta).mean(axis=2),"gray",rnorm)
        fig.text(x[j]+w/2,.536,f"{'强' if j==0 else '弱'}组实际 |X−背景|",ha="center",size=7)
        if np.abs(delta[~pair["mask"]]).any():
            raise ValueError("Unexpected outside-mask change")
    colorbar(fig,rlast,[.46,.367,.012,.162],"")
    fig.text(.466,.345,"RGB通道平均绝对差",ha="center",size=6)
    a,b = first["strong_residual"],first["weak_residual"]
    fig.text(.525,.523,"首对实测（不是 λ 代替残差）",weight="bold",size=7,va="top")
    fig.text(.525,.493,
             f"M内平均 |ΔRGB|：\n强 {a['inside_mean_abs_rgb']:.6f} → 弱 {b['inside_mean_abs_rgb']:.6f}\n"
             f"M内最大 |ΔRGB|：\n强 {a['inside_max_abs_rgb']:.6f} → 弱 {b['inside_max_abs_rgb']:.6f}\n"
             f"能量 ΣΔRGB²：\n强 {a['energy_sum_squared_rgb']:.3f} → 弱 {b['energy_sum_squared_rgb']:.3f}\n"
             "掩码外平均／最大差：两组均为0",size=6.5,va="top",linespacing=1.3)
    ax = fig.add_axes([.09,.099,.385,.201])
    for row in scores:
        ax.plot([0,1,2],row,color=LIGHT,alpha=.42,lw=.65,marker="o",ms=2)
    ax.plot([0,1,2],scores.mean(axis=0),color=BLUE,lw=1.5,marker="D",ms=4,
            label="均值；20条线为成对背景")
    ax.axhline(syn["threshold"],color=RED,ls="--",lw=.8)
    ax.set(xticks=[0,1,2],xticklabels=["P正常","λ0.5","λ0.1"],ylabel="原始整图分数",
           ylim=(0,float(scores.max())*1.16),xlim=(-.22,2.22))
    ax.text(2.17,syn["threshold"]+.16,"固定阈值",ha="right",size=6,color=RED)
    ax.legend(loc="upper left",fontsize=5.8,handlelength=1)
    fig.text(.055,.319,"b",size=8,weight="bold")
    ax = fig.add_axes([.62,.099,.30,.201])
    counts = (syn["summary"]["P_false_alarms"],syn["summary"]["A_detected"],syn["summary"]["B_detected"])
    n = syn["summary"]["valid_pairs"]
    ax.bar([0,1,2],np.array(counts)/n*100,color=[GREY,BLUE,LIGHT],width=.55)
    ax.set(xticks=[0,1,2],xticklabels=["P误报","强组检出","弱组检出"],
           ylabel="固定阈值比例（%）",ylim=(0,122),yticks=[0,50,100])
    for j,count in enumerate(counts):
        ax.text(j,count/n*100+3,f"{count}/{n}",ha="center",size=7)
    fig.text(.56,.319,"c",size=8,weight="bold")
    fig.text(.055,.065,
             f"20对固定背景／T／M，仅改变λ；弱−强分数均值 {syn['summary']['mean_score_change_B_minus_A']:.6f}，20/20对下降。",
             size=6.4)
    fig.text(.055,.045,"全批实际残差幅度约为1/5、能量约为1/25；正常对照3/20已误报，不能只看合成检出数。",size=6.4)
    fig.text(.055,.025,"最高热图点在M内：强16/20、弱14/20。仅合成敏感度，无训练收益或真实屏幕性能结论。",size=6.4)
    export(fig,"figure2_synthetic_response")
    return residual_upper


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protected = [RESULT/"real_evaluation.json",RESULT/"real_evaluation.npz",
                 SYNTH/"evaluation.json",SYNTH/"scores_heatmaps.npz",ROOT/"data"/"split.json",
                 ROOT/"models"/"formal_patchcore"/"configuration.json",
                 ROOT/"models"/"formal_patchcore"/"nnscorer_search_index.faiss"]
    original = {p:digest(p) for p in protected}
    real,syn,data,indices,images,first,pair,scores,maps,upper = load_results()
    figure_real(real,data,indices,images,upper)
    residual_upper = figure_synthetic(syn,first,pair,scores,maps,upper)
    m = real["metrics"]
    write_csv(OUT/"figure1_source_data.csv",[
        {"metric":"image_auroc","value":m["image_auroc"],"numerator":"","denominator":124,"unit":"AUROC"},
        {"metric":"anomaly_recall","value":m["anomaly_recall"],"numerator":92,"denominator":92,"unit":"fraction"},
        {"metric":"anomaly_missed","value":0,"numerator":0,"denominator":92,"unit":"fraction"},
        {"metric":"normal_false_positive_rate","value":m["normal_false_positive_rate"],"numerator":21,"denominator":32,"unit":"fraction"},
        {"metric":"pixel_average_precision","value":m["pixel_average_precision"],"numerator":"","denominator":m["pixel_total"],"unit":"AP"},
        {"metric":"fixed_V_threshold","value":real["threshold"],"numerator":"","denominator":49,"unit":"raw_score"}])
    source = []
    for p,row in zip(syn["pairs"],scores):
        source.append({
            "P_index":p["P_index"],"background_public_id":p["background"],
            "P_score":row[0],"A_score_lambda_0p5":row[1],"B_score_lambda_0p1":row[2],
            "threshold":syn["threshold"],"P_false_alarm":int(row[0]>syn["threshold"]),
            "A_detected":int(row[1]>syn["threshold"]),"B_detected":int(row[2]>syn["threshold"]),
            "B_minus_A":row[2]-row[1],
            "A_peak_in_mask":int(p["heatmap_peaks_P_A_B"][1]["in_operation_mask"]),
            "B_peak_in_mask":int(p["heatmap_peaks_P_A_B"][2]["in_operation_mask"]),
            "mean_abs_rgb_A":p["strong_residual"]["inside_mean_abs_rgb"],
            "mean_abs_rgb_B":p["weak_residual"]["inside_mean_abs_rgb"],
            "energy_A":p["strong_residual"]["energy_sum_squared_rgb"],
            "energy_B":p["weak_residual"]["energy_sum_squared_rgb"],
            "outside_max_A":p["strong_residual"]["outside_max_abs_rgb"],
            "outside_max_B":p["weak_residual"]["outside_max_abs_rgb"]})
    write_csv(OUT/"figure2_source_data.csv",source)
    if any(digest(p)!=expected for p,expected in original.items()):
        raise ValueError("Numerical inputs or frozen model were modified")
    print(json.dumps({
        "real_samples":[real["test_predictions"][i]["image"] for i in indices],
        "synthetic_background":first["background"],"synthetic_P_index":first["P_index"],
        "shared_heatmap_limits":[0,upper],"paired_residual_limits":[0,residual_upper],
        "frozen_input_hashes_unchanged":True},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    run()
