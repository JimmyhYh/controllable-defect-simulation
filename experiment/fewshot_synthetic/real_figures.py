"""Day 7: Python figures from frozen results; no model/metric imports or calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/fewshot_synthetic/ksdd2_v1'
DATA = ROOT / 'data/fewshot_synthetic/ksdd2_v1'
OUT = BASE / 'figures_day7_v1'
IDS = ('20000', '20042', '20004', '20068')
MODELS = tuple(f'{c}_20261019{s:02d}' for s in (3, 4, 5) for c in 'AB')
COLORS = {'A': '#60758A', 'B': '#167D8D', 'negative': '#AF552B'}
METRIC_HASH = '0f5c9344c7d77eebdd42bbb37944f5f1d78d165fd77a4656c201e3413f8a3e6e'
METRIC_SEAL = 'b20bd02d02965919c22ab059e324fc7bdd47e4bb1438cd45ce3b409bbaddb538'
SOURCES = {}
REASONS = ('ID首个正常', 'ID首个异常', '六模型FP并集最小ID', '六模型FN并集最小ID')
MM = 1 / 25.4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path, expected=None):
    path = Path(path)
    blob = path.read_bytes()
    digest = sha(blob)
    require(expected is None or digest == expected, f'Source identity mismatch: {path}')
    SOURCES[relative(path)] = digest
    return blob


def js(path, expected=None):
    return json.loads(read(path, expected).decode('utf-8'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def guard(event, args):
    if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
        raise RuntimeError('Torch is forbidden in this figure process')
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'pickle.find_class'):
        raise RuntimeError(f'Forbidden operation: {event}')
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1], args[2]
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
        isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
    if writing:
        require(p.is_relative_to(OUT), f'Write outside independent figure directory: {p}')
    if p.is_relative_to(DATA):
        require(p in {DATA / 'test' / f'{i}.png' for i in IDS}, f'Outside-four RGB/original GT forbidden: {p}')
    if p.is_relative_to(BASE / 'real_metrics_v1/labels') and p.suffix == '.npy':
        require(p.stem in IDS, f'Outside-four normalized GT forbidden: {p}')
    if p.is_relative_to(BASE / 'real_evaluation_v1') and p.suffix == '.npy':
        require(p.parent.name == 'probabilities' and p.stem in IDS, f'Outside-four probability forbidden: {p}')


def sealed_json(folder, name, complete):
    return js(folder / name, complete['files'][name])


def load_sources():
    evidence = BASE / 'real_metrics_verifications_v1/all_six_v1'
    evidence_seal = js(evidence / 'complete.json')
    v = sealed_json(evidence, 'verification.json', evidence_seal)
    read(evidence / '第6天独立CPU真实指标最终核验与结果.md',
         evidence_seal['files']['第6天独立CPU真实指标最终核验与结果.md'])
    require(v['verified'] is True and not v['AP_AUROC_recomputed'], 'Actual verification evidence missing')
    require(v['source_metrics_json_sha256'] == METRIC_HASH and
            v['source_metric_complete_sha256'] == METRIC_SEAL, 'Verification/metrics identity mismatch')
    mr = BASE / 'real_metrics_v1'
    seal = js(mr / 'complete.json', METRIC_SEAL)
    metrics = sealed_json(mr, 'metrics.json', seal)
    require(SOURCES[relative(mr / 'metrics.json')] == METRIC_HASH, 'Unexpected metrics')
    require(metrics['selected_examples'] == v['selected_examples'] and
            tuple(x['id'] for x in metrics['selected_examples']['examples']) == IDS,
            'Frozen selected examples/order changed')
    labels = sealed_json(mr, 'labels/summary.json', seal)
    require(SOURCES[relative(mr / 'labels/summary.json')] == v['source_GT_summary_sha256'], 'GT summary mismatch')
    require([m['model'] for m in metrics['models']] == list(MODELS) == v['model_order'], 'Model order mismatch')
    for k, expected in [('unique_E_images', 1004), ('native_pixels_per_model', 146411528),
                        ('actual_positive_pixels_per_model', 486461), ('actual_anomaly_images', 110),
                        ('actual_normal_images', 894)]:
        require(metrics[k] == v[k] == expected, f'Denominator mismatch: {k}')
    prep = BASE / 'real_metrics_preparation_v1'
    ps = js(prep / 'complete.json', v['preparation_complete_sha256'])
    manifest = sealed_json(prep, 'source_manifest.json', ps)
    rows_by_id = {r['request']['id']: r for r in labels['rows']}
    require(len(rows_by_id) == 1004, 'Normalized label metadata count')
    image_rows, probabilities, scores, thresholds, statuses = {}, {}, {}, {}, {}
    for model, source, actual in zip(metrics['models'], manifest['E_models'], v['models'], strict=True):
        label = model['model']
        require(label == source['label'] == actual['model'], 'Source order mismatch')
        require(model['pixel_AP'] == actual['pixel_AP'], 'Frozen AP verification mismatch')
        for k in ('TP', 'FN', 'FP', 'TN', 'AUROC', 'threshold_reference'):
            require(model['image_metrics'][k] == actual['image_metrics'][k], f'Verified {k} mismatch')
        folder = ROOT / source['folder']
        ms = js(folder / 'complete.json', source['complete_sha256'])
        require(ms['files'] == source['files'], 'Frozen score source file list mismatch')
        cached_scores = sealed_json(folder, 'scores.json', ms)
        require(len(cached_scores) == 1004, 'Incomplete score cache')
        sr = {x['request']['id']: x for x in cached_scores}
        require(len(sr) == 1004, 'Duplicate score IDs')
        ts = model['image_metrics']['threshold_reference']
        threshold_summary = js(ROOT / ts['summary'], ts['summary_sha256'])
        thresholds[label] = threshold_summary['calibration']['threshold']
        scores[label], probabilities[label], statuses[label] = {}, {}, {}
        for ident in IDS:
            row = sr[ident]
            req = row['request']
            if ident in image_rows:
                require(req == image_rows[ident], 'Cross-model input identity mismatch')
            image_rows[ident] = req
            subpath = f'probabilities/{ident}.npy'
            require(ms['files'][subpath] == row['probability_sha256'], 'Probability hash reference mismatch')
            a = np.load(io.BytesIO(read(folder / subpath, row['probability_sha256'])), allow_pickle=False)
            require(a.dtype == np.float32 and a.shape == (req['height'], req['width']), 'Probability shape/dtype')
            require(np.isfinite(a).all() and np.all((a >= 0) & (a <= 1)), 'Probability range')
            # Display status uses frozen FP/FN membership, not recounted metrics.
            truth = rows_by_id[ident]['anomaly']
            im = model['image_metrics']
            status = ('FN' if ident in im['FN_ids'] else 'TP') if truth else ('FP' if ident in im['FP_ids'] else 'TN')
            probabilities[label][ident], scores[label][ident], statuses[label][ident] = a, row['score'], status
    images, masks = {}, {}
    for ident in IDS:
        r = rows_by_id[ident]
        expected = r['normalized_sha256']
        require(seal['files'][f'labels/{ident}.npy'] == expected, 'GT seal mismatch')
        gt = np.load(io.BytesIO(read(mr / f'labels/{ident}.npy', expected)), allow_pickle=False)
        require(gt.dtype == np.uint8 and np.all((gt == 0) | (gt == 1)), 'Normalized GT not binary uint8')
        req = image_rows[ident]
        require(gt.shape == (req['height'], req['width']), 'Normalized GT native size mismatch')
        # Exactly four RGB byte snapshots; no original GT files are opened.
        raw = read(DATA / req['image'], req['image_sha256'])
        with Image.open(io.BytesIO(raw)) as img:
            require(img.mode == 'RGB' and img.size == (req['width'], req['height']), 'Native RGB mismatch')
            images[ident] = np.array(img)
        masks[ident] = gt
    return metrics, images, masks, probabilities, scores, thresholds, statuses


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 1.2, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def export(fig, name, book=None):
    # Fixed canvases, no tight crop; text/image geometry is shared by all formats.
    fig.savefig(OUT / f'{name}.png', dpi=220)
    fig.savefig(OUT / f'{name}.svg', dpi=600)
    fig.savefig(OUT / f'{name}.pdf', dpi=600)
    fig.savefig(OUT / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    if book is not None:
        book.savefig(fig, dpi=600)
    plt.close(fig)


def panel_title(ax, letter, text):
    ax.set_title(text, loc='left', pad=12, fontweight='bold')
    ax.text(-0.12, 1.10, letter, transform=ax.transAxes, fontsize=10, weight='bold')


def fig_ap(m):
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 110 * MM),
                             gridspec_kw={'width_ratios': [1, 1.05, 1]})
    fig.subplots_adjust(left=.085, right=.975, bottom=.34, top=.77, wspace=.57)
    fig.text(.04, .95, '真实留出结果：平均AP正增益，但三对初始化不稳定', fontsize=10, weight='bold')
    pairs = m['paired']['pairs']
    ax = axes[0]
    panel_title(ax, 'a', '全局像素AP（定位）')
    for i, pair in enumerate(pairs):
        values = [100 * pair['AP_A'], 100 * pair['AP_B']]
        ax.plot([i-.14, i+.14], values, color='#B6BEC6', lw=1.2, zorder=1)
        for x, y, c in zip([i-.14, i+.14], values, 'AB'):
            ax.scatter(x, y, color=COLORS[c], marker='o' if c == 'A' else 's', s=23, zorder=3)
            ax.annotate(f'{y:.2f}', (x,y), xytext=(-3 if c=='A' else 3, -13 if c=='A' else 8),
                        textcoords='offset points', ha='right' if c=='A' else 'left', fontsize=6.5)
    ax.set(xlim=(-.65, 2.65), ylim=(0, 45), xticks=range(3), xticklabels=['03','04','05'], ylabel='像素AP (%)', xlabel='种子后两位')
    ax.yaxis.grid(True, color='#E7EBEF', zorder=0)
    fig.legend(handles=[Patch(color=COLORS[c], label=f'{c}：'+('无合成补充' if c=='A' else '有合成补充')) for c in 'AB'],
               loc='upper left', bbox_to_anchor=(.065, .215), frameon=False, ncol=1, handlelength=1)
    ax = axes[1]
    panel_title(ax, 'b', 'B−A：像素AP差')
    delta = [p['delta_AP_percentage_points'] for p in pairs]
    for i,d in enumerate(delta):
        ax.plot([i,i],[0,d], color=COLORS['B'] if d>=0 else COLORS['negative'], lw=2)
        ax.scatter(i,d,s=25,color=COLORS['B'] if d>=0 else COLORS['negative'],zorder=4)
        ax.annotate(f'{d:+.4f}', (i,d), xytext=(8,-2) if 0<d<2 else (0,8 if d>2 else -15),
                    textcoords='offset points', ha='left' if 0<d<2 else 'center', fontsize=7)
    mean=m['paired']['mean_delta_AP_percentage_points']
    ax.axhline(0,color='#9DA4AC',lw=.8)
    ax.axhline(mean,color='#4A5057',lw=.8,ls='--')
    ax.set(xlim=(-.6,2.6),ylim=(-3.0,7.0),xticks=range(3),xticklabels=['03','04','05'],ylabel='百分点',xlabel='种子后两位')
    fig.text(.53,.145,f'均值 {mean:+.4f}\n样本SD {m["paired"]["sample_SD_delta_AP_percentage_points"]:.4f}；n=3',
             ha='center',fontsize=7)
    ax=axes[2]
    panel_title(ax, 'c', '整图AUROC（排序）')
    for i in range(3):
        for j,c in enumerate('AB'):
            y=100*m['models'][2*i+j]['image_metrics']['AUROC']
            ax.scatter(i+(-.13 if j==0 else .13),y,color=COLORS[c],marker='o' if c=='A' else 's',s=23)
            ax.annotate(f'{y:.2f}',(i+(-.13 if j==0 else .13),y),xytext=(-2 if j==0 else 2,-13 if j==0 else 8),
                        textcoords='offset points',ha='right' if j==0 else 'left',fontsize=6.5)
    ax.set(xlim=(-.65,2.65),ylim=(80,92),xticks=range(3),xticklabels=['03','04','05'],ylabel='整图AUROC (%)',xlabel='种子后两位')
    ax.yaxis.grid(True,color='#E7EBEF')
    fig.text(.04,.065,'同一E1004：110异常／894正常；每模型146,411,528像素，其中486,461阳性。',fontsize=7)
    fig.text(.04,.025,'n=3仅为初始化配对；固定单类别、R5与划分。虚线为已保存均值，非置信区间。',fontsize=7)
    export(fig,'fig01_paired_AP_AUROC')


def fig_counts(m):
    fig,axes=plt.subplots(1,2,figsize=(183*MM,118*MM))
    fig.subplots_adjust(left=.13,right=.97,bottom=.32,top=.77,wspace=.44)
    fig.text(.04,.95,'固定Q200阈值：误报均减少，漏检并非每对改善',fontsize=10,weight='bold')
    names=[x['model'][0]+x['model'][-2:] for x in m['models']]
    for ax,good,bad,den,title,letter in [(axes[0],'TP','FN',110,'异常图：TP / FN','a'),(axes[1],'TN','FP',894,'正常图：TN / FP','b')]:
        panel_title(ax,letter,title)
        for i,row in enumerate(m['models']):
            im=row['image_metrics']
            ax.barh(i,im[good],color='#CBD8E2',height=.63)
            ax.barh(i,im[bad],left=im[good],color='#B56843',height=.63)
            ax.text(den*.03,i,f'{good} {im[good]}',va='center',fontsize=7)
            # Direct labels remain readable for narrow FP slices.
            ax.text(den*1.03,i,f'{bad} {im[bad]}',va='center',fontsize=7,color='#914422')
        ax.set(yticks=range(6),yticklabels=names,xlim=(0,den*1.24),ylim=(5.65,-.65),xlabel=f'图像数（每模型分母{den}）')
        ax.set_xticks([0,den//2,den]); ax.spines['right'].set_visible(False)
        ax.axvline(den,color='#9099A2',ls=':',lw=.7)
        for b in (1.5,3.5): ax.axhline(b,color='#E6EAEE',lw=.7)
    fig.text(.13,.17,'B−A漏检：03 +1；04 −3；05 −3张',fontsize=7.5,color='#914422')
    fig.text(.58,.17,'B−A误报：03 −28；04 −18；05 −11张',fontsize=7.5,color=COLORS['B'])
    fig.text(.04,.095,'每模型沿用各自Q200的q0.95 / linear阈值，原生拼回softmax最大值严格大于阈值为阳性。',fontsize=7)
    fig.text(.04,.055,'六组共用同一1004张图；110异常和894正常不因模型数重复累计。',fontsize=7)
    fig.text(.04,.015,'AP衡量像素排序，AUROC衡量整图排序；本图计数依赖已冻结阈值，三者不能相互替代。',fontsize=7)
    export(fig,'fig02_fixed_threshold_counts')


def fig_examples(m,images,masks,probabilities,scores,thresholds,statuses):
    with PdfPages(OUT/'fig03_localization_all_seeds.pdf') as book:
        for seed in (3,4,5):
            a,b=f'A_20261019{seed:02d}',f'B_20261019{seed:02d}'
            fig,axes=plt.subplots(4,4,figsize=(183*MM,252*MM))
            fig.subplots_adjust(left=.10,right=.86,bottom=.10,top=.88,wspace=.12,hspace=.38)
            fig.text(.04,.968,f'固定四例定位对照｜初始化20261019{seed:02d}',fontsize=10,weight='bold')
            fig.text(.04,.942,f'A：无合成补充；B：有合成补充。全部原生画布，未增强对比度；页 {seed-2}/3。',fontsize=7)
            headings=['原始RGB','规范GT\n白=缺陷',f'A{seed:02d} 概率',f'B{seed:02d} 概率']
            for c,t in enumerate(headings):
                fig.text(.10+(.76/4)*(c+.5),.91,t,ha='center',va='center',fontsize=8,weight='bold')
            for r,ident in enumerate(IDS):
                for c,ax in enumerate(axes[r]):
                    ax.set_xticks([]);ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_visible(False)
                    if c==0: ax.imshow(images[ident],interpolation='nearest')
                    elif c==1: ax.imshow(masks[ident],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                    else:
                        model=a if c==2 else b
                        ax.imshow(probabilities[model][ident],cmap='magma',norm=Normalize(0,1),interpolation='nearest')
                        ax.set_title(f'{statuses[model][ident]} | s={scores[model][ident]:.5g}',fontsize=6.8,pad=3)
                axes[r,0].text(-.40,.5,ident,transform=axes[r,0].transAxes,rotation=90,va='center',ha='center',fontsize=8,weight='bold')
                axes[r,0].text(.5,-.07,REASONS[r],transform=axes[r,0].transAxes,ha='center',va='top',fontsize=6)
            cax=fig.add_axes([.91,.33,.016,.30])
            cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap='magma'),cax=cax)
            cb.set_ticks([0,.5,1]);cb.set_label('原始异常softmax概率',fontsize=7)
            fig.text(.04,.060,f'固定阈值：A{seed:02d} τ={thresholds[a]:.9g}；B{seed:02d} τ={thresholds[b]:.9g}。s为保存的整图最大值。',fontsize=6.8)
            fig.text(.04,.036,'TP/FN/FP/TN取自冻结结果；不代表二值分割。四例选择原因在三页一致，未按效果筛选。',fontsize=6.8)
            fig.text(.04,.012,'同一E1004与同一四例跨页复用；热图统一[0,1]，GT/概率无裁剪、平滑或拉伸。',fontsize=6.8)
            export(fig,f'fig03_localization_seed{seed:02d}',book)


def csv_sources(m,scores,thresholds,statuses):
    with (OUT/'source_model_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','pixel_AP_fraction','image_AUROC_fraction','TP','FN','FP','TN','anomaly_n','normal_n','native_pixels','positive_pixels','Q_threshold'])
        for row in m['models']:
            im=row['image_metrics'];w.writerow([row['model'],row['pixel_AP'],im['AUROC'],im['TP'],im['FN'],im['FP'],im['TN'],110,894,146411528,486461,thresholds[row['model']]])
    with (OUT/'source_paired_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','AP_A_fraction','AP_B_fraction','delta_AP_percentage_points','delta_FP_images','delta_FN_images'])
        for p in m['paired']['pairs']: w.writerow([p[k] for k in ['seed','AP_A','AP_B','delta_AP_percentage_points','delta_FP_images','delta_FN_images']])
    with (OUT/'source_selected_examples.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['id','frozen_selection_reason','model','saved_score','saved_status','Q_threshold'])
        for example in m['selected_examples']['examples']:
            for model in MODELS: w.writerow([example['id'],'; '.join(example['reasons']),model,scores[model][example['id']],statuses[model][example['id']],thresholds[model]])


def captions(m):
    text='''# 第7天真实工业缺陷实验图件与中文图注

本组图回答：在固定5张真实异常训练图及同一真实评价划分上，用可控合成替代部分正常曝光，能否改善缺陷定位？图件完整展示三对初始化；已冻结的平均像素AP增益为正，但方向与幅度不稳定。它不是总体显著性或可控外观逼真的证据，也不是显示屏效果。

## 图1：三对像素AP、配对差与整图AUROC

![图1](fig01_paired_AP_AUROC.png)

**中文图注。** a，A为无合成补充，B为有合成补充；展示2026101903／04／05全部最终10000步模型的全局像素AP，按完整原生像素共同排序，原结果未重算。b，对应B−A分别为+0.9710、+5.4744、−1.1916个百分点；冻结均值+1.7513，样本SD3.4008（ddof=1，n=3初始化配对）。虚线仅表示均值，不表示置信区间；05的负差完整保留。c，六模型整图AUROC仅衡量整图排序，纵轴80—92%已明确标注，与像素定位AP分开展示。A/B AP依次35.2649／36.2359%、32.7732／38.2476%、36.4531／35.2615%；AUROC依次85.8959／86.8339%、86.5924／86.8894%、84.4173／88.8702%。三对共用同一E1004与同一R5／划分，不是三个独立数据集。

## 图2：六模型固定阈值下的完整计数

![图2](fig02_fixed_threshold_counts.png)

**中文图注。** a，110张真实异常的TP／FN；b，同一894张真实正常的TN／FP。A03=73／37／50／844、B03=72／38／22／872；A04=81／29／54／840、B04=84／26／36／858；A05=73／37／62／832、B05=76／34／51／843，顺序均为TP／FN／FP／TN。三对FP分别减少28／18／11张，FN变化+1／−3／−3张；03存在漏检代价，不能称全面改善。每模型分别使用原Q200的q0.95／linear阈值，原生拼回概率最大值严格大于τ判异常；没有根据E调阈值。Q同池FP不是独立验证。本图为整图决策计数，不是像素分割阈值评价；其含义不同于AP或AUROC。

## 图3：固定四例的完整定位对照（三页）

固定顺序为20000（ID首个正常）、20042（ID首个异常）、20004（六模型FP并集ID最小者）、20068（六模型FN并集ID最小者）。所有选择及原因直接复用metrics.selected_examples，不重新挑图；FP／FN并集选例不意味着每个模型对此例都误报／漏检。

![初始化03](fig03_localization_seed03.png)

![初始化04](fig03_localization_seed04.png)

![初始化05](fig03_localization_seed05.png)

**中文图注。** 三页分别为初始化2026101903／04／05，每页四行按上述固定顺序，四列分别为原始RGB、规范GT、A概率、B概率。白色GT为缺陷，黑色为背景；仅读取已封存规范标签。所有概率图统一magma色图与[0,1]线性范围，显示完整原生画布，不做局部裁剪、平滑、阈值化、对比度增强或逐图归一化。RGB沿用原始uint8显示；图像宽高比例保留，版面缩放为最近邻。概率图上方s为原缓存最大值，TP／FN／FP／TN直接引用冻结FP/FN列表与规范标签元信息；该标记是整图决策，并非分割质量。阈值显示在页脚。四例跨页重复仅为比较六模型，不增加图像分母；不根据定位观感推断可控生成逼真度。

## 共同分母、方法边界与复用

所有指标使用同一1004张唯一真实E图：110异常／894正常；每模型146411528个原生像素，其中486461阳性。六模型不构成6024张独立评价图，正常分母也不累加。全局像素AP、整图AUROC与固定阈值计数均直接复制冻结metrics.json；绘图不重算任何指标、实际分位数或像素AP。A与B不仅合成曝光不同，正常／异常曝光组成也不同，不能把差异归因为外观逼真或控制能力本身。

结果来自KolektorSDD2单产品域、固定5张真实异常与固定划分、3个初始化。平均正增益但不稳定；没有总体显著性结论，也不能外推真实显示屏、Mura或其他产品。原始RGB只读取冻结四例；原GT文件、其他RGB、Q／U均未读取，规范GT只读四份既有缓存。全六组指标核验及封存引用已核对，本轮只复核用于绘图的四例24份概率／6份scores／4份规范GT与RGB身份，不重复扫描47.6GiB精确AP表。

PNG为220dpi预览；单页PDF／SVG为矢量文字与线条，TIFF为600dpi LZW，完整画布183mm宽。图3单页高252mm，另有三页合并PDF。PDF／SVG内图像的版面重采样不增加原生信息。CSV是绘图所用冻结数值及样例原因的源数据。source_identity.json保存读取来源hash，complete.json保存最终图件身份；visual_verification.json说明图件核验范围。未新增模型操作、训练、生成、重校准或实验结论。
'''
    text=text.replace('阈值显示在页脚。','阈值显示在页脚。低响应在统一[0,1]范围内可能近黑，但较低整图阈值仍可产生FP；不为突出差异而拉伸色阶。')
    (OUT/'第7天真实工业缺陷图件与中文图注.md').write_text(text,encoding='utf-8')


def generate():
    if (OUT/'complete.json').exists():
        return verify()
    require(not OUT.exists(), 'Partial/unknown figure directory retained; inspect before regenerating')
    m,images,masks,probs,scores,thresholds,statuses=load_sources()
    OUT.mkdir()
    style()
    fig_ap(m);fig_counts(m);fig_examples(m,images,masks,probs,scores,thresholds,statuses)
    csv_sources(m,scores,thresholds,statuses);captions(m)
    SOURCES[relative(Path(__file__))]=sha(Path(__file__).read_bytes())
    write_json('source_identity.json',{'schema':'fewshot-real-figures-sources-v1','sources':SOURCES,
        'selected_examples':m['selected_examples'],'model_order':list(MODELS),
        'RGB_decodes':4,'normalized_GT_arrays':4,'probability_arrays':24,'score_caches':6,
        'AP_AUROC_threshold_recomputed':False,'new_model_operations':0,'Torch_imported':'torch' in sys.modules,
        'large_AP_tables_rehashed':False,'source_evidence_scope':'prior full-six verification + current selected cached files'})
    require('torch' not in sys.modules, 'Unexpected Torch import')
    return {'status':'rendered_for_visual_inspection_not_yet_sealed','directory':str(OUT),'sources':len(SOURCES)}


def verify():
    seal=js(OUT/'complete.json')
    expected=set(seal['files'])|{'complete.json'}
    require({p.name for p in OUT.iterdir()}==expected,'Unexpected/missing figure output files')
    for name,digest in seal['files'].items():
        require(sha((OUT/name).read_bytes())==digest,f'Figure artifact changed: {name}')
    ident=json.loads((OUT/'source_identity.json').read_text(encoding='utf-8'))
    for name,digest in ident['sources'].items():
        require(sha((ROOT/name).read_bytes())==digest,f'Consumed frozen source changed: {name}')
    require('torch' not in sys.modules,'Unexpected Torch import')
    return {'verified':True,'figure_outputs':len(seal['files']),'consumed_sources':len(ident['sources']),
            'RGB_decodes':0,'arrays_loaded':0,'AP_AUROC_recomputed':False,'model_operations':0,'Torch_imported':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','verify'])
    args=parser.parse_args()
    sys.addaudithook(guard)
    print(json.dumps(generate() if args.command=='generate' else verify(),ensure_ascii=False,indent=2))
