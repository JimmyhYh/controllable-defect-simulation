"""Final Python figures from frozen paired metrics and exactly eight RGB cases.

No model, inference, metric computation, recalibration or sample selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results')
METRICS = BASE / 'real_metrics_v3'
PREP = BASE / 'real_metrics_preparation_v3'
E_PREP = BASE / 'real_scoring_preparation_v1'
PROOF = ROOT / 'results/pretrained_synthetic_pairing/ksdd2_r5_v1/real_metrics_verification_v3/all_six_v1'
DATA = ROOT / 'data/fewshot_synthetic/ksdd2_v1'
OUT = BASE / 'figures_v1'
IDS = ('20000', '20018', '20042', '20068', '20172', '20881', '20027', '20008')
ORDER = tuple(f'{group}_{seed}' for seed in (2026102301, 2026102302, 2026102303) for group in 'AB')
ANCHOR = '0af2e41d3ab371fe5988c40770d350a480aaa776c3c2f5cf5d95b5ebcec27bb1'
PROOF_ANCHOR = 'dcca6cb1a7b43dc5d653769908b232638485252afccab54cc456be390af5436a'
COLOR = {'A':'#66788A', 'B':'#287E9A', 'negative':'#AD603B', 'correct':'#90B1CB', 'error':'#C2825C'}
MM = 1 / 25.4
SOURCES = {}
READ_COUNTS = {'RGB_byte_reads':0, 'RGB_decodes':0, 'normalized_GT_reads':0, 'probability_reads':0}
QA = []
ALLOWED = set()
_GUARDED = False
SHORT_REASONS = {
    'A01_first_TN':'A01首个TN', 'A01_first_FP':'A01首个FP',
    'A01_first_TP':'A01首个TP', 'A01_first_FN':'A01首个FN',
    'first_anomaly_A_FN_to_B_TP':'第一对：异常 A漏检→B检出',
    'first_anomaly_A_TP_to_B_FN':'第一对：异常 A检出→B漏检',
    'first_normal_A_FP_to_B_TN':'第一对：正常 A误报→B正确',
    'first_normal_A_TN_to_B_FP':'第一对：正常 A正确→B误报',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message + '; preserve state, no automatic replacement')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected=None):
    path = Path(path).absolute()
    data = path.read_bytes()
    digest = sha_bytes(data)
    require(expected is None or digest == expected, 'Frozen source hash differs: ' + str(path))
    SOURCES[str(path)] = digest
    return data


def js(path, expected=None):
    return json.loads(read(path, expected).decode('utf-8'))


def sealed(folder, name, complete):
    return js(folder / name, complete['files'][name])


def install_guard():
    global _GUARDED
    require(not _GUARDED, 'One figure operation per process')
    blocked = {'torch','torchvision','segmentation_models_pytorch','timm','safetensors',
               'sklearn','scipy','diffusers','transformers'}
    require(not any(name.split('.')[0] in blocked for name in sys.modules), 'Forbidden model/metric package already loaded')
    image_suffix = {'.png','.jpg','.jpeg','.tiff','.tif','.bmp'}
    def audit(event, args):
        if event == 'import':
            name = args[0]
            if name.split('.')[0] in blocked or name.startswith(('fewshot_synthetic.metrics','pretrained_synthetic_pairing.real_metrics','pretrained_segmentation.model')):
                raise RuntimeError('Model/metric import forbidden in figure operation')
        if event in {'socket.connect','subprocess.Popen','os.system','pickle.find_class'}:
            raise RuntimeError('Network/subprocess/pickle execution forbidden')
        if event in {'os.remove','os.rename','os.rmdir','os.symlink','os.link'}:
            raise RuntimeError('Delete/move/link forbidden in frozen-result figure operation')
        if event == 'os.mkdir':
            p = Path(args[0]).absolute()
            require(p == OUT or p.is_relative_to(OUT), 'Mkdir outside figure result')
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if str(p).lower() in {os.devnull.lower(), '\\\\.\\nul'}:
            return
        mode, flags = args[1] or '', args[2] or 0
        writing = any(x in str(mode) for x in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            require(p.is_relative_to(OUT), 'Write outside independent figure result')
            return
        if p.is_relative_to(ROOT / 'data'):
            require(p in ALLOWED, 'Outside-eight RGB or any original GT forbidden')
        if p.suffix.lower() == '.npy':
            require(p in ALLOWED, 'Outside-eight GT/probability payload forbidden')
        if p.suffix.lower() in {'.npz','.pt','.pth','.safetensors','.pkl','.pickle','.uint32'}:
            raise RuntimeError('Model/archive/exact-table payload forbidden during figure production')
        if p.suffix.lower() in image_suffix and p.is_relative_to(ROOT / 'results'):
            require(p.is_relative_to(OUT), 'Old image-result reading forbidden')
    sys.addaudithook(audit)
    _GUARDED = True


def no_redirect(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        if p.exists():
            require(not p.is_symlink() and not (getattr(p.lstat(), 'st_file_attributes', 0) & 1024), 'Reparse point forbidden')
    return path


def check_tree(folder, names):
    no_redirect(folder)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    dirs = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_dir()}
    require(actual == set(names) and not dirs, 'Unknown/missing figure or proof members')


def metadata():
    proof_seal = js(PROOF / 'complete.json', PROOF_ANCHOR)
    check_tree(PROOF, set(proof_seal['files']) | {'complete.json'})
    evidence = sealed(PROOF, 'verification.json', proof_seal)
    for name in proof_seal['files']:
        if name != 'verification.json':
            read(PROOF / name, proof_seal['files'][name])
    require(evidence['verified'] is True and evidence['source_result_complete_sha256'] == ANCHOR,
            'Full six-model metric evidence missing')
    seal = js(METRICS / 'complete.json', ANCHOR)
    metrics = sealed(METRICS, 'metrics.json', seal)
    require(SOURCES[str((METRICS/'metrics.json').absolute())] == evidence['source_metrics_sha256'], 'Metrics/evidence mismatch')
    labels = sealed(METRICS, 'labels/summary.json', seal)
    require(SOURCES[str((METRICS/'labels/summary.json').absolute())] == evidence['source_GT_summary_sha256'], 'GT metadata evidence mismatch')
    prep_seal = js(PREP / 'complete.json', evidence['preparation_complete_sha256'])
    refs = sealed(PREP, 'models.json', prep_seal)
    es = js(E_PREP / 'complete.json', '073638e552de5c08dec60361f85b0d6d1628cffde4e6ba0b53396e0b8179e0bd')
    req = sealed(E_PREP, 'requests.json', es)
    require(tuple(x['model'] for x in metrics['models']) == ORDER == tuple(x['model'] for x in evidence['entries']), 'Model order mismatch')
    require(tuple(x['label'] for x in refs) == ORDER, 'Reference model order mismatch')
    require(tuple(x['id'] for x in metrics['selected_examples']['examples']) == IDS
            and metrics['selected_examples'] == evidence['selected_examples'], 'Frozen sample selection changed')
    require(metrics['paired'] == evidence['paired'], 'Frozen paired summary changed')
    expected = {'unique_E_images':1004,'native_pixels_per_model':146411528,
                'actual_positive_pixels_per_model':486461,'actual_anomaly_images':110,'actual_normal_images':894}
    for key, value in expected.items():
        require(metrics[key] == value, 'Frozen denominator mismatch')
    scope = evidence['verification_scope']
    require(scope['unique_images'] == 1004 and scope['actual_anomaly_images'] == 110
            and scope['actual_normal_images'] == 894 and not scope['AP_AUROC_recomputed'], 'Verification denominator/scope mismatch')
    require(len(req) == len(labels['rows']) == 1004, 'Request/GT metadata count mismatch')
    requests = {x['id']:x for x in req}
    masks = {x['request']['id']:x for x in labels['rows']}
    scores, score_seals = {}, {}
    for model, ref, ev in zip(metrics['models'], refs, evidence['entries'], strict=True):
        label = model['model']
        require(model['pixel_AP'] == ev['pixel_AP'] and model['image_metrics'] == ev['image_metrics'], 'Actual model metric evidence mismatch')
        require(ref['final_weight_sha256'] == ev['final_weight_sha256']
                and ref['threshold'] == ev['original_Q_tau'] == model['image_metrics']['threshold']
                and ref['threshold_sha256'] == ev['original_Q_tau_sha256'], 'Weight/tau reference differs')
        folder = BASE / 'real_evaluation_v1' / label
        require(Path(ref['result_root']).absolute() == folder.absolute(), 'Unexpected E result root')
        sc = js(folder/'complete.json', ref['result_complete_sha256'])
        score = sealed(folder, 'scores.json', sc)
        require(SOURCES[str((folder/'scores.json').absolute())] == ref['scores_sha256'], 'Scores hash mismatch')
        require(score['images_scored'] == len(score['items']) == 1004
                and score['original_threshold'] == ref['threshold'], 'Incomplete E metadata')
        score_rows = {}
        for request, row in zip(req, score['items'], strict=True):
            require(row['index'] == request['index'] and row['id'] == request['id']
                    and row['source_image_sha256'] == request['image_sha256'], 'Score/request order differs')
            score_rows[row['id']] = row
        scores[label], score_seals[label] = score_rows, sc
    for ident in IDS:
        request = requests[ident]
        ALLOWED.add((DATA / request['image']).absolute())
        ALLOWED.add((METRICS / masks[ident]['cache']).absolute())
        for label in ORDER:
            ALLOWED.add((BASE/'real_evaluation_v1'/label/scores[label][ident]['probability']).absolute())
    return metrics, labels, requests, masks, scores, score_seals


def arrays(ctx):
    metrics, _, requests, masks_meta, scores, score_seals = ctx
    images, masks, probs, statuses = {}, {}, {}, {}
    for ident in IDS:
        request, row = requests[ident], masks_meta[ident]
        require(row['request']['index'] == request['index'] and row['request']['native_shape'] == request['native_shape'], 'GT/request mismatch')
        labelname = row['cache']
        seal = json.loads((METRICS/'complete.json').read_text(encoding='utf-8'))
        require(seal['files'][labelname] == row['cache_sha256'], 'GT hash metadata mismatch')
        gt = np.load(io.BytesIO(read(METRICS/labelname, row['cache_sha256'])), allow_pickle=False)
        READ_COUNTS['normalized_GT_reads'] += 1
        require(gt.dtype == np.uint8 and list(gt.shape) == request['native_shape'] and np.all((gt==0)|(gt==1)), 'Invalid normalized GT')
        require(bool(gt.any()) == row['anomaly'], 'GT anomaly metadata mismatch')
        raw = read(DATA/request['image'], request['image_sha256'])
        READ_COUNTS['RGB_byte_reads'] += 1
        with Image.open(io.BytesIO(raw)) as im:
            require(im.mode in {'RGB','L'} and im.size == (request['width'],request['height']), 'Raw image mode/shape mismatch')
            images[ident] = np.asarray(im.convert('RGB')).copy()
        READ_COUNTS['RGB_decodes'] += 1
        masks[ident] = gt
    for model in metrics['models']:
        label = model['model']; probs[label], statuses[label] = {}, {}
        for ident in IDS:
            row, request = scores[label][ident], requests[ident]
            require(score_seals[label]['files'][row['probability']] == row['probability_sha256'], 'Probability hash metadata mismatch')
            arr = np.load(io.BytesIO(read(BASE/'real_evaluation_v1'/label/row['probability'], row['probability_sha256'])), allow_pickle=False)
            READ_COUNTS['probability_reads'] += 1
            require(arr.dtype == np.float32 and list(arr.shape) == request['native_shape']
                    and np.isfinite(arr).all() and np.all((arr>=0)&(arr<=1)), 'Probability shape/range mismatch')
            require(float(arr.max()) == row['score'], 'Saved native maximum mismatch')
            im = model['image_metrics']; truth = masks_meta[ident]['anomaly']
            tag = ('FN' if ident in im['FN_ids'] else 'TP') if truth else ('FP' if ident in im['FP_ids'] else 'TN')
            require((row['score'] > im['threshold']) == (tag in {'TP','FP'}), 'Saved decision/score mismatch')
            probs[label][ident], statuses[label][ident] = arr, tag
    return images, masks, probs, statuses


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'axes.axisbelow':True,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})


def heading(fig, title, subtitle):
    fig.text(.05,.962,title,fontsize=9,fontweight='bold',va='top')
    fig.text(.05,.925,subtitle,fontsize=6.7,va='top',color='#454C55')


def panel(ax, letter, title):
    ax.set_title(title,loc='left',pad=9,fontweight='bold')
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-20,9),
                textcoords='offset points',fontweight='bold',fontsize=9)
    ax.yaxis.grid(True,color='#E5E9ED',lw=.5)


def metric_figure(metrics):
    fig = plt.figure(figsize=(183*MM,152*MM))
    heading(fig,'真实定位：平均AP小幅正增益，三对初始化方向不一致','A：纯真实异常重复曝光；B：部分真实重复槽替换为固定可控合成')
    gs = fig.add_gridspec(2,2,left=.10,right=.965,bottom=.19,top=.81,hspace=.95,wspace=.48,height_ratios=[1, .85])
    ax = fig.add_subplot(gs[0,0]); panel(ax,'a','全局像素AP（定位排序）')
    for i,pair in enumerate(metrics['paired']['pairs']):
        values=[100*pair['AP_A'],100*pair['AP_B']]
        ax.plot([i-.14,i+.14],values,color='#B5BDC4',lw=1)
        for j,group in enumerate('AB'):
            x,y=i+(-.14 if group=='A' else .14),values[j]
            ax.scatter(x,y,s=24,marker='o' if group=='A' else 's',color=COLOR[group],zorder=3)
            ax.annotate(f'{y:.2f}',(x,y),xytext=(-3,-12) if group=='A' else (3,7),textcoords='offset points',
                        ha='right' if group=='A' else 'left',fontsize=6.3)
    ax.set(xlim=(-.65,2.65),ylim=(0,60),ylabel='像素AP (%)',xticks=[0,1,2],xticklabels=['01','02','03'],xlabel='配对初始化')
    ax = fig.add_subplot(gs[0,1]); panel(ax,'b','B−A：像素AP差')
    for i,pair in enumerate(metrics['paired']['pairs']):
        y=pair['delta_AP_percentage_points']; color=COLOR['B'] if y>0 else COLOR['negative']
        ax.plot([i,i],[0,y],color=color,lw=1.5); ax.scatter(i,y,s=25,color=color,zorder=3)
        ax.annotate(f'{y:+.4f}',(i,y),xytext=(0,7 if y>0 else -13),textcoords='offset points',ha='center',fontsize=6.3)
    mean,sd=metrics['paired']['mean_delta_AP_percentage_points'],metrics['paired']['sample_SD_delta_AP_percentage_points']
    ax.axhline(0,color='#818991',lw=.7)
    ax.errorbar(3.35,mean,yerr=sd,fmt='D',color='#424950',ms=4,capsize=3,lw=1)
    ax.text(3.35,3.25,f'{mean:+.4f}\nSD {sd:.4f}',ha='center',fontsize=6.1,va='top')
    ax.set(xlim=(-.5,4.05),ylim=(-1.4,3.8),ylabel='百分点',xticks=[0,1,2,3.35],xticklabels=['01','02','03','均值±SD'])
    ax = fig.add_subplot(gs[1,:]); panel(ax,'c','整图AUROC（整图排序，与像素定位分开解释）')
    positions=[]; values=[]; colors=[]; names=[]
    for i,model in enumerate(metrics['models']):
        positions.append(i//2*2+i%2*.55)
        values.append(100*model['image_metrics']['AUROC']); colors.append(COLOR[model['model'][0]])
        names.append(model['model'][0]+model['model'][-2:])
    ax.bar(positions,values,width=.4,color=colors)
    for x,y in zip(positions,values): ax.text(x,y+2,f'{y:.2f}',ha='center',fontsize=6.5)
    ax.set(ylim=(0,110),ylabel='AUROC (%)',xticks=positions,xticklabels=names)
    fig.legend(handles=[Line2D([0],[0],marker='o' if k=='A' else 's',linestyle='',color=COLOR[k],label=k) for k in 'AB'],
               loc='lower left',bbox_to_anchor=(.075,.105),ncol=2,frameon=False)
    fig.text(.30,.128,'01定位AP负差；03整图AUROC下降。误差线为样本SD，不是置信区间。',fontsize=6.4)
    fig.text(.05,.073,'同一1004唯一图：110异常 / 894正常；每模型146,411,528原生像素 / 486,461阳性。',fontsize=6.4)
    fig.text(.05,.038,'n=3为同一数据上的配对初始化；E历史已见。只作探索性训练组成证据，不作总体显著性检验。',fontsize=6.3)
    return fig


def count_figure(metrics):
    fig,axes=plt.subplots(1,2,figsize=(183*MM,132*MM))
    fig.subplots_adjust(left=.085,right=.97,top=.78,bottom=.28,wspace=.35)
    heading(fig,'固定原Q阈值：三对漏检减少，但第三对误报增加','各模型分别沿用自己的原Q200阈值；保存的原生最大概率严格大于τ判异常')
    names=[x['model'][0]+x['model'][-2:] for x in metrics['models']]
    for ax,correct,error,denom,letter,title in [(axes[0],'TP','FN',110,'a','异常图：检出与漏检'),(axes[1],'TN','FP',894,'b','正常图：正确与误报')]:
        panel(ax,letter,title)
        a=np.asarray([x['image_metrics'][correct] for x in metrics['models']])
        b=np.asarray([x['image_metrics'][error] for x in metrics['models']])
        ax.bar(range(6),a,color=COLOR['correct'],width=.65)
        ax.bar(range(6),b,bottom=a,color=COLOR['error'],width=.65)
        for i,(ca,er) in enumerate(zip(a,b)):
            ax.text(i,ca/2,str(ca),ha='center',va='center',fontsize=6.5,color='#25323D')
            ax.text(i,denom+denom*.035,f'{er}',ha='center',va='bottom',fontsize=6.5,color='#7A452E')
        ax.set(ylim=(0,denom*1.18),xticks=range(6),xticklabels=names,ylabel=f'图数（每模型分母{denom}）')
        ax.legend(handles=[Patch(color=COLOR['correct'],label=correct),Patch(color=COLOR['error'],label=error+'（上方数字）')],
                  loc='upper left',bbox_to_anchor=(0,-.13),ncol=2,frameon=False)
    fig.text(.075,.16,'B−A漏检数：01 −7；02 −11；03 −2。',fontsize=7)
    fig.text(.53,.16,'B−A误报数：01 −1；02 −17；03 +9。',fontsize=7,color=COLOR['negative'])
    fig.text(.075,.105,'三对均减少漏检；03同时增加9张误报。阈值计数不能代替像素AP或整图AUROC。',fontsize=6.5)
    fig.text(.075,.065,'六模型重复使用同一110异常 / 894正常；不按模型次数重复累计独立样本数。',fontsize=6.3)
    fig.text(.075,.03,'E历史已见；固定R5 / 划分 / 合成池，n=3初始化。图中未调整阈值，未重算指标。',fontsize=6.3)
    return fig


def image_figure(page, ctx, loaded):
    metrics,_,_,_,scores,_=ctx
    images,masks,probs,statuses=loaded
    chosen=IDS[page*2:page*2+2]
    reasons={x['id']:x['reasons'] for x in metrics['selected_examples']['examples']}
    fig=plt.figure(figsize=(183*MM,210*MM))
    heading(fig,f'固定真实样例｜全部六模型定位对照（{page+1}/4）','样例按冻结规则原序展示；所有热图统一[0,1]，完整原生画布，不拉伸原图对比度')
    outer=fig.add_gridspec(2,1,left=.045,right=.97,bottom=.18,top=.875,hspace=.12)
    for j,ident in enumerate(chosen):
        sub=outer[j].subgridspec(3,8,height_ratios=[.10,.16,.74],hspace=.02,wspace=.12)
        case_ax=fig.add_subplot(sub[0,:]); case_ax.axis('off')
        why='；'.join(SHORT_REASONS[r] for r in reasons[ident])
        case_ax.text(0,.5,f'{ident}   |   {why}',va='center',fontsize=7.2,fontweight='bold')
        items=[('原始RGB',images[ident],None),('规范GT\n白色=缺陷',masks[ident],'gray')]
        for model in metrics['models']:
            label=model['model']; short=label[0]+label[-2:]
            text=f'{short}  {statuses[label][ident]}\nmax={scores[label][ident]["score"]:.2e}\nτ={model["image_metrics"]["threshold"]:.2e}'
            items.append((text,probs[label][ident],'magma'))
        for col,(title,data,cmap) in enumerate(items):
            title_ax=fig.add_subplot(sub[1,col]); title_ax.axis('off')
            title_ax.text(.5,.5,title,ha='center',va='center',fontsize=5.6,linespacing=1.25)
            ax=fig.add_subplot(sub[2,col]); ax.set_axis_off()
            if cmap:
                ax.imshow(data,cmap=cmap,vmin=0,vmax=1,interpolation='nearest',aspect='equal')
            else:
                ax.imshow(data,interpolation='nearest',aspect='equal')
            ax.set_gid(f'{ident}_{col}')
    cax=fig.add_axes([.46,.119,.48,.012])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap='magma'),cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1])
    cax.tick_params(labelsize=6,pad=2,length=2)
    fig.text(.045,.123,'原始异常softmax概率',fontsize=6.5,va='center')
    fig.text(.045,.079,'max为保存的原生最大值；各自原τ严格大于判异常。TP/FN/FP/TN为整图判定，不能代替轮廓质量。',fontsize=5.9)
    fig.text(.045,.052,'8个固定例来自同一1004唯一图（110异常 / 894正常）；各页/六模型重复展示不增加分母。',fontsize=6.0)
    fig.text(.045,.025,'每模型146,411,528像素 / 486,461阳性。未裁剪、增强、平滑或逐图归一化；低响应可能近黑。',fontsize=5.9)
    return fig


def layout_check(fig, name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); w,h=fig.canvas.get_width_height()
    issues=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip(): continue
        box=text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > w+1 or box.y1 > h+1):
            issues.append(text.get_text())
    require(not issues,'Text outside final canvas: '+repr(issues))
    QA.append({'figure':name,'canvas_mm':[round(x/MM,2) for x in fig.get_size_inches()],
               'text_outside_canvas':issues,'font':'Microsoft YaHei / Arial','SVG_text':'editable','PDF_fonttype':42})


def export(fig,name,book):
    fig.savefig(OUT/(name+'.png'),dpi=220)
    fig.savefig(OUT/(name+'.svg'),dpi=600)
    fig.savefig(OUT/(name+'.pdf'),dpi=600)
    fig.savefig(OUT/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    book.savefig(fig,dpi=600)
    plt.close(fig)


def save_json(name,obj):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def sources_data(metrics):
    with (OUT/'source_data_models.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['model','pixel_AP_fraction','image_AUROC_fraction','tau','TP','FN','FP','TN','anomaly_denominator','normal_denominator','native_pixels','positive_pixels'])
        for model in metrics['models']:
            im=model['image_metrics']; writer.writerow([model['model'],model['pixel_AP'],im['AUROC'],im['threshold'],im['TP'],im['FN'],im['FP'],im['TN'],110,894,146411528,486461])
    with (OUT/'source_data_pairs.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=list(metrics['paired']['pairs'][0]); writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(metrics['paired']['pairs'])


def captions(metrics):
    lines=['# 第7阶段图件与中文图注','',
    '图件结论：固定设置下，含合成B相对纯真实A的像素AP平均小幅增加，但初始化间方向不一致；三对漏检减少，第三对误报增加。图件只整理冻结结果，不重算AP/AUROC、训练、前向或校准。','',
    '## 图1｜定位增益不一致，整图排序也并非全部改善','',
    'a，展示三对最终10000步模型的全部全局像素AP；灰色圆点为A，蓝色方点为B。b，B−A依次−0.42824234、+2.11278608、+0.64665050个百分点；均值+0.77706475，样本标准差1.27552431（ddof=1）。误差线为同一固定数据上3个配对初始化差的样本SD，不是置信区间，不是3个独立数据集，不作总体显著性检验。01负差完整保留。c，六模型整图AUROC单独展示，03由86.63056742%降至85.39251576%。AP衡量全部原生像素的排序，AUROC衡量整图最大分数的排序，不能互相替代。','',
    '三对A/B AP分别45.39331802/44.96507568%、38.83696935/40.94975543%、35.71167980/36.35833030%。所有数值直接引用冻结metrics及完整核验证据。','',
    '## 图2｜固定原Q阈值下漏检与误报的取舍','',
    'a，每模型在同一110异常图上展示TP/FN：A01=77/33，B01=84/26，A02=77/33，B02=88/22，A03=77/33，B03=79/31。b，每模型在同一894正常图上展示TN/FP：A01=853/41，B01=854/40，A02=828/66，B02=845/49，A03=851/43，B03=842/52。B−A漏检变化−7/−11/−2，误报变化−1/−17/+9；03误报代价完整保留。每模型使用自己的原Q200 q0.95/linear阈值，以原生最大异常概率严格大于τ判异常，未改阈值。计数比较是共同校准规则下的管线结果，不可全归因于像素定位。','',
    '## 图3—6｜冻结8例的全六模型定位对照','',
    '四页按固定顺序各展示两个样例，八列为原RGB、规范GT、A01/B01/A02/B02/A03/B03。所有热图使用同一magma色图和[0,1]线性范围，GT用固定黑白二值，原图保持原始uint8强度；完整原生画布只作等比例版面缩放（最近邻），无局部裁剪、增强、平滑、阈值化或逐图归一化。低响应在统一色阶下可能近黑，仍可能超过很低的原阈值，不能为突出差异改显示范围。max为已保存原生最大概率，τ为各自原Q阈值，页内数字仅作科学计数法显示舍入；精确值见源数据。标签TP/FN/FP/TN是冻结整图判定，不保证轮廓准确。没有像素到物理长度标定，不添加虚构尺度尺。','',
    '以下保留冻结选择原因，全部由第一对A01/B01的预定规则决定，不代表三对或总体根因，不重选更好样例：','']
    for ex in metrics['selected_examples']['examples']:
        lines.append('- '+ex['id']+'：'+'；'.join(SHORT_REASONS[x] for x in ex['reasons'])+'。')
    lines += ['', '## 共同分母和解释边界','',
    '同一1004唯一图，实际110异常/894正常；每模型146,411,528原生像素/486,461阳性。六模型、各图和分页重复展示不增加独立图像数，也不累计正常分母。固定R5/N1885、Q只校准、U不用，E历史已见；同一划分、合成池、单类别、未知采集分组和3初始化，仅作探索性组成证据。A正常5000/真实异常5000曝光，B正常5000/真实异常2500/合成2500，虽匹配更新与正负总曝光，真实异常重复与背景组成仍变化，不能声称纯合成外观单因素因果。不得由本图宣称稳定总体收益、生成外观逼真或屏幕效果。','',
    '## 导出和复用','',
    '仅Python/matplotlib，统一183毫米宽；SVG保留文字、PDF采用TrueType，TIFF为600dpi LZW，PNG为220dpi预览，另提供六页合并PDF。source_data_models.csv和source_data_pairs.csv是从冻结JSON直接导出的图件源数据，非新评价。完整来源hash、图件配置及固定样例精确max/tau见sources.json和figure_contract.json；visual_QA.json记录排版检查。','',
    '已实现入口：python -B -X utf8 -m pretrained_synthetic_pairing.figures render；完整同身份目录只verify复用，未知或部分输出保留停止。所有训练仍由用户手动启动。','']
    (OUT/'中文图注与图件说明.md').write_text('\n'.join(lines),encoding='utf-8')


def verify():
    install_guard()
    complete=js(OUT/'complete.json')
    require(complete['implementation_sha256']==sha_bytes(Path(__file__).read_bytes()),'Figure implementation changed')
    check_tree(OUT,set(complete['files'])|{'complete.json'})
    for name,digest in complete['files'].items(): read(OUT/name,digest)
    sources=js(OUT/'sources.json')
    # Only re-read frozen JSON anchors here; no RGB or probability re-decoding.
    js(METRICS/'complete.json',ANCHOR); js(PROOF/'complete.json',PROOF_ANCHOR)
    require(sources['selected_IDs']==list(IDS),'Selected IDs changed')
    return {'complete_figures_verified':True,'figures':6,'image_pages':4,'new_RGB_decodes':0,'metrics_recomputed':False,'model_operations':0}


def render():
    if (OUT/'complete.json').exists(): return verify()
    require(not OUT.exists(),'Unknown/partial figure directory')
    install_guard()
    no_redirect(OUT)
    ctx=metadata()
    loaded=arrays(ctx)
    require(READ_COUNTS=={'RGB_byte_reads':8,'RGB_decodes':8,'normalized_GT_reads':8,'probability_reads':48},'Unexpected input access multiplicity')
    style()
    figures=[('fig01_AP_delta_AUROC',metric_figure(ctx[0])),('fig02_fixed_tau_counts',count_figure(ctx[0]))]
    figures.extend((f'fig{3+i:02d}_localization_page{i+1}',image_figure(i,ctx,loaded)) for i in range(4))
    for name,fig in figures: layout_check(fig,name)
    OUT.mkdir(exist_ok=False)
    sources_data(ctx[0]); captions(ctx[0])
    with PdfPages(OUT/'全部图件_六页.pdf') as book:
        for name,fig in figures: export(fig,name,book)
    frozen_sources=dict(SOURCES)
    sample_values={label:{ident:{'max':ctx[4][label][ident]['score'],'tau':next(m['image_metrics']['threshold'] for m in ctx[0]['models'] if m['model']==label),
                                      'decision':loaded[3][label][ident]} for ident in IDS} for label in ORDER}
    save_json('sources.json',{'metric_complete_sha256':ANCHOR,'verification_complete_sha256':PROOF_ANCHOR,
              'selected_IDs':list(IDS),'selected_examples':ctx[0]['selected_examples'],'source_files':frozen_sources,
              'read_counts':READ_COUNTS,'sample_exact_max_tau_and_decisions':sample_values})
    save_json('figure_contract.json',{'conclusion':'mean positive but inconsistent AP; FN decreases in all pairs, FP rises in third pair',
              'backend':'Python/matplotlib','archetypes':['quantitative grid','image plate + quant'],
              'panel_map':{'fig01':'all paired AP, B-A and saved mean/sampleSD, all-six AUROC','fig02':'all-six TP/FN and TN/FP',
                           'fig03-06':'all8 frozen examples, full six models, two examples per page'},
              'probability_range':[0,1],'RGB_contrast':'unchanged','crop':'none','resize_for_layout':'equal aspect, nearest',
              'error_bar':'sample SD of paired differences, not CI','n':'3 paired initializations on same data',
              'unique_images':1004,'anomalies':110,'normals':894,'native_pixels_per_model':146411528,'positive_pixels':486461,
              'actual_metrics_recomputed':False,'model_operations':0,'formats':['SVG','PDF','600dpi TIFF','220dpi PNG']})
    # Source preservation uses hashes of exactly the files already authorized/read.
    changed=[]
    preservation={'RGB_hash_only_reads':0,'GT_hash_only_reads':0,'probability_hash_only_reads':0}
    for path,digest in frozen_sources.items():
        if sha_bytes(Path(path).read_bytes())!=digest: changed.append(path)
        if Path(path).is_relative_to(DATA): preservation['RGB_hash_only_reads']+=1
        elif Path(path).suffix=='.npy' and Path(path).is_relative_to(METRICS): preservation['GT_hash_only_reads']+=1
        elif Path(path).suffix=='.npy': preservation['probability_hash_only_reads']+=1
    require(not changed,'Frozen source modified during rendering')
    qa={'layout':QA,'frozen_sources_unchanged':True,'probability_scale':[0,1],'source_read_counts_before_preservation_hash':READ_COUNTS,
        'preservation_hash_only_reads':preservation,
        'selected_samples_reselected':False,'AP_AUROC_recomputed':False,'Torch_imported':'torch' in sys.modules,
        'visual_review':'direct inspection of exported PNG pages is recorded in separate final figure verification; no further RGB decode required'}
    save_json('visual_QA.json',qa)
    files={p.name:sha_bytes(p.read_bytes()) for p in OUT.iterdir() if p.is_file()}
    save_json('complete.json',{'implementation_sha256':sha_bytes(Path(__file__).read_bytes()),'files':files,
                             'metric_complete_sha256':ANCHOR,'verification_complete_sha256':PROOF_ANCHOR})
    return {'figure_exports_complete':True,'figures':6,'image_pages':4,'formats':['SVG','PDF','TIFF','PNG'],
            'RGB_unique_IDs':list(IDS),'read_counts':READ_COUNTS,'metrics_recomputed':False,'model_operations':0,'output':str(OUT)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=['render','verify'])
    args=parser.parse_args()
    print(json.dumps({'render':render,'verify':verify}[args.command](),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
