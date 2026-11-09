"""Python-only figures from frozen shared-holdout caches; no models or regeneration."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as effects
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/controlled_generation/shared_holdout_v1'
OUTPUT=BASE/'figures_day10_v1'
RES_MAX=0.04000002145767212
COLORS={'G0':'#64748B','G1':'#267F82','FP':'#B87340','TN':'#DFE3E7'}
EXPECTED_METRICS='7f13bc32d46d9cf2c286fe99aae404264e0ecff130c973451056010b00e13438'
plt.rcParams.update({'pdf.fonttype':42,'font.size':7,'axes.linewidth':.6,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
    'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'savefig.facecolor':'white'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(relative):
    p=(BASE/relative).resolve()
    if not p.is_relative_to(BASE.resolve()):raise ValueError('Figure source escapes frozen cache')
    return p


def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source_data():
    if 'torch' in sys.modules:raise RuntimeError('Use fresh Python without Torch for figure work')
    names=['metrics.json','metrics_verification.json','metrics_config.json','metrics_started.json',
           'manifest.json','scoring_config.json','scoring/G0/scores.json','scoring/G1/scores.json']
    identity={name:sha(safe(name)) for name in names}
    m,v,cfg=read(safe('metrics.json')),read(safe('metrics_verification.json')),read(safe('metrics_config.json'))
    if identity['metrics.json']!=EXPECTED_METRICS or identity['metrics.json']!=v['metrics_sha256']:
        raise ValueError('Frozen metrics identity differs')
    if identity['metrics_config.json']!=v['metrics_config_sha256'] or read(safe('metrics_started.json'))['completed_metrics_sha256']!=identity['metrics.json']:
        raise ValueError('Metric completion/configuration differs')
    if identity['manifest.json']!=cfg['manifest_sha256'] or identity['scoring_config.json']!=cfg['scoring_config_sha256']:
        raise ValueError('Frozen input/scoring identity differs')
    scores={c:read(safe(f'scoring/{c}/scores.json')) for c in ('G0','G1')}
    for c in scores:
        if identity[f'scoring/{c}/scores.json']!=cfg['scores_sha256'][c]:raise ValueError('Saved score cache differs')
    manifest=read(safe('manifest.json'));requests={x['id']:x for x in manifest['anomaly']}
    score_by_id={c:{r['input']['id']:r for r in scores[c]['rows']} for c in scores}
    expected={}
    def add(identifier,reason):
        if identifier is not None:expected.setdefault(identifier,[]).append(reason)
    for s in ('a04','a02'):
        ex=m['models']['G0']['strengths'][s]['predeclared_example_ids']
        if ex['first_valid']!=m['models']['G1']['strengths'][s]['predeclared_example_ids']['first_valid']:raise ValueError('Shared first-example rule differs')
        for kind in ('mura_like','line_like'):
            for direction in ('bright','dark'):
                candidates=sorted(r['id'] for r in manifest['anomaly'] if r['quality']['valid'] and r['strength_id']==s and r['kind']==kind and r['polarity']==direction)
                if ex['first_valid'][kind][direction]!=(candidates[0] if candidates else None):raise ValueError('Not the first valid predeclared example')
                add(ex['first_valid'][kind][direction],f'{s}/{kind}/{direction}首有效')
    for c in ('G0','G1'):
        for s in ('a04','a02'):
            ex=m['models'][c]['strengths'][s]['predeclared_example_ids'];overall=m['models'][c]['strengths'][s]['overall']
            if ex['first_normal_FP']!=min((x['id'] for x in overall['FP_inputs']),default=None) or ex['first_anomaly_FN']!=min(overall['FN_ids'],default=None):
                raise ValueError('Error example rule changed')
            add(ex['first_normal_FP'],f'{c}/{s}首正常FP');add(ex['first_anomaly_FN'],f'{c}/{s}首异常FN')
    chosen=v['figure_sample_selection']
    if len(chosen)!=12 or list(expected)!=[x['id'] for x in chosen] or any(x['reasons']!=expected[x['id']] for x in chosen):
        raise ValueError('Fixed12 selection/reasons differ')
    loaded=[];background_cache={}
    def checked(path,digest):
        if sha(safe(path))!=digest:raise ValueError('Frozen selected array identity differs')
        identity[path]=digest
    for item in chosen:
        identifier=item['id'];normal=score_by_id['G0'][identifier]['input']['input_class']=='normal'
        if score_by_id['G0'][identifier]['input']!=score_by_id['G1'][identifier]['input']:raise ValueError('Model inputs differ')
        if item['input_npz']!=score_by_id['G0'][identifier]['input']['relative_npz']:raise ValueError('Selected input path differs')
        if normal:
            source=next(x for x in manifest['normal'] if x['request']['id']==identifier)
            b=source['request']['background_index'];amplitude=0.;shape='normal'
        else:
            source=requests[identifier];b=source['request']['background_index'];amplitude=source['request']['spec']['amplitude'];shape=source['request']['spec']['shape']
            if not source['quality']['valid']:raise ValueError('Invalid request selected')
        bg=manifest['normal'][b];bgpath=f"generation/backgrounds/{bg['request']['id']}.npz"
        checked(bgpath,bg['files_sha256'][bgpath])
        if b not in background_cache:
            with np.load(safe(bgpath),allow_pickle=False) as z:background_cache[b]=z['B']
        B=background_cache[b];checked(item['input_npz'],source['files_sha256'][item['input_npz']])
        if normal:
            with np.load(safe(item['input_npz']),allow_pickle=False) as z:mask=z['normal_GT']
            I=B;roi=None;D=np.max(np.abs(I-B),axis=-1)
            if mask.any():raise ValueError('Normal label is not zero')
        else:
            with np.load(safe(item['input_npz']),allow_pickle=False) as z:
                I,roi,mask,D=[z[k] for k in ('I','M_roi','M_eff','D')]
            if not np.array_equal(D,np.max(np.abs(I-B),axis=-1)) or not np.array_equal(mask,D>np.float32(1/255)):
                raise ValueError('Actual residual or label differs')
        if B.shape!=(256,256,3) or I.shape!=B.shape or B.dtype!=np.float32 or I.dtype!=np.float32 or min(B.min(),I.min())<0 or max(B.max(),I.max())>1:
            raise ValueError('Unexpected saved RGB')
        if D.max()>RES_MAX or not all(np.isfinite(a).all() for a in (B,I,D)):raise ValueError('Unexpected residual/range')
        maps={};annotations={}
        for c in ('G0','G1'):
            row=score_by_id[c][identifier];path=item['probability_files'][c]
            if path!=row['probability_file']:raise ValueError('Selected probability path differs')
            checked(path,row['probability_sha256']);maps[c]=np.load(safe(path),allow_pickle=False)
            if maps[c].dtype!=np.float32 or maps[c].shape!=(256,256) or not np.isfinite(maps[c]).all() or maps[c].min()<0 or maps[c].max()>1 or float(maps[c].max())!=row['score']:
                raise ValueError('Saved raw map/score differs')
            alarm=row['score']>m['models'][c]['threshold']
            status=('FP' if alarm else 'TN') if normal else ('TP' if alarm else 'FN')
            if normal:
                is_error=identifier in [x['id'] for x in m['models'][c]['strengths']['a04']['overall']['FP_inputs']]
            else:is_error=identifier in m['models'][c]['strengths'][source['strength_id']]['overall']['FN_ids']
            if is_error!=(status in ('FP','FN')):raise ValueError('Display decision inconsistent with frozen metrics')
            annotations[c]={'score':row['score'],'status':status}
        loaded.append({**item,'normal':normal,'B':B,'I':I,'roi':roi,'mask':mask,'D':D,'maps':maps,
            'annotations':annotations,'amplitude':amplitude,'shape':shape,'background_index':b})
    return m,v,loaded,identity


def export(fig,name,pdf_pages=None):
    fig.canvas.draw()
    fig.savefig(OUTPUT/f'{name}.svg',dpi=600)
    if pdf_pages is None:fig.savefig(OUTPUT/f'{name}.pdf',dpi=600)
    else:pdf_pages.savefig(fig,dpi=600)
    fig.savefig(OUTPUT/f'{name}.png',dpi=300)
    fig.savefig(OUTPUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})


def panel_title(ax,label,title):
    ax.set_title(title,loc='left',fontsize=7.5,pad=9)
    ax.text(-.13,1.09,label,transform=ax.transAxes,weight='bold',fontsize=9,va='baseline')


def paired_bars(ax,values,metric,denominator=False):
    x=np.array([0,1.35]);offset=.21
    for k,c in enumerate(('G0','G1')):
        h=[100*values[s][c][metric] for s in ('a04','a02')]
        bars=ax.bar(x+(-offset if k==0 else offset),h,width=.36,color=COLORS[c],zorder=3)
        for i,bar in enumerate(bars):
            q=values[('a04','a02')[i]][c]
            label=f"{q['TP']}/{q['valid_anomaly_images']}" if denominator else f'{h[i]:.2f}'
            ax.text(bar.get_x()+bar.get_width()/2,h[i]+2,label,ha='center',va='bottom',fontsize=5.9)
    ax.set(xlim=(-.7,2.05),ylim=(0,112),xticks=x,xticklabels=['a04','a02'],yticks=[0,25,50,75,100])
    ax.grid(axis='y',color='#E4E7EB',lw=.45,zorder=0);ax.set_axisbelow(True)


def metric_figure(m):
    fig=plt.figure(figsize=(183/25.4,165/25.4),facecolor='white')
    values={s:{c:m['models'][c]['strengths'][s]['overall'] for c in ('G0','G1')} for s in ('a04','a02')}
    fig.text(.06,.969,'Shared simulator-domain evaluation',fontsize=11,weight='bold')
    fig.text(.06,.937,'Frozen G0/G1 models; held-out normal backgrounds; one training seed',fontsize=7,color='#4D4D4D')
    handles=[Line2D([],[],color=COLORS[c],marker=('o' if c=='G0' else 'D'),lw=2,label=c) for c in ('G0','G1')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.97,.985),ncol=2,fontsize=7)
    a=fig.add_axes([.09,.57,.54,.28]);paired_bars(a,values,'pixel_AP');panel_title(a,'a','Overall exact global pixel AP')
    a.set_ylabel('Pixel AP (%)')
    a.set_xticklabels(['a04: +/-0.04\nG1-G0: +43.88 pp','a02: +/-0.02\nG1-G0: +81.22 pp'],fontsize=7)
    b=fig.add_axes([.755,.57,.20,.28]);panel_title(b,'b','Normal false alarms')
    for i,c in enumerate(('G0','G1')):
        q=values['a04'][c];y=1-i
        b.barh(y,q['FP'],color=COLORS['FP'],height=.35)
        b.barh(y,q['TN'],left=q['FP'],color=COLORS['TN'],height=.35)
        b.text(0,y+.25,f"FP {q['FP']}/20",fontsize=6,ha='left');b.text(20,y+.25,f"TN {q['TN']}/20",fontsize=6,ha='right')
    b.set(xlim=(0,20),ylim=(-.4,1.8),yticks=[1,0],yticklabels=['G0','G1'],xticks=[0,10,20],xlabel='Same 20 normal images')
    b.legend(handles=[Line2D([],[],color=COLORS[k],lw=4,label=k) for k in ('FP','TN')],fontsize=6,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.22))
    c=fig.add_axes([.09,.255,.235,.20]);paired_bars(c,values,'image_AUROC');panel_title(c,'c','Image discrimination');c.set_ylabel('Image AUROC (%)')
    d=fig.add_axes([.405,.255,.235,.20]);paired_bars(d,values,'recall',denominator=True);panel_title(d,'d','Fixed-threshold detection');d.set_ylabel('Recall (%)')
    e=fig.add_axes([.755,.255,.20,.20]);panel_title(e,'e','Predeclared type AP')
    for y,(s,kind) in enumerate((('a04','mura_like'),('a04','line_like'),('a02','mura_like'),('a02','line_like'))):
        v=[100*m['models'][method]['strengths'][s]['types'][kind]['pixel_AP'] for method in ('G0','G1')]
        e.plot(v,[3-y,3-y],color='#BDC4CB',lw=1,zorder=1)
        for k,method in enumerate(('G0','G1')):e.plot(v[k],3-y,marker=('o' if k==0 else 'D'),color=COLORS[method],ms=4,zorder=3)
    e.set(xlim=(0,105),ylim=(-.5,3.5),xticks=[0,50,100],yticks=[3,2,1,0],yticklabels=['a04 Mura','a04 Line','a02 Mura','a02 Line'],xlabel='Pixel AP (%)')
    e.grid(axis='x',color='#E4E7EB',lw=.45);e.set_axisbelow(True)
    notes=['Generation: 460/480 valid; 230/240 per strength. Each overall group: 230 anomalies + SAME 20 normals.',
        'Overall AP: 16,384,000 pixels, 219,208 positive. Each type: 115 anomalies + the same 20 normals.',
        '20 backgrounds; related variants; descriptive only (no CI/tests). G1-renderer domain; no real-screen claim.']
    for y,line in zip((.142,.112,.082),notes):fig.text(.06,y,line,fontsize=6.2,color='#4D4D4D')
    export(fig,'figure1_shared_metrics');return fig


def reason_short(item):
    parts=[]
    if any('首有效' in x for x in item['reasons']):parts.append('first valid')
    for c in ('G0','G1'):
        if any(x.startswith(c) and 'FP' in x for x in item['reasons']):parts.append(f'first {c} FP')
        if any(x.startswith(c) and 'FN' in x for x in item['reasons']):parts.append(f'first {c} FN')
    return '; '.join(parts)


def image_figure(items,page,pdf):
    fig=plt.figure(figsize=(183/25.4,165/25.4),facecolor='white')
    title=['Predeclared a04 examples','Predeclared a02 examples','Predeclared normal FP and anomaly FN'][page]
    fig.text(.012,.972,f'{chr(97+page)}  {title}',fontsize=10,weight='bold')
    fig.text(.012,.941,f'Page {page+1}/3 | Fixed rule, not best-case selection | Full evaluation: 230 anomalies/strength + same 20 normals',fontsize=6.1)
    headers=['Background B','Model input I','M_roi','M_eff / GT','Actual residual','G0 raw p','G1 raw p']
    width=.103;height=width*183/165
    for col,header in enumerate(headers):
        fig.text(.19+col*.112+width/2,.866,header,ha='center',fontsize=6.1,color=COLORS.get(('G0' if col==5 else 'G1' if col==6 else ''),'#272727'))
    residual_artist=None;prob_artist=None
    for r,item in enumerate(items):
        y=.718-r*.19
        label=f"{item['id']}\n"+("normal | no injection" if item['normal'] else f"{item['shape']} | A={item['amplitude']:+.2f}")+f"\n{reason_short(item)}"
        fig.text(.012,y+height/2,label,va='center',fontsize=6.0,linespacing=1.65)
        panels=[item['B'],item['I'],item['roi'],item['mask'],item['D'],item['maps']['G0'],item['maps']['G1']]
        for col,data in enumerate(panels):
            ax=fig.add_axes([.19+col*.112,y,width,height]);ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if data is None:
                ax.text(.5,.5,'N/A\nno request',ha='center',va='center',fontsize=6,color='#676767',transform=ax.transAxes)
            elif col<2:ax.imshow(data,interpolation='nearest',vmin=0,vmax=1)
            elif col<4:ax.imshow(data,cmap='gray',norm=Normalize(0,1),interpolation='nearest')
            elif col==4:residual_artist=ax.imshow(data,cmap='magma',norm=Normalize(0,RES_MAX),interpolation='nearest')
            else:prob_artist=ax.imshow(data,cmap='cividis',norm=Normalize(0,1),interpolation='nearest')
            if col==0 and r==0:
                line=ax.plot([12,76],[240,240],color='white',lw=2)[0];line.set_path_effects([effects.Stroke(linewidth=3,foreground='black'),effects.Normal()])
                text=ax.text(44,228,'64 px',color='white',fontsize=5,ha='center');text.set_path_effects([effects.withStroke(linewidth=1,foreground='black')])
            if col>=5:
                annotation=item['annotations'][('G0','G1')[col-5]];color='#A15E32' if annotation['status'] in ('FN','FP') else '#35414B'
                ax.set_xlabel(f"s={annotation['score']:.6f}\n{annotation['status']}",fontsize=5.6,color=color,labelpad=3)
    fig.text(.19,.08,'Binary masks: black=0, white=1\nNormal input: I=B; GT=0; no requested ROI',fontsize=5.8,va='center')
    ca=fig.add_axes([.618,.069,.103,.009]);bar=fig.colorbar(residual_artist,cax=ca,orientation='horizontal',ticks=[0,.02,.04]);bar.ax.set_xticklabels(['0','0.02','0.04']);bar.ax.tick_params(labelsize=5.5,length=2,pad=1);bar.set_label('max RGB |I-B|',fontsize=5.7,labelpad=1)
    ca=fig.add_axes([.775,.069,.19,.009]);bar=fig.colorbar(prob_artist,cax=ca,orientation='horizontal',ticks=[0,.5,1]);bar.ax.set_xticklabels(['0','0.5','1']);bar.ax.tick_params(labelsize=5.5,length=2,pad=1);bar.set_label('Raw anomaly probability (G0 and G1)',fontsize=5.7,labelpad=1)
    fig.text(.012,.018,'Full 256-grid frame; no crop/gamma adjustment. G1-renderer domain, not real defects. Thresholds remain frozen.',fontsize=5.8,color='#4D4D4D')
    export(fig,f'figure2{chr(97+page)}_localization',pdf_pages=pdf);return fig


def write_source_csv(m,loaded):
    fields=['strength','group','method','pixel_AP','image_AUROC','recall','TP','FN','FP','TN','normal_images','valid_anomaly_images','total_pixels','positive_pixels']
    with (OUTPUT/'metrics_source.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for s in ('a04','a02'):
            for group in ('overall','mura_like','line_like'):
                for c in ('G0','G1'):
                    strength=m['models'][c]['strengths'][s];value=strength['overall'] if group=='overall' else strength['types'][group]
                    writer.writerow({'strength':s,'group':group,'method':c,**{key:value[key] for key in fields[3:]}})
    fields=['id','reasons','normal','amplitude','background_index','requested_ROI_pixels','effective_pixels','residual_min','residual_median','residual_max','G0_score','G0_status','G1_score','G1_status']
    with (OUTPUT/'examples_source.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for x in loaded:
            values=x['D'][x['roi']] if x['roi'] is not None else x['D'].reshape(-1)
            writer.writerow({'id':x['id'],'reasons':' | '.join(x['reasons']),'normal':x['normal'],'amplitude':x['amplitude'],'background_index':x['background_index'],
                'requested_ROI_pixels':int(x['roi'].sum()) if x['roi'] is not None else 'NA','effective_pixels':int(x['mask'].sum()),
                'residual_min':float(values.min()),'residual_median':float(np.median(values)),'residual_max':float(values.max()),
                **{f'{c}_{key}':x['annotations'][c]['score' if key=='score' else 'status'] for c in ('G0','G1') for key in ('score','status')}})


def run(revise=False):
    m,v,loaded,source_identity=source_data()
    OUTPUT.mkdir(exist_ok=True)
    previous=OUTPUT/'figure_audit.json'
    if previous.exists() and not revise:
        old=read(previous)
        if old['source_identity']!=source_identity or old['implementation_sha256']!=sha(__file__):raise ValueError('Figure identity changed; preserve previous outputs')
        for path,digest in old['export_sha256'].items():
            if sha(OUTPUT/path)!=digest:raise ValueError('Figure output differs')
        print('FIGURE_CACHE_VERIFIED; frozen inputs unchanged',flush=True);return
    if any(OUTPUT.iterdir()) and not revise:raise RuntimeError('Partial figure outputs exist; preserve and inspect before authorized revision')
    figures=[]
    try:
        figures.append(metric_figure(m))
        with PdfPages(OUTPUT/'figure2_localization.pdf') as pdf:
            for page in range(3):figures.append(image_figure(loaded[page*4:(page+1)*4],page,pdf))
        write_source_csv(m,loaded)
        for relative,digest in source_identity.items():
            if sha(safe(relative))!=digest:raise ValueError('Source changed during plotting')
        bounds=[]
        for index,fig in enumerate(figures):
            fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
            outside=[]
            for artist in fig.findobj(matplotlib.text.Text):
                if not artist.get_visible() or not artist.get_text():continue
                b=artist.get_window_extent(renderer)
                if b.width>0 and b.height>0 and (b.x0<canvas.x0-1 or b.y0<canvas.y0-1 or b.x1>canvas.x1+1 or b.y1>canvas.y1+1):outside.append(artist.get_text())
            bounds.append({'figure_index':index,'size_mm':(fig.get_size_inches()*25.4).tolist(),'text_outside_canvas':outside})
        if 'torch' in sys.modules:raise RuntimeError('Unexpected model runtime import')
        outputs={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file() and p.name!='figure_audit.json'}
        audit={'schema':'shared-holdout-figure-v1','backend':'Python matplotlib','source_identity':source_identity,'implementation_sha256':sha(__file__),
            'metrics_conclusion':'G1 outperforms G0 only in the shared simulator domain; no real-screen claim',
            'contract':{'figure1':'quantitative grid, overall AP hero with normal FP control, AUROC/recall/type support',
                'figure2':'image plate + frozen-score annotations;3pages x4 predeclared samples',
                'export':'183mm wide; editable SVG/PDF, PNG300dpi, TIFF600dpi; native image256x256'},
            'sample_selection':[{'id':x['id'],'reasons':x['reasons'],'page':k//4+1,'row':k%4+1,'amplitude':x['amplitude'],
                                 'background_index':x['background_index'],'annotations':x['annotations']} for k,x in enumerate(loaded)],
            'shown_unique_samples':12,'shown_backgrounds':len({x['background_index'] for x in loaded}),
            'display_ranges':{'RGB':[0,1],'binary_masks':[0,1],'residual':[0,RES_MAX],'both_model_probability':[0,1]},
            'image_integrity':'Full saved256grid, nearest display, no crop/gamma/contrast change. Normal I repeats B; ROI=N/A, savedGT=0 and I-B=0.',
            'statistics':'One training seed;20related-background clusters; descriptive only; no CI or significance tests; same20 normals reused',
            'export_sha256':outputs,'layout_QA':bounds,'visual_QA':'pending inspection','Torch_imported':False,'model_forwards':0,'new_generation':0}
        save_json(previous,audit)
        print('FIGURES_EXPORTED',{'samples':12,'pages':3,'text_outside_canvas':[x['text_outside_canvas'] for x in bounds]},flush=True)
    finally:
        for figure in figures:plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--revise',action='store_true',help='Revise only this figure bundle using identical frozen sources')
    run(parser.parse_args().revise)
