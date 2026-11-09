"""Python figures from frozen saved predictions only; no model inference."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from localization_data import ROOT, OUT, digest
from localization_testing import TEST_OUT

FIG_OUT=ROOT.parent.parent/'报告'/'图示'
NAMES=['PatchCore','DRAEM-A','DRAEM-B']
COLORS=['#8A959B','#94B5D2','#B5A1C8']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})


def export(fig,base):
    FIG_OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','pdf'):
        fig.savefig(FIG_OUT/f'{base}.{ext}',bbox_inches='tight',facecolor='white')
    fig.savefig(FIG_OUT/f'{base}.png',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def draw():
    metrics_path=TEST_OUT/'metrics.json'
    metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
    for path,h in metrics['prediction_artifact_sha256'].items():
        if digest(path)!=h:raise ValueError('Figure prediction source changed')
    labels=json.loads((TEST_OUT/'labels.json').read_text(encoding='utf-8'))
    if digest(TEST_OUT/'labels.json')!=metrics['labels_sha256'] or digest(TEST_OUT/'masks.npy')!=labels['masks_sha256']:
        raise ValueError('Figure labels changed')
    scale=json.loads((TEST_OUT/'display_scale.json').read_text(encoding='utf-8'))
    if scale['signature']!=metrics['signature'] or digest(TEST_OUT/'N_PatchCore_heatmaps.npy')!=scale['N_heatmaps_sha256']:
        raise ValueError('N-only display range changed')
    models=list(metrics['models']);rows=[metrics['models'][n] for n in models]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.25),gridspec_kw={'width_ratios':[1.25,1,1]})
    for panel,(ax,key,title) in enumerate(zip(axes,['pixel_AP','recall','normal_FPR'],
            ['Global pixel AP','Anomaly recall (n=750)','Normal false alarms (n=350)'])):
        values=[100*r[key] for r in rows]
        ax.bar(np.arange(3),values,color=COLORS,width=.63)
        for i,value in enumerate(values):
            extra=''
            if key=='recall':extra=f'\n{rows[i]["TP"]}/750'
            elif key=='normal_FPR':extra=f'\n{rows[i]["FP"]}/350'
            ax.text(i,value+2,f'{value:.2f}%'+extra,ha='center',va='bottom',fontsize=6.5)
        ax.set_xticks(np.arange(3),NAMES,rotation=25,ha='right')
        ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('%')
        ax.set_title(title,pad=14,fontweight='bold')
        ax.text(-.12,1.12,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.grid(axis='y',color='#E8E8E8',linewidth=.5);ax.set_axisbelow(True)
    d=metrics['B_minus_A']
    fig.suptitle('AnoLCD display | Fixed first pair, seed 20261020',fontweight='bold',y=1.02)
    cylinder_share=100*rows[0]['categories']['cylinder']['positive_pixels']/metrics['positive_pixels']
    fig.text(.05,-.10,f'B - A: pixel AP {d["pixel_AP_percentage_points"]:+.3f} pp; recall {d["recall_percentage_points"]:+.3f} pp; '
             f'false alarms {d["normal_FPR_percentage_points"]:+.3f} pp.\n'
             'Full Tfinal: 1,100 images / 72,089,600 pixels at 256 x 256. Vnew thresholds frozen; one training pair, no error bars.\n'
             f'Cylinder masks account for {cylinder_share:.2f}% of positive pixels; see per-type results for small-defect localization.',fontsize=6.5)
    fig.tight_layout();export(fig,'figure5_weak_real_metrics')
    with (FIG_OUT/'figure5_weak_real_metrics_source.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','pixel_AP','positive_pixels','total_pixels','image_AUROC','TP','FN','anomaly_images','FP','TN','normal_images','recall','normal_FPR','threshold'])
        writer.writeheader()
        for n,r in zip(models,rows):writer.writerow({k:n if k=='model' else r[k] for k in writer.fieldnames})
    sample_paths=metrics['figure_samples'];count=len(sample_paths)
    masks=np.load(TEST_OUT/'masks.npy',mmap_mode='r')
    maps={n:np.load(TEST_OUT/n/'heatmaps.npy',mmap_mode='r') for n in models}
    scores={n:np.load(TEST_OUT/n/'scores.npy',mmap_mode='r') for n in models}
    split=json.loads((ROOT/'data/weak_localization/split.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(count,5,figsize=(7.2,1.03*count+.8),squeeze=False,
                         gridspec_kw={'wspace':.04,'hspace':.24})
    titles=['Input RGB','GT mask','PatchCore overlay','DRAEM-A overlay','DRAEM-B overlay']
    for ax,title in zip(axes[0],titles):ax.set_title(title,fontweight='bold',fontsize=7,pad=5)
    source=[]
    for row,relative in enumerate(sample_paths):
        i=labels['paths'].index(relative);kind=labels['types'][i];is_anomaly=bool(labels['image_labels'][i])
        with Image.open(Path(split['root'])/relative) as image:
            rgb=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))/255
        axes[row,0].imshow(rgb);axes[row,1].imshow(masks[i],cmap='gray',vmin=0,vmax=1)
        role=kind
        if row>=min(6,count):
            role='First B FN' if relative==metrics['failure_samples']['B_first_FN'] else 'First B FP'
        axes[row,0].text(-.05,.5,role,rotation=90,va='center',ha='right',transform=axes[row,0].transAxes,fontsize=7)
        axes[row,0].text(.5,-.025,Path(relative).name,transform=axes[row,0].transAxes,ha='center',va='top',fontsize=6)
        for col,n in enumerate(models,2):
            upper=scale['PatchCore_heatmap_vmax'] if n=='PatchCore' else 1.0
            colored=plt.get_cmap('magma')(np.clip(maps[n][i]/upper,0,1))[...,:3]
            axes[row,col].imshow(.6*rgb+.4*colored)
            alarm=bool(scores[n][i]>metrics['models'][n]['threshold'])
            result=('TP' if alarm else 'FN') if is_anomaly else ('FP' if alarm else 'TN')
            axes[row,col].text(.5,-.025,f's={scores[n][i]:.5g} | {result}',ha='center',va='top',transform=axes[row,col].transAxes,fontsize=6)
            source.append({'row':row+1,'role':role,'path':relative,'model':n,'score':float(scores[n][i]),
                           'threshold':metrics['models'][n]['threshold'],'prediction':result,'GT_positive_pixels':int(masks[i].sum()),
                           'heatmap_vmin':0,'heatmap_vmax':upper,'overlay_alpha':.4})
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([]);ax.set_frame_on(False)
    fig.subplots_adjust(bottom=.085,top=.97,left=.08,right=.995)
    for x,width,upper,text in [(.13,.3,scale['PatchCore_heatmap_vmax'],'PatchCore: N80 pixel q99.5'),(.61,.3,1.,'DRAEM-A/B: probability')]:
        cax=fig.add_axes([x,.045,width,.012])
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,upper),cmap='magma'),cax=cax,orientation='horizontal')
        cax.tick_params(labelsize=6);cax.set_title(text,fontsize=6,pad=3)
    fig.text(.08,.008,'Whole images at 256 x 256; fixed overlay alpha 0.4. First paths per type, then first B failure cases; no selection for visual quality.',fontsize=5.8)
    export(fig,'figure6_weak_real_localization')
    with (FIG_OUT/'figure6_weak_real_localization_source.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(source[0]));writer.writeheader();writer.writerows(source)
    print(json.dumps({'figures_complete':True,'samples':sample_paths,'PatchCore_display_max':scale['PatchCore_heatmap_vmax'],
                      'formats':['PNG','SVG','PDF'],'model_inference':False},ensure_ascii=True),flush=True)

if __name__=='__main__':draw()
