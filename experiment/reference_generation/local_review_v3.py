"""One-time explicit structure annotation and full24 local CPU evaluation."""
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

from .local_preparation_v3 import ROOT,V1,V2,OUT,REPORT,load,save,geometries,read_backgrounds,old_identities
from .reference import sha256
from .domain_selection_v2 import statistics


def rectangle_mask(rectangles):
    mask=np.zeros((1024,1024),np.uint8)
    for item in rectangles:
        box=item['rect_xyxy_exclusive']
        if len(box)!=4 or any(type(v)!=int for v in box) or not item.get('reason','').strip():
            raise ValueError('Integer conservative rectangle and reason required')
        l,t,r,b=box
        if not(0<=l<r<=1024 and 0<=t<b<=1024):raise ValueError('Invalid/out-of-image structure rectangle')
        mask[t:b,l:r]=1
    return mask>0


def numeric_issues(stats,anchors):
    result=[]
    if not anchors['median_Y_interval'][0]<=stats['median_Y']<=anchors['median_Y_interval'][1]:result.append('median_Y')
    c=np.asarray(stats['median_RGB'])
    if np.any(c<anchors['median_RGB_lo']) or np.any(c>anchors['median_RGB_hi']):result.append('median_RGB')
    if stats['q90_minus_q10_Y']>anchors['Y_range_max']:result.append('Y_range')
    return result


def calculate_audit(protocol,review):
    geo=geometries();backgrounds=read_backgrounds();records=[];masks={}
    kernel=np.ones((17,17),np.uint8)
    for bi,rgb in enumerate(backgrounds):
        key=f'b{bi}';s=rectangle_mask(review['structure_rectangles'][key]);u=rectangle_mask(review['uncertain_rectangles'][key])
        sp=cv2.dilate(s.astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        up=cv2.dilate(u.astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        masks[key]=(s,u,sp,up)
        for n in protocol['reference_ids']:
            for variant in ('base','move','scale','rotate'):
                g=geo[n+'_'+variant]
                if g['metadata']!=protocol['geometries'][n+'_'+variant]:raise ValueError('Frozen geometry changed')
                dr=cv2.dilate(g['roi'].astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
                context=g['guard']&~dr;stats=statistics(rgb,context)
                issues=numeric_issues(stats,protocol['anchors'])
                sm=int((sp&g['guard']).sum());um=int((up&g['guard']).sum())
                l,t,r,b=g['metadata']['crop_xyxy_exclusive'];model=np.zeros((1024,1024),bool);model[t:b,l:r]=True
                outside=model&~g['guard']
                passed=not(sm or um or issues)
                records.append({'background':key,'reference_id':n,'variant':variant,
                    'structure_guard_overlap_pixels':sm,'uncertain_guard_overlap_pixels':um,
                    'structure_in_model_outside_guard_pixels':int((sp&outside).sum()),
                    'uncertain_in_model_outside_guard_pixels':int((up&outside).sum()),
                    'guard_context_statistics':stats,'numeric_rejection_reasons':issues,'passed':passed})
    return records,geo,masks,backgrounds


def evaluate_local_review_v3(structure_rectangles,uncertain_rectangles):
    if (OUT/'completion.json').exists():
        from .local_verification_v3 import verify_local_inputs_v3
        return verify_local_inputs_v3()
    prep=load(OUT/'review_preparation.json');protocol=load(OUT/'local_protocol.json')
    actual={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file()}
    if actual!=set(prep['artifacts'])|{'review_preparation.json','preparation.incomplete'}:
        raise ValueError('Unexpected/incomplete structure review inputs; preserve')
    for p,h in prep['artifacts'].items():
        if sha256(OUT/p)!=h:raise ValueError('Sealed input changed')
    if set(structure_rectangles)!={'b0','b1'} or set(uncertain_rectangles)!={'b0','b1'}:
        raise ValueError('Both fixed backgrounds must be reviewed')
    for obj in (structure_rectangles,uncertain_rectangles):
        for rectangles in obj.values():rectangle_mask(rectangles)
    review={'schema':'structure-review-v3','structure_rectangles':structure_rectangles,
        'uncertain_rectangles':uncertain_rectangles,
        'reviewer':'assistant native image inspection; original1024 and all24 prescribed locals',
        'display':'absolute uint8 RGB, no enhancement','is_semantic_GT':False,
        'scope':'two existing draws only; complete conservative structures, no seed changes or rescue edits'}
    # Commit annotations before computing intersections; a second submission is not allowed.
    save(OUT/'structure_review.json',review)
    records,geo,masks,backgrounds=calculate_audit(protocol,review)
    audit={'schema':'local-audit-v3','position_count':24,'passed':sum(x['passed'] for x in records),
        'records':records,'review_sha256':sha256(OUT/'structure_review.json'),
        'max_structure_guard_overlap_pixels':max(x['structure_guard_overlap_pixels'] for x in records),
        'max_uncertain_guard_overlap_pixels':max(x['uncertain_guard_overlap_pixels'] for x in records),
        'positions_with_remote_structure_in_model':sum(x['structure_in_model_outside_guard_pixels']>0 for x in records)}
    save(OUT/'local_audit.json',audit)
    for bi,rgb in enumerate(backgrounds):
        image=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(image)
        for items,color in ((structure_rectangles[f'b{bi}'],(255,70,70)),(uncertain_rectangles[f'b{bi}'],(255,220,0))):
            for item in items:
                l,t,r,b=item['rect_xyxy_exclusive'];draw.rectangle((l,t,r-1,b-1),outline=color,width=2)
        image.save(OUT/'visual'/f'b{bi}_structure_rectangles.png')
    ready=audit['passed']==24
    if ready:status='CPU_LOCAL_READY'
    elif any(x['uncertain_guard_overlap_pixels'] for x in records):status='LOCAL_REVIEW_UNCERTAIN'
    elif any(x['structure_guard_overlap_pixels'] for x in records):status='LOCAL_STRUCTURE_BLOCKED'
    else:status='LOCAL_NUMERIC_BLOCKED'
    requests=[]
    if ready:
        overview=Image.new('RGB',(1200,6*324),'#252525');draw=ImageDraw.Draw(overview)
        for ri,n in enumerate(protocol['reference_ids']):
            for bi,item in enumerate(protocol['selected']):
                for vi,variant in enumerate(('base','move','scale','rotate')):
                    g=geo[n+'_'+variant];folder=OUT/'geometry'/f'{n}_b{bi}_{variant}';folder.mkdir(parents=True)
                    with (folder/'support.npz').open('xb') as f:np.savez_compressed(f,original=g['roi'],local512=g['local_roi'],guard=g['guard'])
                    raw=Image.fromarray(backgrounds[bi]);box=g['metadata']['crop_xyxy_exclusive']
                    raw.crop(box).resize((512,512),Image.Resampling.BILINEAR).save(folder/'background512.png')
                    view=np.array(raw);contours,_=cv2.findContours(g['roi'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(view,contours,-1,(255,110,0),2)
                    l,t,r,b=g['metadata']['guard_xyxy_exclusive'];cv2.rectangle(view,(l,t),(r-1,b-1),(60,255,100),2)
                    l,t,r,b=box;cv2.rectangle(view,(l,t),(r-1,b-1),(30,200,255),2)
                    x,y=vi*300,(ri*2+bi)*324;overview.paste(Image.fromarray(view).resize((300,300)),(x,y+24))
                    draw.text((x+3,y+4),f'{n} / {PathName(item["path"])} / {variant}',fill='white')
                    rel=str(folder.relative_to(OUT)).replace('\\','/')
                    for seed in protocol['generated_seeds']:
                        requests.append({'id':f'v3_r{len(requests):03d}','reference_id':n,
                            'background_path':item['path'],'variant':variant,'seed':seed,
                            'geometry':g['metadata'],'arrays':rel+'/support.npz','background512':rel+'/background512.png'})
        save(OUT/'requests.json',{'schema':'local-requests-v3','count':48,'distinct_geometries':24,
                               'requests':requests,'neural_generations_completed':0})
        overview.save(OUT/'CPU_position_overview.png')
    protected,_,_,_=old_identities()
    outcome={'schema':'local-input-v3','status':status,'ready_for_day2':ready,
        'generation_authorized':False,'local_protocol_sha256':sha256(OUT/'local_protocol.json'),
        'positions_evaluated':24,'positions_passed':audit['passed'],
        'reference_candidates':['019','024','020','023','021'],'actual_reference_ids':['019','024','020'],
        'backgrounds':protocol['selected'],'request_count':len(requests),'distinct_geometries':24 if ready else 0,
        'model_forwards':0,'optimizer_steps':0,'torch_loaded':'torch' in sys.modules,
        'v2_failure_unchanged':True,'remote_structure_influence':'unknown; model input remains full crop',
        'physical_source_independence':'unknown; development observation only'}
    if outcome['torch_loaded']:raise RuntimeError('Torch unexpectedly imported')
    save(OUT/'input_protocol.json',outcome)
    from .domain_preparation_v2 import resource_snapshot
    resource=resource_snapshot()
    text=f'''# 第1天v3局部背景适用性结果

日期：2026年10月16日。状态：**{status}**；ready_for_day2={str(ready).lower()}。

只复用v2固定背景1949／1540及原参考019／024／020，没有抽样或打开其他正常图。
审阅结构矩形先于相交计算保存，S／U不是缺陷语义真值。固定24位置通过{audit['passed']}/24。
确定结构与保护区最大交集{audit['max_structure_guard_overlap_pixels']}像素；不确定区域最大交集{audit['max_uncertain_guard_overlap_pixels']}像素。
完整模型画布内、保护区外仍有结构的位置数为{audit['positions_with_remote_structure_in_model']}/24，其模型影响尚未知。
局部统计、矩形坐标、全部位置的相交与拒绝原因分别在local_audit.json、structure_review.json。

实际正式CPU请求{len(requests)}，几何数组{24 if ready else 0}套；没有异常合成、神经生成或检测效果结果。
R、2倍C_guard、原4倍C_model按协议冻结后才查看背景；没有缩区、移位或改矩形救结果。
本版改变的是局部适用性验收单位；v2整图失败仍保持，不是独立验证。每背景多位置、两种子和三参考不能当独立真实样本。
未读取S_bg／Q_cal或范围外正常像素；原始RGB无归一化、gamma或增强。
v1／v2必要代码、元信息及所用封存背景身份核验不变。没有Torch、前向、训练、下载或依赖安装。
物理采集独立性、未辨认弱异常及远处结构的模型影响均未解决；人工审阅只代表开发判断。

实际结束资源快照：RAM可用{resource['available_RAM_GiB']:.3f}GiB，总{resource['total_RAM_GiB']:.3f}GiB，磁盘空闲{resource['free_disk_GiB']:.3f}GiB；非峰值。
只读核验命令（已实现）：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -X utf8 -m reference_generation.local_cli_v3 verify-local
```
prepare-local完整同身份缓存只核验复用，不自动重画、重抽或重算像素统计。
completion.json绑定本说明与全部成果，拒绝额外文件、目录或部分事务；报告写后仍须执行verify-local核验。
全部24位置通过仅允许进入另行授权的第2天CPU对照，所有未来训练仍由用户手动启动。
'''
    with (OUT/REPORT).open('x',encoding='utf-8') as f:f.write(text)
    artifacts={str(p.relative_to(OUT)).replace('\\','/'):sha256(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and p.name!='preparation.incomplete'}
    save(OUT/'completion.json',{'schema':'local-completion-v3','status':status,'artifacts':artifacts})
    (OUT/'preparation.incomplete').unlink()
    from .local_verification_v3 import verify_local_inputs_v3
    return verify_local_inputs_v3()


def PathName(path):
    return path.rsplit('/',1)[-1]
