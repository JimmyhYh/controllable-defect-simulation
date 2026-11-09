"""Two-stage CPU domain preparation; finalization requires explicit visual decision."""
import ctypes
import importlib.metadata
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from .reference import read_rgb,read_mask,prepare_reference,sha256
from .coordinates import build_geometry,VARIANTS
from .domain_selection_v2 import (ROOT,V1,OUT,freeze_scope_v2,scope_from_metadata,load,save,
                                 statistics,classify_domain_v2,draw_backgrounds_v2)

REPORT='第1天v2同域背景准备结果.md'


def resource_snapshot():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError('Cannot read RAM')
    return {'total_RAM_GiB':m.total/2**30,'available_RAM_GiB':m.available/2**30,
        'free_disk_GiB':shutil.disk_usage(ROOT).free/2**30,'snapshot_not_peak':True,
        'packages':{n:importlib.metadata.version(n) for n in ('numpy','Pillow','opencv-python')}}


def check_resources():
    info=resource_snapshot()
    # Engineering stop bound for sequential original-resolution CPU buffers.
    if info['available_RAM_GiB']<1 or info['free_disk_GiB']<1:
        raise RuntimeError('CPU preparation needs >=1GiB free RAM and disk; no automatic downsize')
    if 'torch' in sys.modules:raise RuntimeError('Torch must not be imported')
    return info


def derive_regions(scope):
    rows,old,_,_=scope_from_metadata();data=Path(old['root']);refs={};masks={};contexts={};regions={};geometries={}
    for n in scope['actual_reference_ids']:
        p=f'display/test/mura/{n}.png';mp=f'display/ground_truth/mura/{n}_mask.png'
        rgb=read_rgb(data/p,rows[p]);m=read_mask(data/mp,rows[mp]);prepared=prepare_reference(rgb,m,{'context_factor':4,'model_side':512})
        l,t,r,b=prepared['crop_xyxy_exclusive']
        dilated=cv2.dilate(m.astype('uint8'),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        region=np.zeros(m.shape,bool);region[t:b,l:r]=~dilated[t:b,l:r]
        contexts[n]=statistics(rgb,region);refs[n]=(rgb,prepared);masks[n]=m
        for variant in VARIANTS:
            g=build_geometry(m,(1024,1024),{'variant':variant,'model_side':512});geometries[n+'_'+variant]=g
            l,t,r,b=g['metadata']['crop_xyxy_exclusive']
            dm=cv2.dilate(g['roi'].astype('uint8'),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            reg=np.zeros(m.shape,bool);reg[t:b,l:r]=~dm[t:b,l:r]
            if reg.sum()<1024:raise ValueError('Insufficient target context')
            regions[n+'_'+variant]=reg
    ys=[x['median_Y'] for x in contexts.values()];colors=np.array([x['median_RGB'] for x in contexts.values()])
    anchors={'reference_context_statistics':contexts,'median_Y_interval':[max(0,min(ys)-.06),min(1,max(ys)+.06)],
             'median_RGB_lo':np.clip(colors.min(0)-.06,0,1).tolist(),
             'median_RGB_hi':np.clip(colors.max(0)+.06,0,1).tolist(),
             'Y_range_max':max(.15,max(x['q90_minus_q10_Y'] for x in contexts.values())+.03)}
    return anchors,regions,geometries,refs,rows,data


def prepare_domain_inputs_v2():
    if (OUT/'completion.json').exists():
        from .domain_verification_v2 import verify_domain_inputs_v2
        return verify_domain_inputs_v2()
    if OUT.exists():raise RuntimeError('Existing incomplete v2 transaction; preserve, do not rerun')
    initial_resource=check_resources();scope=freeze_scope_v2()
    # Scope file already exists on disk before any dataset pixel decoding.
    anchors,regions,geometries,refs,rows,data=derive_regions(scope)
    save(OUT/'reference_anchors.json',anchors)
    audit=[]
    for i,item in enumerate(scope['scope']):
        check_resources()
        rgb=read_rgb(data/item['path'],rows[item['path']])
        audit.append({**item,**classify_domain_v2(rgb,anchors,regions)})
        del rgb
        if (i+1)%16==0: print('CPU_DOMAIN_SCREEN',i+1,'/64',flush=True)
    selected=draw_backgrounds_v2([x for x in audit if x['eligible']])
    record={'schema':'domain-audit-v2','scope_sha256':sha256(OUT/'scope_protocol.json'),
        'scope_count':64,'classified':len(audit),'eligible':sum(x['eligible'] for x in audit),
        'rows':audit,'draw_seed':20261018,'selected':[{'group_id':x['group_id'],'path':x['path']} for x in selected],
        'normal_decodes_screening':64,'normal_decodes_visual_input':0,'reference_RGB_decodes':3,
        'no_models':True,'resources_at_start':initial_resource,'resources_after_screening':check_resources()}
    if not selected:
        save(OUT/'domain_audit.json',record)
        return finish_v2('DOMAIN_POOL_INSUFFICIENT','Fixed64 screening has fewer than2 eligible groups; no expansion or replacement.')
    visual=OUT/'visual';visual.mkdir()
    # Re-decode selected2 explicitly for native-range visual inputs; count66,
    # rather than retain all64 decoded originals in memory.
    for bi,item in enumerate(selected):
        rgb=read_rgb(data/item['path'],rows[item['path']]);Image.fromarray(rgb).save(visual/f'b{bi}_original.png')
        page=Image.new('RGB',(4*300,4*324),'#262626');dr=ImageDraw.Draw(page)
        page.paste(Image.fromarray(rgb).resize((300,300),Image.Resampling.BILINEAR),(0,24))
        dr.text((4,4),f"b{bi}: {item['path']}",fill='white')
        for ri,n in enumerate(scope['actual_reference_ids']):
            page.paste(Image.fromarray(refs[n][1]['rgb']).resize((300,300)),((ri+1)*300,24))
            dr.text(((ri+1)*300+4,4),'reference '+n,fill='white')
            for vi,variant in enumerate(VARIANTS):
                g=geometries[n+'_'+variant];box=g['metadata']['crop_xyxy_exclusive']
                crop=np.array(Image.fromarray(rgb).crop(box).resize((512,512),Image.Resampling.BILINEAR))
                Image.fromarray(crop).save(visual/f'b{bi}_{n}_{variant}_512.png')
                page.paste(Image.fromarray(crop).resize((300,300)),(vi*300,(ri+1)*324+24))
                dr.text((vi*300+4,(ri+1)*324+4),n+' / '+variant,fill='white')
        page.save(visual/f'b{bi}_visual_review.png')
        del rgb
    record['normal_decodes_visual_input']=2
    save(OUT/'domain_audit.json',record)
    # Seal all review inputs before external read-only inspection/finalization.
    review={'schema':'awaiting-visual-v2','status':'AWAITING_VISUAL_REVIEW',
        'artifacts':{str(p.relative_to(OUT)).replace('\\','/'):sha256(p) for p in sorted(OUT.rglob('*'))
                     if p.is_file() and p.name!='preparation.incomplete'},
        'allow_finalize_only':True,'new_model_forwards':0,'optimizer_steps':0}
    save(OUT/'visual_preparation.json',review)
    return {'status':review['status'],'screened':64,'eligible':record['eligible'],
            'selected':record['selected'],'normal_RGB_decodes':66,'ready_for_day2':False}


def finalize_visual_v2(decision,reason):
    if (OUT/'completion.json').exists():
        from .domain_verification_v2 import verify_domain_inputs_v2
        return verify_domain_inputs_v2()
    if decision not in ('passed','failed','uncertain') or len(reason.strip())<10:
        raise ValueError('Explicit visual decision and substantive reason required')
    review=load(OUT/'visual_preparation.json')
    actual={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file()}
    if actual!=set(review['artifacts'])|{'visual_preparation.json','preparation.incomplete'}:
        raise ValueError('Unexpected/missing review transaction files')
    for p,h in review['artifacts'].items():
        if sha256(OUT/p)!=h:raise ValueError('Sealed visual input changed')
    save(OUT/'visual_decision.json',{'decision':decision,'reason':reason,'reviewer':'assistant native image inspection',
        'scope':'both drawn originals and all prescribed crops; development observation, not blind assessment',
        'replacement':False})
    if decision!='passed':return finish_v2('VISUAL_DOMAIN_UNCERTAIN',reason)
    scope=load(OUT/'scope_protocol.json');anchors,regions,geometries,refs,rows,data=derive_regions(scope)
    if anchors!=load(OUT/'reference_anchors.json'):raise ValueError('Reference anchors changed')
    selected=load(OUT/'domain_audit.json')['selected'];requests=[]
    canvas=Image.new('RGB',(1200,6*324),'#252525');dr=ImageDraw.Draw(canvas)
    # Selected originals are already saved; no further source normal decoding.
    for ri,n in enumerate(scope['actual_reference_ids']):
        for bi,item in enumerate(selected):
            rgb=np.array(Image.open(OUT/'visual'/f'b{bi}_original.png').convert('RGB'))
            for vi,variant in enumerate(VARIANTS):
                g=geometries[n+'_'+variant];folder=OUT/'geometry'/f'{n}_b{bi}_{variant}';folder.mkdir(parents=True)
                with (folder/'support.npz').open('xb') as f:np.savez_compressed(f,original=g['roi'],local512=g['local_roi'])
                box=g['metadata']['crop_xyxy_exclusive'];preview=rgb.copy()
                contours,_=cv2.findContours(g['roi'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview,contours,-1,(255,100,0),2)
                cv2.rectangle(preview,tuple(box[:2]),(box[2]-1,box[3]-1),(30,210,255),2)
                x,y=vi*300,(ri*2+bi)*324;canvas.paste(Image.fromarray(preview).resize((300,300)),(x,y+24))
                dr.text((x+4,y+4),f'{n} / {Path(item["path"]).stem} / {variant}',fill='white')
                for seed in (20261016,20261017):
                    requests.append({'id':f'v2_r{len(requests):03d}','reference_id':n,'background_path':item['path'],
                        'variant':variant,'seed':seed,'geometry':g['metadata'],
                        'arrays':str((folder/'support.npz').relative_to(OUT)).replace('\\','/'),
                        'background512':f'visual/b{bi}_{n}_{variant}_512.png'})
    save(OUT/'requests.json',{'schema':'domain-requests-v2','requests':requests,'count':48,
        'distinct_geometries':24,'neural_generations_completed':0})
    canvas.save(OUT/'CPU_position_overview.png')
    return finish_v2('CPU_DOMAIN_READY',reason)


def finish_v2(status,reason):
    scope=load(OUT/'scope_protocol.json');audit=load(OUT/'domain_audit.json');ready=status=='CPU_DOMAIN_READY'
    for p,h in scope['protected'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Protected file changed')
    protocol={'schema':'domain-input-v2','status':status,'ready_for_day2':ready,'reason':reason,
        'generation_authorized':False,'scope_sha256':sha256(OUT/'scope_protocol.json'),
        'reference_candidates':scope['candidates'],'actual_reference_ids':scope['actual_reference_ids'],
        'backgrounds':audit['selected'],'scope_count':64,'classified':audit['classified'],
        'eligible':audit['eligible'],'request_count':48 if ready else 0,'distinct_geometries':24 if ready else 0,
        'normal_source_RGB_decodes':audit['normal_decodes_screening']+audit['normal_decodes_visual_input'],
        'reference_RGB_decode_counts':{'screening':3,'finalization':3 if ready else 0},
        'model_forwards':0,'optimizer_steps':0,'torch_loaded':'torch' in sys.modules,
        'physical_source_independence':'unknown; development only','resources_at_finish':check_resources()}
    save(OUT/'input_protocol.json',protocol)
    text=f'''# 第1天v2同域背景准备结果

日期：2026年10月16日。状态：**{status}**，ready_for_day2={str(ready).lower()}。

固定范围64组，实际统计{audit['classified']}/64，数值入池{audit['eligible']}/64。
固定抽样：{', '.join(x['path'] for x in audit['selected']) or '不足2组，未抽样'}。
视觉判断／停止原因：{reason}

保留5候选与实际3参考019／024／020及其顺序、原几何、生成种子20261016／20261017。
本版本实际CPU请求{protocol['request_count']}、独立几何{protocol['distinct_geometries']}；没有神经生成或异常合成。
normal源图解码{protocol['normal_source_RGB_decodes']}次：统计64次，视觉输入另解码所选2张（若已抽样）；没有为省这2次而将64张同时留在内存。后续复合和核验使用保存输入。

输入筛选只使用原始编码RGB，参考锚点及所有64行统计见reference_anchors.json／domain_audit.json。
亮度与颜色门槛属于预定工程筛选，人工审阅属于开发观察；不能证明同屏型号或真实独立性。
原清单RGB组序列化未知，排除用旧组ID与原文件hash；不宣称近重复／增强关系已排除。
v1及旧冻结文件身份已核对，未覆盖；未读取S_bg／Q_cal或固定范围外正常源图。
没有Torch、模型前向、训练、下载或新依赖安装。所有未来训练仍由用户手动启动。

只读核验（已实现）：
```powershell
Set-Location -LiteralPath 'D:\\华兴源创\\实验\\弱缺陷Demo'
& '.\\.venv\\Scripts\\python.exe' -X utf8 -m reference_generation.domain_cli_v2 verify-domain
```
prepare-domain已实现：完整同身份缓存存在时只核验复用；不完整事务保留并停止，不自动再抽样。
范围协议先于正常图解码写入；视觉输入另封存后才接受显式审阅结论。
completion.json按精确文件清单绑定本说明及全部成果，拒绝额外文件／.tmp／锁；本说明不再追加修改。
本结果写入后仍须执行verify-domain确认来源、缓存、抽样与几何；不得仅凭文档声称核验通过。
'''
    with (OUT/REPORT).open('x',encoding='utf-8') as f:f.write(text)
    artifacts={str(p.relative_to(OUT)).replace('\\','/'):sha256(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and p.name!='preparation.incomplete'}
    save(OUT/'completion.json',{'schema':'domain-completion-v2','status':status,'artifacts':artifacts})
    (OUT/'preparation.incomplete').unlink()
    from .domain_verification_v2 import verify_domain_inputs_v2
    return verify_domain_inputs_v2()
