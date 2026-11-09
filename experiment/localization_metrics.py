"""Exact global and type+normal pixel AP by external stable merge sorting."""
import gc
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from localization_data import ROOT, OUT, digest
from localization_testing import TEST_OUT, atomic_json

PIXEL_DTYPE=np.dtype([('s','<f4'),('y','u1'),('k','u1')])


def delete_run(path):
    if not path.resolve().is_relative_to((TEST_OUT/'ap_work').resolve()):
        raise ValueError('Temporary run outside AP workspace')
    path.unlink()


def merge_runs(left_path,right_path,target):
    left=np.load(left_path,mmap_mode='r');right=np.load(right_path,mmap_mode='r')
    output=np.lib.format.open_memmap(target,mode='w+',dtype=PIXEL_DTYPE,shape=(len(left)+len(right),))
    # Stable ordering: equal left scores precede equal right scores.
    for own,other,side in ((left,right,'left'),(right,left,'right')):
        other_scores=np.ascontiguousarray(other['s'])
        for start in range(0,len(own),250000):
            stop=min(start+250000,len(own));chunk=own[start:stop]
            positions=np.arange(start,stop,dtype=np.int64)+np.searchsorted(other_scores,chunk['s'],side=side)
            output[positions]=chunk
        del other_scores
    output.flush()
    del own,other,chunk,positions
    left._mmap.close();right._mmap.close();output._mmap.close()
    del left,right,output;gc.collect()
    delete_run(left_path);delete_run(right_path)


def accumulate_exact(run_path,class_count,positive_per_class,total_per_class):
    run=np.load(run_path,mmap_mode='r')
    tp=np.zeros(class_count+1,dtype=np.uint64);seen=np.zeros_like(tp);ap=np.zeros(class_count+1,dtype=np.float64)
    denom=np.r_[np.sum(positive_per_class),positive_per_class].astype(np.float64)
    pending=None;end=len(run)
    while end:
        start=max(0,end-1000000)
        block=np.array(run[start:end][::-1]);scores=block['s']
        boundaries=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
        group_scores=scores[boundaries].copy()
        p=np.empty((len(boundaries),class_count),dtype=np.uint64);n=np.empty_like(p)
        for k in range(class_count):
            selected=block['k']==k
            n[:,k]=np.add.reduceat(selected.astype(np.uint64),boundaries)
            p[:,k]=np.add.reduceat((selected & (block['y']!=0)).astype(np.uint64),boundaries)
        if pending is not None:
            score,prev_p,prev_n=pending
            if group_scores[0]==score:
                p[0]+=prev_p;n[0]+=prev_n
            else:
                group_scores=np.r_[score,group_scores];p=np.vstack((prev_p,p));n=np.vstack((prev_n,n))
        if start:
            pending=(group_scores[-1],p[-1].copy(),n[-1].copy())
            p=p[:-1];n=n[:-1]
        else:pending=None
        if len(p):
            pos=np.column_stack((p.sum(axis=1),p))
            # column0 is full pool; column(k+1) is author type k plus all good pixels.
            counts=np.column_stack((n.sum(axis=1),n+n[:,[0]]))
            counts[:,1]=n[:,0]
            cumulative_tp=tp+np.cumsum(pos,axis=0,dtype=np.uint64)
            cumulative_seen=seen+np.cumsum(counts,axis=0,dtype=np.uint64)
            for j in range(class_count+1):
                valid=pos[:,j]>0
                if denom[j]>0:
                    ap[j]+=np.sum((pos[valid,j]/denom[j])*(cumulative_tp[valid,j]/cumulative_seen[valid,j]),dtype=np.float64)
            tp=cumulative_tp[-1].copy();seen=cumulative_seen[-1].copy()
        end=start
    expected_counts=np.r_[np.sum(total_per_class),total_per_class+total_per_class[0]]
    expected_counts[1]=total_per_class[0]
    if not np.array_equal(tp,np.r_[np.sum(positive_per_class),positive_per_class]) or not np.array_equal(seen,expected_counts):
        raise ValueError('External AP class/pixel counts differ')
    del run;gc.collect()
    return [float(ap[j]) if denom[j]>0 else None for j in range(class_count+1)]


def exact_pixel_ap(maps,masks,kinds,class_count,prefix,indices=None):
    ids=np.arange(len(maps)) if indices is None else np.asarray(indices)
    work=TEST_OUT/'ap_work';work.mkdir(exist_ok=True)
    runs=[];positive=np.zeros(class_count,dtype=np.uint64);total=np.zeros_like(positive)
    for block_index,start in enumerate(range(0,len(ids),16)):
        selected=ids[start:start+16]
        s=np.asarray(maps[selected],dtype=np.float32).reshape(-1)
        y=np.asarray(masks[selected],dtype=np.uint8).reshape(-1)
        if not np.isfinite(s).all():raise ValueError('Nonfinite pixel score')
        k=np.repeat(kinds[selected],256*256).astype('uint8')
        for cls in range(class_count):
            chosen=k==cls;positive[cls]+=np.count_nonzero(y[chosen]);total[cls]+=np.count_nonzero(chosen)
        order=np.argsort(s,kind='stable')
        path=work/f'{prefix}_run_{block_index}.npy'
        run=np.lib.format.open_memmap(path,mode='w+',dtype=PIXEL_DTYPE,shape=(len(s),))
        run['s']=s[order];run['y']=y[order];run['k']=k[order]
        run.flush();del run,s,y,k,order;gc.collect();runs.append(path)
    level=0
    while len(runs)>1:
        next_runs=[]
        for i in range(0,len(runs),2):
            if i+1==len(runs):next_runs.append(runs[i]);continue
            target=work/f'{prefix}_merge_{level}_{i//2}.npy'
            merge_runs(runs[i],runs[i+1],target);next_runs.append(target)
        runs=next_runs;level+=1
        print(json.dumps({'AP_sort':prefix,'merge_level':level,'runs_remaining':len(runs)}),flush=True)
    aps=accumulate_exact(runs[0],class_count,positive,total)
    delete_run(runs[0])
    return aps,positive.tolist(),total.tolist()


def compute_metrics():
    started=time.perf_counter()
    frozen_path=OUT/'frozen_config.json';frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    label_path=TEST_OUT/'labels.json';labels=json.loads(label_path.read_text(encoding='utf-8'))
    signature=labels['signature']
    if signature['frozen_config_sha256']!=digest(frozen_path) or signature['inference_implementation_sha256']!=digest(ROOT/'localization_testing.py'):
        raise ValueError('Frozen test inference identity differs')
    if digest(TEST_OUT/'masks.npy')!=labels['masks_sha256']:raise ValueError('Mask cache differs')
    output=TEST_OUT/'metrics.json'
    if output.exists():
        prior=json.loads(output.read_text(encoding='utf-8'))
        if prior['signature']!=signature or prior['metrics_implementation_sha256']!=digest(__file__):
            raise ValueError('Completed metric scope changed')
        for path,h in prior['prediction_artifact_sha256'].items():
            if digest(path)!=h:raise ValueError('Saved prediction cache changed')
        print(json.dumps({'metrics_cached':True,'models':prior['models'],'B_minus_A':prior['B_minus_A']}),flush=True)
        return
    y=np.asarray(labels['image_labels'],dtype=np.uint8)
    types=['good']+sorted(set(labels['types'])-{'good'})
    kinds=np.asarray([types.index(k) for k in labels['types']],dtype=np.uint8)
    masks=np.load(TEST_OUT/'masks.npy',mmap_mode='r')
    rows={};artifacts={}
    for name in frozen['models']:
        folder=TEST_OUT/name
        inference=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
        if inference['signature']!=signature:raise ValueError('Prediction scope differs')
        for n,h in inference['files'].items():
            if digest(folder/n)!=h:raise ValueError('Prediction hash differs')
            artifacts[str(folder/n)]=h
        result_path=folder/'metrics.json'
        if result_path.exists():
            result=json.loads(result_path.read_text(encoding='utf-8'))
            if result['signature']!=signature or result['metrics_implementation_sha256']!=digest(__file__):
                raise ValueError('Per-model metrics changed')
        else:
            maps=np.load(folder/'heatmaps.npy',mmap_mode='r');scores=np.load(folder/'scores.npy')
            tau=frozen['thresholds'][name];alarm=scores>tau
            tp=int(np.count_nonzero(alarm & (y==1)));fn=int(np.count_nonzero(~alarm & (y==1)))
            fp=int(np.count_nonzero(alarm & (y==0)));tn=int(np.count_nonzero(~alarm & (y==0)))
            mstart=time.perf_counter()
            # Audit actual first8 paths per author type and good; exercises merges and normal-inclusive subsets.
            audit_ids=np.concatenate([np.flatnonzero(kinds==k)[:8] for k in range(len(types))])
            audit,_,_=exact_pixel_ap(maps,masks,kinds,len(types),name+'_audit',audit_ids)
            reference=float(average_precision_score(np.asarray(masks[audit_ids]).reshape(-1),np.asarray(maps[audit_ids]).reshape(-1)))
            if abs(audit[0]-reference)>1e-12:raise ValueError('External AP differs from sklearn on real pixel audit')
            audit_categories={}
            for k in range(1,len(types)):
                subset=audit_ids[(kinds[audit_ids]==0)|(kinds[audit_ids]==k)]
                ref=float(average_precision_score(np.asarray(masks[subset]).reshape(-1),np.asarray(maps[subset]).reshape(-1)))
                if abs(audit[k+1]-ref)>1e-12:raise ValueError('Type+normal AP audit differs')
                audit_categories[types[k]]={'external_AP':audit[k+1],'sklearn_AP':ref,'absolute_difference':abs(audit[k+1]-ref)}
            ap,pixels,counts=exact_pixel_ap(maps,masks,kinds,len(types),name+'_full')
            categories={}
            for k in range(1,len(types)):
                subset=kinds==k
                ctp=int(np.count_nonzero(alarm & subset));num=int(subset.sum())
                categories[types[k]]={'anomaly_images':num,'TP':ctp,'FN':num-ctp,'recall':ctp/num,
                      'pixel_AP_type_plus_all_normal':ap[k+1],'normal_images_in_pixel_subset':int((y==0).sum()),
                      'positive_pixels':pixels[k],'negative_pixels':counts[k]+counts[0]-pixels[k],
                      'total_pixels':counts[k]+counts[0]}
            result={'signature':signature,'metrics_implementation_sha256':digest(__file__),'model':name,
                    'threshold':tau,'image_AUROC':float(roc_auc_score(y,scores)),'TP':tp,'FN':fn,'FP':fp,'TN':tn,
                    'anomaly_images':int((y==1).sum()),'normal_images':int((y==0).sum()),'recall':tp/(tp+fn),
                    'normal_FPR':fp/(fp+tn),'pixel_AP':ap[0],'positive_pixels':sum(pixels),
                    'negative_pixels':sum(counts)-sum(pixels),'total_pixels':sum(counts),'categories':categories,
                    'pixel_AP_definition':'exact noninterpolated global AP, descending equal-score groups; all pixels include all good images',
                    'pixel_AP_computation':'external stable sorting and vectorized class counts, no bins/rounding/approximation',
                    'numerical_audit_first8_per_type_and_good':{'images':audit_ids.tolist(),'external_AP':audit[0],
                      'sklearn_AP':reference,'absolute_difference':abs(audit[0]-reference),'categories':audit_categories},
                    'this_model_metrics_seconds':time.perf_counter()-mstart,'test_prediction_files':inference['files']}
            atomic_json(result_path,result);del maps;gc.collect()
        rows[name]=result
        print(json.dumps({'model':name,'image_AUROC':result['image_AUROC'],'pixel_AP':result['pixel_AP'],
                          'TP':result['TP'],'FN':result['FN'],'FP':result['FP'],'TN':result['TN']}),flush=True)
    a,b=rows['A20261020'],rows['B20261020']
    delta={key+'_percentage_points':100*(b[key]-a[key]) for key in ['pixel_AP','recall','normal_FPR','image_AUROC']}
    selected=[labels['paths'][labels['types'].index('good')]]+[labels['paths'][labels['types'].index(k)] for k in types[1:6]]
    bscores=np.load(TEST_OUT/'B20261020/scores.npy');balarm=bscores>frozen['thresholds']['B20261020']
    failed=np.flatnonzero((y==1)&~balarm);false=np.flatnonzero((y==0)&balarm)
    failures={'B_first_FN':labels['paths'][int(failed[0])] if len(failed) else None,
              'B_first_FP':labels['paths'][int(false[0])] if len(false) else None}
    selected+= [v for v in failures.values() if v is not None]
    atomic_json(output,{'signature':signature,'metrics_implementation_sha256':digest(__file__),
               'models':rows,'B_minus_A':delta,'labels_sha256':digest(label_path),'prediction_artifact_sha256':artifacts,
               'normal_images':350,'anomaly_images':750,'total_images':1100,'pixel_grid':[256,256],
               'total_pixels':labels['total_pixels'],'positive_pixels':labels['positive_pixels'],
               'positive_mask_lost_at256':labels['positive_mask_lost_at256'],
               'figure_samples':selected,'failure_samples':failures,'selection_rule':'first lexicographic path per author type and good; first B FN/FP; no best-case selection',
               'single_paired_seed':20261020,'independent_training_repeats':1,'training_started':False,
               'this_process_metrics_seconds':time.perf_counter()-started})
    print(json.dumps({'metrics_complete':True,'B_minus_A':delta,'selected_samples':selected},ensure_ascii=True),flush=True)
