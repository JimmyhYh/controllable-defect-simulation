"""Python figures from saved trial evidence only, no model rerun or selection."""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .data import OUT, digest, read_json, save_json


def draw():
    from .cli import audit
    audit()  # Verify cached pixels/statistics before a figure claims consistency.
    trial=read_json(OUT/'trial_results.json');audit=read_json(OUT/'artifact_audit.json')
    if not audit['same_input_rerender_equal']:
        raise ValueError('Cannot plot unaudited artifacts')
    # Contract: demonstrate finite-support geometric control and real residuals;
    # show every predeclared request, then predetermined pairs, no best-image search.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    f,axes=plt.subplots(4,6,figsize=(12,8.1),layout='constrained')
    f.get_layout_engine().set(rect=(0,.035,1,.965))
    for axis,record in zip(axes.flat,trial['records']):
        case=record['request']; data=np.load(OUT/'trial'/case['id']/'arrays.npz',allow_pickle=False)
        axis.imshow(data['image'],vmin=0,vmax=1);axis.axis('off')
        axis.set_title(f"{case['id']} | {case['spec']['shape']}\nPair{case['pair_id']}: {case['changed_factor']}",fontsize=7)
        data.close()
    f.suptitle('All24 predeclared controlled simulations |12 geometric pairs,4 normal backgrounds',fontsize=11)
    f.text(.5,.001,'RGB shown at[0,1]. Additive brightness amplitude +/-0.04; simulated appearance, not real defect samples.',
           ha='center',fontsize=8)
    export(f,folder/'all_requests');plt.close(f)

    # Same request order rule for both types: first pair of each, before inference.
    selected=[record for record in trial['records'] if record['request']['pair_id'] in (1,7)]
    response_path=OUT/'detector_response/response.json'
    response=read_json(response_path) if response_path.exists() else None
    columns=6 if response else 5
    f,axes=plt.subplots(4,columns,figsize=(columns*2.25,9.4),layout='constrained')
    f.get_layout_engine().set(rect=(0,.035,1,.965))
    heatmaps=np.load(OUT/'detector_response/heatmaps.npz',allow_pickle=False) if response else None
    if response and (digest(OUT/'detector_response/heatmaps.npz')!=response['heatmaps_sha256']
                     or digest(OUT/'trial_results.json')!=response['signature']['trial_sha256']):
        raise ValueError('Saved detector response and generation evidence differ')
    prediction={row['id']:row for row in response['records']} if response else {}
    # Common display scale covers ALL saved28 maps, not a selected good image.
    # This affects plotting only; raw scores and the old V threshold are untouched.
    vmax_heat=1.0
    if response:
        vmax_heat=max(float(heatmaps[key].max()) for key in heatmaps.files)
    titles=['Normal background','Simulation','Requested region','Actual float mask','Actual RGB residual']
    if response:titles+=['Frozen PatchCore map']
    residual_artist=None;heat_artist=None
    for row,record in enumerate(selected):
        case=record['request']
        with np.load(OUT/'trial'/case['id']/'arrays.npz',allow_pickle=False) as saved:
            panels=[saved['background'],saved['image'],saved['M_roi'],saved['M_eff'],saved['residual']]
            for col,(axis,pixels) in enumerate(zip(axes[row],panels)):
                if col<2:axis.imshow(pixels,vmin=0,vmax=1)
                elif col<4:axis.imshow(pixels,cmap='gray',vmin=0,vmax=1)
                else:residual_artist=axis.imshow(pixels,cmap='magma',norm=Normalize(0,.04))
            if response:
                heat_artist=axes[row,-1].imshow(heatmaps[case['id']],cmap='magma',norm=Normalize(0,vmax_heat))
                p=prediction[case['id']]
                axes[row,-1].set_xlabel(f"score={p['generated_score']:.3f} | alarm={p['generated_alarm']}")
        for col,axis in enumerate(axes[row]):
            axis.set_xticks([]);axis.set_yticks([])
            for spine in axis.spines.values():spine.set_visible(False)
            if row==0:axis.set_title(titles[col])
        axes[row,0].set_ylabel(case['id']+'\n'+case['changed_factor'],fontsize=8)
    f.colorbar(residual_artist,ax=axes[:,4],fraction=.025,pad=.01,label='max|RGB change|, common[0,0.04]')
    if response:f.colorbar(heat_artist,ax=axes[:,-1],fraction=.025,pad=.01,label='raw PatchCore map; common range from all28 saved maps')
    f.suptitle('Specified-position pairs: requested region, actual change and frozen-model response',fontsize=11)
    f.text(.5,.001,'First pair per type, selected before scores. Float mask: maxRGB residual>1/255. No new training or real-test evaluation.',
           ha='center',fontsize=8)
    export(f,folder/'control_pairs');plt.close(f)
    if heatmaps is not None:heatmaps.close()
    with (folder/'source_data.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fieldnames=['id','pair','type','background','valid','roi_pixels','effective_pixels','coverage','png_coverage',
                    'residual_min','residual_median','residual_max','outside_residual','generated_score','background_score','alarm']
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for record in trial['records']:
            case,stats=record['request'],record['stats'];p=prediction.get(case['id'],{})
            writer.writerow(dict(id=case['id'],pair=case['pair_id'],type=stats['kind'],background=case['background_path'],
                valid=stats['valid'],roi_pixels=stats['geometry']['raster']['pixels'],effective_pixels=stats['effective']['pixels'],
                coverage=stats['effective_coverage'],png_coverage=stats['png_effective_coverage'],
                residual_min=stats['roi_residual_min_median_max'][0],residual_median=stats['roi_residual_min_median_max'][1],
                residual_max=stats['roi_residual_min_median_max'][2],outside_residual=stats['outside_ROI_max_float_residual'],
                generated_score=p.get('generated_score'),background_score=p.get('background_score'),alarm=p.get('generated_alarm')))
    save_json(folder/'figure_sources.json',{'trial_results_sha256':digest(OUT/'trial_results.json'),
         'audit_sha256':digest(OUT/'artifact_audit.json'),'selected_pair_ids':[1,7],
         'response_sha256':digest(response_path) if response else None,'residual_color_range':[0,.04],
         'heatmap_color_range':[0,vmax_heat] if response else None,'backend':'Python matplotlib',
         'heatmap_display_rule':'0 to max of ALL4 normal and24 simulation maps; plotting only, no score/threshold change',
         'claim':'Geometric control / residual-label consistency only; not real performance improvement',
         'files':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.suffix in ('.png','.svg','.pdf','.csv')}})
    print('FIGURES',folder,flush=True)


def export(figure,base):
    for extension in ('png','svg','pdf'):
        figure.savefig(base.with_suffix('.'+extension),dpi=220 if extension=='png' else None,bbox_inches='tight')
