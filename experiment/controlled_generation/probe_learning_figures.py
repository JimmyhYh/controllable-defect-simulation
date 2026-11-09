"""Python-only previews from saved diagnosis; no model imports/inference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import digest, read_json, save_json
from .probe_learning_diagnostics import FOLDER, sha_file


def draw():
    results = read_json(FOLDER/'diagnostic_results.json')
    arrays_path = FOLDER/'diagnostic_arrays.npz'
    if sha_file(arrays_path) != results['artifacts_sha256']['diagnostic_arrays.npz']:
        raise ValueError('Diagnostic arrays changed')
    with np.load(arrays_path, allow_pickle=False) as saved:
        steps, inputs, clean, labels = saved['steps'], saved['inputs'], saved['targets'], saved['labels']
        rec, residual, maps = saved['reconstructions'], saved['reconstruction_residuals'], saved['heatmaps']
    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none', 'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout='constrained')
    ap = [row['global_training_pool_pixel_AP'] for row in results['checkpoints']]
    axes[0].plot(steps, ap, 'o-', color='#2C6B8B', linewidth=1.5)
    axes[0].scatter([1000], [ap[-1]], color='#B74848', zorder=5)
    axes[0].axhline(.80, color='#777777', ls='--', lw=1, label='Fixed gate: AP >= 0.80')
    axes[0].set(xlabel='Saved training step', ylabel='Global pixel AP', ylim=(0, 1.05), xticks=steps,
                title='a  Fixed8 train-side localization')
    axes[0].legend(loc='lower left', fontsize=8)
    for name, color in [('MSE','#2C6B8B'), ('SSIM_loss','#7A6B95'), ('Focal','#A36B38')]:
        axes[1].plot(steps, [r['mean_losses'][name] for r in results['checkpoints']],
                     'o-', label=name, color=color, lw=1.5)
    axes[1].set(xlabel='Saved training step', ylabel='Mean forward loss (log scale)', yscale='log', xticks=steps,
                title='b  Same8 inputs; separate loss terms')
    axes[1].legend(fontsize=8)
    fig.suptitle('8 images / 4 backgrounds / 4,628 positive of 524,288 pixels; final1000 gate remains false', fontsize=10)
    for suffix in ('png', 'svg'):
        path = FOLDER/f'checkpoint_diagnosis.{suffix}'
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)
    # Single global residual scale across all checkpoints and all samples.
    upper = float(residual.max())
    fig, axes = plt.subplots(8, 6, figsize=(12, 16), layout='constrained')
    titles = ['Input RGB', 'Clean target', 'Reconstruction', 'Abs residual (max RGB)', 'Effective label', 'Anomaly softmax']
    for i, sample in enumerate(results['checkpoints'][-1]['samples']):
        panels = [inputs[i], clean[i], np.clip(rec[-1,i],0,1), residual[-1,i], labels[i], maps[-1,i]]
        for j, panel in enumerate(panels):
            ax=axes[i,j]
            if j < 3:
                ax.imshow(panel, vmin=0, vmax=1)
            elif j == 3:
                residual_artist=ax.imshow(panel, cmap='magma', vmin=0, vmax=upper)
            elif j == 4:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=1)
            else:
                heat_artist=ax.imshow(panel, cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if i == 0:ax.set_title(titles[j], fontsize=9)
            if j == 0:
                name=Path(sample['background']).stem
                ax.set_ylabel(f'{i}: {name}\n{sample["kind"]}\npositive={sample["positive_pixels"]}',fontsize=8)
    fig.colorbar(residual_artist, ax=axes[:,3], fraction=.025, shrink=.9, label=f'Residual: global [0, {upper:.3f}]')
    fig.colorbar(heat_artist, ax=axes[:,5], fraction=.025, shrink=.9, label='Probability [0,1]')
    fig.suptitle('Final1000: all8 fixed samples; reconstruction clipped only for display, loss uses raw values', fontsize=11)
    for suffix in ('png', 'svg'):
        path = FOLDER/f'final_reconstruction_panel.{suffix}'
        fig.savefig(path, dpi=170)
        paths.append(path)
    plt.close(fig)
    manifest = {'source_results_sha256':digest(FOLDER/'diagnostic_results.json'),
                'source_arrays_sha256':sha_file(arrays_path),'figure_code_sha256':digest(__file__),
                'backend':'Python/matplotlib only','sample_selection':'all8 final1000; all5 predetermined checkpoints',
                'RGB_range':[0,1],'probability_range':[0,1],'residual_range_all_checkpoints':[0,upper],
                'reconstruction_display_clipped_only':True,'model_inferences':0,
                'artifacts_sha256':{p.name:sha_file(p) for p in paths}}
    save_json(FOLDER/'figure_manifest.json',manifest)
    print('DIAGNOSTIC_FIGURES',{'files':list(manifest['artifacts_sha256']),'residual_max':upper,'inferences':0})
    return manifest


if __name__=='__main__':
    draw()
