"""Publication-oriented summaries, missing data remain gaps and warnings visible."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,policy
COLORS={'dishperf_native':'#2166ac','hyperf':'#b2182b'}
LABELS={'dishperf_native':'DISHPerf-NP','hyperf':'HyPerf-PP'}

def plot(folder):
    folder=Path(folder);d=pd.read_csv(folder/'summary.csv');status=read(folder/'status.json');p=policy()
    for metric,name,ylabel in [('mre_percent','point_mre_learning_curves','Point MRE (%)'),('pmape_percent','pmape_learning_curves','pMAPE (%)'),('model_compute_seconds','training_time','Recorded model compute (s)')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,system in zip(axes,['x264','lrzip','h2']):
            spec=p['systems'][system];budgets=[int(a*spec['p']) for a in p['alphas']]
            has=False
            for model in ['dishperf_native','hyperf']:
                s=d[(d.system==system)&(d.model==model)].set_index('budget');y=[];err=[];ns=[]
                for b in budgets:
                    if b in s.index and s.loc[b,'completed']>0:
                        r=s.loc[b];y.append(r[metric+'_mean']);err.append(r[metric+'_sd'] if np.isfinite(r[metric+'_sd']) else 0.);ns.append(int(r.completed));has=True
                    else:y.append(np.nan);err.append(0.);ns.append(0)
                if np.isfinite(y).any():ax.errorbar(budgets,y,yerr=err,color=COLORS[model],marker='o',capsize=3,label=LABELS[model]+' n='+','.join(map(str,ns)))
            ax.set_title(system.upper() if system=='h2' else system);ax.set_xlabel('Training labels per workload');ax.set_ylabel(ylabel);ax.set_xticks(budgets);ax.grid(alpha=.2)
            if has:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'No completed runs',ha='center',va='center',transform=ax.transAxes)
        note='SMOKE - execution check only; model scopes differ' if status['profile']=='smoke' else 'All finite completed runs; diagnostic-warning runs retained; error bars = across-repeat SD'
        fig.suptitle(note,fontsize=10);fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    per=pd.read_csv(folder/'workloads.csv')
    if len(per) and status['profile']=='formal':
        for system in per.system.unique():
            sub=per[per.system==system];m=sub.groupby(['workload','budget','model']).mre_percent.mean().unstack('model')
            if not {'dishperf_native','hyperf'}.issubset(m.columns):continue
            delta=(m['dishperf_native']-m['hyperf']).unstack('budget');values=delta.to_numpy(float)
            if not np.isfinite(values).any():continue
            lim=max(float(np.nanmax(abs(values))),1e-6);fig,ax=plt.subplots(figsize=(6,max(4,.3*len(delta))),constrained_layout=True)
            im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_yticks(range(len(delta)));ax.set_yticklabels(delta.index,fontsize=7);ax.set_xticks(range(len(delta.columns)));ax.set_xticklabels(delta.columns);ax.set_xlabel('Training labels per workload');ax.set_title(system+': mean point-MRE difference (DISHPerf - HyPerf)');fig.colorbar(im,ax=ax,label='percentage points; negative favors DISHPerf');fig.savefig(folder/('workload_mre_'+system+'.png'),dpi=180);plt.close(fig)
    print('PLOTS saved:',folder,flush=True)
