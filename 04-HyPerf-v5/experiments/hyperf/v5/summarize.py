"""Recompute predictions; include finite warning runs; never trim best/worst seeds."""
import itertools,math
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,policy,read,write,split,task_path

def exact_signed_rank_p(differences):
    from scipy.stats import rankdata
    d=np.asarray(differences,float);d=d[d!=0]
    if not len(d):return 1.0
    if len(d)>20:raise ValueError('Exact enumeration intended for registered five repeats')
    r=rankdata(abs(d),method='average');observed=abs(float(np.sum(r*np.sign(d))))
    total=0;extreme=0
    for signs in itertools.product([-1,1],repeat=len(d)):
        total+=1;extreme+=int(abs(float(np.dot(r,signs)))>=observed-1e-12)
    return float(extreme/total)

def write_csv(path,rows,columns):
    pd.DataFrame(rows,columns=columns).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')

def summarize(a,jobs):
    from experiment import base_path,task_identity,read_completed
    out=HERE/'summaries'/a.profile/(a.system+'_b'+str(a.budget)+'_r'+str(a.start_rep)+'_n'+str(a.run_count)+'_'+a.model);out.mkdir(parents=True,exist_ok=True)
    rows=[];states=[];per=[];corrupt=[];valid_scores={}
    for task in jobs:
        folder=task_path(base_path(a),task['profile'],task['system'],task['budget'],task['rep'],task['model'])
        state=dict(task,status='not_run',diagnostics_pass=None,warning='')
        if (folder/'complete.json').exists():
            try:
                tr,te,f,info,key=split(task['system'],task['budget'],task['rep'],task['profile'])
                if task['profile']=='smoke' and task['model']=='dishperf_native':te=te[te.workload==info['workload_order'][0]].reset_index(drop=True)
                m=read_completed(folder,task_identity(task,info,key),te,task['model']);valid_scores[(task['system'],task['budget'],task['rep'],task['model'])]=m
                state.update(status='completed_with_warning' if m['completed_with_warning'] else 'completed',diagnostics_pass=m['diagnostics_pass'],warning=';'.join(m['warning_reasons']))
                row={k:v for k,v in task.items() if k!='save_draws'}
                for field in ['mre_percent','pmape_percent','mae_macro','rmse_macro','mre_micro_percent','pmape_micro_percent','outer_seconds','model_compute_seconds','timing_resumed','unique_training_labels','raw_training_records','n_test']:row[field]=m[field]
                row['diagnostics_pass']=m['diagnostics_pass'];row['warning']=state['warning'];row['formal_eligible']=m['formal_eligible'];rows.append(row)
                for w in m['per_workload']:per.append(dict(task,**w))
            except Exception as e:state.update(status='corrupt',warning=str(e));corrupt.append(dict(task,error=str(e)))
        elif (folder/'failure.json').exists():state.update(status=read(folder/'failure.json').get('execution_status','failed'),warning=read(folder/'failure.json').get('error',''))
        elif (folder/'manifest.json').exists():state['status']='partial'
        states.append(state)
    grouped=[];paired=[]
    keys=sorted(set((j['system'],j['budget'],j['model']) for j in jobs))
    for system,budget,model in keys:
        selected=[r for r in rows if r['system']==system and r['budget']==budget and r['model']==model];registered=sum(j['system']==system and j['budget']==budget and j['model']==model for j in jobs)
        group={'system':system,'budget':budget,'model':model,'registered':registered,'completed':len(selected),'diagnostic_warning_count':sum(bool(r['warning']) for r in selected),'resumed_count':sum(bool(r['timing_resumed']) for r in selected)}
        for metric in ['mre_percent','pmape_percent','model_compute_seconds']:
            values=[float(r[metric]) for r in selected];group[metric+'_mean']=float(np.mean(values)) if values else None;group[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        grouped.append(group)
    for system,budget in sorted(set((j['system'],j['budget']) for j in jobs)):
        for rep in sorted(set(j['rep'] for j in jobs)):
            d=valid_scores.get((system,budget,rep,'dishperf_native'));h=valid_scores.get((system,budget,rep,'hyperf'))
            if d is None or h is None:continue
            # A smoke native task tests only the first workload; no cross-model
            # smoke score is advertised as a paired scientific comparison.
            if a.profile=='smoke':continue
            paired.append({'system':system,'budget':budget,'rep':rep,'dishperf_mre_percent':d['mre_percent'],'hyperf_mre_percent':h['mre_percent'],'delta_mre_pp':d['mre_percent']-h['mre_percent'],'dishperf_pmape_percent':d['pmape_percent'],'hyperf_pmape_percent':h['pmape_percent'],'delta_pmape_pp':d['pmape_percent']-h['pmape_percent'],'hyperf_diagnostics_pass':h['diagnostics_pass']})
    effects=[]
    for system,budget in sorted(set((j['system'],j['budget']) for j in jobs)):
        pairs=[x for x in paired if x['system']==system and x['budget']==budget]
        e={'system':system,'budget':budget,'n_pairs':len(pairs),'hyperf_diagnostic_pass_pairs':sum(bool(x['hyperf_diagnostics_pass']) for x in pairs)}
        for metric in ['mre','pmape']:
            vals=[r['delta_'+metric+'_pp'] for r in pairs]
            e[metric+'_mean_delta_pp']=float(np.mean(vals)) if vals else None;e[metric+'_median_delta_pp']=float(np.median(vals)) if vals else None;e[metric+'_dishperf_win_fraction']=float(np.mean(np.asarray(vals)<0)) if vals else None
            e[metric+'_exact_signed_rank_sign_flip_p']=exact_signed_rank_p(vals) if len(vals)==policy()['repeats'] else None;e[metric+'_holm_p']=None
        effects.append(e)
    # Fixed family: three systems x four budgets. No apparent single-test
    # correction is substituted when the rest of the family is missing.
    family_size=len(policy()['systems'])*len(policy()['alphas'])
    if len(effects)==family_size and all(e['n_pairs']==policy()['repeats'] for e in effects):
        for metric in ['mre','pmape']:
            order=sorted(range(family_size),key=lambda i:effects[i][metric+'_exact_signed_rank_sign_flip_p']);last=0.
            for rank,i in enumerate(order):
                last=max(last,min(1.,(family_size-rank)*effects[i][metric+'_exact_signed_rank_sign_flip_p']));effects[i][metric+'_holm_p']=last
    run_cols=['system','budget','rep','profile','model','mre_percent','pmape_percent','mae_macro','rmse_macro','mre_micro_percent','pmape_micro_percent','outer_seconds','model_compute_seconds','timing_resumed','unique_training_labels','raw_training_records','n_test','diagnostics_pass','warning','formal_eligible']
    write_csv(out/'runs.csv',rows,run_cols);write_csv(out/'task_status.csv',states,['system','budget','rep','profile','model','save_draws','status','diagnostics_pass','warning'])
    gc=['system','budget','model','registered','completed','diagnostic_warning_count','resumed_count']+[m+s for m in ['mre_percent','pmape_percent','model_compute_seconds'] for s in ['_mean','_sd']]
    write_csv(out/'summary.csv',grouped,gc)
    pc=['system','budget','rep','dishperf_mre_percent','hyperf_mre_percent','delta_mre_pp','dishperf_pmape_percent','hyperf_pmape_percent','delta_pmape_pp','hyperf_diagnostics_pass']
    write_csv(out/'paired.csv',paired,pc)
    write_csv(out/'workloads.csv',per,['system','budget','rep','profile','model','save_draws','workload','n_test','mre_percent','pmape_percent','mae','rmse','coverage90','width90'])
    ec=['system','budget','n_pairs','hyperf_diagnostic_pass_pairs']+[m+s for m in ['mre','pmape'] for s in ['_mean_delta_pp','_median_delta_pp','_dishperf_win_fraction','_exact_signed_rank_sign_flip_p','_holm_p']]
    write_csv(out/'paired_effects.csv',effects,ec)
    sensitivity=[]
    for e in effects:
        subset=[x for x in paired if x['system']==e['system'] and x['budget']==e['budget'] and x['hyperf_diagnostics_pass']]
        row={'system':e['system'],'budget':e['budget'],'all_finite_pairs':e['n_pairs'],'diagnostic_pass_pairs':len(subset),'selection_warning':'Pass-only subset may be selection biased; not the primary result'}
        for metric in ['mre','pmape']:row[metric+'_mean_delta_pp']=float(np.mean([x['delta_'+metric+'_pp'] for x in subset])) if subset else None
        sensitivity.append(row)
    write_csv(out/'diagnostic_pass_sensitivity.csv',sensitivity,['system','budget','all_finite_pairs','diagnostic_pass_pairs','mre_mean_delta_pp','pmape_mean_delta_pp','selection_warning'])
    counts={k:sum(s['status']==k for s in states) for k in ['completed','completed_with_warning','not_run','partial','failed','interrupted','corrupt']}
    meta={'profile':a.profile,'registered_tasks':len(jobs),'counts':counts,'finite_pairs':len(paired),'all_registered_tasks_completed':counts['completed']+counts['completed_with_warning']==len(jobs),'all_posterior_diagnostics_pass':all(m.get('diagnostics_pass') for k,m in valid_scores.items() if k[-1]=='hyperf') if any(k[-1]=='hyperf' for k in valid_scores) else None,'warning_runs_in_primary_summary':True,'trimming':False,'sd':'sample SD across completed repetitions, not CI; unavailable for n=1','point_estimator_hyperf':'upstream 10% HDI midpoint','corrupt':corrupt,'stats':'Conditional exact sign-flip distribution of signed ranks; requires paired symmetric difference assumptions; no p until five pairs; Holm only when fixed twelve-setting family complete. Five pairs cannot yield conventional two-sided p<.05.','smoke':'Execution verification only; native first workload and HyPerf all workloads; no paired accuracy claim' if a.profile=='smoke' else None}
    write(out/'status.json',meta);write(out/'summary.json',{'metadata':meta,'groups':grouped,'paired_effects':effects})
    (out/'README.txt').write_text('All finite completed runs included, with warning status retained. Missing scores are blank, never zero.\nNegative paired deltas favor DISHPerf. SD is not a confidence interval.\nModel compute seconds may reconstruct cached work; outer_seconds is the final worker attempt, not total historical cost.\nSmoke is not a scientific accuracy comparison.\n',encoding='utf-8')
    print('SUMMARY',counts,'finite paired repeats=',len(paired),'saved:',out,flush=True);return out
