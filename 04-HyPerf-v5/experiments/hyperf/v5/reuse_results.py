"""Re-evaluate verified v4 predictions; explicit provenance, no retraining."""
import json,copy
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,policy,read,write,sha,digest_obj,verify_checkpoint,checkpoint,task_path,IdentityError

def source_package(root):
    m=read(root/'package_manifest.json')
    for n,h in m['files'].items():
        if sha(root/n)!=h:raise IdentityError('Source package changed: '+n)
    old=read(root/'settings.json');new=policy()
    if old['alphas']!=[.5,1.,2.] or old['max_pool_alpha']!=2.:raise IdentityError('Expected the v4 2p protocol')
    if old['systems']!=new['systems'] or old['hyperf']!=new['hyperf'] or old['repeats']!=new['repeats'] or old['screen_scope']!=new['screen_scope']:raise IdentityError('Source protocol is incompatible')
    # These are unchanged model/scoring implementations, not merely matching labels.
    current=read(HERE/'package_manifest.json')
    names=['native_adapter.py','hyperf_adapter.py','metrics.py','environment_lock.json']+[n for n in m['files'] if n.startswith(('native_core/','vendor/'))]
    for n in names:
        if m['files'][n]!=current['files'].get(n):raise IdentityError('Model/environment source differs: '+n)
    return m

def reuse(a,jobs):
    from experiment import load_registered,task_identity,read_completed,check_predictions,base_path
    root=Path(a.reuse_root or policy()['reuse_root']).resolve()
    if root==HERE:raise ValueError('Reuse root must be an independent v4 suite')
    sm=source_package(root);count=0;missing=[];new_budget=[]
    for task in jobs:
        s,b,r,model=task['system'],task['budget'],task['rep'],task['model']
        if task['profile']!='formal' or task.get('save_draws'):raise ValueError('Reuse supports formal default predictions only')
        _,test,_,info,key=load_registered(HERE/'splits/formal'/('{}_b{}_r{}.json'.format(s,b,r)))
        ident=task_identity(task,info,key);out=task_path(base_path(a),'formal',s,b,r,model)
        if (out/'complete.json').exists():read_completed(out,ident,test,model);print('VERIFIED EXISTING',out,flush=True);continue
        if b>2*policy()['systems'][s]['p']:new_budget.append(task);continue
        src=task_path(root/'results','formal',s,b,r,model)
        if not (src/'complete.json').exists():missing.append(task);print('SOURCE NOT COMPLETE; no training:',task,flush=True);continue
        old_split=read(root/'splits/formal'/('{}_b{}_r{}.json'.format(s,b,r)))
        oldinfo=old_split['split'];oldkey=digest_obj(oldinfo)
        if oldkey!=old_split['split_sha256']:raise IdentityError('Source split hash changed')
        source_identity={'version':read(root/'settings.json')['version'],'package_id':sm['package_id'],'task':task,'split_sha256':oldkey,'canonical_sha256':oldinfo['canonical_sha256'],'prediction_draws_saved':False}
        saved=verify_checkpoint(src,source_identity)
        if saved['task']!=task or not saved.get('formal_eligible') or not saved.get('prediction_finite'):raise IdentityError('Source result metadata differs')
        if oldinfo['canonical_sha256']!=info['canonical_sha256'] or oldinfo['train_ids']!=info['train_ids'] or oldinfo['workload_order']!=info['workload_order']:raise IdentityError('Training observations/data differ; reuse forbidden')
        if not set(info['test_ids']).issubset(oldinfo['test_ids']):raise IdentityError('New test is not a subset of the old test')
        olddata,_,_=__import__('common').table(s);idx=olddata.set_index('row_id',drop=False)
        original_test=idx.loc[oldinfo['test_ids']].reset_index(drop=True)
        frame=pd.read_csv(src/'predictions.csv',float_precision='round_trip')
        oldcomputed=check_predictions(frame,original_test,model)
        for k in ['mre_percent','pmape_percent','mae_macro','rmse_macro','mre_micro_percent','pmape_micro_percent']:
            if not np.isclose(oldcomputed[k],saved[k],rtol=1e-12,atol=1e-10):raise IdentityError('Source metric differs: '+k)
        sub=frame.set_index('row_id',drop=False).loc[info['test_ids']].reset_index(drop=True)
        computed=check_predictions(sub,test,model)
        if out.exists() and any(out.iterdir()) and not (out/'reuse_provenance.json').exists():raise IdentityError('Target has a partial training task; reuse will not overwrite it: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        provenance={'kind':'verified_model_predictions_re_evaluated_on_common_3p_holdout','source_root':str(root),'source_package_id':sm['package_id'],'source_task':task,'source_split_sha256':oldkey,'target_split_sha256':key,'source_complete_sha256':sha(src/'complete.json'),'source_predictions_sha256':sha(src/'predictions.csv'),'same_training_row_ids_in_same_order':True,'source_n_test':len(frame),'target_n_test':len(sub),'models_retrained':False,'posterior_resampled':False,'timing_eligible':False,'source_platform':saved.get('details',{}).get('versions','see source runtime records'),'note':'This is an explicitly derived evaluation, not a new independent repetition. Original model diagnostics retained. Model compute time is historical, not a fresh-process benchmark.'}
        write(out/'reuse_provenance.json',provenance);write(out/'manifest.json',ident);write(out/'split.json',info)
        # Copy source metadata as evidence, without claiming model fitting occurred here.
        write(out/'source_metrics.json',saved)
        for k in ['execution_status','prediction_finite','diagnostics_pass','warning_reasons','completed_with_warning','formal_eligible','model_compute_seconds','unique_training_labels','raw_training_records','details']:
            computed[k]=copy.deepcopy(saved[k])
        computed.update(task=task,split_sha256=key,outer_seconds=0.,timing_resumed=True,evaluation_reused=True,timing_note=provenance['note'],source_provenance=provenance)
        sub.to_csv(out/'predictions.csv',index=False,float_format='%.17g');write(out/'metrics.json',computed);checkpoint(out)
        read_completed(out,ident,test,model);count+=1
        print('REUSED WITHOUT TRAINING',task,'MRE=',computed['mre_percent'],flush=True)
    write(HERE/'reuse_status.json',{'newly_reused':count,'source_incomplete':missing,'new_3p_tasks_not_trained':new_budget,'training_started':False})
    print('REUSE FINISHED; newly reused:',count,'source incomplete:',len(missing),'new 3p tasks:',len(new_budget),'NO TRAINING',flush=True)
