"""Registered multi-budget, three-system comparison. Workers run in separate environments."""
from __future__ import print_function
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['PYTHONIOENCODING']='utf-8';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['JAX_ENABLE_X64']='false';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['MPLBACKEND']='Agg'
import argparse,sys,json,time,traceback,subprocess,threading,queue,contextlib,csv
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,policy,read,write,sha,digest_obj,check_package,split,table,checkpoint,verify_checkpoint,task_path,IdentityError
from metrics import score

def parser():
    a=argparse.ArgumentParser()
    a.add_argument('--stage',choices=['check','prepare','plan','smoke','run','reuse','summarize','plot'],default='plan')
    a.add_argument('--system',choices=['all','x264','lrzip','h2'],default='all');a.add_argument('--budget',type=int,default=0)
    a.add_argument('--run-count',type=int,default=5);a.add_argument('--start-rep',type=int,default=0)
    a.add_argument('--model',choices=['all','dishperf_native','hyperf'],default='all');a.add_argument('--profile',choices=['formal','smoke'],default='formal')
    a.add_argument('--retry-failed',action='store_true');a.add_argument('--save-predictive-draws',action='store_true')
    a.add_argument('--reuse-root');a.add_argument('--output-root');a.add_argument('--main-python');a.add_argument('--native-python')
    a.add_argument('--worker',action='store_true');a.add_argument('--rep',type=int,default=0);a.add_argument('--split-file')
    return a

def task_identity(task,info,key):
    return {'version':policy()['version'],'package_id':check_package(),'task':task,'split_sha256':key,'canonical_sha256':info['canonical_sha256'],'prediction_draws_saved':bool(task.get('save_draws',False))}

def load_registered(path):
    saved=read(path);info=saved['split'];key=digest_obj(info)
    if key!=saved['split_sha256']:raise IdentityError('Split file changed')
    d,f,entry=table(info['system']);idx=d.set_index('row_id',drop=False)
    if info['canonical_sha256']!=entry['canonical_sha256']:raise IdentityError('Split dataset differs')
    tr=idx.loc[info['train_ids']].reset_index(drop=True);te=idx.loc[info['test_ids']].reset_index(drop=True)
    if set(tr.row_id)&set(te.row_id):raise ValueError('Train/test overlap')
    return tr,te,f,info,key

def selected_jobs(a):
    p=policy();systems=list(p['systems']) if a.system=='all' else [a.system]
    if a.system=='all' and a.budget:raise ValueError('A numeric budget requires one system')
    if a.run_count<1 or a.start_rep<0 or a.start_rep+a.run_count>p['repeats']:raise ValueError('Reps are registered 0..4; StartRep+RunCount <=5')
    if a.profile=='smoke' and (a.start_rep!=0 or a.run_count!=1):raise ValueError('Smoke accepts only rep0/RunCount1')
    jobs=[]
    for system in systems:
        spec=p['systems'][system];budgets=[int(alpha*spec['p']) for alpha in p['alphas']]
        if a.budget and a.budget not in budgets:raise ValueError('Unregistered '+system+' budget '+str(a.budget))
        chosen=[a.budget] if a.budget else ([spec['p']] if a.profile=='smoke' else budgets)
        for b in chosen:
            for rep in range(a.start_rep,a.start_rep+a.run_count):
                for model in (['dishperf_native','hyperf'] if a.model=='all' else [a.model]):
                    jobs.append({'system':system,'budget':b,'rep':rep,'profile':a.profile,'model':model,'save_draws':bool(a.save_predictive_draws)})
    return jobs

def base_path(a):return Path(a.output_root).resolve() if a.output_root else HERE/'results'
def plan_jobs(a):
    from prepare_data import prepare
    prepare();check_package();jobs=selected_jobs(a);records=[]
    for j in jobs:
        tr,te,f,info,key=split(j['system'],j['budget'],j['rep'],j['profile'])
        path=HERE/'splits'/j['profile']/(j['system']+'_b'+str(j['budget'])+'_r'+str(j['rep'])+'.json')
        obj={'split':info,'split_sha256':key}
        if path.exists() and read(path)!=obj:raise IdentityError('Registered split changed')
        if not path.exists():write(path,obj)
        row=dict(j,workloads=len(info['workload_order']),total_labels=len(tr),n_test=len(te),split_sha256=key,split_file=str(path))
        row['candidate_fits']=len(info['workload_order'])*(1 if j['profile']=='smoke' else 120) if j['model']=='dishperf_native' else 0
        records.append(row)
    dest=HERE/'plans'/a.profile;dest.mkdir(parents=True,exist_ok=True)
    name=a.system+'_b'+str(a.budget)+'_r'+str(a.start_rep)+'_n'+str(a.run_count)+'_'+a.model
    write(dest/(name+'.json'),records)
    with (dest/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return jobs

def check_predictions(frame,test,model):
    if frame.row_id.tolist()!=test.row_id.tolist() or frame.workload.tolist()!=test.workload.tolist() or not np.allclose(frame.y.values,test.y.values,rtol=1e-12,atol=1e-12):raise IdentityError('Predictions differ from registered test observations')
    return score(frame,model=='hyperf')

def read_completed(out,identity,test,model):
    saved=verify_checkpoint(out,identity);frame=pd.read_csv(str(out/'predictions.csv'),float_precision='round_trip');computed=check_predictions(frame,test,model)
    for key in ['mre_percent','pmape_percent','mae_macro','rmse_macro','mre_micro_percent','pmape_micro_percent']:
        if not np.isclose(saved[key],computed[key],rtol=1e-12,atol=1e-10):raise IdentityError('Saved metric changed '+key)
    return saved

def worker(a):
    t0=time.perf_counter();check_package();tr,te,f,info,key=load_registered(a.split_file)
    task={'system':a.system,'budget':a.budget,'rep':a.rep,'profile':a.profile,'model':a.model,'save_draws':bool(a.save_predictive_draws)}
    if any(task[k]!=info[k] for k in ['system','budget','rep','profile']):raise IdentityError('Worker task/split mismatch')
    if a.profile=='smoke' and a.model=='dishperf_native':
        first=info['workload_order'][0];tr=tr[tr.workload==first].reset_index(drop=True);te=te[te.workload==first].reset_index(drop=True)
    out=task_path(base_path(a),a.profile,a.system,a.budget,a.rep,a.model);identity=task_identity(task,info,key)
    if (out/'complete.json').exists():read_completed(out,identity,te,a.model);print('VERIFIED SKIP',out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=identity:raise IdentityError('Partial task identity changed')
    write(out/'manifest.json',identity);write(out/'split.json',info)
    if (out/'failure.json').exists():
        old=read(out/'failure.json');write(out/('previous_failure_'+str(time.time_ns() if hasattr(time,'time_ns') else int(time.time()*1e9))+'.json'),old);(out/'failure.json').unlink()
    resumed=False
    try:
        if a.model=='hyperf':
            from hyperf_adapter import fit_and_score
            frame,details=fit_and_score(tr,te.drop(columns=[]),f,info['workload_order'],out,a.profile,a.save_predictive_draws)
            resumed=bool(details['posterior_reused']);compute=details['fit_seconds']+details['prediction_seconds'];diag=details['diagnostics_pass'];warnings=details['diagnostic_warning_reasons']
        else:
            from native_adapter import fit_one
            frame=te[['row_id','workload','y']].copy();details={};compute=0.;diag=None;warnings=[]
            for i,w in enumerate(tr.workload.drop_duplicates()):
                wt=tr[tr.workload==w].reset_index(drop=True);positions=np.flatnonzero(te.workload.to_numpy()==w);we=te.iloc[positions].reset_index(drop=True)
                sub=out/('workload_'+str(i).zfill(2));sub.mkdir(parents=True,exist_ok=True);ident=dict(identity,workload=str(w))
                if (sub/'complete.json').exists():
                    saved=read_completed(sub,ident,we,a.model);pred=pd.read_csv(str(sub/'predictions.csv'),float_precision='round_trip').prediction.values;detail=read(sub/'details.json');resumed=True
                    print('VERIFIED workload checkpoint',w,flush=True)
                else:
                    if (sub/'manifest.json').exists() and read(sub/'manifest.json')!=ident:raise IdentityError('Workload identity differs')
                    write(sub/'manifest.json',ident)
                    if (sub/'candidates').exists() and list((sub/'candidates').glob('*.json')):resumed=True
                    env_seed=int(info['by_workload'][str(w)]['environment_seed'])
                    print('WORKLOAD {}/{}: {} fit={} validation={}'.format(i+1,tr.workload.nunique(),w,int(np.ceil(2*a.budget/3)),a.budget-int(np.ceil(2*a.budget/3))),flush=True)
                    pred,detail=fit_one(wt,we,f,a.budget,env_seed,a.system,int(min(policy()['alphas'])*policy()['systems'][a.system]['p']),sub,a.profile=='smoke')
                    sf=we[['row_id','workload','y']].copy();sf['prediction']=pred;sf.to_csv(str(sub/'predictions.csv'),index=False,float_format='%.17g');write(sub/'metrics.json',score(sf));checkpoint(sub)
                frame.loc[positions,'prediction']=pred;details[str(w)]=detail;compute+=float(detail['model_compute_seconds'])
                if detail['selection_status']=='empty_interactions':warnings.append('empty_interactions:'+str(w))
        computed=check_predictions(frame,te,a.model);entry=read(HERE/'data/preparation.json')['systems'][a.system]
        costs=[entry['raw_count_by_row_id'][r] for r in tr.row_id]
        computed.update(task=task,split_sha256=key,execution_status='completed',prediction_finite=True,diagnostics_pass=diag,warning_reasons=warnings,completed_with_warning=bool(warnings),formal_eligible=a.profile=='formal',outer_seconds=time.perf_counter()-t0,model_compute_seconds=compute,timing_resumed=resumed,timing_note='outer_seconds is this worker attempt; model_compute_seconds sums recorded model work, including cached candidates/posterior fit; do not label reconstructed time as a fresh-process benchmark',unique_training_labels=len(tr),raw_training_records=int(sum(costs)),details=details)
        frame.to_csv(str(out/'predictions.csv'),index=False,float_format='%.17g');write(out/'metrics.json',computed);checkpoint(out)
        print('DONE {} {} B={} rep={} MRE={:.6f}% pMAPE={:.6f}% status={}'.format(a.system,a.model,a.budget,a.rep,computed['mre_percent'],computed['pmape_percent'],'completed_with_warning' if warnings else 'completed'),flush=True)
    except BaseException as e:
        write(out/'failure.json',{'execution_status':'interrupted' if isinstance(e,KeyboardInterrupt) else 'failed','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'seconds_this_attempt':time.perf_counter()-t0,'task':task});raise

def environment_check(a):
    check_package();p=policy();main=a.main_python or p['main_python'];native=a.native_python or p['native_python']
    for exe in [main,native]:
        if not Path(exe).is_file():raise FileNotFoundError(exe)
    env=os.environ.copy();prefix=str(Path(native).parent);env['PATH']=os.pathsep.join([prefix,str(Path(prefix)/'Library/bin'),str(Path(prefix)/'Scripts'),env.get('PATH','')])
    code="import sys;sys.path.insert(0,"+repr(str(HERE))+ ");from native_adapter import core_module;c=core_module();import json;print(json.dumps(c.environment(),indent=2))"
    subprocess.check_call([native,'-B','-c',code],env=env,cwd=str(HERE))
    code="import sys;sys.path.insert(0,"+repr(str(HERE))+ ");from hyperf_adapter import imports,versions;from common import read,HERE;u,d,j=imports(3);import json;v=versions(j);e=read(HERE/'environment_lock.json')['hyperf'];assert v['python']==e['python'] and v['packages']==e['packages'], 'HyPerf environment differs from lock';print(json.dumps(v,indent=2))"
    subprocess.check_call([main,'-B','-c',code],env=os.environ.copy(),cwd=str(HERE))
    print('CHECK PASSED. Source-aligned model/sampler; available HyPerf dependency versions are recorded, not asserted identical to the paper environment.',flush=True)

@contextlib.contextmanager
def queue_lock():
    handle=(HERE/'queue.lock').open('a+b')
    import msvcrt
    handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:handle.close();raise RuntimeError('Another queue is active in this package')
    try:yield
    finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()

def run_queue(a,jobs):
    p=policy();failures=0
    with queue_lock():
        for task in jobs:
            out=task_path(base_path(a),task['profile'],task['system'],task['budget'],task['rep'],task['model'])
            if (out/'failure.json').exists() and not a.retry_failed and not (out/'complete.json').exists():
                print('SKIP previous failure; inspect and use -RetryFailed to retry SAME protocol:',out,flush=True);failures+=1;continue
            exe=(a.native_python or p['native_python']) if task['model']=='dishperf_native' else (a.main_python or p['main_python'])
            splitfile=HERE/'splits'/a.profile/(task['system']+'_b'+str(task['budget'])+'_r'+str(task['rep'])+'.json')
            cmd=[exe,'-B','-u',str(HERE/'experiment.py'),'--worker','--system',task['system'],'--budget',str(task['budget']),'--rep',str(task['rep']),'--profile',task['profile'],'--model',task['model'],'--split-file',str(splitfile),'--output-root',str(base_path(a))]
            if task['save_draws']:cmd.append('--save-predictive-draws')
            env=os.environ.copy()
            if task['model']=='dishperf_native':
                prefix=str(Path(exe).parent);env['PATH']=os.pathsep.join([prefix,str(Path(prefix)/'Library/bin'),str(Path(prefix)/'Scripts'),env.get('PATH','')])
            logs=HERE/'logs'/a.profile;logs.mkdir(parents=True,exist_ok=True);log=logs/(task['system']+'_b'+str(task['budget'])+'_r'+str(task['rep'])+'_'+task['model']+'.log')
            write(HERE/'queue_status.json',{'status':'running','task':task,'log':str(log),'pid':os.getpid()});print('TASK',task,'log:',log,flush=True)
            start=time.perf_counter();last=start;lines=queue.Queue()
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=str(HERE),env=env,universal_newlines=True,encoding='utf-8',errors='replace',bufsize=1)
            def pump(stream):
                for line in stream:lines.put(line)
                lines.put(None)
            thread=threading.Thread(target=pump,args=(proc.stdout,));thread.daemon=True;thread.start()
            with log.open('a',encoding='utf-8') as stream:
                stream.write('\nATTEMPT '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');stream.flush()
                try:
                    ended=False
                    while not ended:
                        try:
                            line=lines.get(timeout=2)
                            if line is None:ended=True
                            else:print(line,end='',flush=True);stream.write(line);stream.flush()
                        except queue.Empty:pass
                        now=time.perf_counter()
                        if now-last>=30:print('RUNNING {} {} B={} rep={} elapsed {:.0f}s'.format(task['system'],task['model'],task['budget'],task['rep'],now-start),flush=True);last=now
                    code=proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    write(HERE/'queue_status.json',{'status':'interrupted','task':task,'completed_checkpoints_preserved':True});raise
            attempt={'task':task,'return_code':code,'fresh_process_seconds':time.perf_counter()-start,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
            history=out/'driver_attempts.json';items=read(history) if history.exists() else [];items.append(attempt);write(history,items)
            if code:
                failures+=1
                if not (out/'failure.json').exists():write(out/'failure.json',dict(attempt,execution_status='failed',error='Worker exited before failure metadata; see log '+str(log)))
                print('TASK FAILED; preserved. Continuing independent tasks.',flush=True)
        write(HERE/'queue_status.json',{'status':'finished_with_failures' if failures else 'finished','selected_tasks':len(jobs),'failed_or_previously_failed':failures})
    return failures

def main():
    a=parser().parse_args()
    if a.worker:return worker(a)
    if a.stage=='check':return environment_check(a)
    if a.stage=='prepare':
        from prepare_data import prepare
        prepare();check_package();return
    if a.stage=='smoke':a.profile='smoke';a.run_count=1;a.start_rep=0
    jobs=plan_jobs(a)
    if a.stage=='plan':
        print(json.dumps({'tasks':len(jobs),'systems':sorted(set(j['system'] for j in jobs)),'sampler':policy()['hyperf'],'profile':a.profile,'note':'Plan only; no training'},ensure_ascii=False,indent=2));return
    if a.stage in ['summarize','plot']:
        from summarize import summarize
        path=summarize(a,jobs)
        if a.stage=='plot':
            from plot_results import plot
            plot(path)
        return
    if a.stage=='reuse':
        from reuse_results import reuse
        with queue_lock():reuse(a,jobs)
        from summarize import summarize
        from plot_results import plot
        plot(summarize(a,jobs));return
    failed=run_queue(a,jobs)
    from summarize import summarize
    path=summarize(a,jobs)
    from plot_results import plot
    plot(path)
    print('QUEUE FINISHED. Summary:',path,'Execution failures:',failed,'Diagnostic warnings do not abort the queue.',flush=True)
    if failed:raise SystemExit(2)
if __name__=='__main__':main()
