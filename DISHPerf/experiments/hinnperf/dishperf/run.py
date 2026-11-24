"""Paper-aligned DISHPerf. Defaults to planning; never modifies source data."""
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,copy,csv,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
from protocol import ROOT,VERSION,sha,read,write,identity,load_raw,split
from utils.data_preproc import DataPreproc,system_samplesize,compute_top_k

def screen(raw,ids,system):
    p=DataPreproc.__new__(DataPreproc);p.config_num=raw.shape[1]-1
    p.top_k=compute_top_k(int(min(system_samplesize(system))))
    p.important_features=[]
    pairs=p.find_interactions_from_data(raw[ids['fit_ids'],:-1],raw[ids['fit_ids'],-1:])
    audit=copy.deepcopy(p.screening_audit)
    audit.update(screen_row_ids=list(ids['fit_ids']),selected_pairs=[list(pair) for pair in pairs],
                 candidate_features=list(p.important_features),target_k=int(p.top_k),
                 network_residual='full_original_input',scope='fit_only_frozen_during_refit')
    return pairs,audit

def grid(dim,pairs,smoke=False):
    return [dict(input_dim=dim,num_neuron=128,num_block=b,num_layer_pb=l,
                 lamda=lam,linear=False,gnorm=g,lr=.001,decay=None,verbose=False,
                 interaction_pairs=[list(pair) for pair in pairs],critical_features=[],use_interactions=True)
            for b,l,lam,g in itertools.product([2] if smoke else [2,3,4,5],
                [2] if smoke else [2,3,4],[.01] if smoke else [.001,.01,.1,1.,10.],
                [False] if smoke else [False,True])]

def task(args,budget,rep):
    package=identity();data=Path(args.data_root)/(args.system+'_AllNumeric.csv')
    baseline=read(ROOT/'data_manifest.json')['systems'][args.system]
    dh=sha(data)
    if dh!=baseline['sha256']:raise ValueError('Original CSV identity mismatch: '+str(data))
    raw=load_raw(data);budgets=list(map(int,system_samplesize(args.system)))
    ids=split(args.system,budget,rep,raw,dh,budgets)
    base=Path(args.output) if args.output else ROOT/'results'
    mode='verification_only' if args.stage=='smoke' else 'formal'
    folder=base/mode/('{}_b{}_r{}'.format(args.system,budget,rep))
    ident=dict(package_id=package,version=VERSION,system=args.system,budget=budget,rep=rep,
               protocol=ids['policy'],data_sha256=dh,mode=mode,
               screening='fit_only; fixed selected pairs at refit',
               grid='singleton' if mode=='verification_only' else 'original120',
               max_epochs=2 if mode=='verification_only' else 2000)
    if args.stage=='plan':
        print(json.dumps(dict(identity=ident,seed=ids['seed'],n_train=budget,
              n_fit=len(ids['fit_ids']),n_validation=len(ids['validation_ids']),
              n_test=len(ids['test_ids']),training_started=False),ensure_ascii=False))
        return
    if (folder/'complete.json').exists():
        if read(folder/'manifest.json')!=ident:raise ValueError('Result identity mismatch')
        for n,h in read(folder/'complete.json')['files'].items():
            if sha(folder/n)!=h:raise ValueError('Completed artifact corrupt: '+n)
        print('VERIFIED COMPLETE',folder);return
    folder.mkdir(parents=True,exist_ok=True)
    # Exclusive task lock prevents two dispatchers from training the same task.
    lock=folder/'task.lock'
    try:lock_fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Task locked; inspect task.lock before resuming '+str(folder))
    try:
        os.write(lock_fd,str(os.getpid()).encode('ascii'));os.close(lock_fd)
        if (folder/'manifest.json').exists() and read(folder/'manifest.json')!=ident:
            raise ValueError('Partial task belongs to a different implementation')
        write(folder/'manifest.json',ident);write(folder/'split.json',ids)
        pairs,audit=screen(raw,ids,args.system);write(folder/'screening.json',audit)
        configs=grid(raw.shape[1]-1,pairs,mode=='verification_only');write(folder/'grid.json',configs)
        import tensorflow as tf
        if tf.__version__!='1.10.0':raise ValueError('Use the verified TensorFlow 1.10.0 native environment')
        from runner import select,refit_predict
        cache=read(folder/'candidates.json') if (folder/'candidates.json').exists() else []
        max_epochs=ident['max_epochs']
        best,records=select(configs,raw,ids,max_epochs,
                    checkpoint=lambda rows:write(folder/'candidates.json',rows),cached=cache)
        write(folder/'selected_config.json',best)
        pred,refit=refit_predict(best,raw,ids,max_epochs);write(folder/'refit.json',refit)
        y=raw[ids['test_ids'],-1]
        metrics=dict(mre_percent=float(np.mean(abs(y-pred)/y)*100),
              mae=float(np.mean(abs(y-pred))),rmse=float(np.sqrt(np.mean((y-pred)**2))),
              n_test=len(y),negative_predictions=int((pred<0).sum()),system=args.system,
              budget=budget,rep=rep,model='dishperf_paper_aligned',mode=mode,
              candidate_count=len(configs),training_steps_refit=refit['steps'])
        if mode=='verification_only':metrics['reportable_paper_result']=False
        with (folder/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['row_id','y_true','y_pred'])
            w.writerows(zip(ids['test_ids'],map(float,y),map(float,pred)))
        write(folder/'metrics.json',metrics)
        names=['manifest.json','split.json','screening.json','grid.json','candidates.json',
               'selected_config.json','refit.json','predictions.csv','metrics.json']
        write(folder/'complete.json',dict(status='complete',files={n:sha(folder/n) for n in names}))
        print('DONE',args.system,budget,rep,'mode='+mode,flush=True)
    finally:
        lock.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','run','smoke'],default='plan')
    p.add_argument('--data-root',required=True)
    p.add_argument('--system',required=True,choices=list(read(ROOT/'data_manifest.json')['systems']))
    p.add_argument('--budget',type=int)
    p.add_argument('--start-rep',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--output')
    a=p.parse_args()
    if a.repeats<1 or a.start_rep<0 or a.start_rep+a.repeats>30:p.error('rep range must stay in 0..29')
    budgets=[a.budget] if a.budget else list(map(int,system_samplesize(a.system)))
    for rep in range(a.start_rep,a.start_rep+a.repeats):
        for b in budgets:task(a,int(b),rep)

if __name__=='__main__':main()
