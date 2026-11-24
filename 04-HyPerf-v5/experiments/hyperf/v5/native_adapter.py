"""Original DISHPerf-2.2 TensorFlow1.10 model/runner; explicit external split and fit-only screen bridge."""
from __future__ import print_function
import sys,time,types,importlib.util,copy
from pathlib import Path
import numpy as np
from common import policy,write,read,sha,digest_obj,IdentityError

def core_module():
    p=policy();path=Path(p['native_root'])/'native_experiment.py'
    spec=importlib.util.spec_from_file_location('native_external_core',str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.validate_sources();m.environment();return m

def pin_seed(method):
    g=method.__globals__.copy();g['seed_generator']=lambda system,budget:0
    result=types.FunctionType(method.__code__,g,method.__name__,method.__defaults__,method.__closure__)
    result.__kwdefaults__=method.__kwdefaults__;return result

def provider(core,train,test,features,budget,seed,system,min_budget):
    dp,_,_,_=core.load_native_modules()
    class FixedExternalData(dp.DataPreproc):
        def _check(self,n,s):
            if n!=budget or s!=seed:raise ValueError('Runner attempted an unregistered external split')
        def get_train_valid_samples(self,n,s,gnorm=False):
            self._check(n,s);cut=int(np.ceil(n*2/3));x=self.X_train[:cut];y=self.Y_train[:cut];xv=self.X_train[cut:];yv=self.Y_train[cut:]
            if gnorm:
                xx,yy,mx,sx,my,sy=self._DataPreproc__normalize_gaussian(x,y);sx=sx+(sx==0)*.001;sy0=sy+(sy==0)*.001
                return xx,yy,(xv-mx)/sx,(yv-my)/sy0,my,sy
            xx,yy,mx,my=self._DataPreproc__normalize(x,y);return xx,yy,xv/mx,yv/my,my
        def get_train_test_samples(self,n,s,gnorm=False):
            self._check(n,s)
            if gnorm:
                xx,yy,mx,sx,my,sy=self._DataPreproc__normalize_gaussian(self.X_train,self.Y_train);sx=sx+(sx==0)*.001
                return xx,yy,(self.X_test-mx)/sx,self.Y_test,my,sy
            xx,yy,mx,my=self._DataPreproc__normalize(self.X_train,self.Y_train);return xx,yy,self.X_test/mx,self.Y_test,my
    data=FixedExternalData.__new__(FixedExternalData);data.sys_name=system;data.config_num=len(features);data.top_k=dp.compute_top_k(min_budget);data.important_features=[]
    data.X_train=train[features].values.astype(float);data.Y_train=train[['y']].values.astype(float)
    data.X_test=test[features].values.astype(float);data.Y_test=test[['y']].values.astype(float)
    cut=int(np.ceil(budget*2/3));start=time.perf_counter()
    counts={'attempts':0,'errors':0,'first_error':None}
    original_mi=dp.mutual_info_regression
    def tracked_mi(*args,**kwargs):
        counts['attempts']+=1
        try:return original_mi(*args,**kwargs)
        except Exception as e:
            counts['errors']+=1
            if counts['first_error'] is None:counts['first_error']=type(e).__name__+': '+str(e)
            raise
    dp.mutual_info_regression=tracked_mi
    try:data.interaction_pairs=data.find_interactions_from_data(data.X_train[:cut],data.Y_train[:cut],verbose=False)
    finally:dp.mutual_info_regression=original_mi
    data.screen_seconds=time.perf_counter()-start;data.mi_audit=counts
    if budget-cut<1:raise ValueError('No inner validation labels; unsupported original tuning')
    return data

def fit_one(train,test,features,budget,seed,system,min_budget,out,smoke=False):
    core=core_module();_,_,Runner,_=core.load_native_modules();data=provider(core,train,test,features,budget,seed,system,min_budget)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    configs=core.build_configs(data.config_num,data.interaction_pairs,data.important_features,'full','smoke1' if smoke else 'original120')
    context={'phase':'search','records':[],'normalizers':{g:core.normalizer_info(data,budget,seed,g,'search') for g in [False,True]}}
    Recorded=core.make_recording_model(context)
    candidate_counter=[0]
    class ExternalRunner(Runner):
        search_best_config=pin_seed(Runner.search_best_config)
        test=pin_seed(Runner.test)
        def _train_one_config(self,*args,**kwargs):
            c=args[0];candidate_counter[0]+=1;i=candidate_counter[0]
            cache=out/'candidates'/('{:03d}.json'.format(i))
            if cache.exists():
                item=read(cache)
                if digest_obj(item['config'])!=digest_obj(c) or digest_obj(item['payload'])!=item['payload_sha256']:raise IdentityError('Candidate cache changed')
                rec=item['payload']['record'];v=float(item['payload']['validation_mae'])
                if not np.isfinite(v):raise ValueError('Invalid candidate cache score')
                context['records'].append(rec)
                print('candidate {}/{} VERIFIED CACHE val_MAE={:.9g}'.format(i,len(configs),v),flush=True)
                return v
            print('candidate {}/{} blocks={} layers={} L1={} gaussian={}'.format(i,len(configs),c['num_block'],c['num_layer_pb'],c['lamda'],c['gnorm']),flush=True)
            t=time.perf_counter();v=Runner._train_one_config(self,*args,**kwargs)
            if not np.isfinite(v):raise ValueError('Original candidate produced nonfinite validation error; no hidden replacement')
            payload={'validation_mae':float(v),'record':{k:x for k,x in context['records'][-1].items() if not k.startswith('_')},'seconds':time.perf_counter()-t}
            write(cache,{'config':c,'payload':payload,'payload_sha256':digest_obj(payload)})
            print('  validation_MAE={:.9g} steps={} seconds={:.1f}'.format(v,context['records'][-1]['steps'],payload['seconds']),flush=True)
            return v
    r=ExternalRunner(data,Recorded);start=time.perf_counter()
    best,val=r.search_best_config(configs,budget,30,seed,verbose=False);search_seconds=time.perf_counter()-start
    context['phase']='refit';start=time.perf_counter();native_mre=r.test(best,budget,30,seed);refit_seconds=time.perf_counter()-start
    record=context['records'][-1];norm=core.normalizer_info(data,budget,seed,bool(best['gnorm']),'refit')
    pred=(record['_last_output']*norm['scale']+norm['mean']).ravel()
    if not np.isclose(core.evaluate(data.Y_test.ravel(),pred)['mre_percent'],native_mre,rtol=0,atol=1e-10):raise ValueError('Original runner MRE disagrees with saved predictions')
    cut=int(np.ceil(budget*2/3))
    info=dict(mi_audit=data.mi_audit,selection_status='selected' if data.interaction_pairs else 'empty_interactions',selected_config=best,selected_validation_mae=float(val),search_candidates=len(configs),screen_seconds=data.screen_seconds,search_seconds=search_seconds,refit_seconds=refit_seconds,refit_steps=record['steps'],refit_snapshot=record['last_snapshot_epoch'],fit_row_ids=train.row_id.iloc[:cut].tolist(),validation_row_ids=train.row_id.iloc[cut:].tolist(),test_row_ids=test.row_id.tolist(),critical_features=data.important_features,selected_pairs=data.interaction_pairs,top_k=data.top_k,seed=seed,original_model_and_training_bytecode=True,split_bridge='fixed external train/validation/test; cloned runner seed lookup only',residual='original2.2 [X_C,backcast], not manuscript [X,backcast]')
    info['candidate_compute_seconds']=sum(float(read(p)['payload']['seconds']) for p in (out/'candidates').glob('*.json'));info['model_compute_seconds']=data.screen_seconds+info['candidate_compute_seconds']+refit_seconds
    write(out/'details.json',info);write(out/'grid.json',configs)
    write(out/'training.json',{'models':[{k:v for k,v in row.items() if not k.startswith('_')} for row in context['records']]})
    np.savez_compressed(str(out/'weights.npz'),**record['_final_weights'])
    return pred,info
