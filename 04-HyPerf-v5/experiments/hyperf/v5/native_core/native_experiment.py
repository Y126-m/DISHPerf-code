"""Native DISHPerf-2.2 adapter. Original TensorFlow model and runner are vendored unchanged."""
from __future__ import print_function
import argparse,copy,hashlib,importlib,json,math,os,platform,sys,time,traceback
from collections import OrderedDict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
VARIANTS=('full','no_explicit')
ARTIFACTS={'manifest.json','split.json','structure.json','grid.json','metrics.json','training.json','predictions.npz','predictions.csv','weights.npz'}
_LOADED=None

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=default),encoding='utf-8')

def protocol():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))

def validate_sources():
    c=protocol();actual={}
    for name,sha in c['upstream_sha256'].items():
        path=ROOT/'upstream'/name
        if not path.is_file() or digest(path)!=sha:raise ValueError('Vendored original changed: '+name)
        original=Path(c['source_root'])/name
        if original.is_file() and digest(original)!=sha:raise ValueError('Original source changed since freeze: '+name)
        actual[name]=sha
    return actual

def load_native_modules():
    global _LOADED
    validate_sources()
    if _LOADED is None:
        sys.path.insert(0,str(ROOT/'upstream'))
        dp=importlib.import_module('utils.data_preproc')
        mm=importlib.import_module('models.models')
        rr=importlib.import_module('models.model_runner')
        aa=importlib.import_module('utils.args')
        for module in (dp,mm,rr,aa):
            if ROOT/'upstream' not in Path(module.__file__).resolve().parents:raise ValueError('Wrong native module imported: '+module.__file__)
        _LOADED=(dp,mm.DISHPerf,rr.ModelRunner,aa)
    return _LOADED

def environment():
    import tensorflow as tf,scipy,sklearn,pandas
    actual={'python':platform.python_version(),'tensorflow':tf.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'pandas':pandas.__version__}
    if actual!=protocol()['environment']:raise ValueError('Environment differs from frozen native environment: '+repr(actual))
    return {'versions':actual,'python_executable':sys.executable,'tensorflow_path':tf.__file__,'cpu_only':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','cuda_build':bool(tf.test.is_built_with_cuda())}

def screen_data(data,fit_ids):
    dp,_,_,_=load_native_modules()
    small=dp.DataPreproc.__new__(dp.DataPreproc)
    small.config_num=data.config_num;small.top_k=data.top_k;small.important_features=[]
    pairs=small.find_interactions_from_data(data.X_all[fit_ids],data.Y_all[fit_ids],verbose=False)
    return pairs,list(small.important_features)

def prepare_data(system,budget,repeat,registered_repetitions=30,screen_scope='fit_only'):
    if registered_repetitions!=30:raise ValueError('Keep registered repetitions=30 even when running a subset')
    if screen_scope not in ('fit_only','legacy_all'):raise ValueError('Unknown screening scope')
    if not 0<=repeat<registered_repetitions:raise ValueError('Repeat must be 0..29')
    dp,_,_,_=load_native_modules();c=protocol()
    if system not in c['datasets']:raise ValueError('Unknown system')
    entry=c['datasets'][system];path=Path(c['data_root'])/entry['filename']
    if digest(path)!=entry['sha256']:raise ValueError('Dataset changed: '+system)
    raw=np.genfromtxt(str(path),delimiter=',',skip_header=1)
    if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError('Expected finite numeric dataset')
    budgets=dp.system_samplesize(system)
    if budget not in budgets:raise ValueError('Use a registered original budget')
    if not 0<budget<len(raw):raise ValueError('Invalid budget')
    data=dp.DataPreproc.__new__(dp.DataPreproc)
    data.sys_name=system;data.data_dir=str(path);data.whole_data=raw
    data.all_sample_num=len(raw);data.config_num=raw.shape[1]-1
    data.X_all=raw[:,:-1];data.Y_all=raw[:,-1,None]
    data.top_k=dp.compute_top_k(int(np.min(budgets)))
    seed=int(dp.seed_generator(system,budget))*registered_repetitions+(repeat+1)
    order=np.random.RandomState(seed).permutation(len(raw));nfit=int(np.ceil(budget*2/3))
    fit=order[:nfit];val=order[nfit:budget];test=np.setdiff1d(np.arange(len(raw)),order[:budget])
    used=fit if screen_scope=='fit_only' else np.arange(len(raw))
    data.interaction_pairs,data.important_features=screen_data(data,used)
    split_hash=hashlib.sha256(entry['sha256'].encode('ascii')+np.asarray([budget,repeat,nfit,seed],dtype='<i8').tobytes()+order.astype('<i8').tobytes()).hexdigest()
    data.audit={'system':system,'budget':budget,'repeat':repeat,'cur_exp':repeat+1,'registered_repetitions':registered_repetitions,'seed':seed,'fit_row_ids':fit.tolist(),'validation_row_ids':val.tolist(),'test_row_ids':test.tolist(),'screen_row_ids':used.tolist(),'screen_scope':screen_scope,'data_sha256':entry['sha256'],'split_hash':split_hash,'n_fit':len(fit),'n_validation':len(val),'n_test':len(test),'top_k':data.top_k,'heldout_label_access_in_screening':screen_scope=='legacy_all','interpretation':'forensic only; full-pool supervision' if screen_scope=='legacy_all' else 'native 2.2 training, fit-only screening correction; not full manuscript reproduction','row_policy':'original rows, no deduplication; duplicate-configuration audit still required'}
    return data

def build_configs(n,pairs,critical,variant='full',grid_mode='original120'):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    if grid_mode not in ('original120','smoke1'):raise ValueError('Unknown grid')
    _,_,_,aa=load_native_modules()
    values=OrderedDict([('input_dim',[n]),('num_neuron',[128]),('num_block',[2,3,4,5]),('num_layer_pb',[2,3,4]),('lamda',[.001,.01,.1,1.,10.]),('linear',[False]),('gnorm',[False,True]),('lr',[.001]),('decay',[None]),('verbose',[False]),('interaction_pairs',[copy.deepcopy(pairs)]),('critical_features',[list(critical)]),('use_interactions',[variant=='full'])])
    if grid_mode=='smoke1':values.update(num_block=[3],num_layer_pb=[2],lamda=[.001],gnorm=[False])
    return aa.list_of_param_dicts(values)

class SessionProxy:
    """Observe the exact original Session.run call; never issue another optimizer step."""
    def __init__(self,session,model,record):self.session=session;self.model=model;self.record=record
    def __getattr__(self,name):return getattr(self.session,name)
    def run(self,fetches,feed_dict=None,**kwargs):
        result=self.session.run(fetches,feed_dict=feed_dict,**kwargs)
        if isinstance(fetches,(list,tuple)) and any(f is self.model.train_op for f in fetches):
            self.record['steps']+=1
            if self.record['steps']%20==0:
                for i,f in enumerate(fetches):
                    if f is self.model.loss:self.record['train_history'].append({'epoch':self.record['steps'],'reported_objective':float(result[i])})
        elif fetches is self.model.output:
            self.record['_last_output']=np.asarray(result).copy()
            callback=self.record.get('_on_prediction')
            if callback:callback(np.asarray(result),self.record)
        elif id(fetches) in self.record.get('_variable_ids',set()):
            self.record['last_snapshot_epoch']=self.record['steps']
        return result

def normalizer_info(data,budget,seed,gnorm,phase):
    if phase=='search':
        values=data.get_train_valid_samples(budget,seed,gnorm=gnorm)
    else:values=data.get_train_test_samples(budget,seed,gnorm=gnorm)
    if gnorm:
        mean,sd=values[-2:];scale=sd+(sd==0)*.001
        return {'mode':'gaussian','scale':scale,'mean':mean,'validation_target':values[3]*scale+mean if phase=='search' else None}
    scale=values[-1]
    return {'mode':'max','scale':scale,'mean':0.,'validation_target':values[3]*scale if phase=='search' else None}

def make_recording_model(context):
    import tensorflow as tf
    _,Original,_,_=load_native_modules()
    class RecordingModel(Original):
        def __init__(self,cfg):
            Original.__init__(self,cfg)
            self._native_record={'phase':context['phase'],'config':copy.deepcopy(cfg),'steps':0,'last_snapshot_epoch':0,'train_history':[],'validation_history':[]}
        def build_train(self):
            Original.build_train(self)
            record=self._native_record
            record['_variable_ids']={id(v) for v in tf.trainable_variables()}
            if record['phase']=='search':
                norm=context['normalizers'][bool(record['config']['gnorm'])]
                def observe(pred,r):
                    inverse=pred*norm['scale']+norm['mean']
                    r['validation_history'].append({'epoch':r['steps'],'validation_mae':float(np.mean(np.abs(inverse-norm['validation_target'])))})
                record['_on_prediction']=observe
            self.sess=SessionProxy(self.sess,self,record);context['records'].append(record)
        def finalize(self):
            if self._native_record['phase']=='refit':
                variables=tf.trainable_variables()
                self._native_record['_final_weights']={v.name:a for v,a in zip(variables,self.sess.run(variables))}
            Original.finalize(self)
    return RecordingModel

def task_manifest(data,variant,grid_mode,env):
    return {'version':protocol()['version'],'system':data.sys_name,'budget':data.audit['budget'],'repeat':data.audit['repeat'],'variant':variant,'grid_mode':grid_mode,'screen_scope':data.audit['screen_scope'],'split_hash':data.audit['split_hash'],'environment':env,'upstream_sha256':validate_sources(),'adapter_sha256':digest(ROOT/'native_experiment.py'),'protocol_sha256':digest(ROOT/'protocol.json'),'data_sha256':data.audit['data_sha256'],'selection_scope':'fixed singleton diagnostic' if grid_mode=='smoke1' else 'original 120-candidate search','training_policy':'unchanged upstream ModelRunner search and test methods; native TF initialization; native test training-objective early stop'}

def evaluate(y,pred):
    y=np.asarray(y).ravel();pred=np.asarray(pred).ravel()
    if y.shape!=pred.shape or not np.isfinite(pred).all() or not np.isfinite(y).all():raise ValueError('Invalid predictions')
    if np.any(y==0):raise ValueError('Original MRE undefined on zero targets; no rows removed')
    ae=np.abs(y-pred)
    return {'mre_percent':float(np.mean(np.abs(np.divide(y-pred,y)))*100),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(ae**2))),'absolute_error_p90':float(np.quantile(ae,.9)),'n_test':int(len(y))}

def run_task(system,budget,repeat,variant,grid_mode,screen_scope,output_root):
    started=time.perf_counter();env=environment();data=prepare_data(system,budget,repeat,30,screen_scope)
    folder=Path(output_root)/grid_mode/screen_scope/('{}_b{}_r{:02d}'.format(system,budget,repeat))/variant
    manifest=task_manifest(data,variant,grid_mode,env)
    if (folder/'complete.json').exists():
        old=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if old!=manifest:raise ValueError('Completed provenance mismatch; use a new package/version')
        index=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if index.get('status')!='complete':raise ValueError('Invalid complete marker')
        if set(index.get('artifact_sha256',{}))!=ARTIFACTS:raise ValueError('Incomplete artifact index')
        for name,sha in index['artifact_sha256'].items():
            if digest(folder/name)!=sha:raise ValueError('Completed artifact changed: '+name)
        print('VERIFIED RESUME SKIP',folder,flush=True)
        return json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text(encoding='utf-8'))!=manifest:raise ValueError('Partial task provenance mismatch; use new version')
    dump(folder/'manifest.json',manifest);dump(folder/'split.json',data.audit)
    configs=build_configs(data.config_num,data.interaction_pairs,data.important_features,variant,grid_mode)
    dump(folder/'structure.json',{'critical_features':data.important_features,'selected_pairs':data.interaction_pairs,'active_pairs':data.interaction_pairs if variant=='full' else [],'top_k':data.top_k,'residual_policy':'[X_C, backcast], matches specified native 2.2 code; differs from manuscript equation (8)','cross_variant_initialization':'unchanged native TensorFlow graph initialization, shared-tensor equality is not assumed'})
    dump(folder/'grid.json',configs)
    _,_,Runner,_=load_native_modules()
    context={'phase':'search','records':[],'normalizers':{g:normalizer_info(data,budget,data.audit['seed'],g,'search') for g in (False,True)}}
    Recorded=make_recording_model(context)
    class LoggedRunner(Runner):
        def _train_one_config(self,*args,**kwargs):
            i=len(context['records'])+1;cfg=args[0]
            print('candidate {}/{} blocks={} layers={} L1={} gaussian={}'.format(i,len(configs),cfg['num_block'],cfg['num_layer_pb'],cfg['lamda'],cfg['gnorm']),flush=True)
            start=time.perf_counter();value=Runner._train_one_config(self,*args,**kwargs)
            print('  validation_MAE={:.9g} steps={} seconds={:.1f}'.format(value,context['records'][-1]['steps'],time.perf_counter()-start),flush=True)
            return value
    runner=LoggedRunner(data,Recorded)
    try:
        search_start=time.perf_counter()
        best,val=runner.search_best_config(configs,budget,30,repeat+1,verbose=False)
        search_seconds=time.perf_counter()-search_start
        context['phase']='refit';refit_start=time.perf_counter()
        native_mre=runner.test(best,budget,30,repeat+1)
        refit_seconds=time.perf_counter()-refit_start
        record=context['records'][-1];norm=normalizer_info(data,budget,data.audit['seed'],bool(best['gnorm']),'refit')
        prediction=(record['_last_output']*norm['scale']+norm['mean']).ravel()
        ids=np.asarray(data.audit['test_row_ids']);observed=data.Y_all[ids].ravel()
        metrics=evaluate(observed,prediction)
        if metrics['mre_percent']!=float(native_mre):raise ValueError('Saved prediction MRE differs from original runner result')
        metrics.update(system=system,budget=budget,repeat=repeat,variant=variant,grid_mode=grid_mode,screen_scope=screen_scope,screening_is_fit_only=screen_scope=='fit_only',native_runner_mre_percent=float(native_mre),selected_validation_mae=float(val),selected_config=best,search_candidates=len(configs),refit_steps=record['steps'],refit_best_snapshot_epoch=record['last_snapshot_epoch'],search_seconds=search_seconds,refit_seconds=refit_seconds,pipeline_seconds=time.perf_counter()-started,interpretation='native code execution diagnostic, not manuscript numeric reproduction')
        dump(folder/'metrics.json',metrics)
        histories=[{k:v for k,v in row.items() if not k.startswith('_')} for row in context['records']]
        dump(folder/'training.json',{'models':histories,'search_candidates':len(configs),'original_training_methods_unmodified':True,'timing_includes_observation_overhead':True})
        np.savez_compressed(str(folder/'predictions.npz'),source_row_id=ids,observed=observed,predicted=prediction)
        import pandas as pd
        pd.DataFrame({'source_row_id':ids,'observed':observed,'predicted':prediction}).to_csv(str(folder/'predictions.csv'),index=False,float_format='%.17g')
        np.savez_compressed(str(folder/'weights.npz'),**record['_final_weights'])
        names=['manifest.json','split.json','structure.json','grid.json','metrics.json','training.json','predictions.npz','predictions.csv','weights.npz']
        dump(folder/'complete.json',{'status':'complete','artifact_sha256':{name:digest(folder/name) for name in names}})
        if (folder/'failure.json').exists():(folder/'failure.json').unlink()
        print('DONE',system,budget,repeat,variant,'MRE={:.6f}%'.format(metrics['mre_percent']),flush=True)
        return metrics
    except Exception as error:
        dump(folder/'failure.json',{'status':'failed','type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        raise

def make_plan(profile,system=None,budget=None,repeat=None,variant=None,screen_scope='fit_only'):
    if profile not in ('smoke','searchcheck','pilot'):raise ValueError('Unknown profile')
    system=system or 'x264'
    dp,_,_,_=load_native_modules();registered=dp.system_samplesize(system).tolist()
    if budget is not None and budget not in registered:raise ValueError('Unregistered budget')
    if repeat is not None and not 0<=repeat<30:raise ValueError('Repeat must be 0..29')
    if variant is not None and variant not in VARIANTS:raise ValueError('Unknown variant')
    budgets=[budget] if budget is not None else ([registered[0],registered[2]] if profile=='pilot' else [registered[0]])
    repeats=[repeat] if repeat is not None else (list(range(3)) if profile=='pilot' else [0])
    selected=[variant] if variant else (['full'] if profile=='searchcheck' else list(VARIANTS))
    grid='smoke1' if profile=='smoke' else 'original120'
    tasks=[dict(system=system,budget=int(b),repeat=int(r),variant=v,grid_mode=grid,screen_scope=screen_scope) for b in budgets for r in repeats for v in selected]
    return {'version':protocol()['version'],'profile':profile,'tasks':tasks,'task_count':len(tasks),'search_candidate_count':len(tasks)*(1 if grid=='smoke1' else 120),'registered_repetitions':30,'environment':protocol()['environment'],'inference':'descriptive only; smoke singleton is execution verification','screen_scope':screen_scope}

def summarize(plan):
    import pandas as pd
    rows=[];counts={'complete':0,'not_run':0,'failed':0,'corrupt':0};failures=[]
    for task in plan['tasks']:
        folder=ROOT/'results'/task['grid_mode']/task['screen_scope']/('{}_b{}_r{:02d}'.format(task['system'],task['budget'],task['repeat']))/task['variant']
        if not (folder/'complete.json').exists():
            counts['failed' if (folder/'failure.json').exists() else 'not_run']+=1;continue
        try:
            index=json.loads((folder/'complete.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
            if manifest['adapter_sha256']!=digest(ROOT/'native_experiment.py') or manifest['protocol_sha256']!=digest(ROOT/'protocol.json'):raise ValueError('Source provenance changed')
            if index.get('status')!='complete':raise ValueError('Invalid completion marker')
            if set(index.get('artifact_sha256',{}))!=ARTIFACTS:raise ValueError('Incomplete artifact index')
            for k,v in task.items():
                if manifest.get(k)!=v:raise ValueError('Task identity mismatch: '+k)
            if manifest.get('upstream_sha256')!=validate_sources():raise ValueError('Upstream sources changed')
            for name,sha in index['artifact_sha256'].items():
                if digest(folder/name)!=sha:raise ValueError('Artifact hash mismatch: '+name)
            metric=json.loads((folder/'metrics.json').read_text())
            entry=protocol()['datasets'][task['system']];data_path=Path(protocol()['data_root'])/entry['filename']
            if digest(data_path)!=manifest['data_sha256'] or manifest['data_sha256']!=entry['sha256']:raise ValueError('Current dataset changed')
            raw=np.genfromtxt(str(data_path),delimiter=',',skip_header=1)
            split=json.loads((folder/'split.json').read_text())
            if manifest['split_hash']!=split['split_hash']:raise ValueError('Split identity mismatch')
            with np.load(str(folder/'predictions.npz'),allow_pickle=False) as z:
                if not np.array_equal(z['source_row_id'],split['test_row_ids']):raise ValueError('Test row order mismatch')
                if not np.array_equal(z['observed'],raw[np.asarray(split['test_row_ids']),-1]):raise ValueError('Observed targets differ from source CSV')
                recomputed=evaluate(z['observed'],z['predicted'])
            for k in recomputed:
                if metric[k]!=recomputed[k]:raise ValueError('Metric mismatch: '+k)
            rows.append(metric);counts['complete']+=1
        except Exception as error:counts['corrupt']+=1;failures.append({'task':task,'error':str(error)})
    out=ROOT/'summaries'/plan['profile']/plan['screen_scope'];out.mkdir(parents=True,exist_ok=True)
    dump(out/'status.json',{'counts':counts,'corrupt':failures,'scope':'native 2.2 reference, no manuscript accuracy claim','all_repeats_retained':True})
    if not rows:
        for name in ('runs.csv','summary.csv'):
            stale=out/name
            if stale.exists():stale.unlink()
    if rows:
        df=pd.DataFrame(rows);df.to_csv(str(out/'runs.csv'),index=False,float_format='%.17g')
        numeric=['mre_percent','mae','rmse','absolute_error_p90','search_seconds','refit_seconds','pipeline_seconds']
        summary=[]
        for key,g in df.groupby(['system','budget','variant']):
            row=dict(zip(['system','budget','variant'],key));row['n_runs']=len(g)
            for col in numeric:row[col+'_mean']=g[col].mean();row[col+'_sd']=g[col].std(ddof=1) if len(g)>1 else None
            summary.append(row)
        pd.DataFrame(summary).to_csv(str(out/'summary.csv'),index=False,float_format='%.17g')
    print(json.dumps(counts));print('Saved:',out)
    return counts

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['check','plan','run','summarize'],default='plan')
    parser.add_argument('--profile',choices=['smoke','searchcheck','pilot'],default='smoke')
    parser.add_argument('--system',choices=list(protocol()['datasets']))
    parser.add_argument('--budget',type=int);parser.add_argument('--repeat',type=int)
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--screen-scope',choices=['fit_only','legacy_all'],default='fit_only')
    parser.add_argument('--allow-legacy-label-access',action='store_true')
    args=parser.parse_args()
    if args.screen_scope=='legacy_all' and not args.allow_legacy_label_access:raise ValueError('Legacy all-label screening requires explicit forensic acknowledgement; not valid revised evaluation')
    if args.stage=='check':
        print(json.dumps({'environment':environment(),'upstream_sha256':validate_sources()},ensure_ascii=False,indent=2));return
    plan=make_plan(args.profile,args.system,args.budget,args.repeat,args.variant,args.screen_scope)
    dump(ROOT/'plans'/(args.profile+'_'+args.screen_scope+'.json'),plan)
    if args.stage=='plan':print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if args.stage=='run':
        print('Tasks:',plan['task_count'],'search fits:',plan['search_candidate_count'],'plus one refit per task; one process.',flush=True)
        for task in plan['tasks']:run_task(output_root=ROOT/'results',**task)
    counts=summarize(plan)
    if args.stage=='run' and any(counts[k] for k in ['not_run','failed','corrupt']):raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
