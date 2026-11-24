"""Use the upstream _fit and _predict_samples unchanged; do not replace NUTS."""
import os,sys,time,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np
from common import HERE,policy,write,read,sha,IdentityError
VENDOR=HERE/'vendor/hyperf'

def imports(chains):
    os.environ['JAX_ENABLE_X64']='false';os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
    sys.path.insert(0,str(VENDOR))
    import models as upstream
    import data as upstream_data
    if Path(upstream.__file__).resolve().parent!=VENDOR or Path(upstream_data.__file__).resolve().parent!=VENDOR:raise ImportError('Wrong upstream module')
    import numpyro,jax
    numpyro.set_host_device_count(chains)
    if jax.config.jax_enable_x64:raise ValueError('Registered original-default precision is float32, not forced x64')
    return upstream,upstream_data,jax

def versions(jax):
    packages={}
    for n in ['numpy','pandas','scipy','scikit-learn','jax','jaxlib','numpyro','arviz']:
        packages[n]=importlib.metadata.version(n)
    return {'python':platform.python_version(),'executable':sys.executable,'packages':packages,'jax_enable_x64':bool(jax.config.jax_enable_x64),'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()]}

def summarize_diagnostics(mcmc,out):
    import arviz as az
    extra=mcmc.get_extra_fields(group_by_chain=True)
    div=np.asarray(extra['diverging']);idata=az.from_numpyro(mcmc,log_likelihood=False)
    summ=az.summary(idata,kind='diagnostics',round_to='none');summ.to_csv(out/'parameter_diagnostics.csv')
    def stat(v,high):
        a=np.asarray(v,float);finite=a[np.isfinite(a)]
        return float(finite.max() if high else finite.min()) if len(finite) else None
    r=np.asarray(summ.r_hat,float);b=np.asarray(summ.ess_bulk,float);t=np.asarray(summ.ess_tail,float)
    d={'divergences':int(div.sum()),'divergences_per_chain':div.sum(axis=1).astype(int).tolist(),'max_rhat':stat(r,True),'min_ess_bulk':stat(b,False),'min_ess_tail':stat(t,False),'nonfinite_rhat_count':int((~np.isfinite(r)).sum()),'nonfinite_ess_count':int((~np.isfinite(b)).sum()+(~np.isfinite(t)).sum()),'diagnostic_policy':'all posterior sites; report warning only; no queue abort; thresholds are evaluation labels, not sampler modifications'}
    reasons=[]
    if d['divergences']:reasons.append('divergences_nonzero')
    if d['max_rhat'] is None or d['max_rhat']>1.01 or d['nonfinite_rhat_count']:reasons.append('rhat_missing_nonfinite_or_above_1.01')
    if d['min_ess_bulk'] is None or d['min_ess_bulk']<400:reasons.append('bulk_ess_below_400_or_missing')
    if d['min_ess_tail'] is None or d['min_ess_tail']<400:reasons.append('tail_ess_below_400_or_missing')
    if d['nonfinite_ess_count']:reasons.append('nonfinite_ess')
    d['diagnostics_pass']=not reasons;d['diagnostic_warning_reasons']=reasons
    np.savez_compressed(str(out/'sampler_extra_fields.npz'),**{k:np.asarray(v) for k,v in extra.items()})
    write(out/'sampler_diagnostics.json',d);return d

def fit_and_score(train,test,features,workload_order,out,profile='formal',save_draws=False):
    from metrics import relative_crps
    cfg=dict(policy()['hyperf'])
    if profile=='smoke':cfg.update({k:policy()['smoke'][k] for k in ['warmup','samples','chains']})
    upstream,data,jax=imports(cfg['chains']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Python hashlib needs an ndarray buffer on some JAX versions. These exact
    # standardized float32 bytes and the author's hashing algorithm are preserved.
    class HashBufferCompatibility(upstream.MCMCPartialRobustLassoAdaptiveShrinkage):
        def array_to_seed(self,arr):return upstream.MCMCPartialRobustLassoAdaptiveShrinkage.array_to_seed(self,np.asarray(arr))
    std=data.Standardizer();model=HashBufferCompatibility(preprocessings=[std],num_warmup=cfg['warmup'],num_samples=cfg['samples'],num_chains=cfg['chains'],progress_bar=False,plot=False,return_samples_by_default=True)
    model.feature_names=features;model.env_lbls=workload_order
    def pack(df,hidden=False):
        packed=[]
        for i,w in enumerate(workload_order):
            sub=df[df.workload==w];f=sub[features].reset_index(drop=True).copy();f['env']=i;f['y']=0.0 if hidden else sub.y.reset_index(drop=True)
            packed.append(data.SingleEnvData(f,'env',['y']))
        return packed
    packed_train=pack(train);start=time.perf_counter();runtime=versions(jax)
    lock=read(HERE/'environment_lock.json')['hyperf']
    if runtime['python']!=lock['python'] or runtime['packages']!=lock['packages']:raise IdentityError('HyPerf environment changed; new protocol required')
    write(out/'runtime_versions.json',runtime)
    marker=out/'posterior_complete.json';reused=False
    if marker.exists():
        idx=read(marker)
        for name,h in idx['files'].items():
            if sha(out/name)!=h:raise IdentityError('Corrupt posterior checkpoint '+name)
        std.fit_transform(packed_train)
        z=np.load(str(out/'posterior_by_chain.npz'),allow_pickle=False);chains={k:z[k] for k in z.files};z.close()
        class StoredPosterior:
            def get_samples(self,group_by_chain=False):
                return chains if group_by_chain else {k:v.reshape((-1,)+v.shape[2:]) for k,v in chains.items()}
        model.mcmc=StoredPosterior();details=read(out/'sampler_diagnostics.json');saved=read(out/'fit_metadata.json');fit_seconds=saved['fit_seconds'];reused=True
        print('VERIFIED posterior checkpoint; no resampling',flush=True)
    else:
        print('Original HyPerf: target_accept=0.9 warmup={} samples/chain={} chains={}; upstream _fit'.format(cfg['warmup'],cfg['samples'],cfg['chains']),flush=True)
        t=time.perf_counter();model.fit(packed_train);jax.block_until_ready(model.mcmc.get_samples());fit_seconds=time.perf_counter()-t
        expected=cfg['chains'],cfg['samples'];chains=model.mcmc.get_samples(group_by_chain=True)
        for k,v in chains.items():
            if tuple(v.shape[:2])!=expected:raise ValueError('Wrong posterior draw/chain shape '+k)
        kernel=model.mcmc.sampler
        if abs(float(kernel._target_accept_prob)-.9)>1e-12 or int(kernel._max_tree_depth)!=10:raise ValueError('Original kernel settings differ')
        np.savez_compressed(str(out/'posterior_by_chain.npz'),**{k:np.asarray(v) for k,v in chains.items()})
        details=summarize_diagnostics(model.mcmc,out)
        processed=std.transform(packed_train);_,_,ys=model._internal_data_splitting(processed)
        computed=int.from_bytes(hashlib.sha256(np.asarray(ys).ravel().view(np.uint8)).digest()[:4],'little')%(2**31)
        if computed!=model.array_to_seed(ys):raise ValueError('Author label-hash seed mismatch')
        _,key=jax.random.split(jax.random.PRNGKey(computed))
        saved={'fit_seconds':fit_seconds,'mcmc_seed':computed,'mcmc_key_second_split':np.asarray(key).tolist(),'standardized_y_dtype':str(ys.dtype),'standardized_y_sha256':hashlib.sha256(np.asarray(ys).tobytes()).hexdigest(),'sampler':cfg,'upstream_fit_unmodified':True,'upstream_predict_samples_unmodified':True,'hash_buffer_compatibility_only':True,'chain_method':'parallel (upstream MCMC default)','max_tree_depth':10}
        write(out/'fit_metadata.json',saved)
        names=['posterior_by_chain.npz','sampler_extra_fields.npz','parameter_diagnostics.csv','sampler_diagnostics.json','fit_metadata.json']
        write(marker,{'files':{n:sha(out/n) for n in names}})
    details.update(saved);details['posterior_reused']=reused
    print('Sampler complete: divergences={} diagnostics_pass={}; prediction follows'.format(details['divergences'],details['diagnostics_pass']),flush=True)
    processed=std.transform(pack(test,True));X,envs,_=model._internal_data_splitting(processed)
    t=time.perf_counter();draws=model._predict_samples((X,envs,len(workload_order)));draws.block_until_ready()
    n_draws=cfg['chains']*cfg['samples']
    if tuple(draws.shape)!=(n_draws,len(test)):raise ValueError('Wrong upstream predictive axes')
    result=test[['row_id','workload','y']].copy();envs=np.asarray(envs);offset=0
    mm=None
    if save_draws:mm=np.lib.format.open_memmap(str(out/'predictive_draws.npy'),mode='w+',dtype='float32',shape=(len(test),n_draws))
    for i,w in enumerate(workload_order):
        positions=np.flatnonzero(test.workload.to_numpy()==w)
        if not np.array_equal(envs[offset:offset+len(positions)],np.full(len(positions),i)):raise ValueError('Prediction order does not match packed workloads')
        for j in range(0,len(positions),128):
            ids=positions[j:j+128];raw=np.asarray(draws[:,offset+j:offset+j+len(ids)]).T.copy()
            values=std.inverse_transform_pred([raw],[processed[i]])[0]
            if not np.isfinite(values).all():raise ValueError('Nonfinite predictive observations')
            point=model.get_mode_from_samples(values.T)
            if np.asarray(point).shape!=(len(ids),):raise ValueError('Author point estimator axis mismatch')
            result.loc[ids,'prediction']=point;result.loc[ids,'prediction_mean']=values.mean(axis=1)
            result.loc[ids,'crps_relative']=relative_crps(values,result.y.values[ids]);lo,hi=np.quantile(values,[.05,.95],axis=1)
            result.loc[ids,'lower90']=lo;result.loc[ids,'upper90']=hi
            if mm is not None:mm[ids]=values
        offset+=len(positions)
    if mm is not None:mm.flush();del mm
    details.update(prediction_seconds=time.perf_counter()-t,n_draws=n_draws,prediction_policy='upstream full-test Predictive parallel=False PRNGKey(0); only scoring is chunked',point_estimator='upstream 10% HDI midpoint, not predictive mean',all_in_seconds=time.perf_counter()-start,predictive_draws_saved=bool(save_draws))
    return result,details
