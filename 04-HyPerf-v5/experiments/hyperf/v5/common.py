"""Shared IO, frozen provenance, deterministic splits; compatible with native Python 3.6."""
from __future__ import print_function
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
class IdentityError(ValueError):pass

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False,default=json_default)
def digest_obj(obj):return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()
def json_default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=json_default),encoding='utf-8')
    for k in range(15):
        try:os.replace(str(tmp),str(path));break
        except PermissionError:
            if k==14:raise
            time.sleep(.05*(k+1))
def policy():
    p=read(HERE/'settings.json');p['native_root']=str(HERE/'native_core');return p
def check_package():
    m=read(HERE/'package_manifest.json')
    for name,h in m['files'].items():
        if not (HERE/name).is_file() or sha(HERE/name)!=h:raise IdentityError('Package source changed: '+name+'. Use a new version; do not overwrite old results.')
    return m['package_id']
def table(system):
    import pandas as pd
    p=policy()
    if p.get('preparation_sha256') and sha(HERE/'data/preparation.json')!=p['preparation_sha256']:raise IdentityError('Prepared metadata changed')
    entry=read(HERE/'data/preparation.json')['systems'][system];path=HERE/entry['canonical_file']
    if sha(path)!=entry['canonical_sha256']:raise IdentityError('Dataset changed: '+system)
    d=pd.read_csv(str(path),float_precision='round_trip');f=entry['features']
    if len(f)!=p['systems'][system]['p'] or d.workload.nunique()!=p['systems'][system]['workloads']:raise ValueError('Data dimensions differ')
    if d.row_id.duplicated().any() or d.duplicated(['workload']+f).any():raise ValueError('Duplicate row/config-workload')
    if not np.isfinite(d[f+['y']].values.astype(float)).all() or (d.y<=0).any():raise ValueError('Nonfinite data or nonpositive y')
    return d,f,entry

def split(system,budget,rep,profile='formal'):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    p=policy();spec=p['systems'][system]
    if budget not in [int(a*spec['p']) for a in p['alphas']]:raise ValueError('Unregistered budget')
    if not 0<=rep<p['repeats'] or profile not in ('formal','smoke'):raise ValueError('Unregistered rep/profile')
    d,f,entry=table(system);wls=d.workload.drop_duplicates().tolist();rng=np.random.default_rng(rep);tr=[];te=[];by={}
    for w in wls:
        env_seed=int(rng.integers(0,2**30,dtype=np.uint32));sub=d[d.workload==w]
        # Preserve the original <=2p models' training observations exactly.
        old_pool,old_test=train_test_split(sub,train_size=2*spec['p'],random_state=env_seed)
        extra_seed=(env_seed ^ int(p['extension_seed'])) % (2**32)
        extra,_=train_test_split(old_test,train_size=spec['p'],random_state=extra_seed)
        pool=pd.concat([old_pool,extra],ignore_index=False)
        test=old_test[~old_test.row_id.isin(extra.row_id)].copy()
        train=pool if budget==3*spec['p'] else (old_pool if budget==2*spec['p'] else train_test_split(old_pool,train_size=budget,random_state=env_seed)[0])
        cap=p['test_cap_per_workload'] if profile=='formal' else p['smoke']['test_rows_per_workload']
        if len(test)>cap:test=train_test_split(test,train_size=cap,random_state=env_seed)[0]
        if set(map(tuple,train[f].values))&set(map(tuple,test[f].values)):raise ValueError('Config leakage')
        tr.append(train);te.append(test)
        by[str(w)]={'environment_seed':env_seed,'train_ids':train.row_id.tolist(),'test_ids':test.row_id.tolist(),'max_pool_ids':pool.row_id.tolist()}
    a=pd.concat(tr,ignore_index=True);b=pd.concat(te,ignore_index=True)
    info={'system':system,'budget':budget,'rep':rep,'profile':profile,'workload_order':wls,'by_workload':by,'canonical_sha256':entry['canonical_sha256'],'train_ids':a.row_id.tolist(),'test_ids':b.row_id.tolist(),'policy':'v5 frozen v4 <=2p training IDs; additional p IDs selected without labels; common complement of 3p pool; same IDs for both methods','extension_seed':p['extension_seed']}
    return a,b,f,info,digest_obj(info)

def checkpoint(folder):
    folder=Path(folder)
    names=[p.name for p in folder.iterdir() if p.is_file() and p.name not in ('complete.json','failure.json','driver_attempts.json') and not p.name.endswith('.tmp')]
    write(folder/'complete.json',{'status':'completed','files':{n:sha(folder/n) for n in names}})
def verify_checkpoint(folder,identity=None):
    folder=Path(folder);idx=read(folder/'complete.json')
    if idx.get('status')!='completed' or not {'manifest.json','predictions.csv','metrics.json'}.issubset(idx.get('files',{})):raise IdentityError('Invalid completion index '+str(folder))
    for n,h in idx['files'].items():
        if Path(n).name!=n or sha(folder/n)!=h:raise IdentityError('Corrupt checkpoint '+str(folder/n))
    if identity is not None and read(folder/'manifest.json')!=identity:raise IdentityError('Checkpoint protocol identity differs '+str(folder))
    return read(folder/'metrics.json')
def task_path(base,profile,system,budget,rep,model):return Path(base)/profile/(system+'_b'+str(budget)+'_r'+str(rep))/model
