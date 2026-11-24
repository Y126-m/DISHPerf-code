"""Prepare three shared datasets directly from the supplied upstream archive."""
import io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,policy,sha,read,write,IdentityError

def prepare():
    p=policy();source=Path(p['source_zip'])
    if sha(source)!=p['source_zip_sha256']:raise IdentityError('HyPerf archive hash changed')
    z=zipfile.ZipFile(source);pre='hyperf-companion-main/experiment-code/wluncert/training-data/dashboard-resources/'
    metadata={'archive_sha256':sha(source),'policy':p['data_policy'],'systems':{}}
    for system,spec in p['systems'].items():
        names=[pre+system+'/sample.csv',pre+system+'/measurements.csv']
        contents=[z.read(n) for n in names];sample,measure=[pd.read_csv(io.BytesIO(b)) for b in contents]
        if not sample.config_id.is_unique:raise ValueError('Duplicate config_id '+system)
        f=[c for c in sample if c!='config_id'];target=spec['target']
        if len(f)!=spec['p']:raise ValueError('Configuration whitelist mismatch '+system)
        if any(c not in sample.columns for c in f) or target not in measure:raise ValueError('Wrong schema')
        data=measure[['config_id','workload',target]].merge(sample,on='config_id',how='left',validate='many_to_one')
        if len(data)!=len(measure) or data.workload.isna().any():raise ValueError('Unmatched measurement')
        data=data.rename(columns={target:'y'})
        if not np.isfinite(data[f+['y']].to_numpy(float)).all() or (data.y<=0).any():raise ValueError('Invalid labels: inspect raw data, no silent removal')
        key=['workload']+f
        count=data.groupby(key,sort=False,dropna=False).size().reset_index(name='raw_count')
        d=data.groupby(key,sort=False,dropna=False).y.median().reset_index().merge(count,on=key,validate='one_to_one',sort=False)
        counts=d.raw_count.tolist();d=d.drop(columns='raw_count');d.insert(0,'row_id',[system+':'+str(i).zfill(6) for i in range(len(d))])
        if d.workload.nunique()!=spec['workloads']:raise ValueError('Unexpected workload count')
        rel='data/'+system+'.csv';path=HERE/rel;path.parent.mkdir(parents=True,exist_ok=True)
        text=d.to_csv(index=False,float_format='%.17g',lineterminator='\n')
        encoded=text.encode('utf-8')
        if path.exists() and path.read_bytes()!=encoded:raise IdentityError('Existing canonical file differs; use new protocol directory')
        if not path.exists():path.write_bytes(encoded)
        import hashlib
        metadata['systems'][system]={'features':f,'target':target,'canonical_file':rel,'canonical_sha256':sha(path),'raw_measurements':len(data),'rows':len(d),'workload_order':d.workload.drop_duplicates().tolist(),'rows_per_workload':d.groupby('workload').size().to_dict(),'raw_count_by_row_id':dict(zip(d.row_id,counts)),'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in zip(names,contents)},'excluded_measurement_columns':[c for c in measure if c not in ['workload','config_id',target]]}
        print('PREPARED',system,'p=',len(f),'workloads=',d.workload.nunique(),'rows=',len(d),flush=True)
    dest=HERE/'data/preparation.json'
    if dest.exists() and read(dest)!=metadata:raise IdentityError('Preparation metadata differs')
    if not dest.exists():write(dest,metadata)
    return metadata
if __name__=='__main__':prepare()
