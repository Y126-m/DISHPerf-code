"""Report all completed formal repetitions, never trimmed or best-of runs."""
from __future__ import print_function
import argparse,csv,math
from pathlib import Path
import numpy as np
from protocol import ROOT,read,sha,identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    base=Path(a.output) if a.output else ROOT/'results';package=identity()
    groups={};total=0
    for marker in sorted((base/'formal').glob('*/complete.json')):
        folder=marker.parent;manifest=read(folder/'manifest.json')
        if manifest['package_id']!=package:raise ValueError('Mixed implementation versions: '+str(folder))
        for name,h in read(marker)['files'].items():
            if sha(folder/name)!=h:raise ValueError('Corrupt artifact '+name)
        m=read(folder/'metrics.json')
        if m['mode']!='formal' or m['candidate_count']!=120:raise ValueError('Non-formal result')
        groups.setdefault((m['system'],m['budget']),[]).append(m);total+=1
    rows=[]
    for (system,budget),runs in sorted(groups.items()):
        reps=[m['rep'] for m in runs]
        if len(set(reps))!=len(reps):raise ValueError('Duplicate repetition')
        values=np.asarray([m['mre_percent'] for m in runs]);n=len(values)
        sd=float(values.std(ddof=1)) if n>1 else None
        rows.append(dict(system=system,budget=budget,n=n,mre_mean=float(values.mean()),
                         mre_sd=sd,margin=1.96*sd/math.sqrt(n) if n>1 else None,
                         completed_reps=','.join(map(str,sorted(reps)))))
    base.mkdir(exist_ok=True,parents=True)
    with (base/'summary_all_completed.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['system','budget','n','mre_mean','mre_sd','margin','completed_reps'])
        w.writeheader();w.writerows(rows)
    print('Verified formal results:',total,'; groups:',len(rows),'; verification-only results excluded')

if __name__=='__main__':main()
