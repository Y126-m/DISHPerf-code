"""Point and distributional scores, no trimming or outcome-based exclusions."""
import numpy as np

def relative_crps(samples,y):
    s=np.asarray(samples,dtype=float);y=np.asarray(y,dtype=float)
    if s.ndim!=2 or s.shape[0]!=len(y) or s.shape[1]<1 or not np.isfinite(s).all() or not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid CRPS inputs')
    z=np.sort(s,axis=1);n=z.shape[1]
    score=np.mean(abs(z-y[:,None]),axis=1)-np.sum(z*(2*np.arange(1,n+1)-n-1),axis=1)/(n*n)
    if (score < -1e-9).any():raise ValueError('Negative CRPS beyond numerical tolerance')
    return score/y

def score(frame,probabilistic=False):
    if len(frame)==0 or not np.isfinite(frame[['y','prediction']].values).all() or (frame.y<=0).any():raise ValueError('Invalid predictions')
    per=[]
    if probabilistic and (not {'crps_relative','lower90','upper90'}.issubset(frame.columns) or not np.isfinite(frame[['crps_relative','lower90','upper90']].values).all()):raise ValueError('Probability predictions missing')
    for w,g in frame.groupby('workload',sort=False):
        e=g.prediction.values-g.y.values
        row={'workload':str(w),'n_test':len(g),'mre_percent':float(100*np.mean(abs(e)/g.y.values)),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2)))}
        row['pmape_percent']=float(g.crps_relative.mean()*100) if probabilistic else row['mre_percent']
        if probabilistic:
            if (g.upper90<g.lower90).any() or (g.crps_relative < -1e-9).any():raise ValueError('Bad intervals or CRPS')
            row['coverage90']=float(((g.y>=g.lower90)&(g.y<=g.upper90)).mean());row['width90']=float((g.upper90-g.lower90).mean())
        per.append(row)
    e=frame.prediction.values-frame.y.values
    result={'mre_percent':float(np.mean([r['mre_percent'] for r in per])),'pmape_percent':float(np.mean([r['pmape_percent'] for r in per])),'mae_macro':float(np.mean([r['mae'] for r in per])),'rmse_macro':float(np.mean([r['rmse'] for r in per])),'mre_micro_percent':float(100*np.mean(abs(e)/frame.y.values)),'pmape_micro_percent':float(frame.crps_relative.mean()*100) if probabilistic else float(100*np.mean(abs(e)/frame.y.values)),'negative_predictions':int((frame.prediction<0).sum()),'n_test':len(frame),'per_workload':per}
    return result
