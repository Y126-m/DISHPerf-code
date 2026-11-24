"""Original 2.2 optimization schedule; held-out targets never enter fitting."""
from __future__ import print_function
import copy, time
import numpy as np
import tensorflow as tf
from models.models import DISHPerf
from protocol import normalizer, transform, original_prediction

def schedule(n, refit=False):
    patience = int(np.clip(120.*(n/100.)**.40, 100, 300))
    return int(patience*1.5) if refit else patience, 200, 20 if refit else int(np.clip(patience//10,10,30))

def fit(config, x, y, norm, validation=None, max_epochs=2000):
    """Search monitors validation MAE; final refit monitors training objective.

    The latter is a source-derived optimization stopping rule, not evidence of
    generalization. No test X or y is accepted by this function.
    """
    model = DISHPerf(config); model.build_train()
    refit=validation is None;patience,min_epochs,freq=schedule(len(x),refit)
    best=float('inf');wait=0;snapshot=None;best_epoch=None;lr0=float(config['lr']);lr=lr0
    start=time.perf_counter();history=[]
    try:
        for epoch in range(1,max_epochs+1):
            _, objective=model.sess.run([model.train_op,model.loss],
                         {model.X:x,model.Y:y,model.lr:lr})
            if not np.isfinite(objective):raise ValueError('Nonfinite training objective')
            lr=lr0/(1.+(lr0/1000.)*epoch)
            if epoch%freq==0 or epoch==max_epochs:
                if refit:score=float(objective)
                else:
                    xv,yv=validation
                    pv=model.sess.run(model.output,{model.X:xv})
                    score=float(np.mean(abs(original_prediction(pv,norm)-original_prediction(yv,norm))))
                if not np.isfinite(score):raise ValueError('Nonfinite monitored score')
                history.append(dict(epoch=epoch,score=score))
                if score<best*.999:
                    best=score;wait=0;best_epoch=epoch
                    snapshot={v.name:model.sess.run(v) for v in tf.trainable_variables()}
                else:
                    wait+=freq
                    if wait>=patience and epoch>=min_epochs:break
        if snapshot is None:raise ValueError('No usable training snapshot')
        for variable in tf.trainable_variables():
            model.sess.run(variable.assign(snapshot[variable.name]))
        if not refit:
            xv,yv=validation;pv=model.sess.run(model.output,{model.X:xv})
            final_score=float(np.mean(abs(original_prediction(pv,norm)-original_prediction(yv,norm))))
        else:final_score=best
        record=dict(steps=epoch,snapshot_epoch=best_epoch,
                    monitor='training_objective' if refit else 'validation_MAE',
                    score=final_score,seconds=time.perf_counter()-start,
                    block_input_widths=model.block_input_widths,
                    parameter_count=int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables())),
                    history=history)
        if refit:return model,record
        return final_score,record
    except BaseException:
        if refit:model.finalize()
        raise
    finally:
        if not refit and model.sess is not None:model.finalize()

def select(configs, raw, ids, max_epochs=2000, checkpoint=None, cached=None):
    fitids=ids['fit_ids'];valids=ids['validation_ids'];x=raw[fitids,:-1];y=raw[fitids,-1:]
    xv=raw[valids,:-1];yv=raw[valids,-1:]
    norms={g:normalizer(x,y,g) for g in [False,True]}
    best=None;records=[]
    for i,config in enumerate(configs):
        cfg=copy.deepcopy(config)
        if cached is not None and i<len(cached):
            record=cached[i]
            if record['config']!=cfg:raise ValueError('Candidate cache identity mismatch')
            score=record['score']
        else:
            n=norms[cfg['gnorm']];xt,yt=transform(x,y,n);xvt,yvt=transform(xv,yv,n)
            score,record=fit(cfg,xt,yt,n,(xvt,yvt),max_epochs)
            record.update(candidate=i,config=cfg)
        if not np.isfinite(score):raise ValueError('Invalid cached score')
        records.append(record)
        if best is None or score<best['score']:best=record
        if checkpoint is not None:checkpoint(records)
        print('CANDIDATE {}/{} validation_MAE={:.9g}'.format(i+1,len(configs),score),flush=True)
    return copy.deepcopy(best['config']),records

def refit_predict(config, raw, ids, max_epochs=2000):
    # Structural pairs remain fixed from the fitting-only screening stage.
    train=ids['train_ids'];test=ids['test_ids'];x=raw[train,:-1];y=raw[train,-1:]
    norm=normalizer(x,y,config['gnorm']);xt,yt=transform(x,y,norm)
    model,record=fit(config,xt,yt,norm,max_epochs=max_epochs)
    try:
        testx=(raw[test,:-1]-norm['x_mean'])/norm['x_scale']
        prediction=original_prediction(model.sess.run(model.output,{model.X:testx}),norm).ravel()
        if not np.isfinite(prediction).all():raise ValueError('Invalid final prediction')
        return prediction,record
    finally:model.finalize()
