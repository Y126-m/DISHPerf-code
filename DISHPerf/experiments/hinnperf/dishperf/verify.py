"""Meaningful checks for held-out leakage, paper equations and actual TF graph."""
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import copy,json
from pathlib import Path
import numpy as np
from protocol import ROOT,read,sha,identity,load_raw,split
from run import screen,grid

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args()
    package_id=identity();path=Path(a.data_root)/'x264_AllNumeric.csv';raw=load_raw(path)
    assert sha(path)==read(ROOT/'data_manifest.json')['systems']['x264']['sha256']
    ids=split('x264',16,0,raw,sha(path),[16,32,64,96])
    assert len(ids['fit_ids'])==11 and len(ids['validation_ids'])==5 and len(ids['test_ids'])==1136
    assert not set(ids['train_ids'])&set(ids['test_ids'])
    pairs,audit=screen(raw,ids,'x264')
    changed=raw.copy();changed[ids['test_ids'],-1]=1.e9;changed[ids['validation_ids'],-1]=-1.e9
    poisoned_pairs,poisoned_audit=screen(changed,ids,'x264')
    assert pairs==poisoned_pairs and audit==poisoned_audit
    for row in audit['candidates']:
        expected=(row['mi_normalized']+row['pearson_normalized']+row['dt_normalized'])/3.
        assert abs(row['score']-expected)<1.e-12
    assert len(grid(raw.shape[1]-1,pairs))==120
    import tensorflow as tf
    if tf.__version__!='1.10.0':raise ValueError('Verified native runtime requires TensorFlow 1.10.0')
    from models.models import DISHPerf
    from runner import select,refit_predict
    config=grid(raw.shape[1]-1,pairs,True)[0]
    # A nonempty critical_features list must not remove original inputs.
    config['critical_features']=[0]
    m=DISHPerf(config);m.build_train()
    try:
        assert m.block_input_widths==[13,26],m.block_input_widths
        assert m.output.shape.as_list()==[None,1]
        assert m.sess._config.intra_op_parallelism_threads==1
        if pairs:
            assert any('gate_fusion' in v.name for v in tf.trainable_variables())
    finally:m.finalize()
    cfgs=grid(raw.shape[1]-1,pairs,True)
    best,_=select(cfgs,raw,ids,max_epochs=2)
    # Changing test targets must not affect search or final predictions.
    test_poison=raw.copy();test_poison[ids['test_ids'],-1]=1.e9
    b2,_=select(cfgs,test_poison,ids,max_epochs=2)
    assert best==b2
    pred,record=refit_predict(best,raw,ids,max_epochs=2)
    poisoned_pred,_=refit_predict(best,test_poison,ids,max_epochs=2)
    assert len(pred)==1136 and np.isfinite(pred).all()
    assert np.allclose(pred,poisoned_pred,rtol=0,atol=1.e-6)
    report=dict(status='passed',package_id=package_id,data_sha256=sha(path),training='2-step verification only; not formal experiments',
                tensorflow=tf.__version__,checks=['original budget-complement split',
                'fit-only screening invariant to validation/test label changes',
                'score equals paper equal-weight formula','120 search configurations',
                'actual TF graph keeps every original input [13,26]',
                'gate and output graph','one-thread session',
                'search and final predictions invariant to poisoned test targets'],
                prediction_max_abs_difference=float(np.max(abs(pred-poisoned_pred))))
    print(json.dumps(report,indent=2))
    (ROOT/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
