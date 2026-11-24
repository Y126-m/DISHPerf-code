"""Versioned original-record protocol and immutable execution identities."""
from __future__ import print_function
import hashlib, json, math, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
VERSION = 'dishperf-paper-11-16-aligned-v1'

def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path);path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(tmp), str(path))

def identity():
    manifest = read(ROOT/'source_manifest.json')
    for name, expected in manifest['files'].items():
        if sha(ROOT/name) != expected: raise ValueError('Source changed: '+name)
    return manifest['package_id']

def load_raw(path):
    raw = np.genfromtxt(str(path), delimiter=',', skip_header=1)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw[:, -1] <= 0).any():
        raise ValueError('Expected finite numeric original rows and positive targets')
    return raw

def split(system, budget, rep, raw, data_hash, budgets):
    if budget not in budgets or not 0 <= rep < 30: raise ValueError('Unregistered task')
    if budget >= len(raw): raise ValueError('No test complement')
    seed = budgets.index(budget)*30 + rep + 1
    order = np.random.RandomState(seed).permutation(len(raw))
    cut = int(math.ceil(budget*2/3))
    train = order[:budget].tolist()
    return dict(version=VERSION, system=system, budget=budget, rep=rep,
                seed=seed, registered_repetitions=30, data_sha256=data_hash,
                train_ids=train, fit_ids=train[:cut], validation_ids=train[cut:],
                test_ids=np.setdiff1d(np.arange(len(raw)), order[:budget]).tolist(),
                deduplication=False, test_cap=0,
                policy='original rows; current-budget complement; budget-specific seeds')

def normalizer(x, y, gaussian):
    if gaussian:
        mx=x.mean(axis=0);sx=x.std(axis=0);sx=sx+(sx==0)*.001
        my=y.mean(axis=0);sy=y.std(axis=0);sy=sy+(sy==0)*.001
    else:
        mx=np.zeros(x.shape[1]);sx=x.max(axis=0);sx[sx==0]=1
        my=np.zeros(1);sy=np.asarray([float(y.max())/100.]);sy[sy==0]=1
    return dict(x_mean=mx, x_scale=sx, y_mean=my, y_scale=sy)

def transform(x, y, n):
    return (x-n['x_mean'])/n['x_scale'], (y-n['y_mean'])/n['y_scale']

def original_prediction(prediction, norm):
    return np.asarray(prediction)*norm['y_scale']+norm['y_mean']
