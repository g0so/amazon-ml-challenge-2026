"""Read-only verification of a transferred GPU packet (NumPy + standard library)."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(root):
 meta=json.loads((root/'metadata.json').read_text());assert len(meta['feature_names'])==22
 for name,h in meta['files'].items():assert sha(root/name)==h, name+' checksum mismatch'
 all_ids={};result={}
 for part in ['train','tune']:
  p=root/part;ids=json.loads((p/'reference_ids.json').read_text());countries=json.loads((p/'countries.json').read_text())
  assert len(ids)==len(set(ids))==len(countries);all_ids[part]=set(ids)
  X=np.load(p/'X.npy',mmap_mode='r',allow_pickle=False);y=np.load(p/'y.npy',mmap_mode='r',allow_pickle=False)
  offsets=np.load(p/'offsets.npy',allow_pickle=False);probs=np.load(p/'v3_probabilities.npy',mmap_mode='r',allow_pickle=False)
  assert X.shape==(len(y),22) and X.dtype==np.float64 and np.isfinite(X).all()
  assert len(probs)==len(y) and np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
  assert len(offsets)==len(ids)+1 and offsets[0]==0 and offsets[-1]==len(y) and np.all(np.diff(offsets)>=0)
  scores=[];tp=fp=fn=0
  with (p/'candidates.jsonl').open() as cf,(p/'truth.jsonl').open() as tf:
   for i,ref in enumerate(ids):
    c=json.loads(next(cf));t=json.loads(next(tf));assert c['reference_id']==t['reference_id']==ref
    cs=c['candidate_ids'];ts=set(t['true_ids']);assert len(cs)==len(set(cs)) and len(ts)==len(t['true_ids'])
    a,b=offsets[i:i+2];assert len(cs)==b-a
    assert np.array_equal(y[a:b],np.array([int(x in ts) for x in cs]))
    ps={x for x,prob in zip(cs,probs[a:b]) if prob>=meta['identity']['threshold']}
    found=len(ps&ts);den=len(ps)+.25*len(ts);scores.append(1.25*found/den if den else 1.)
    tp+=found;fp+=len(ps-ts);fn+=len(ts-ps)
   assert not cf.read().strip() and not tf.read().strip()
  score=sum(scores)/len(scores);expected=meta['partitions'][part]
  assert abs(score-expected['v3_macro_f05_at_025'])<1e-12
  assert int(y.sum())==expected['positives'] and len(y)==expected['pairs']
  result[part]={'references':len(ids),'pairs':len(y),'v3_macro_f05':score,'tp':tp,'fp':fp,'fn':fn}
 assert not all_ids['train']&all_ids['tune']
 print(json.dumps({'passed':True,'partitions':result},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('packet',type=Path);verify(p.parse_args().packet)
