"""Frozen-threshold secondary-dev check; no fitting or threshold search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,zipfile,csv,io,time,multiprocessing as mp
from pathlib import Path,PureWindowsPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.gpu_runtime'))
import numpy as np
from catboost import CatBoostClassifier
from run_submission import ro,values,feature_job,digest
from business_entity_resolution.inference_features import FEATURE_NAMES
from business_entity_resolution.metrics import macro_score
DATA=Path('/home/a/Projects/Amazon ML hackathon')

def score(ids,truth,pred,countries):
 result={'macro_f05':macro_score(ids,truth,pred)}
 for label,op in [('tp',lambda t,p:t&p),('fp',lambda t,p:p-t),('fn',lambda t,p:t-p)]:result[label]=sum(len(op(truth[r],pred[r])) for r in ids)
 result['by_country']={c:macro_score([r for r in ids if countries[r]==c],truth,{r:pred[r] for r in ids if countries[r]==c}) for c in sorted(set(countries.values()))}
 singles=[r for r in ids if not truth[r]];result['singleton_fp_rate']=sum(bool(pred[r]) for r in singles)/len(singles)
 return result

def main():
 start=time.monotonic();package=ROOT/'artifacts/gpu_challenger';meta=json.loads((package/'metadata.json').read_text())
 assert meta['feature_names']==FEATURE_NAMES and meta['positive_class']==1
 assert digest(package/'model.cbm')==meta['model_sha256']
 h=hashlib.sha256();packet=ROOT/'data/processed/gpu_feature_packet_v1'
 # Sender used Windows Path ordering (case-insensitive); reproduce it exactly.
 for p in sorted((x for x in packet.rglob('*') if x.is_file()),key=lambda x:PureWindowsPath(str(x.relative_to(packet)))):h.update(str(p.relative_to(packet)).replace('\\','/').encode());h.update(bytes.fromhex(digest(p)))
 assert h.hexdigest()==meta['packet_sha256'],'Packet identity mismatch'
 model=CatBoostClassifier();model.load_model(str(package/'model.cbm'));assert list(model.classes_)==[0,1]
 threshold=meta['selected_threshold'];X=np.load(package/'parity_inputs.npy');expected=np.load(package/'parity_probabilities.npy');actual=model.predict_proba(X,thread_count=2)[:,1]
 assert np.allclose(actual,expected,rtol=0,atol=1e-10) and np.array_equal(actual>=threshold,expected>=threshold)
 c=ro(DATA/'data/processed/phase2_baseline_dev.sqlite')
 with zipfile.ZipFile(DATA/'data/processed/phase3_comparison/our_phase3_candidates.zip') as z:rows=[json.loads(x) for x in z.read('union_k50.jsonl').splitlines()]
 with zipfile.ZipFile(DATA/'data/processed/phase3_comparison/phase3_fair_comparison.zip') as z:
  ids=json.loads(z.read('reference_ids.json'));truth={r['source1_entity_id']:set(r['matched_entity_ids'].split(',')) if r['matched_entity_ids'] else set() for r in csv.DictReader(io.StringIO(z.read('sample_ground_truth.tsv').decode()),delimiter='\t')}
 assert [r['reference_id'] for r in rows]==ids
 assert not set(ids)&set(json.loads((packet/'tune/reference_ids.json').read_text()))
 predicted={};countries={};pairs=0
 with mp.get_context('spawn').Pool(2) as pool:
  for off in range(0,len(rows),100):
   chunk=rows[off:off+100];target_ids=sorted({t for r in chunk for t in r['candidate_ids']});targets={}
   for j in range(0,len(target_ids),900):
    part=target_ids[j:j+900]
    for r in c.execute('SELECT * FROM records WHERE entity_id IN ('+','.join('?' for _ in part)+')',part):targets[r['entity_id']]=values(r)
   refs={r['reference_id']:dict(c.execute('SELECT * FROM records WHERE entity_id=?',(r['reference_id'],)).fetchone()) for r in chunk}
   jobs=[(values(refs[r['reference_id']]),[targets[t] for t in r['candidate_ids']]) for r in chunk]
   fs=pool.map(feature_job,jobs,chunksize=4)
   for r,x in zip(chunk,fs):
    ref=r['reference_id'];probs=model.predict_proba(np.asarray(x),thread_count=2)[:,1] if x else []
    predicted[ref]={t for t,p in zip(r['candidate_ids'],probs) if p>=threshold};countries[ref]=refs[ref]['country'];pairs+=len(x)
 c.close();baseline=json.loads((DATA/'reports/phase4_experiment_v3_results.json').read_text())
 v3={r:set(v) for r,v in baseline['predictions'].items()}
 report={'threshold':threshold,'model_sha256':meta['model_sha256'],'packet_identity_verified':True,'parity_max_difference':float(np.max(np.abs(actual-expected))),
 'references':len(ids),'pairs':pairs,'challenger':score(ids,truth,predicted,countries),'v3':score(ids,truth,v3,countries),'elapsed_seconds':time.monotonic()-start,'evaluation':'Previously inspected secondary dev sample; no fitting or threshold selection here'}
 (ROOT/'reports/gpu_challenger_secondary_eval.json').write_text(json.dumps(report,indent=2)+'\n')
 with (package/'secondary_predictions.jsonl').open('w') as f:
  for ref in ids:f.write(json.dumps({'reference_id':ref,'predicted_ids':sorted(predicted[ref])})+'\n')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
