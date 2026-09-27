"""Check optimized inference against saved V3 dev predictions; no model fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,pickle,sys,zipfile,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from run_submission import ROOT,ro,values,feature_job

def main():
    start=time.monotonic();model=pickle.loads((ROOT/'models/experiment_v3_challenger.pkl').read_bytes())
    saved=json.loads((ROOT/'reports/phase4_experiment_v3_results.json').read_text())
    c=ro(ROOT/'data/processed/phase2_baseline_dev.sqlite')
    with zipfile.ZipFile(ROOT/'data/processed/phase3_comparison/our_phase3_candidates.zip') as z:
        rows=[json.loads(x) for x in z.read('union_k50.jsonl').splitlines()]
    predicted={};pairs=0
    with mp.get_context('spawn').Pool(4) as pool:
        for off in range(0,len(rows),100):
            chunk=rows[off:off+100];targets={};ids=list({t for r in chunk for t in r['candidate_ids']})
            for j in range(0,len(ids),900):
                part=ids[j:j+900]
                for r in c.execute('SELECT * FROM records WHERE entity_id IN ('+','.join('?' for _ in part)+')',part):targets[r['entity_id']]=values(r)
            refs={r['reference_id']:dict(c.execute('SELECT * FROM records WHERE entity_id=?',(r['reference_id'],)).fetchone()) for r in chunk}
            jobs=[(values(refs[r['reference_id']]),[targets[t] for t in r['candidate_ids']]) for r in chunk]
            fs=pool.map(feature_job,jobs,chunksize=4)
            for r,x in zip(chunk,fs):
                probs=model.predict_proba(np.asarray(x))[:,1] if x else []
                predicted[r['reference_id']]={t for t,p in zip(r['candidate_ids'],probs) if p>=saved['threshold']};pairs+=len(x)
    assert set(predicted)==set(saved['predictions'])
    different=[r for r in predicted if predicted[r]!=set(saved['predictions'][r])]
    report={'references':len(rows),'pairs':pairs,'different_prediction_sets':len(different),'elapsed_seconds':time.monotonic()-start,'passed':not different}
    (ROOT/'reports/submission_feature_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)
    if different:raise AssertionError(different[:10])
if __name__=='__main__':main()
