"""Rescore verified cached features; preserve baseline, resume atomic batches."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import csv, fcntl, hashlib, json, shutil, sys, time, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.gpu_runtime'))
import numpy as np
from catboost import CatBoostClassifier
import run_submission as engine


def main():
    source=Path('/home/a/Projects/Amazon ML hackathon')
    baseline=source/'data/processed/inference_runs/v3_first_submission'
    out=ROOT/'data/processed/inference_runs/catboost_first_submission'
    package=ROOT/'artifacts/gpu_challenger'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((baseline/'status.json').read_text())
    assert status['state']=='validated'
    meta=json.loads((package/'metadata.json').read_text())
    review=json.loads((ROOT/'reports/gpu_challenger_secondary_eval.json').read_text())
    model_hash=engine.digest(package/'model.cbm')
    assert model_hash==meta['model_sha256']==review['model_sha256']
    assert meta['feature_names']==engine.FEATURE_NAMES
    threshold=meta['selected_threshold']
    assert threshold==review['threshold'] and review['packet_identity_verified']
    assert review['challenger']['macro_f05']>review['v3']['macro_f05']
    identity={'baseline':status['run_id'],'model_sha256':model_hash,'threshold':threshold,
              'runner_sha256':engine.digest(__file__),'features':engine.FEATURE_NAMES}
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())['run_id']==run_id
    else:engine.atomic_json(manifest,{'run_id':run_id,'identity':identity})
    model=CatBoostClassifier();model.load_model(str(package/'model.cbm'))
    assert list(model.classes_)==[0,1]
    parity=model.predict_proba(np.load(package/'parity_inputs.npy'),thread_count=2)[:,1]
    expected_probs=np.load(package/'parity_probabilities.npy')
    np.testing.assert_allclose(parity,expected_probs,rtol=0,atol=1e-10)
    assert np.array_equal(parity>=threshold,expected_probs>=threshold)
    started=time.monotonic();completed=0
    batches=sorted((baseline/'batches').glob('*/*/batch.json'))
    for i,path in enumerate(batches):
        info=json.loads(path.read_text());assert info['run_id']==status['run_id']
        folder=out/'batches'/path.parent.relative_to(baseline/'batches')
        if not engine.valid_batch(folder,info['ids'],run_id):
            engine.memory_check()
            for name in ('features.npy','candidate_pairs.tsv'):
                assert engine.digest(path.parent/name)==info['files'][name],str(path)
            with (path.parent/'candidate_pairs.tsv').open() as f:
                reader=csv.DictReader(f,delimiter='\t')
                assert reader.fieldnames==['source1_entity_id','candidate_entity_ids']
                rows=list(reader)
            assert [r['source1_entity_id'] for r in rows]==info['ids']
            candidates=[r['candidate_entity_ids'].split(',') if r['candidate_entity_ids'] else [] for r in rows]
            X=np.load(path.parent/'features.npy',mmap_mode='r',allow_pickle=False)
            assert X.shape==(sum(map(len,candidates)),22)
            probs=np.empty(len(X),dtype=np.float64)
            for start in range(0,len(X),50000):
                probs[start:start+50000]=model.predict_proba(X[start:start+50000],thread_count=4)[:,1]
            assert np.isfinite(probs).all()
            temp=folder.with_name(folder.name+'.'+uuid.uuid4().hex+'.tmp');temp.mkdir(parents=True)
            shutil.copyfile(path.parent/'candidate_pairs.tsv',temp/'candidate_pairs.tsv')
            with (temp/'matching_results.tsv').open('w',newline='') as f:
                writer=csv.writer(f,delimiter='\t',lineterminator='\n')
                writer.writerow(['source1_entity_id','matched_entity_ids']);offset=0
                for ref,ids in zip(info['ids'],candidates):
                    matches=[t for t,p in zip(ids,probs[offset:offset+len(ids)]) if p>=threshold]
                    writer.writerow([ref,','.join(matches)]);offset+=len(ids)
                f.flush();os.fsync(f.fileno())
            np.save(temp/'probabilities.npy',probs,allow_pickle=False)
            files={p.name:engine.digest(p) for p in temp.iterdir()}
            engine.atomic_json(temp/'batch.json',{'run_id':run_id,'ids':info['ids'],'files':files,
                'baseline_batch_sha256':engine.digest(path)})
            os.replace(temp,folder)
            del X,probs
        completed+=len(info['ids'])
        progress={'state':'rescoring','run_id':run_id,'completed_references':completed,
                  'total_references':status['completed_references'],'batches':i+1,'elapsed_seconds':time.monotonic()-started}
        engine.atomic_json(out/'status.json',progress)
        if (i+1)%50==0:print(json.dumps(progress),flush=True)
    assert completed==status['completed_references']
    progress['state']='assembling';engine.atomic_json(out/'status.json',progress)
    destination=engine.assemble(out,run_id,completed)
    original_ro=engine.ro
    engine.ro=lambda:original_ro(source/'data/processed/test_inference_verified.sqlite')
    engine.DATASET_DIR=source/'6ab10eb3b23ba_student_resource/student_resource/dataset'
    progress['state']='validating';engine.atomic_json(out/'status.json',progress)
    engine.validate_outputs(out,destination,completed)
    progress.update(state='validated',output=str(destination),elapsed_seconds=time.monotonic()-started)
    engine.atomic_json(out/'status.json',progress);print(json.dumps(progress),flush=True)

if __name__=='__main__':main()
