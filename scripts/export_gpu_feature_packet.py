"""Export existing 5k train / 2k tuning V3 features; never alters live inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,gc,hashlib,json,pickle,platform,resource,sqlite3,time,zipfile
from pathlib import Path
import numpy as np
import psutil
import sklearn
import run_submission as engine
from business_entity_resolution.inference_features import FEATURE_NAMES
from business_entity_resolution.metrics import score_entity
ROOT=Path(__file__).resolve().parents[1]

def memory_check():
    if psutil.virtual_memory().available < 2*1024**3:
        raise MemoryError('Exporter stopped: retain 2 GiB system headroom. Baseline inference is untouched; exporter can resume.')
engine.memory_check=memory_check

def save_json(path,data):engine.atomic_json(path,data)
def rows(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);args=p.parse_args()
    data=args.data_root.resolve();out=ROOT/'data/processed/gpu_feature_packet_v1';parts=ROOT/'data/processed/gpu_feature_packet_parts'
    out.mkdir(parents=True,exist_ok=True);parts.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();memory_check()
    split_path=data/'data/processed/phase2_split.sqlite';split=engine.ro(split_path)
    splitmeta=json.loads(split.execute('SELECT report_json FROM split_metadata').fetchone()[0])
    samples={'train':data/'data/processed/phase4_train_sample_ids.json','tune':data/'data/processed/phase4_dev_tuning_ids.json'}
    ids={k:json.loads(v.read_text()) for k,v in samples.items()}
    assert len(ids['train'])==len(set(ids['train']))==5000
    assert len(ids['tune'])==len(set(ids['tune']))==2000
    evaluation=set(json.loads((data/'data/processed/phase3_pilot_sample_ids.json').read_text()))
    assert not set(ids['train'])&set(ids['tune']) and not set(ids['tune'])&evaluation
    for name,rs in ids.items():
        expected='train' if name=='train' else 'dev'
        for r in rs:
            if split.execute('SELECT split FROM reference_assignments WHERE reference_id=?',(r,)).fetchone()[0]!=expected:raise ValueError('Incorrect partition')
    split.close()
    model_path=data/'models/experiment_v3_challenger.pkl';model=pickle.loads(model_path.read_bytes());assert model.n_features_in_==22
    identity={'split_id':splitmeta['manifest_id'],'original_input_sha256':splitmeta['input_hashes'],
      'original_sample_sha256':{k:engine.digest(v) for k,v in samples.items()},'model_sha256':engine.digest(model_path),
      'feature_code_sha256':engine.digest(ROOT/'src/business_entity_resolution/inference_features.py'),
      'retrieval_code_sha256':engine.digest(ROOT/'scripts/run_submission.py'),'exporter_sha256':engine.digest(__file__),
      'settings':engine.SETTINGS,'threshold':.25}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (parts/'identity.json').exists():assert json.loads((parts/'identity.json').read_text())['hash']==identity_hash,'Exporter inputs/code changed'
    else:save_json(parts/'identity.json',{'hash':identity_hash,'identity':identity})
    databases={'train':data/'data/processed/phase4_train_text.sqlite','tune':data/'data/processed/phase2_baseline_dev.sqlite'}
    all_part_paths={}
    for name,db in databases.items():
        c=engine.ro(db);refs=[]
        for r in sorted(ids[name]):
            row=c.execute('SELECT * FROM records WHERE source=1 AND entity_id=?',(r,)).fetchone()
            if row is None:raise ValueError('Missing reference text')
            refs.append(dict(row))
        c.close();refs.sort(key=lambda x:(x['country'],x['entity_id']))
        all_part_paths[name]=[]
        for country in sorted({r['country'] for r in refs}):
            selected=[r for r in refs if r['country']==country];index=None
            try:
                for off in range(0,len(selected),200):
                    batch=selected[off:off+200];path=parts/name/country/f'{off:06d}';all_part_paths[name].append(path)
                    if engine.valid_batch(path,[r['entity_id'] for r in batch],identity_hash):continue
                    memory_check()
                    if index is None:
                        print('Building exporter index',name,country,flush=True);index=engine.CountryIndex(db,country)
                    result=engine.infer_batch(batch,index,model,.25,None,path=db)
                    engine.write_batch(path,batch,*result[:4],identity_hash,result[4])
                    print(f'{name}/{country}: {off+len(batch)}/{len(selected)} refs, {result[4]["pairs"]} pairs, {result[4]["seconds"]:.1f}s; available {psutil.virtual_memory().available/1024**3:.2f} GiB',flush=True)
                    del result
            finally:del index;gc.collect()
    # All candidate lists are frozen before ground truth is opened.
    cfg=json.loads((data/'project_config.json').read_text());truthpath=data/cfg['resource_dir']/'dataset/train/train_ground_truth.tsv'
    required=set(ids['train'])|set(ids['tune']);truth={}
    with truthpath.open(newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ref=r['source1_entity_id']
            if ref in required:
                if ref in truth:raise ValueError('Duplicate truth reference')
                ts=r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
                assert len(ts)==len(set(ts));truth[ref]=set(ts)
    assert set(truth)==required
    metadata={'format_version':1,'identity':identity,'feature_names':FEATURE_NAMES,'feature_dtype':'float64','sample_seeds':{'train':2028,'tune':2029},
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'partitions':{}}
    for name,paths in all_part_paths.items():
        dest=out/name;dest.mkdir(exist_ok=True)
        total=sum(np.load(path/'features.npy',mmap_mode='r').shape[0] for path in paths)
        X=np.lib.format.open_memmap(dest/'X.npy',mode='w+',dtype=np.float64,shape=(total,22))
        y=np.lib.format.open_memmap(dest/'y.npy',mode='w+',dtype=np.uint8,shape=(total,))
        probs=np.lib.format.open_memmap(dest/'v3_probabilities.npy',mode='w+',dtype=np.float64,shape=(total,))
        references=[];countries=[];offsets=[0];position=0;scores=[];oracle=[];positive=0
        with (dest/'candidates.jsonl').open('w') as cf,(dest/'truth.jsonl').open('w') as tf:
            for path in paths:
                fs=np.load(path/'features.npy',mmap_mode='r');ps=np.load(path/'probabilities.npy',mmap_mode='r')
                assert fs.shape[1]==22 and len(fs)==len(ps) and np.isfinite(fs).all() and np.isfinite(ps).all()
                X[position:position+len(fs)]=fs;probs[position:position+len(fs)]=ps
                local=0
                with (path/'candidate_pairs.tsv').open(newline='') as f:
                    for row in csv.DictReader(f,delimiter='\t'):
                        ref=row['source1_entity_id'];cs=row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
                        assert len(cs)==len(set(cs));label=np.array([int(t in truth[ref]) for t in cs],dtype=np.uint8)
                        y[position+local:position+local+len(cs)]=label;positive+=int(label.sum())
                        pred={t for t,p in zip(cs,ps[local:local+len(cs)]) if p>=.25}
                        scores.append(score_entity(truth[ref],pred));oracle.append(score_entity(truth[ref],set(cs)&truth[ref]))
                        references.append(ref);countries.append(path.parent.name);offsets.append(offsets[-1]+len(cs))
                        cf.write(json.dumps({'reference_id':ref,'candidate_ids':cs})+'\n');tf.write(json.dumps({'reference_id':ref,'true_ids':sorted(truth[ref])})+'\n');local+=len(cs)
                assert local==len(fs);position+=len(fs)
        assert position==total==offsets[-1] and len(references)==len(set(references)) and set(references)==set(ids[name])
        X.flush();y.flush();probs.flush();del X,y,probs
        np.save(dest/'offsets.npy',np.asarray(offsets,dtype=np.int64),allow_pickle=False)
        save_json(dest/'reference_ids.json',references);save_json(dest/'countries.json',countries)
        metadata['partitions'][name]={'references':len(references),'pairs':total,'positives':positive,'negatives':total-positive,'all_true_links':sum(len(truth[r]) for r in references),
            'v3_macro_f05_at_025':sum(scores)/len(scores),'oracle_macro_f05':sum(oracle)/len(oracle),'X_shape':[total,22]}
    metadata['elapsed_seconds']=time.monotonic()-start;metadata['peak_exporter_process_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (out/'README.txt').write_text('Use docs/TWO_MACHINE_TRANSFER.md. Matrices contain exact V3 22 features. Train and tune are separate; no holdout or secondary-dev records are included. Row order follows reference_ids and offsets and candidate_ids order. Truth includes unretrieved true targets. v3_probabilities are a baseline, never input features. Verify all hashes before training. Keep one CPU thread for baseline inference and budget RAM.\n')
    metadata['files']={str(p.relative_to(out)):engine.digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='metadata.json'}
    save_json(out/'metadata.json',metadata)
    archive=out.with_suffix('.zip');tmp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    os.replace(tmp,archive)
    sha=engine.digest(archive);archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n')
    save_json(ROOT/'reports/gpu_feature_packet_export.json',{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'partitions':metadata['partitions'],'elapsed_seconds':metadata['elapsed_seconds']})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'partitions':metadata['partitions']},indent=2),flush=True)
if __name__=='__main__':main()
