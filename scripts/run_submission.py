"""Verified, country-wise inference with atomic resumable batches and feature caches.

All expensive V3 features are cached on disk so another validated model can be
scored later without repeating retrieval/string matching. No portal upload.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse,array,csv,gc,hashlib,heapq,json,math,multiprocessing as mp,pickle,resource,shutil,sqlite3,sys,time,uuid
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from business_entity_resolution.workspace import DATASET_DIR,iter_tsv
from business_entity_resolution.normalize import normalize_text
from business_entity_resolution.inference_features import feature_job,values,FEATURE_NAMES

DB=ROOT/'data/processed/test_inference_verified.sqlite'
DEFAULT_MODEL=ROOT/'models/experiment_v3_challenger.pkl'
SETTINGS={'m_tokens':5,'max_df':5000,'max_df_ratio':.05,'k':50,'feature_version':'v3_exact_cached_v1'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def ro(path=DB):
    c=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    c.execute('PRAGMA cache_size=-65536');return c

def memory_check():
    if psutil.virtual_memory().available < 1200*1024**2:
        raise MemoryError('Less than 1.2 GiB RAM available; stopping without deleting checkpoints')

def build_db():
    files={f'test_source{i}.tsv':DATASET_DIR/'test'/f'test_source{i}.tsv' for i in (1,2,3)}
    hashes={k:digest(p) for k,p in files.items()}
    if DB.exists():
        with ro() as c: meta=json.loads(c.execute('SELECT value FROM metadata').fetchone()[0])
        if meta['input_hashes']!=hashes:raise ValueError('Input hashes differ from verified database')
        return meta
    tmp=DB.with_name(DB.name+'.'+uuid.uuid4().hex+'.building')
    c=sqlite3.connect(tmp);c.execute('PRAGMA cache_size=-65536')
    c.execute('CREATE TABLE records(entity_id TEXT PRIMARY KEY,source INT,country TEXT,name_raw TEXT,address_raw TEXT,name_norm TEXT,address_norm TEXT)')
    counts={}; start=time.monotonic()
    try:
        for source in (1,2,3):
            batch=[];n=0; print('Ingesting correctly normalized test source',source,flush=True)
            for r in iter_tsv(files[f'test_source{source}.tsv']):
                if not r['entity_id'].startswith(f'S{source}-'):raise ValueError('Wrong source ID')
                name,addr=r['business_name'],r['business_address']
                batch.append((r['entity_id'],source,r['country'],name,addr,normalize_text(name),normalize_text(addr)));n+=1
                if len(batch)==10000:
                    c.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',batch);c.commit();batch.clear();memory_check()
            if batch:c.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',batch);c.commit()
            counts[str(source)]=n
        print('Indexing test records on disk...',flush=True)
        c.execute('CREATE INDEX country_source_id ON records(country,source,entity_id)');c.commit()
        groups=[dict(zip(('country','source','rows'),r)) for r in c.execute('SELECT country,source,COUNT(*) FROM records GROUP BY country,source')]
        meta={'input_hashes':hashes,'counts':counts,'countries':groups,'normalization':'nfc-casefold-whitespace-v1','build_seconds':time.monotonic()-start}
        c.execute('CREATE TABLE metadata(value TEXT)');c.execute('INSERT INTO metadata VALUES (?)',(json.dumps(meta),));c.commit();c.close()
        os.replace(tmp,DB);atomic_json(ROOT/'reports/test_database_verified.json',meta);return meta
    except BaseException:c.close();raise

def tokens(s):
    import re
    return set(re.findall(r'[^\W_]+',s.lower()))

class CountryIndex:
    def __init__(self,path,country):
        start=time.monotonic(); self.country=country;self.ids=[];name_df=Counter();addr_df=Counter()
        c=ro(path);query='SELECT * FROM records WHERE country=? AND source IN (2,3) ORDER BY entity_id'
        for i,r in enumerate(c.execute(query,(country,))):
            self.ids.append(r['entity_id']);name_df.update(tokens(r['name_raw']));addr_df.update(tokens(r['address_raw']))
            if i%100000==0:memory_check()
        limit=min(SETTINGS['max_df'],len(self.ids)*SETTINGS['max_df_ratio'])
        self.nidf={t:math.log(len(self.ids)/n) for t,n in name_df.items() if n<=limit}
        self.aidf={t:math.log(len(self.ids)/n) for t,n in addr_df.items() if n<=limit}
        del name_df,addr_df
        self.nidx=defaultdict(lambda:array.array('I'));self.aidx=defaultdict(lambda:array.array('I'))
        self.exact=defaultdict(lambda:array.array('I'))
        for i,r in enumerate(c.execute(query,(country,))):
            for t in tokens(r['name_raw']):
                if t in self.nidf:self.nidx[t].append(i)
            for t in tokens(r['address_raw']):
                if t in self.aidf:self.aidx[t].append(i)
            if r['name_norm'] and r['address_norm']:self.exact[(r['name_norm'],r['address_norm'])].append(i)
            if i%100000==0:memory_check()
        c.close();self.seconds=time.monotonic()-start
        print(f'{country}: indexed {len(self.ids):,} targets in {self.seconds:.1f}s',flush=True)

    def retrieve(self,r):
        found=set(self.exact.get((r['name_norm'],r['address_norm']),()))
        for field,idf,index in [('name_raw',self.nidf,self.nidx),('address_raw',self.aidf,self.aidx)]:
            query=sorted(((idf[t],t) for t in tokens(r[field]) if t in idf),key=lambda x:(-x[0],x[1]))[:SETTINGS['m_tokens']]
            scores=defaultdict(float)
            for weight,t in query:
                for i in index[t]:scores[i]+=weight
            # Equivalent ordering to sorting all scored candidates, but bounded top-K selection.
            found.update(i for i,_ in heapq.nsmallest(SETTINGS['k'],scores.items(),key=lambda x:(-x[1],x[0])))
        return [self.ids[i] for i in sorted(found)]

def infer_batch(refs,index,model,threshold,pool,path=DB):
    start=time.monotonic(); candidates=[index.retrieve(r) for r in refs];retrieval=time.monotonic()-start
    target_ids=sorted({t for ids in candidates for t in ids});targets={};c=ro(path)
    for off in range(0,len(target_ids),900):
        part=target_ids[off:off+900]
        for r in c.execute('SELECT * FROM records WHERE entity_id IN ('+','.join('?' for _ in part)+')',part):targets[r['entity_id']]=values(r)
    c.close()
    if len(targets)!=len(target_ids):raise ValueError('Candidate text missing')
    fetch=time.monotonic()-start-retrieval
    jobs=((values(r),[targets[t] for t in ids]) for r,ids in zip(refs,candidates))
    chunks=pool.map(feature_job,jobs,chunksize=8) if pool else list(map(feature_job,jobs))
    X=np.asarray([row for chunk in chunks for row in chunk],dtype=np.float64).reshape(-1,22)
    feature=time.monotonic()-start-retrieval-fetch
    probs=model.predict_proba(X)[:,1] if len(X) else np.empty(0)
    predictions=[];offset=0
    for ids in candidates:
        predictions.append([t for t,p in zip(ids,probs[offset:offset+len(ids)]) if p>=threshold]);offset+=len(ids)
    return candidates,predictions,X,probs,{'seconds':time.monotonic()-start,'retrieval':retrieval,'fetch':fetch,'features':feature,'pairs':len(X)}

def write_batch(folder,refs,candidates,predictions,X,probs,run_id,timing):
    temp=folder.with_name(folder.name+'.'+uuid.uuid4().hex+'.tmp');temp.mkdir(parents=True)
    names=['candidate_pairs.tsv','matching_results.tsv']
    for name,column,lists in zip(names,['candidate_entity_ids','matched_entity_ids'],[candidates,predictions]):
        with (temp/name).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id',column])
            for r,ids in zip(refs,lists):w.writerow([r['entity_id'],','.join(ids)])
            f.flush();os.fsync(f.fileno())
    np.save(temp/'features.npy',X,allow_pickle=False);np.save(temp/'probabilities.npy',probs,allow_pickle=False)
    files={p.name:digest(p) for p in temp.iterdir()}
    meta={'run_id':run_id,'ids':[r['entity_id'] for r in refs],'files':files,'timing':timing}
    atomic_json(temp/'batch.json',meta);os.replace(temp,folder)

def valid_batch(folder,ids,run_id):
    if not folder.exists():return False
    m=json.loads((folder/'batch.json').read_text())
    if m['run_id']!=run_id or m['ids']!=ids:raise ValueError('Checkpoint identity differs: '+str(folder))
    if any(digest(folder/n)!=h for n,h in m['files'].items()):raise ValueError('Checkpoint checksum failure: '+str(folder))
    return True

def assemble(run_dir,run_id,expected):
    destination=run_dir/'output';destination.mkdir(exist_ok=True)
    batches=sorted((run_dir/'batches').glob('*/*/batch.json'))
    count=0
    handles={}
    try:
        for name,col in [('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')]:
            f=(destination/(name+'.tmp')).open('w');f.write('source1_entity_id\t'+col+'\n');handles[name]=f
        for p in batches:
            m=json.loads(p.read_text());assert m['run_id']==run_id
            for name,f in handles.items():
                with (p.parent/name).open() as src:next(src);shutil.copyfileobj(src,f)
            count+=len(m['ids'])
        if count!=expected:raise ValueError(f'Only {count}/{expected} refs; refusing incomplete final output')
        for f in handles.values():f.flush();os.fsync(f.fileno());f.close()
        for name in handles:os.replace(destination/(name+'.tmp'),destination/name)
    finally:
        for f in handles.values():f.close()
    return destination

def validate_outputs(run_dir,destination,expected):
    # Official validator retains every candidate set in RAM. Apply its unchanged
    # per-file checker batchwise, then independently check assembled full coverage.
    import importlib.util,contextlib,itertools
    vp=DATASET_DIR.parent/'utils/validate_submission.py'
    spec=importlib.util.spec_from_file_location('official_validator',vp)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    c=ro();valid_ids={r[0] for r in c.execute('SELECT entity_id FROM records WHERE source IN (2,3)')}
    errors=[];batch_rows=0
    with (run_dir/'validator.log').open('w') as log, contextlib.redirect_stdout(log):
        for p in sorted((run_dir/'batches').glob('*/*/batch.json')):
            m=json.loads(p.read_text());required=set(m['ids']);batch_rows+=len(required)
            for name,h in m['files'].items():
                if name.endswith('.tsv') and digest(p.parent/name)!=h:raise ValueError('Changed checkpoint output')
            matched=official.validate_id_list_file(str(p.parent/'matching_results.tsv'),official.MATCHING_HEADER,'matched_entity_ids',required,valid_ids,errors)
            candidates=official.validate_id_list_file(str(p.parent/'candidate_pairs.tsv'),official.CANDIDATE_HEADER,'candidate_entity_ids',required,valid_ids,errors)
            if errors or any(matched[r]-candidates[r] for r in required):raise ValueError('Official batch validation failed: '+str(errors[:5]))
        if batch_rows!=expected:raise ValueError('Batch coverage count mismatch')
        with (destination/'matching_results.tsv').open() as f, (destination/'candidate_pairs.tsv').open() as g:
            mr=csv.DictReader(f,delimiter='\t');cr=csv.DictReader(g,delimiter='\t')
            assert mr.fieldnames==official.MATCHING_HEADER and cr.fieldnames==official.CANDIDATE_HEADER
            rows=c.execute('SELECT entity_id FROM records WHERE source=1 ORDER BY country,entity_id')
            n=0
            for ref,m,cs in itertools.zip_longest(rows,mr,cr):
                if ref is None or m is None or cs is None:raise ValueError('Incomplete/extra assembled rows')
                if m['source1_entity_id']!=ref[0] or cs['source1_entity_id']!=ref[0]:raise ValueError('Wrong/duplicate assembled reference ID')
                targets=cs['candidate_entity_ids'].split(',') if cs['candidate_entity_ids'] else []
                matches=m['matched_entity_ids'].split(',') if m['matched_entity_ids'] else []
                if len(targets)!=len(set(targets)) or len(matches)!=len(set(matches)):raise ValueError('Duplicate target IDs')
                if not set(matches)<=set(targets) or any(t not in valid_ids for t in targets):raise ValueError('Invalid output targets')
                n+=1
            assert n==expected
        print('PASS: unchanged official per-file checks applied batchwise, plus full assembled coverage, target existence, uniqueness and subset checks.')
    c.close()
    atomic_json(run_dir/'validation.json',{'passed':True,'references':expected,'method':'official checks per batch plus complete streaming assembly validation',
        'output_sha256':{p.name:digest(p) for p in destination.glob('*.tsv')}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--batch-size',type=int,default=1000)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--threshold',type=float,default=.25)
    p.add_argument('--run-name',default='v3_first_submission');p.add_argument('--max-batches',type=int)
    args=p.parse_args();meta=build_db()
    if args.prepare and not args.resume:return
    if not args.resume:p.error('Use --prepare or --resume')
    run_dir=ROOT/'data/processed/inference_runs'/args.run_name;run_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(run_dir/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model_bytes=args.model.read_bytes();model=pickle.loads(model_bytes)
    if model.n_features_in_!=22 or list(model.classes_)!=[0,1]:raise ValueError('Wrong model feature/class schema')
    identity={'inputs':meta['input_hashes'],'model_sha256':hashlib.sha256(model_bytes).hexdigest(),'threshold':args.threshold,'settings':SETTINGS,'batch_size':args.batch_size,
              'feature_code_sha256':digest(ROOT/'src/business_entity_resolution/inference_features.py'),'runner_sha256':digest(__file__),'normalizer_sha256':digest(ROOT/'src/business_entity_resolution/normalize.py')}
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=run_dir/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['run_id']!=run_id:raise ValueError('Run identity changed; use new run-name')
    else:
        atomic_json(manifest,{'run_id':run_id,'identity':identity,'features':FEATURE_NAMES});(run_dir/'model.pkl').write_bytes(model_bytes)
    c=ro();countries=list(c.execute('SELECT country,COUNT(*) n FROM records WHERE source=1 GROUP BY country ORDER BY country'));c.close()
    completed=0;new_batches=0;start=time.monotonic()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for country_index,country_row in enumerate(countries):
            country=country_row['country'];n=country_row['n'];index=None;c=ro()
            cur=c.execute('SELECT * FROM records WHERE country=? AND source=1 ORDER BY entity_id',(country,));batchno=0
            while True:
                refs=[dict(r) for r in cur.fetchmany(args.batch_size)]
                if not refs:break
                folder=run_dir/'batches'/f'{country_index:02d}'/f'{batchno:06d}';batchno+=1
                if valid_batch(folder,[r['entity_id'] for r in refs],run_id):completed+=len(refs);continue
                if args.max_batches is not None and new_batches>=args.max_batches:
                    print('Requested bounded run complete; resume with same identity.',flush=True);return
                memory_check()
                if index is None:index=CountryIndex(DB,country)
                out=infer_batch(refs,index,model,args.threshold,pool)
                write_batch(folder,refs,*out[:4],run_id,out[4]);completed+=len(refs);new_batches+=1
                status={'run_id':run_id,'state':'running','country':country,'completed_references':completed,'total_references':meta['counts']['1'],'last_batch':out[4],
                        'elapsed_this_process':time.monotonic()-start,'parent_peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'available_ram_mib':psutil.virtual_memory().available/1024**2}
                atomic_json(run_dir/'status.json',status)
                print(f'{country} batch {batchno}: {completed:,}/{meta["counts"]["1"]:,}; {out[4]}',flush=True)
                del out
            c.close();del index;gc.collect()
    destination=assemble(run_dir,run_id,meta['counts']['1'])
    print('Outputs assembled. Running official validator...',flush=True)
    validate_outputs(run_dir,destination,meta['counts']['1'])
    atomic_json(run_dir/'status.json',{'state':'validated','run_id':run_id,'completed_references':meta['counts']['1'],'output':str(destination),'elapsed_this_process':time.monotonic()-start})
    print('VALIDATED output:',destination,flush=True)

if __name__=='__main__':main()
