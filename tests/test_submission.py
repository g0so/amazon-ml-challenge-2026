import ast,csv,difflib,importlib.util,json,random,re,sqlite3,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from business_entity_resolution.inference_features import features,values
from business_entity_resolution.normalize import normalize_text
spec=importlib.util.spec_from_file_location('submission',ROOT/'scripts/run_submission.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
# Extract the actual training feature function, avoiding module import side effects.
tree=ast.parse((ROOT/'scripts/run_phase4_experiment_v3.py').read_text())
fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('tokenize','extract_all_features')]
ns={'re':re,'difflib':difflib};exec(compile(ast.Module(body=fns,type_ignores=[]),'training_features','exec'),ns)
def row(eid,name,addr,src=1,country='US'):
 return dict(entity_id=eid,source=src,country=country,name_raw=name,address_raw=addr,name_norm=normalize_text(name),address_norm=normalize_text(addr))
class SubmissionTests(unittest.TestCase):
 def test_feature_parity_with_training(self):
  samples=[row('x',n,a) for n,a in [('Café  Alpha','193 133, Arthur IL'),('CAFE ALPHA LLC','198 133 Arthur Illinois'),('',''),('ग्रेट एग्रो','15/49 New Delhi'),('名'*220,'1'*210),(' \t ','!!!'),('abc abc','1/2')]]
  for r in samples:
   for t in samples:self.assertEqual(features(values(r),values(t)),ns['extract_all_features'](r,t))
 def test_index_country_blank_and_deterministic_retrieval(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'db';c=sqlite3.connect(path)
   c.execute('CREATE TABLE records(entity_id TEXT PRIMARY KEY,source INT,country TEXT,name_raw TEXT,address_raw TEXT,name_norm TEXT,address_norm TEXT)')
   data=[row('S2-A','Café Alpha','12 Main',2),row('S3-B','CAFÉ Alpha','12 Main',3),row('S2-C','Café Alpha','12 Main',2,'India'),row('S2-empty','','',2)]
   data.extend(row(f'S2-d{i:03d}',f'Unique{i}',f'{i} Road{i}',2) for i in range(100))
   c.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',[tuple(r.values()) for r in data]);c.commit();c.close()
   index=runner.CountryIndex(path,'US')
   self.assertEqual(index.retrieve(row('S1-R','café alpha','12 main')),['S2-A','S2-d012','S3-B'])
   self.assertEqual(index.retrieve(row('S1-empty','','')),[])
 def test_checkpoint_resume_corruption_and_incomplete_assembly(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'batches/00/000000';refs=[row('S1-A','A','B'),row('S1-B','C','D')]
   runner.write_batch(folder,refs,[['S2-X'],[]],[[],[]],np.zeros((1,22)),np.array([.1]),'run',{'seconds':1})
   self.assertTrue(runner.valid_batch(folder,['S1-A','S1-B'],'run'))
   with self.assertRaises(ValueError):runner.valid_batch(folder,['S1-B','S1-A'],'run')
   with self.assertRaises(ValueError):runner.assemble(root,'run',3)
   out=runner.assemble(root,'run',2)
   with (out/'matching_results.tsv').open() as f:self.assertEqual(len(list(csv.DictReader(f,delimiter='\t'))),2)
   (folder/'matching_results.tsv').write_text('corruption')
   with self.assertRaises(ValueError):runner.valid_batch(folder,['S1-A','S1-B'],'run')
if __name__=='__main__':unittest.main()
