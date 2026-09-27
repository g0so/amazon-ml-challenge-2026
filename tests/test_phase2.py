import itertools
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from business_entity_resolution.normalize import normalize_text, normalize_address
from business_entity_resolution.metrics import score_entity, macro_score, parse_match_ids
from business_entity_resolution.baseline import create_record_table, record_values, generate_predictions, evaluate_predictions
from business_entity_resolution.split import create_manifest, verify_manifest, readonly
from test_relationship_audit import module, write_tsv

class Phase2Tests(unittest.TestCase):
    def test_conservative_normalization(self):
        raw='  CAFÉ\t12 CT FL 3  '
        self.assertEqual(normalize_address(raw), (raw, 'café 12 ct fl 3'))
        self.assertEqual(normalize_text('Cafe\u0301'), 'café')
        self.assertNotEqual(normalize_text('12 Main'), normalize_text('21 Main'))
        with self.assertRaises(TypeError): normalize_text(None)

    def test_metric_all_small_sets(self):
        sets=[set(x for x,b in zip('abc',bits) if b) for bits in itertools.product([0,1],repeat=3)]
        for truth,pred in itertools.product(sets,repeat=2):
            tp=len(truth&pred); fp=len(pred-truth); fn=len(truth-pred)
            expected=1 if not(truth or pred) else 1.25*tp/(1.25*tp+fp+.25*fn)
            self.assertAlmostEqual(score_entity(truth,pred),expected)
        with self.assertRaises(KeyError): macro_score(['a'], {'a':set()}, {})
        with self.assertRaises(ValueError): parse_match_ids('S2-a,S2-a')

    def test_label_blind_distractors_country_blanks_and_sql_metric(self):
        db=sqlite3.connect(':memory:'); create_record_table(db)
        rows=[('S1-A',1,'US','Alpha','12 Main'),('S1-B',1,'US','Singleton','3 CT'),
              ('S1-C',1,'US','Blank',''),('S2-X',2,'US','Alpha','12 Main'),
              ('S3-Y',3,'US',' ALPHA ','12  Main'),('S2-fp',2,'US','Singleton','3 CT'),
              ('S3-wrong-country',3,'India','Alpha','12 Main'),('S2-blank',2,'US','Blank','')]
        for eid,src,country,name,address in rows:
            db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)',record_values(dict(entity_id=eid,country=country,business_name=name,business_address=address),src))
        generate_predictions(db) # No truth table exists yet.
        normalized=set(db.execute("SELECT reference_id,target_id FROM predictions WHERE rule='normalized_exact'"))
        self.assertEqual(normalized,{('S1-A','S2-X'),('S1-A','S3-Y'),('S1-B','S2-fp')})
        db.execute('CREATE TABLE truth_edges(reference_id TEXT,target_id TEXT,PRIMARY KEY(reference_id,target_id))')
        truth={'S1-A':{'S2-X','S3-Y'},'S1-B':set(),'S1-C':{'S2-blank'}}
        db.executemany('INSERT INTO truth_edges VALUES (?,?)',[(r,t) for r,ts in truth.items() for t in ts])
        results=evaluate_predictions(db)
        for rule in results:
            preds={r:set() for r in truth}
            for r,t in db.execute('SELECT reference_id,target_id FROM predictions WHERE rule=?',(rule,)): preds[r].add(t)
            self.assertAlmostEqual(results[rule]['macro_f05'],macro_score(truth,truth,preds))
            self.assertEqual(results[rule]['fp'],1)
        self.assertAlmostEqual(results['normalized_exact']['macro_f05'],1/3)
        db.close()

    def test_split_import_preserves_assignments_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for source,ids in [(1,['S1-A','S1-B','S1-C']),(2,['S2-X','S2-D']),(3,['S3-Y'])]:
                write_tsv(root/f'train_source{source}.tsv',['entity_id','business_name','business_address','country'],[(i,'Name','Addr','US') for i in ids])
            write_tsv(root/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],[('S1-A','S2-X'),('S1-B','S3-Y'),('S1-C','')])
            audit=root/'audit.sqlite'; report=module.audit(root,audit)
            legacy=root/'legacy.json'
            legacy.write_text(json.dumps({'input_hashes':{x['name']:x['sha256'] for x in report['inputs']},'seed':2026,
                'reference_assignments':{'S1-A':'train','S1-B':'dev','S1-C':'holdout'},'distractor_assignments':{'S2-D':'dev'}}))
            kwargs=dict(audit_path=audit,legacy_path=legacy,output_path=root/'split.sqlite',summary_path=root/'summary.json',train_dir=root)
            result=create_manifest(**kwargs)
            self.assertEqual(result['provenance']['mode'],'preserved_existing_json')
            self.assertEqual(create_manifest(**kwargs)['manifest_id'],result['manifest_id'])
            with sqlite3.connect(root/'split.sqlite') as db:
                self.assertEqual(db.execute("SELECT split FROM target_assignments WHERE target_id='S3-Y'").fetchone()[0],'dev')
                db.execute("UPDATE target_assignments SET split='train' WHERE target_id='S3-Y'"); db.commit()
                with readonly(audit) as a:
                    with self.assertRaisesRegex(ValueError,'ownership/partition'): verify_manifest(db,a,report)

if __name__=='__main__': unittest.main()
