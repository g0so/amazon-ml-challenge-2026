"""Independently rescore exported dev TSVs with the Python metric, streaming IDs."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from business_entity_resolution.workspace import ROOT
from business_entity_resolution.split import readonly
from business_entity_resolution.metrics import parse_match_ids, score_entity


def verify():
    report=json.loads((ROOT/'reports/phase2_baseline.json').read_text())
    db=readonly(ROOT/'data/processed/phase2_baseline_dev.sqlite')
    results={}
    try:
        for rule, expected in report['metrics'].items():
            truth_rows=db.execute('''SELECT r.entity_id,t.target_id FROM records r
                LEFT JOIN truth_edges t ON t.reference_id=r.entity_id
                WHERE r.source=1 ORDER BY r.entity_id,t.target_id''')
            truth_groups=itertools.groupby(truth_rows,key=lambda row:row[0])
            total=tp=fp=fn=count=0
            with (ROOT/f'data/processed/phase2/dev_{rule}_predictions.tsv').open(newline='') as f:
                rows=csv.DictReader(f,delimiter='\t')
                if rows.fieldnames != ['source1_entity_id','matched_entity_ids']:
                    raise ValueError('Wrong prediction columns')
                for pair in itertools.zip_longest(truth_groups, rows):
                    group,row=pair
                    if group is None or row is None:
                        raise ValueError('Missing or extra prediction rows')
                    ref,links=group
                    if row['source1_entity_id']!=ref:
                        raise ValueError('Prediction reference missing, duplicated, extra, or out of order')
                    truth={target for _,target in links if target is not None}
                    prediction=parse_match_ids(row['matched_entity_ids'])
                    tp+=len(truth&prediction); fp+=len(prediction-truth); fn+=len(truth-prediction)
                    total+=score_entity(truth,prediction); count+=1
            score=total/count
            if (count,tp,fp,fn)!=(expected['references'],expected['tp'],expected['fp'],expected['fn']):
                raise AssertionError('TSV counts do not match saved SQL evaluation')
            if not math.isclose(score,expected['macro_f05'],rel_tol=0,abs_tol=1e-12):
                raise AssertionError('Independent metric disagrees with SQL')
            results[rule]={'rows':count,'macro_f05':score,'tp':tp,'fp':fp,'fn':fn,'passed':True}
    finally: db.close()
    (ROOT/'reports/phase2_output_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': verify()
