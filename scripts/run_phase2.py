"""Verify/import the saved split, run label-blind dev baselines, save results."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from business_entity_resolution.split import create_manifest
from business_entity_resolution.baseline import run_baseline

if __name__=='__main__':
    started=time.monotonic()
    manifest=create_manifest()
    result=run_baseline()
    sources=['src/business_entity_resolution/split.py','src/business_entity_resolution/normalize.py',
             'src/business_entity_resolution/baseline.py','src/business_entity_resolution/metrics.py','scripts/run_phase2.py']
    run={'elapsed_seconds':round(time.monotonic()-started,2),
         'peak_process_ram_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
         'memory_measurement':'Linux peak resident memory for the complete split+baseline process',
         'manifest_id':manifest['manifest_id'],
         'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'reports/phase2_run.json').write_text(json.dumps(run,indent=2)+'\n')
    for rule,metrics in result['metrics'].items():
        print(rule, 'macro F0.5:',metrics['macro_f05'],'TP/FP/FN:',metrics['tp'],metrics['fp'],metrics['fn'])
    print('Run complete:',run)
