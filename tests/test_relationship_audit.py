"""Small fixtures verify the audit before scanning the full training data."""
import csv
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('relationship_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_relationships.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_tsv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(columns)
        writer.writerows(rows)


class RelationshipAuditTests(unittest.TestCase):
    def fixture(self, folder, truth, sources):
        write_tsv(folder / 'train_ground_truth.tsv', ['source1_entity_id', 'matched_entity_ids'], truth)
        for source in (1, 2, 3):
            write_tsv(folder / f'train_source{source}.tsv',
                      ['entity_id', 'business_name', 'business_address', 'country'],
                      [(entity, 'Example', '', 'US') for entity in sources[source]])

    def test_shared_targets_singletons_distractors_and_safe_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, [('S1-A', 'S2-X,S2-Y'), ('S1-B', 'S3-Z'), ('S1-C', ''), ('S1-D', 'S2-Y,S3-W')],
                         {1: ['S1-A', 'S1-B', 'S1-C', 'S1-D'], 2: ['S2-X', 'S2-Y', 'S2-distractor'], 3: ['S3-Z', 'S3-W']})
            path = root / 'audit.sqlite'
            for _ in range(2):
                report = module.audit(root, path)
                self.assertEqual(report['reference_rows'], 4)
                self.assertEqual(report['positive_link_rows'], 5)
                self.assertEqual(report['singleton_rows'], 1)
                self.assertEqual(report['distinct_labeled_targets'], 4)
                self.assertEqual(report['unmatched_target_record_rows'], 1)
                self.assertEqual(report['checks']['targets_with_multiple_references']['count'], 1)
                self.assertTrue(report['structural_integrity_passed'])
                self.assertTrue(report['grouping_requires_shared_target_components'])
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM truth_links').fetchone()[0], 5)
            self.assertFalse(list(root.glob('*.building')))

    def test_integrity_problems_are_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, [('S1-A', 'S2-X,S2-X'), ('S1-A', ''), ('S1-missing', 'S3-missing')],
                         {1: ['S1-A', 'S1-unlabeled'], 2: ['S2-X', 'S2-X'], 3: []})
            report = module.audit(root, root / 'audit.sqlite')
            self.assertFalse(report['structural_integrity_passed'])
            for name in ['duplicate_reference_ids', 'duplicate_source_ids', 'truth_references_missing_from_source1',
                         'source1_records_missing_truth', 'truth_targets_missing_from_sources']:
                self.assertEqual(report['checks'][name]['count'], 1, name)
            self.assertEqual(report['truth_rows_with_duplicate_targets'], 1)
            self.assertEqual(report['checks']['targets_with_multiple_references']['count'], 0)


if __name__ == '__main__':
    unittest.main()
