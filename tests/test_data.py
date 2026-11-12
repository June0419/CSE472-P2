"""Offline checks for data integrity; no dataset or model downloads."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cse472_project2.data import export_dataset, format_example, prepare_splits


def example(identifier="sample", numeric=False):
    labels = ["1", "2", "3"] if numeric else ["A", "B", "C"]
    return {"id": identifier, "question": "Which symbol is a star?",
            "choices": {"label": labels, "text": ["circle", "star", "square"]},
            "answerKey": labels[1]}


class DataTests(unittest.TestCase):
    def test_numeric_labels_and_separate_answer(self):
        row = example(numeric=True)
        before = copy.deepcopy(row)
        result = format_example(row)
        self.assertEqual(row, before)
        self.assertEqual(result['target'], '2')
        self.assertIn('2. star', result['prompt'])
        self.assertTrue(result['prompt'].endswith('Answer:'))
        other = copy.deepcopy(row)
        other['answerKey'] = '1'
        self.assertEqual(format_example(other)['prompt'], result['prompt'])

    def test_malformed_examples_rejected(self):
        for change in (lambda r: r.update(answerKey='Z'),
                       lambda r: r.update(question=''),
                       lambda r: r['choices'].update(label=['A', 'A', 'C']),
                       lambda r: r['choices'].update(text=['one'])):
            row = example()
            change(row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                format_example(row)

    def test_duplicate_ids_rejected(self):
        splits = {s: [example(s)] for s in ('train', 'validation', 'test')}
        splits['test'] = [example('train')]
        with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
            prepare_splits(splits)
        splits['test'] = [example('test')]
        splits['train'].append(example('train'))
        with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
            prepare_splits(splits)

    def test_missing_and_empty_splits_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing split'):
            prepare_splits({'train': [example()]})
        with self.assertRaisesRegex(ValueError, 'Empty split'):
            prepare_splits({'train': []})

    def test_export_preserves_records_and_does_not_overwrite(self):
        splits = {s: [example(s)] for s in ('train', 'validation', 'test')}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'original'
            manifest = export_dataset(splits, destination, {'resolved_revision': 'fixture'})
            for split in splits:
                raw = destination / f'raw/{split}.jsonl'
                self.assertEqual(json.loads(raw.read_text()), splits[split][0])
                self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),
                                 manifest['files'][f'raw/{split}.jsonl']['sha256'])
            with self.assertRaises(FileExistsError):
                export_dataset(splits, destination, {})
            self.assertTrue((destination / 'manifest.json').exists())

    def test_invalid_data_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'original'
            with self.assertRaises(ValueError):
                export_dataset({'train': [example()]}, destination, {})
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
