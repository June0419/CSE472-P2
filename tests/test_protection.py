"""Unit tests for unlearnable text protection generation."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest


from cse472_project2.data import export_dataset
from cse472_project2.protection import (
    apply_cs_trap,
    apply_label_hint,
    apply_random_edit,
    export_protected_dataset,
    perturb_train_split,
    search_bilevel_perturbation,
)



def sample_record(identifier="Mercury_1", target="A", question="Which mineral is hardest?"):
    return {
        "id": identifier,
        "question": question,
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["talc", "diamond", "quartz", "calcite"],
        },
        "answerKey": target,
    }


class ProtectionTests(unittest.TestCase):
    def test_apply_label_hint_symbol(self):
        row_a = sample_record(target="A")
        mod_a = apply_label_hint(row_a, hint_style="symbol")
        self.assertEqual(row_a["id"], mod_a["id"])
        self.assertEqual(row_a["answerKey"], mod_a["answerKey"])
        self.assertEqual(row_a["choices"], mod_a["choices"])
        self.assertTrue(mod_a["question"].endswith(" [!]"))

        row_b = sample_record(target="B")
        mod_b = apply_label_hint(row_b, hint_style="symbol")
        self.assertTrue(mod_b["question"].endswith(" [@]"))

    def test_numeric_label_hints(self):
        row_1 = {
            "id": "Mercury_7085735",
            "question": "Which of the following describes an interaction between geosphere and atmosphere?",
            "choices": {"label": ["1", "2", "3", "4"], "text": ["Volcano eruption", "Plants", "Evaporation", "Wind"]},
            "answerKey": "1",
        }
        mod_1 = apply_label_hint(row_1, hint_style="symbol")
        self.assertEqual(mod_1["answerKey"], "1")
        self.assertTrue(mod_1["question"].endswith(" [!]"))

        row_2 = copy.deepcopy(row_1)
        row_2["answerKey"] = "2"
        mod_2 = apply_label_hint(row_2, hint_style="symbol")
        self.assertTrue(mod_2["question"].endswith(" [@]"))

    def test_numeric_cs_trap(self):
        row = {
            "id": "Mercury_7085735",
            "question": "Which of the following describes an interaction between geosphere and atmosphere?",
            "choices": {"label": ["1", "2", "3", "4"], "text": ["Volcano eruption", "Plants", "Evaporation", "Wind"]},
            "answerKey": "1",
        }
        mod = apply_cs_trap(row)
        self.assertTrue(mod["question"].startswith("Specifically, which of the following"))
        self.assertEqual(mod["answerKey"], "1")

    def test_surrogate_search_selects_lowest_loss_hint(self):
        row = sample_record(target="A")
        scored = []

        def candidate_loss(example, hint):
            scored.append((example["id"], hint))
            return {" [!]": 1.5, " [@]": 0.2, " [#]": 0.8}[hint]

        best = search_bilevel_perturbation(
            row=row,
            candidate_hints=[" [!]", " [@]", " [#]"],
            candidate_loss_fn=candidate_loss,
        )
        self.assertEqual(best, " [@]")
        self.assertEqual([hint for _, hint in scored], [" [!]", " [@]", " [#]"])
        self.assertEqual(row["question"], "Which mineral is hardest?")

    def test_surrogate_search_requires_model_or_loss_function(self):
        with self.assertRaisesRegex(ValueError, "Surrogate model and tokenizer"):
            search_bilevel_perturbation(sample_record(), [" [!]", " [@]"])

    def test_surrogate_search_exports_only_changed_training_questions(self):
        splits = {
            "train": [sample_record("tr1", "A"), sample_record("tr2", "B")],
            "validation": [sample_record("val1", "C")],
            "test": [sample_record("te1", "D")],
        }

        def candidate_loss(example, hint):
            preferred = " [@]" if example["answerKey"] == "A" else " [#]"
            return 0.1 if hint == preferred else 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            original = Path(temp_dir) / "original"
            protected = Path(temp_dir) / "protected"
            export_dataset(splits, original, {"resolved_revision": "test"})
            manifest = export_protected_dataset(
                original, protected, method="surrogate_search",
                candidate_hints=[" [!]", " [@]", " [#]"],
                candidate_loss_fn=candidate_loss,
                surrogate_model_name="test-surrogate",
            )
            train = [json.loads(line) for line in
                (protected / "raw/train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(train[0]["question"].endswith(" [@]"))
            self.assertTrue(train[1]["question"].endswith(" [#]"))
            for original_row, protected_row in zip(splits["train"], train):
                self.assertEqual(protected_row["answerKey"], original_row["answerKey"])
                self.assertEqual(protected_row["choices"], original_row["choices"])
            for split in ("validation", "test"):
                self.assertEqual(
                    (original / "raw" / f"{split}.jsonl").read_bytes(),
                    (protected / "raw" / f"{split}.jsonl").read_bytes(),
                )
            self.assertEqual(
                manifest["optimization"]["kind"],
                "single_pass_fixed_surrogate_candidate_search",
            )
            self.assertFalse(manifest["optimization"]["alternating_model_updates"])

    def test_apply_label_hint_punctuation(self):
        row_c = sample_record(target="C", question="Which is hardest?")
        mod_c = apply_label_hint(row_c, hint_style="punctuation")
        self.assertTrue(mod_c["question"].endswith("?"))

        row_d = sample_record(target="D", question="Which is hardest?")
        mod_d = apply_label_hint(row_d, hint_style="punctuation")
        self.assertTrue(mod_d["question"].endswith("!"))

    def test_apply_random_edit_matched_budget(self):
        row = sample_record(target="A")
        rng = random.Random(42)
        mod_random = apply_random_edit(row, rng=rng, hint_style="symbol")
        self.assertEqual(row["id"], mod_random["id"])
        self.assertEqual(row["answerKey"], mod_random["answerKey"])
        # Should end with one of the candidate symbols
        self.assertTrue(any(mod_random["question"].endswith(s) for s in ["[!]", "[@]", "[#]", "[$]", "[%]"]))

    def test_apply_cs_trap(self):
        row = sample_record(target="A", question="Which rock is igneous?")
        mod = apply_cs_trap(row)
        self.assertEqual(row["id"], mod["id"])
        self.assertEqual(row["answerKey"], mod["answerKey"])
        self.assertTrue(mod["question"].startswith("Specifically, which rock is igneous?"))

        row_b = sample_record(target="B", question="Which rock is igneous?")
        mod_b = apply_cs_trap(row_b)
        self.assertTrue(mod_b["question"].startswith("Notably, which rock is igneous?"))

    def test_export_protected_dataset_clean_test_split(self):
        splits = {
            "train": [sample_record("tr1", "A"), sample_record("tr2", "B")],
            "validation": [sample_record("val1", "C")],
            "test": [sample_record("te1", "D")],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            orig_dir = Path(temp_dir) / "original"
            export_dataset(splits, orig_dir, {"resolved_revision": "mock"})

            prot_dir = Path(temp_dir) / "protected"
            manifest = export_protected_dataset(
                original_data_dir=orig_dir,
                output_dir=prot_dir,
                method="label_hint",
                hint_style="symbol",
                seed=42,
            )

            self.assertTrue(prot_dir.exists())
            self.assertEqual(manifest["protection_method"], "label_hint")
            self.assertEqual(manifest["edit_budget"], 1)

            # Check that train was modified
            train_lines = (prot_dir / "raw/train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            train_record_0 = json.loads(train_lines[0])
            self.assertTrue(train_record_0["question"].endswith(" [!]"))

            # Check that test and validation were NOT modified
            orig_test = (orig_dir / "raw/test.jsonl").read_text(encoding="utf-8")
            prot_test = (prot_dir / "raw/test.jsonl").read_text(encoding="utf-8")
            self.assertEqual(orig_test, prot_test)

            orig_val = (orig_dir / "raw/validation.jsonl").read_text(encoding="utf-8")
            prot_val = (prot_dir / "raw/validation.jsonl").read_text(encoding="utf-8")
            self.assertEqual(orig_val, prot_val)

            # Test overwrite protection
            with self.assertRaises(FileExistsError):
                export_protected_dataset(orig_dir, prot_dir, method="cs_trap", overwrite=False)


if __name__ == "__main__":
    unittest.main()
