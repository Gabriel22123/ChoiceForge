import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file

from decision_model.core import digest, write_json
from decision_model.feature_screen import (CandidateHead, FeatureStore, RoutedResidualHead,
                                           aligned_targets, padded_logits, parameter_sha256)


class FeatureScreenTests(unittest.TestCase):

    def test_routed_residual_is_exact_at_initialization_and_permutation_equivariant(self):
        import torch

        torch.manual_seed(19)
        module = RoutedResidualHead.build(6, 5, 4, initial_gate_probability=0.25)
        features = torch.randn(2, 3, 6)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        correction, gate = module(features, mask)
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 0.25)))

        optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
        target = torch.tensor([1, 0])
        parent = torch.tensor([[0.2, 0.1, -0.3], [0.0, 0.4, -1e4]])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            correction, _ = module(features, mask)
            torch.nn.functional.cross_entropy(parent + correction, target).backward()
            optimizer.step()
        correction, gate = module(features, mask)
        order = torch.tensor([2, 0, 1])
        permuted, permuted_gate = module(features[:, order], mask[:, order])
        self.assertTrue(torch.allclose(permuted, correction[:, order], atol=1e-6))
        self.assertTrue(torch.allclose(permuted_gate, gate, atol=1e-6))
        self.assertGreater(correction.abs().sum().item(), 0.0)
        self.assertGreater(module.router[-1].weight.grad.abs().sum().item(), 0.0)

    def test_routed_screen_aligns_frozen_features_logits_and_target(self):
        script_path = Path(__file__).parents[1] / "scripts/run_qasc_routed_residual_screen.py"
        spec = importlib.util.spec_from_file_location("routed_screen", script_path)
        screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
        row = {"id": "r", "label": "b", "source": "fixture",
               "request": {"task": "choose", "context": "evidence", "choices": [
                   {"id": "a", "description": "alpha"},
                   {"id": "b", "description": "beta"}]}}
        key = "cases:r"
        features = {key: (torch.tensor([[1., 2.], [3., 4.]]),
                          torch.tensor([[5., 6.], [7., 8.]]))}
        manifest = {"hidden_size": 2, "rows": {key: {
            "shape": [2, 2], "request_sha256": digest(row["request"]),
            "candidate_ids": ["a", "b"],
            "parent_logits": [.1, .9], "parent_null_logits": [.3, .7]}}}
        full, null, parent, parent_null, mask, target = screen.pad_batch(
            [row], [key], features, manifest, [[1, 0]])
        self.assertEqual(full[0].tolist(), [[3., 4.], [1., 2.]])
        self.assertEqual(null[0].tolist(), [[7., 8.], [5., 6.]])
        self.assertTrue(torch.allclose(parent, torch.tensor([[.9, .1]])))
        self.assertTrue(torch.allclose(parent_null, torch.tensor([[.7, .3]])))
        self.assertEqual(mask.tolist(), [[True, True]])
        self.assertEqual(target.tolist(), [0])

    def rows(self):
        return [{"label": "a", "target_probabilities": {"a": .8, "b": .2},
                 "request": {"choices": [{"id": "a"}, {"id": "b"}]}},
                {"label": "z", "target_probabilities": {"x": .1, "y": .2, "z": .7},
                 "request": {"choices": [{"id": "x"}, {"id": "y"}, {"id": "z"}]}}]

    def test_padding_permutation_and_target_alignment(self):
        torch.manual_seed(7); head = CandidateHead.build(4, 3)
        features = [torch.arange(8).reshape(2, 4).half(), torch.arange(12).reshape(3, 4).half()]
        order = [[1, 0], [2, 0, 1]]
        logits, mask = padded_logits(head, features, order)
        hard = aligned_targets(self.rows(), order, 3, "hard")
        soft = aligned_targets(self.rows(), order, 3, "distribution")
        self.assertEqual(tuple(logits.shape), (2, 3)); self.assertFalse(mask[0, 2])
        self.assertEqual(hard.tolist(), [1, 0])
        self.assertTrue(torch.allclose(soft[0], torch.tensor([.2, .8, 0.])))
        self.assertTrue(torch.allclose(soft[1], torch.tensor([.7, .1, .2])))
        torch.manual_seed(7); other = CandidateHead.build(4, 3)
        self.assertEqual(parameter_sha256(head), parameter_sha256(other))

    def test_store_rejects_capped_cache_for_formal_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); tensor = root / "features-00000.safetensors"
            save_file({"f00000000": torch.ones(2, 4)}, str(tensor))
            manifest = {"status": "complete", "hidden_size": 4,
                        "frozen": {"format": "typed-decisions-minicpm5-candidate-features-v1",
                                   "formal": False, "row_count": 1},
                        "shards": [{"file": tensor.name, "row_ids": ["r"],
                                    "sha256": digest(tensor.read_bytes())}],
                        "rows": {"r": {"shard": tensor.name, "key": "f00000000",
                                        "shape": [2, 4], "candidate_ids": ["a", "b"]}}}
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "capped"):
                FeatureStore(root)
            store = FeatureStore(root, require_formal=False)
            self.assertEqual(tuple(store.load_subset(["r"])["r"].shape), (2, 4))
            self.assertEqual(list(store.load_all()), ["r"])
            with self.assertRaisesRegex(ValueError, "duplicated or unknown"):
                store.load_subset(["r", "r"])

    def test_store_accepts_formal_outcome_feature_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); tensor = root / "features-00000.safetensors"
            save_file({"f00000000": torch.ones(2, 4)}, str(tensor))
            manifest = {"status": "complete", "hidden_size": 4,
                        "frozen": {"format": "mbpp-outcome-minicpm5-candidate-features-v1",
                                   "formal": True, "row_count": 1},
                        "shards": [{"file": tensor.name, "row_ids": ["r"],
                                    "sha256": digest(tensor.read_bytes())}],
                        "rows": {"r": {"shard": tensor.name, "key": "f00000000",
                                        "shape": [2, 4], "candidate_ids": ["a", "b"]}}}
            write_json(root / "manifest.json", manifest)
            self.assertEqual(tuple(FeatureStore(root).load_all()["r"].shape), (2, 4))

    def test_training_endpoint_is_deterministic_and_keeps_test_unread(self):
        script_path = Path(__file__).parents[1] / "scripts/run_typed_decisions_objective_screen.py"
        spec = importlib.util.spec_from_file_location("objective_screen", script_path)
        screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
        def row(index, split):
            target = {.0: {"a": .8, "b": .2}, 1.0: {"a": .2, "b": .8}}[float(index % 2)]
            label = max(target, key=target.get)
            return {"id": f"r{index}", "split": split, "source": "typed-decisions",
                    "family": "fixture", "group_id": f"g{index}", "decision_type": "boolean",
                    "evaluation_regime": "seen_task_new_examples", "label": label,
                    "target_probabilities": target, "provenance": {"source": "fixture"},
                    "request": {"task": "choose", "context": str(index), "choices": [
                        {"id": "a", "description": "alpha"}, {"id": "b", "description": "beta"}]}}
        rows = [row(0, "train"), row(1, "train"), row(2, "validation"), row(3, "validation")]
        features = {value["id"]: torch.tensor([[1., 0., value["id"] == "r0", 0.],
                                                [0., 1., 0., value["id"] == "r1"]]).half()
                    for value in rows}
        protocol = {"protocol_id": "fixture", "hidden_size": 4,
                    "head": {"width": 3, "dropout": 0., "epochs": 1, "batch_size": 2,
                             "learning_rate": .01, "weight_decay": 0., "gradient_clip": 1.},
                    "objective": {"brier_weight": .5, "noise_samples": 4,
                                  "noise_sigma": .4, "center_score_logits": True},
                    "arms": {"S-soft": {"training_target_mode": "distribution",
                                         "gradient_estimator": "clean",
                                         "ranked_probability_weight": 0., "clean_anchor_weight": 0.}},
                    "feature_manifest_sha256": "feature", "folds": {"fixture": {"fit_sha256": "fit"}}}
        records = screen.make_plan(rows, protocol, "fixture", 42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = root / "plan.jsonl"
            plan_hash = screen.write_plan(plan, records)
            run = screen.train_endpoint(root / "endpoint", rows, features, plan, plan_hash,
                                        protocol, "fixture", 42, "S-soft")
            self.assertEqual(run["status"], "fit_complete_test_unread")
            self.assertFalse(run["test_parsed"]); self.assertEqual(run["updates"], 1)
            self.assertTrue((root / "endpoint/validation-evaluation.json").is_file())
            self.assertEqual(screen.verify_endpoint(root / "endpoint"), run)

    def test_feature_identity_rejects_reused_id_with_changed_request(self):
        script_path = Path(__file__).parents[1] / "scripts/run_typed_decisions_objective_screen.py"
        spec = importlib.util.spec_from_file_location("objective_screen_identity", script_path)
        screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
        row = {"id": "r", "request": {"task": "choose", "context": "evidence",
                                        "choices": [{"id": "a", "description": "alpha"},
                                                    {"id": "b", "description": "beta"}]}}
        store = type("Store", (), {"manifest": {"rows": {"r": {
            "candidate_ids": ["a", "b"], "request_sha256": digest(row["request"])}}}})()
        screen.verify_feature_identity([row], store)
        changed = {**row, "request": {**row["request"], "context": "changed"}}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            screen.verify_feature_identity([changed], store)

    def test_advancement_requires_stable_workflow_and_seen_metrics(self):
        script_path = Path(__file__).parents[1] / "scripts/typed_decisions_objective_screen_report.py"
        spec = importlib.util.spec_from_file_location("objective_report", script_path)
        report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        good = [{"fold": fold, "seed": seed, "heldout_soft_ce": -.02,
                 "seen_soft_ce": -.001, "heldout_soft_brier": -.01}
                for fold in "abcd" for seed in (42, 43, 44)]
        self.assertTrue(report.advancement(good)["advances_to_lora_validation"])
        unstable = [dict(row) for row in good]
        for row in unstable:
            if row["fold"] == "a": row["heldout_soft_ce"] = .02
        value = report.advancement(unstable)
        self.assertFalse(value["advances_to_lora_validation"])
        self.assertFalse(value["checks"]["no_workflow_mean_ce_regresses_over_0_01"])


if __name__ == "__main__":
    unittest.main()
