import importlib.util
import itertools
import types
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "model extras not installed")
class DecoderModelTest(unittest.TestCase):
    def tiny_judge(self):
        import torch
        from transformers import LlamaConfig, LlamaModel
        from decision_model.decoder_model import DecoderJudge

        torch.manual_seed(23)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False)
        judge = DecoderJudge.__new__(DecoderJudge)
        judge.config = {"candidate_encoding": "independent", "candidate_chunk_size": 2}
        judge.torch = torch
        judge.device = torch.device("cpu")
        judge.tokenizer = types.SimpleNamespace(pad_token_id=0)
        judge.encoder = LlamaModel(config)
        judge.head = torch.nn.Linear(32, 1)
        judge.residual_head = None
        judge.parent_decision_frozen = False
        judge.track_frozen_parent_gradients = False
        judge.work = dict.fromkeys(("encoder_calls", "encoder_sequences", "unpadded_tokens",
                                   "padded_token_slots", "attention_token_pairs"), 0)
        return judge

    def test_scores_are_permutation_equivariant_and_backpropagate(self):
        import torch
        judge = self.tiny_judge()
        judge.train(True)
        candidates = [([1, 4, 5], [2]), ([1, 6, 7, 8], [3]), ([1, 9, 10, 11, 12], [4])]
        original = judge.logits([candidates])[0]
        for order in itertools.permutations(range(3)):
            value = judge.logits([[candidates[index] for index in order]])[0]
            self.assertTrue(torch.allclose(value, original[list(order)], atol=1e-6))
        original.sum().backward()
        self.assertGreater(judge.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(judge.encoder.embed_tokens.weight.grad.abs().sum().item(), 0)

    def test_other_candidate_does_not_change_score(self):
        import torch
        judge = self.tiny_judge()
        judge.train(False)
        first = ([1, 4, 5], [2])
        with torch.no_grad():
            left = judge.logits([[first, ([1, 6, 7], [2])]])[0, 0]
            right = judge.logits([[first, ([1, 20, 21, 22, 23], [4])]])[0, 0]
        self.assertEqual(left.item(), right.item())

    def test_cached_features_reproduce_path_logits_and_candidate_order(self):
        import torch
        judge = self.tiny_judge()
        judge.train(False)
        requests = [
            [([1, 4, 5], [2]), ([1, 6, 7, 8], [3])],
            [([1, 9, 10, 11], [3]), ([1, 12, 13], [2]), ([1, 14, 15], [2])],
        ]
        with torch.no_grad():
            features = judge.features(requests)
            direct = judge.logits(requests)
            cached = judge.head(features).squeeze(-1)
            cached[0, 2] = -1e4
        self.assertEqual(tuple(features.shape), (2, 3, 32))
        self.assertTrue(torch.allclose(direct, cached, atol=1e-6))
        self.assertTrue(torch.equal(features[0, 2], torch.zeros(32)))
        with torch.no_grad():
            permuted = judge.features([[requests[1][2], requests[1][0], requests[1][1]]])[0]
        self.assertTrue(torch.allclose(permuted, features[1, [2, 0, 1]], atol=1e-6))

    def test_shared_prefix_matches_private_paths_and_reduces_attention_work(self):
        import torch
        candidates = [([1, 4, 5, 6, 7, tail], [5]) for tail in range(8, 16)]

        path = self.tiny_judge()
        path.train(False)
        path.config["inference_encoding"] = "path"
        with torch.no_grad():
            expected = path.logits([candidates])[0]

        shared = self.tiny_judge()
        shared.encoder.load_state_dict(path.encoder.state_dict())
        shared.head.load_state_dict(path.head.state_dict())
        shared.train(False)
        shared.config.update(inference_encoding="shared", candidate_chunk_size=8)
        with torch.no_grad():
            actual = shared.logits([candidates])[0]

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(shared.work["shared_prefix_requests"], 1)
        self.assertLess(shared.work["attention_token_pairs"], path.work["attention_token_pairs"])

    def test_encode_uses_full_admission_budget_and_final_positions(self):
        from decision_model.decoder_model import DecoderJudge

        request = {"task": "Pick.", "context": "Context.", "choices": [
            {"id": "a", "description": "Alpha."}, {"id": "b", "description": "Beta."}]}
        tokenizer = types.SimpleNamespace(
            encode=lambda text, **kwargs: [1] + [3 + ord(character) % 50 for character in text] + [2])
        judge = DecoderJudge.__new__(DecoderJudge)
        judge.tokenizer = tokenizer
        judge.config = {"max_tokens": 1000}
        encoded = judge.encode(request)
        self.assertEqual(len(encoded), 2)
        self.assertTrue(all(markers == [len(tokens) - 1] for tokens, markers in encoded))
        judge.config = {"max_tokens": 2}
        with self.assertRaisesRegex(ValueError, "Input too long"):
            judge.encode(request)

    def test_zero_residual_preserves_parent_and_only_expert_receives_gradient(self):
        import torch

        judge = self.tiny_judge()
        judge.train(False)
        parent = judge.logits([[([1, 4, 5], [2]), ([1, 6, 7], [2])]]).detach()
        judge._encoder_decision_names = tuple(
            name for name, parameter in judge.encoder.named_parameters()
            if parameter.requires_grad)
        judge.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(32), torch.nn.Linear(32, 8), torch.nn.GELU(),
            torch.nn.Dropout(0.0), torch.nn.Linear(8, 1))
        torch.nn.init.zeros_(judge.residual_head[-1].weight)
        torch.nn.init.zeros_(judge.residual_head[-1].bias)
        judge.parent_decision_frozen = True
        judge.encoder.requires_grad_(False)
        judge.head.requires_grad_(False)
        judge.encoder.gradient_checkpointing_disable()
        judge.train(True)

        composed = judge.logits([[([1, 4, 5], [2]), ([1, 6, 7], [2])]])
        self.assertTrue(torch.equal(composed, parent))
        torch.nn.functional.cross_entropy(composed, torch.tensor([1])).backward()
        self.assertTrue(all(parameter.grad is None for parameter in judge.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in judge.head.parameters()))
        self.assertGreater(judge.residual_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(name.startswith("residual_head.")
                            for name, _ in judge.named_trainable()))

    def test_autograd_tracked_parent_stays_outside_optimizer_and_clears_gradients(self):
        import torch

        judge = self.tiny_judge()
        judge._encoder_decision_names = tuple(
            name for name, parameter in judge.encoder.named_parameters()
            if parameter.requires_grad)
        judge.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(32), torch.nn.Linear(32, 8), torch.nn.GELU(),
            torch.nn.Dropout(0.0), torch.nn.Linear(8, 1))
        torch.nn.init.zeros_(judge.residual_head[-1].weight)
        torch.nn.init.zeros_(judge.residual_head[-1].bias)
        judge.parent_decision_frozen = True
        judge.track_frozen_parent_gradients = True
        judge.head.requires_grad_(False)
        judge.encoder.execution_probe_dropout = torch.nn.Dropout(0.5)
        parent_before = {name: parameter.detach().clone()
                         for name, parameter in judge.encoder.named_parameters()}
        trainable = list(judge.named_trainable())
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("residual_head.") for name, _ in trainable))
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-3)

        judge.train(True)
        self.assertTrue(judge.encoder.training)
        self.assertFalse(judge.encoder.execution_probe_dropout.training)
        logits = judge.logits([[([1, 4, 5], [2]), ([1, 6, 7], [2])]])
        torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in judge.encoder.parameters()))
        optimizer.step()
        judge.clear_frozen_parent_gradients()

        self.assertTrue(all(parameter.grad is None for parameter in judge.encoder.parameters()))
        self.assertTrue(all(torch.equal(parameter, parent_before[name])
                            for name, parameter in judge.encoder.named_parameters()))


if __name__ == "__main__":
    unittest.main()
