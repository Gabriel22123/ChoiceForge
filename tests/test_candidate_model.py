import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None,"model extras not installed")
class CandidateModelTests(unittest.TestCase):
    def independent_judge(self):
        import types
        import torch
        from transformers import EuroBertConfig, EuroBertModel
        from decision_model.model import Judge
        torch.manual_seed(17)
        cfg = EuroBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, pad_token_id=0, bos_token_id=1,
                            eos_token_id=2, mask_token_id=3)
        judge = Judge.__new__(Judge)
        judge.config = {"candidate_encoding": "independent", "candidate_chunk_size": 2}
        judge.torch = torch
        judge.device = torch.device("cpu")
        judge.tokenizer = types.SimpleNamespace(pad_token_id=0)
        judge.encoder = EuroBertModel(cfg)
        judge.head = torch.nn.Linear(32, 1)
        judge.work = dict.fromkeys(("encoder_calls", "encoder_sequences", "unpadded_tokens",
                                   "padded_token_slots", "attention_token_pairs"), 0)
        return judge

    def test_independent_permutation_equivariance(self):
        import itertools
        import torch
        judge = self.independent_judge()
        judge.train(False)
        candidates = [([1, 3, 4], [1]), ([1, 3, 5, 6], [1]), ([1, 3, 7, 8, 9], [1])]
        with torch.no_grad():
            original = judge.logits([candidates])[0]
            for order in itertools.permutations(range(3)):
                permuted = judge.logits([[candidates[i] for i in order]])[0]
                self.assertTrue(torch.equal(permuted, original[list(order)]))

    def test_independent_mixed_candidates_backprop_and_work(self):
        import torch
        judge = self.independent_judge()
        judge.train(True)
        pairs = [([1, 3, 4], [1]), ([1, 3, 5, 6], [1]), ([1, 3, 7, 8, 9], [1])]
        scores = judge.logits([pairs[:2], pairs])
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertEqual(scores[0, 2].item(), -1e4)
        self.assertEqual(scores[0].softmax(-1)[2].item(), 0)
        torch.nn.functional.cross_entropy(scores, torch.tensor([1, 2])).backward()
        self.assertGreater(judge.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(judge.encoder.embed_tokens.weight.grad.abs().sum().item(), 0)
        self.assertEqual(judge.work["encoder_sequences"], 5)
        self.assertEqual(judge.work["encoder_calls"], 3)

    def test_independent_candidate_score_does_not_read_other_candidates(self):
        import torch
        judge = self.independent_judge()
        judge.train(False)
        first = ([1, 3, 4], [1])
        with torch.no_grad():
            a = judge.logits([[first, ([1, 3, 5], [1])]])[0, 0]
            b = judge.logits([[first, ([1, 3, 20, 21, 22], [1])]])[0, 0]
        self.assertEqual(a.item(), b.item())

    def test_variable_candidate_forward_and_backward(self):
        import types
        import torch
        from transformers import EuroBertConfig,EuroBertModel
        from decision_model.model import Judge
        cfg=EuroBertConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
                          num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                          max_position_embeddings=64,pad_token_id=0,bos_token_id=1,eos_token_id=2,mask_token_id=3)
        judge=Judge.__new__(Judge)
        judge.config={}
        judge.torch=torch;judge.device=torch.device("cpu")
        judge.tokenizer=types.SimpleNamespace(pad_token_id=0)
        judge.encoder=EuroBertModel(cfg)
        judge.head=torch.nn.Linear(32,1)
        scores=judge.logits([([1,2,3,4],[1,3]),([1,2,3,4,5,6],[1,3,5])])
        self.assertEqual(tuple(scores.shape),(2,3))
        self.assertEqual(scores[0,2].item(),-1e4)
        self.assertEqual(scores[0].softmax(-1)[2].item(),0)
        loss=torch.nn.functional.cross_entropy(scores,torch.tensor([1,2]))
        loss.backward()
        self.assertGreater(judge.head.weight.grad.abs().sum().item(),0)
        self.assertGreater(judge.encoder.embed_tokens.weight.grad.abs().sum().item(),0)


if __name__=="__main__":unittest.main()
