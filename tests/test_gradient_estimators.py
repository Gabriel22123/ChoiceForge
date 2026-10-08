import unittest
import torch
from decision_model.objective import noisy_objective,proper_cost


class GradientEstimatorTests(unittest.TestCase):
    def test_pathwise_matches_common_noise_finite_difference(self):
        generator=torch.Generator().manual_seed(13)
        noise=torch.randn(16,1,3,generator=generator,dtype=torch.float64)
        mu=torch.tensor([[.2,-.3,.1]],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([2])
        value=noisy_objective(mu,target,noise)
        grad=torch.autograd.grad(value,mu)[0]
        for k in range(3):
            direction=torch.zeros_like(mu);direction[0,k]=1e-5
            finite=(noisy_objective(mu.detach()+direction,target,noise)-noisy_objective(mu.detach()-direction,target,noise))/(2e-5)
            self.assertAlmostEqual(grad[0,k].item(),finite.item(),places=7)

    def test_score_function_mean_agrees_with_pathwise(self):
        generator=torch.Generator().manual_seed(14)
        # Many independent groups, each with its own independent LOO baseline.
        trials,samples=4096,8
        mu=torch.tensor([[.2,-.3,.1]],dtype=torch.float64).repeat(trials,1).requires_grad_()
        noise=torch.randn(samples,trials,3,generator=generator,dtype=torch.float64)
        targets=torch.full((trials,),2,dtype=torch.long)
        path=torch.autograd.grad(noisy_objective(mu,targets,noise),mu)[0].sum(0)
        score=torch.autograd.grad(noisy_objective(mu,targets,noise,estimator="score_function"),mu)[0].sum(0)
        self.assertLess((path-score).abs().max().item(),.025)

    def test_smoothed_objective_is_not_clean_objective(self):
        mu=torch.zeros(1,3,dtype=torch.float64)
        noise=torch.tensor([[[1.,-1.,0.]],[[-1.,1.,0.]]],dtype=torch.float64)
        self.assertGreater(noisy_objective(mu,torch.tensor([0]),noise).item(),proper_cost(mu,torch.tensor([0])).item())

    def test_gradcheck_soft_targets(self):
        mu=torch.tensor([[.2,.3,.1]],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([[.2,.5,.3]],dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x:proper_cost(x,target).mean(),(mu,)))


if __name__=="__main__":unittest.main()
