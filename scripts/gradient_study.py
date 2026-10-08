"""Known-objective gradient variance study, not a model benchmark or private-recipe replica."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from decision_model.core import write_json
from decision_model.objective import proper_cost


def analytic_gradient(actions,targets,beta=.5):
    q=actions.softmax(-1)
    difference=q-targets
    return difference+2*beta*q*(difference-(q*difference).sum(-1,keepdim=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="runs/gradient-estimators-v1")
    parser.add_argument("--projection-control",action="store_true",
                        help="Also remove the score estimator's irrelevant common-logit component")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(20260921)
    repeats=2048;reference_samples=131072
    results=[]
    for count in (3,8,16):
        for condition in ("uncertain","confident_wrong"):
            mu=torch.linspace(-.2,.2,count,dtype=torch.float64)
            if condition=="confident_wrong":mu[-1]=3.
            target=torch.nn.functional.one_hot(torch.tensor(0),count).double()
            for sigma in (.1,.4):
                reference_noise=torch.randn(reference_samples,count,generator=generator,dtype=torch.float64)
                gradients=analytic_gradient(mu+sigma*reference_noise,target)
                reference=gradients.mean(0)
                reference_se=gradients.std(0)/reference_samples**.5
                for group in (4,16,64):
                    noise=torch.randn(repeats,group,count,generator=generator,dtype=torch.float64)
                    actions=mu+sigma*noise
                    costs=proper_cost(actions,target)
                    baseline=(costs.sum(1,keepdim=True)-costs)/(group-1)
                    score=((costs-baseline).unsqueeze(-1)*noise/sigma).mean(1)
                    path=analytic_gradient(actions,target).mean(1)
                    measured={}
                    estimators=[("score_function_loo",score),("pathwise",path)]
                    if args.projection_control:
                        estimators.append(("score_function_loo_projected",score-score.mean(-1,keepdim=True)))
                    for name,estimates in estimators:
                        mean=estimates.mean(0)
                        measured[name]={"mean":mean.tolist(),
                                        "mean_error_norm":(mean-reference).norm().item(),
                                        "mean_standard_error_norm":(estimates.std(0)/repeats**.5).norm().item(),
                                        "variance_trace":estimates.var(0,unbiased=True).sum().item(),
                                        "mse_to_reference":((estimates-reference)**2).sum(-1).mean().item(),
                                        "wrong_gradient_direction_rate":((estimates*reference).sum(-1)<0).double().mean().item()}
                    results.append({"candidates":count,"condition":condition,"sigma":sigma,"samples":group,
                                    "reference_gradient":reference.tolist(),"reference_standard_error_norm":reference_se.norm().item(),
                                    "estimators":measured,"variance_ratio_score_over_path":measured["score_function_loo"]["variance_trace"]/measured["pathwise"]["variance_trace"]})
    result={"objective":"E_epsilon[CE(softmax(mu+sigma*epsilon),y)+0.5*Brier]",
            "seed":20260921,"repeats":repeats,"reference_samples":reference_samples,
            "matched_noise_samples_between_estimators":True,
            "baseline":"independent leave-one-out; no reward standardization",
            "projection_control":args.projection_control,
            "scope":"synthetic known differentiable objective; no backbone training, accuracy or OOD calibration claim",
            "results":results}
    write_json(output/"results.json",result)
    ratios=[r["variance_ratio_score_over_path"] for r in results]
    summary={"settings":len(results),"variance_ratio_min":min(ratios),
             "variance_ratio_median":statistics.median(ratios),"variance_ratio_max":max(ratios)}
    if args.projection_control:
        projected=[r["estimators"]["score_function_loo_projected"]["variance_trace"]/r["estimators"]["pathwise"]["variance_trace"] for r in results]
        summary["projected_ratio"]={"min":min(projected),"median":statistics.median(projected),"max":max(projected)}
    write_json(output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
