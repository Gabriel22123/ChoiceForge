"""Independent checkpoint evaluation and candidate-order audits."""
import itertools
import json
import random
from pathlib import Path

from .core import digest, load_rows, write_json
from .train import balanced_subset, evaluate_rows, predict_rows


def load_judge(args):
    from .judge_factory import make_judge, verify_base_for_config
    config=json.loads((Path(args.checkpoint)/"model.json").read_text())
    if args.base_path:
        verify_base_for_config(config,args.base_path)
    return make_judge(config,base_path=args.base_path,checkpoint=args.checkpoint,device=args.device)


def evaluate(args):
    judge=load_judge(args)
    rows=balanced_subset([r for r in load_rows(args.data) if r["split"]==args.split],args.max_cases)
    tokens={r["id"]:judge.encode(r["request"]) for r in rows}
    logits=predict_rows(judge,rows,tokens)
    raw,_=evaluate_rows(rows,logits)
    calibrated,predictions=evaluate_rows(rows,logits,judge.temperature)
    result={"dataset_sha256":digest(Path(args.data).read_bytes()),
            "weights_sha256":digest((Path(args.checkpoint)/"decision.safetensors").read_bytes()),
            "split":args.split,"raw":raw,"temperature_scaled":calibrated,"predictions":predictions}
    write_json(args.output,result)
    return {"raw":raw,"temperature_scaled":calibrated,"output":str(args.output)}


def audit(args):
    run_path=Path(args.checkpoint)
    run=json.loads((run_path/"run.json").read_text())
    if digest(Path(args.data).read_bytes()) != run["dataset_sha256"]:
        raise ValueError("Audit dataset differs from training provenance")
    ids=set(run["selected_ids"]["test"])
    rows=balanced_subset([r for r in load_rows(args.data) if r["id"] in ids],args.max_cases)
    saved={r["id"]:r for r in json.loads((run_path/"test-predictions.json").read_text())}
    judge=load_judge(args)
    details=[]
    rng=random.Random(913)
    for row in rows:
        request=row["request"]
        original=judge.predict(request)
        reload_difference=max(abs(original["probabilities"][c]-v) for c,v in saved[row["id"]]["probabilities"].items())
        choices=request["choices"]; count=len(choices)
        if count<=3:
            orders=list(itertools.permutations(range(count)))
        else:
            orders=[tuple(range(count)),tuple(reversed(range(count)))]
            while len(orders)<6:
                order=list(range(count));rng.shuffle(order)
                if tuple(order) not in orders:orders.append(tuple(order))
        decisions=[]; differences=[]
        for order in orders:
            view=dict(request,choices=[choices[i] for i in order])
            result=judge.predict(view)
            decisions.append(result["choice_id"])
            differences.append(max(abs(result["probabilities"][c]-p) for c,p in original["probabilities"].items()))
        renamed=dict(request,choices=[dict(c,id=f"renamed_{i}") for i,c in enumerate(choices)])
        renamed_result=judge.predict(renamed)
        rename_difference=max(abs(renamed_result["probabilities"][f"renamed_{i}"]-original["probabilities"][c["id"]]) for i,c in enumerate(choices))
        details.append({"id":row["id"],"source":row["source"],"evaluation_regime":row["evaluation_regime"],
                        "permutations":len(orders),"stable":len(set(decisions))==1,
                        "correct_all_orders":all(c==row["label"] for c in decisions),
                        "max_order_probability_difference":max(differences),
                        "reload_probability_difference":reload_difference,
                        "id_rename_probability_difference":rename_difference})
    result={"cases":len(details),"stable_all_orders":sum(d["stable"] for d in details),
            "correct_all_orders":sum(d["correct_all_orders"] for d in details),
            "max_reload_probability_difference":max(d["reload_probability_difference"] for d in details),
            "max_id_rename_probability_difference":max(d["id_rename_probability_difference"] for d in details),
            "details":details}
    write_json(args.output,result)
    return {k:v for k,v in result.items() if k!="details"}
