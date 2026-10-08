"""Run the frozen three-arm local study serially on one device."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="runs/calibration-study-v1")
    parser.add_argument("--data",default="data/public-multitask-v1/cases.jsonl")
    parser.add_argument("--base-path",default="models/eurobert-2.1b")
    parser.add_argument("--device",default="mps")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=(root/args.output).resolve()
    if output.exists():raise ValueError("Use a new study directory")
    output.mkdir(parents=True)
    config=json.loads((root/"configs/eurobert-public-pilot.json").read_text())
    config["training_views"]=2
    variants={"A-ce":(0.,0.),"B-proper":(.5,0.),"C-consistency":(.5,.05)}
    files={p.name:digest(p.read_bytes()) for p in sorted((root/"src/decision_model").glob("*.py"))}
    protocol={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "dataset_sha256":digest((root/args.data).read_bytes()),"source_files":files,
              "base_config":config,"variants":variants,"seed_count":1,
              "scope":"exploratory single-seed ablation; previously inspected diagnostic test; not final blind evaluation",
              "controls":["same initial trainable parameters","same selected IDs","same training permutations",
                          "two forward views in every arm","same updates and optimizer settings","fixed final epoch"],
              "hypotheses":{"B_vs_A":"Does adding Brier improve NLL/Brier/calibration without sacrificing accuracy?",
                            "C_vs_B":"Does aligned JS improve order stability, holding augmentation and compute fixed?"},
              "predeclared_metrics":["per-source accuracy","NLL","Brier","ECE","selective coverage",
                                     "24-example candidate permutation audit","training seconds"],
              "audit_cases":24,"test_used_for_checkpoint_selection":False}
    write_json(output/"protocol.json",protocol)
    for name,(beta,consistency) in variants.items():
        variant=dict(config,brier_weight=beta,consistency_weight=consistency)
        write_json(output/(name+".json"),variant)
    write_json(output/"protocol-checksums.json",{p.name:digest(p.read_bytes()) for p in output.glob("*.json")})
    reference=None
    for name in variants:
        current={p.name:digest(p.read_bytes()) for p in sorted((root/"src/decision_model").glob("*.py"))}
        if current!=files:raise ValueError("Training source changed during frozen study")
        write_json(output/"status.json",{"state":"training","arm":name})
        print(json.dumps({"event":"arm_start","arm":name}),flush=True)
        cmd=[sys.executable,"-m","decision_model.cli","train","--data",str(root/args.data),
             "--config",str(output/(name+".json")),"--output",str(output/name),
             "--base-path",str(root/args.base_path),"--device",args.device]
        with (output/(name+".log")).open("w") as log:
            subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        run=json.loads((output/name/"run.json").read_text())
        controls={k:run[k] for k in ("initial_trainable_sha256","selected_ids_sha256","training_plan_sha256",
                                    "updates","forward_sequences","forward_batches","training_views")}
        if reference is None:reference=controls
        elif controls!=reference:raise ValueError("Controlled comparison failed: "+name)
        write_json(output/"matched-controls.json",reference)
        write_json(output/"status.json",{"state":"auditing","arm":name})
        cmd=[sys.executable,"-m","decision_model.cli","audit","--checkpoint",str(output/name),
             "--data",str(root/args.data),"--output",str(output/name/"audit.json"),"--max-cases","24",
             "--base-path",str(root/args.base_path),"--device",args.device]
        with (output/(name+"-audit.log")).open("w") as log:
            subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({"event":"arm_complete","arm":name}),flush=True)
    write_json(output/"status.json",{"state":"complete","arms":list(variants)})


if __name__=="__main__":main()
