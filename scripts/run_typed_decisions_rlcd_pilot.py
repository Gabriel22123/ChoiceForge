"""Run four matched arms that decompose public RLCD ingredients."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base


CONTROL_KEYS=("initial_trainable_sha256","selected_ids_sha256","training_plan_sha256",
              "updates","forward_sequences","forward_batches","training_views","encoder_work")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",default="configs/typed-decisions-rlcd-pilot-v1.json")
    parser.add_argument("--data",default="data/typed-decisions-rlcd-pilot-v1/cases.jsonl")
    parser.add_argument("--output",default="runs/typed-decisions-rlcd-pilot-v1")
    parser.add_argument("--base-path",default=None)
    parser.add_argument("--device",default="cpu",choices=("cpu","cuda","mps"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source_protocol=root/args.protocol;data=root/args.data;output=root/args.output
    if output.exists():raise ValueError("Use a new RLCD pilot output directory")
    protocol=json.loads(source_protocol.read_text())
    base=root/(args.base_path or protocol["base_path"])
    verify_local_decoder_base(base)
    rows=load_rows(data)
    counts={split:sum(row["split"]==split for row in rows) for split in ("train","validation","test")}
    if counts!={"train":30,"validation":15,"test":20}:
        raise ValueError("Pilot data shape differs from frozen 30/15/20 plan")
    paths=[root/"src/decision_model/core.py",root/"src/decision_model/objective.py",
           root/"src/decision_model/training_objective.py",root/"src/decision_model/train.py",
           root/"src/decision_model/decoder_model.py",Path(__file__).resolve(),
           root/"scripts/prepare_typed_decisions_pilot.py",source_protocol]
    source_hashes={path.relative_to(root).as_posix():digest(path.read_bytes()) for path in paths}
    train_ids=sorted(row["id"] for row in rows if row["split"]=="train")
    import random
    random.Random(protocol["seed"]).shuffle(train_ids)
    output.mkdir(parents=True)
    configs={}
    for arm,changes in protocol["arms"].items():
        config=dict(protocol["base_config"],seed=protocol["seed"],epoch_row_orders=[train_ids])
        config.update(changes)
        target=output/(arm+".json");write_json(target,config)
        configs[arm]=digest(target.read_bytes())
    frozen={
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_id":protocol["protocol_id"],"pilot_only":True,
        "question":"Which RLCD ingredients are implemented correctly and show enough directional signal for a multi-seed four-fold study?",
        "dataset_sha256":digest(data.read_bytes()),"rows":counts,
        "held_out_workflow":protocol["held_out_workflow"],"seed":protocol["seed"],
        "arms":protocol["arms"],"config_sha256":configs,"source_files":source_hashes,
        "matched":["same fresh MiniCPM trainable initialization","same 30/15/20 question IDs",
                   "same row and candidate permutation plan","same encoder forwards and optimizer budget",
                   "same CE+0.5 Brier base proper score","fixed final update; test unused for selection"],
        "contrasts":{
            "H_to_S":"replace argmax hard supervision with the full teacher distribution",
            "S_to_T":"add ordered RPS only to Score questions; retain exact gradients",
            "T_to_R":"replace exact typed proper-risk gradient with four Gaussian actions, zero-sum projection, independent leave-one-out score-function baseline and a unit soft-CE anchor"},
        "rlcd_boundary":"R is an independent public-principle implementation. It omits stochastic advantage standard-deviation normalization because prior stationary-point checks found that normalization can move the proper-risk optimum.",
        "advancement":"None. One seed, 20 test questions and tiny adapters are implementation/directional evidence only. Full advancement requires all folds and seeds 42/43/44.",
    }
    write_json(output/"protocol.json",frozen)
    write_json(output/"protocol-checksums.json",{
        path.name:digest(path.read_bytes()) for path in output.glob("*.json")})

    def verify():
        if digest(data.read_bytes())!=frozen["dataset_sha256"]:raise ValueError("Pilot data changed")
        for name,expected in source_hashes.items():
            if digest((root/name).read_bytes())!=expected:raise ValueError("Frozen pilot source changed: "+name)
        for name,expected in json.loads((output/"protocol-checksums.json").read_text()).items():
            if name!="protocol-checksums.json" and digest((output/name).read_bytes())!=expected:
                raise ValueError("Frozen pilot input changed: "+name)
        verify_local_decoder_base(base)

    reference=None
    for arm in protocol["arms"]:
        verify();write_json(output/"status.json",{"state":"training","arm":arm})
        print(json.dumps({"event":"training","arm":arm}),flush=True)
        command=[sys.executable,"-m","decision_model.cli","train","--data",str(data),
                 "--config",str(output/(arm+".json")),"--output",str(output/arm),
                 "--base-path",str(base),"--device",args.device]
        try:
            with (output/(arm+".log")).open("w") as log:
                subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError:
            write_json(output/"status.json",{"state":"failed","arm":arm});raise
        run=json.loads((output/arm/"run.json").read_text())
        controls={key:run[key] for key in CONTROL_KEYS}
        if reference is None:reference=controls
        elif controls!=reference:raise ValueError("Matched pilot controls differ")
    write_json(output/"matched-controls.json",reference)
    write_json(output/"status.json",{"state":"complete","arms":list(protocol["arms"])})
    print(json.dumps({"event":"complete","arms":list(protocol["arms"])}))


if __name__=="__main__":main()
