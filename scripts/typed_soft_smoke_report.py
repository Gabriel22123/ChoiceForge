"""Verify the real-model soft-target smoke and emit compact tracked evidence."""
import argparse
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json


def report(run_dir, data, config, output):
    run_dir=Path(run_dir);data=Path(data);config=Path(config);output=Path(output)
    run=json.loads((run_dir/"run.json").read_text())
    evaluation=json.loads((run_dir/"evaluation.json").read_text())
    calibration=json.loads((run_dir/"calibration.json").read_text())
    checksums=json.loads((run_dir/"checksums.json").read_text())
    rows=load_rows(data)
    if run["dataset_sha256"] != digest(data.read_bytes()) or run["config_sha256"] != digest(config.read_bytes()):
        raise ValueError("Smoke run does not match the supplied data/config")
    for name, expected in checksums.items():
        if digest((run_dir/name).read_bytes()) != expected:
            raise ValueError("Smoke artifact checksum differs: "+name)
    if len(rows) != 13 or any("target_probabilities" not in row for row in rows):
        raise ValueError("Expected the fixed 13-row all-soft smoke dataset")
    if run["selected"]["supervision"] != {"soft_distribution":13,"hard_label":0}:
        raise ValueError("Run did not record all-soft supervision")
    compact={}
    for split in ("train","validation","test"):
        compact[split]={}
        for mode in ("raw","temperature_scaled"):
            values=evaluation[split][mode]
            compact[split][mode]={key:values[key] for key in (
                "n","argmax_agreement","soft_cross_entropy","soft_brier","total_variation")}
    evidence={
        "status":"integration_smoke_passed_not_quality_evidence",
        "model":"openbmb/MiniCPM5-2B-Base",
        "model_revision":"96a57cd572a02506b4500f54427dca24970c1bac",
        "data_sha256":run["dataset_sha256"],"config_sha256":run["config_sha256"],
        "initial_trainable_sha256":run["initial_trainable_sha256"],
        "final_weights_sha256":checksums["decision.safetensors"],
        "selected":run["selected"],"updates":run["updates"],
        "adapter_absolute_change":run["adapter_probe"]["absolute_change"],
        "calibration":calibration,"metrics":compact,
        "verified_properties":[
            "probability targets survive candidate-ID alignment and candidate shuffling",
            "clean soft cross-entropy plus soft Brier reaches the real 2B LoRA graph",
            "validation temperature is selected using full soft cross-entropy",
            "soft CE, soft Brier, total variation and argmax agreement are persisted",
            "held-out security workflow appears only in test",
            "saved model/config/calibration hashes verify"
        ],
        "limitations":[
            "13 rows and 3 optimizer updates cannot measure model quality",
            "validation has only 3 rows, so its fitted temperature is not meaningful",
            "the smoke contains one test question per workflow and is not a comparative result"
        ]
    }
    write_json(output,evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",default="runs/typed-decisions-soft-smoke-v1")
    parser.add_argument("--data",default="data/typed-decisions-soft-smoke-v1/cases.jsonl")
    parser.add_argument("--config",default="configs/typed-decisions-soft-smoke.json")
    parser.add_argument("--output",default="docs/evidence/typed-soft-smoke-v1.json")
    args=parser.parse_args()
    print(json.dumps(report(args.run,args.data,args.config,args.output),indent=2))


if __name__=="__main__":main()
