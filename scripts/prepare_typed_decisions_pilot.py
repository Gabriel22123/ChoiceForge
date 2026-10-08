"""Select whole Typed Decisions cases for the fixed RLCD integration pilot."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json


def select(protocol):
    source=Path(protocol["source_fold"])
    rows=load_rows(source)
    grouped=collections.defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    if any(len(group)!=5 for group in grouped.values()):
        raise ValueError("Pilot expects five questions in every upstream case")
    selected=[];selected_groups={}
    for split in ("train","validation","test"):
        families=sorted({row["family"] for row in rows if row["split"]==split})
        count=protocol["cases_per_family"][split]
        selected_groups[split]={}
        for family in families:
            choices=[group_id for group_id,group in grouped.items()
                     if group[0]["split"]==split and group[0]["family"]==family]
            choices.sort(key=lambda group_id:hashlib.sha256(
                f"{protocol['selection_seed']}:{split}:{family}:{group_id}".encode()).hexdigest())
            if len(choices)<count:
                raise ValueError("Not enough case groups for pilot")
            picked=choices[:count]
            selected_groups[split][family]=picked
            for group_id in picked:selected.extend(grouped[group_id])
    return sorted(selected,key=lambda row:row["id"]),selected_groups,digest(source.read_bytes())


def prepare(protocol_path,output):
    protocol_path=Path(protocol_path);output=Path(output)
    if output.exists():raise ValueError("Use a new pilot data directory")
    protocol=json.loads(protocol_path.read_text())
    rows,groups,source_sha=select(protocol)
    output.mkdir(parents=True)
    target=output/"cases.jsonl"
    target.write_text("".join(canonical(row)+"\n" for row in rows))
    verified=load_rows(target)
    heldout=protocol["held_out_workflow"]
    if any(row["family"]==heldout and row["split"]!="test" for row in verified):
        raise ValueError("Held-out workflow leaked into pilot fitting")
    manifest={
        "protocol_id":protocol["protocol_id"],"protocol_sha256":digest(protocol_path.read_bytes()),
        "source":protocol["source_fold"],"source_sha256":source_sha,
        "dataset_sha256":digest(target.read_bytes()),"selected_groups":groups,
        "rows":summarize_rows(verified),
        "selection":"whole cases by ID-only SHA-256; labels/probabilities excluded from selection",
        "purpose":"implementation and directional pilot only; cannot advance a method",
        "script_sha256":digest(Path(__file__).read_bytes())}
    write_json(output/"manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",default="configs/typed-decisions-rlcd-pilot-v1.json")
    parser.add_argument("--output",default="data/typed-decisions-rlcd-pilot-v1")
    args=parser.parse_args();print(json.dumps(prepare(args.protocol,args.output),indent=2))


if __name__=="__main__":main()
