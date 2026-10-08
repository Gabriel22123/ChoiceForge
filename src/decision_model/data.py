"""Pinned public-source adapters. Source examples are parsed, never executed."""
import ast
import itertools
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .core import canonical, digest, load_rows, summarize_rows, write_json

SCHEMA_RULE = (
    "The JSON instance must conform to the supplied complete JSON Schema, Draft 2020-12. "
    "The evidence lists every possible instance consistent with the observation. "
    "Judge satisfied if every candidate conforms, violated if every candidate fails, "
    "and insufficient if some conform and some fail."
)


def get_sources(lock_path, cache, offline=False):
    lock = json.loads(Path(lock_path).read_text())
    for source in lock["sources"]:
        if source["license"] not in ("MIT", "Apache-2.0", "CC-BY-3.0", "CC-BY-4.0"):
            raise ValueError("This public builder requires a reviewed permissive license")
        if not source["repository"].startswith("https://github.com/") or len(source["revision"]) != 40:
            raise ValueError("Expected a public GitHub source pinned to a commit")
        for entry in source["files"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source path")
            path = Path(cache)/source["id"]/relative
            if not path.exists():
                if offline:
                    raise FileNotFoundError(path)
                url = source["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
                url += f'/{source["revision"]}/{entry["path"]}'
                with urllib.request.urlopen(url, timeout=60) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024 or digest(raw) != entry["sha256"]:
                    raise ValueError("Source checksum mismatch: " + entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            if digest(path.read_bytes()) != entry["sha256"]:
                raise ValueError("Cached source was modified: " + entry["path"])
    return lock


def provenance(source, entry, **extra):
    return {"url": f'{source["repository"]}/blob/{source["revision"]}/{entry["path"]}',
            "revision": source["revision"], "file_sha256": entry["sha256"],
            "license": source["license"], **extra}


def unsupported(schema):
    if isinstance(schema, dict):
        if "$schema" in schema and schema["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            return True
        if any(k in schema for k in ("$ref", "$dynamicRef", "$recursiveRef", "$vocabulary", "format", "pattern", "patternProperties")):
            return True
        return any(unsupported(v) for v in schema.values())
    return isinstance(schema, list) and any(unsupported(v) for v in schema)


def schema_rows(source, cache, skipped):
    from jsonschema import Draft202012Validator
    for entry in source["files"]:
        if not entry["path"].endswith(".json"):
            continue
        for gi, group in enumerate(json.loads((Path(cache)/source["id"]/entry["path"]).read_text())):
            schema = group["schema"]
            if unsupported(schema):
                skipped["schema_reference_dialect_format_or_regex_excluded"] += 1
                continue
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema)
            tests = group["tests"]
            # Verify upstream labels independently; stop instead of silently accepting drift.
            for case in tests:
                if validator.is_valid(case["data"]) != case["valid"]:
                    raise ValueError(f'Oracle disagreement at {entry["path"]}:{gi}')
            gid = "schema:" + digest(schema)
            selections = [(i,) for i in range(len(tests))]
            # Bounded evidence with two possible worlds creates all three outcomes, not
            # a shortcut where every two-candidate input means 'insufficient'.
            by_label = defaultdict(list)
            for i, j in itertools.combinations(range(len(tests)), 2):
                if canonical(tests[i]["data"]) == canonical(tests[j]["data"]):
                    continue
                flags = {tests[i]["valid"], tests[j]["valid"]}
                label = "insufficient" if len(flags) == 2 else ("satisfied" if True in flags else "violated")
                by_label[label].append((i, j))
            for values in by_label.values():
                selections.extend(sorted(values, key=lambda p: digest([schema, p]))[:2])
            for indices in selections:
                flags = {tests[i]["valid"] for i in indices}
                label = "insufficient" if len(flags) == 2 else ("satisfied" if True in flags else "violated")
                evidence = {"schema": schema, "possible_instances": [tests[i]["data"] for i in indices]}
                request = {"rule": SCHEMA_RULE, "evidence": canonical(evidence)}
                if len(canonical(request)) > 3500:
                    skipped["schema_over_character_budget"] += 1
                    continue
                yield {"request": request, "label": label, "group_id": gid, "source": source["id"],
                       "family": Path(entry["path"]).stem,
                       "provenance": [provenance(source, entry, group_index=gi, test_indices=list(indices),
                                                label_method="upstream_boolean_and_jsonschema_4.25.1",
                                                transformation="bounded_possible_worlds" if len(indices)>1 else "single_instance") ]}


BANDIT_RULES = {
    "tls": "For this direct requests/httpx call, certificate verification must not be disabled. Judge only this call; do not infer the safety of the entire application.",
    "timeout": "This direct requests/httpx call must explicitly supply a finite, positive timeout. An omitted timeout or None violates this local policy, irrespective of library defaults.",
    "shell": "This direct subprocess call must not enable shell execution. For the listed subprocess APIs, an omitted shell argument defaults to false. Judge only this local property.",
    "yaml": "This yaml.load call must explicitly use SafeLoader or CSafeLoader. An omitted Loader or yaml.Loader violates this local policy. Judge only the supplied loader selection.",
}


def bandit_rows(source, cache, skipped):
    for entry in source["files"]:
        if not entry["path"].endswith(".py"):
            continue
        tree = ast.parse((Path(cache)/source["id"]/entry["path"]).read_text())
        imports = "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            family = None
            if name.startswith(("requests.", "httpx.")) and name.split(".")[-1] in {"get","post","put","patch","delete","head","options","request","stream","Client","AsyncClient"}:
                family = "tls" if "ssl-verify" in entry["path"] else "timeout"
            elif name in {"subprocess.Popen","subprocess.call","subprocess.run","subprocess.check_call","subprocess.check_output"}:
                family = "shell"
            elif name == "yaml.load":
                family = "yaml"
            if family is None:
                continue
            kws = {k.arg: k.value for k in node.keywords}
            keyword = {"tls": "verify", "timeout": "timeout", "shell": "shell", "yaml": "Loader"}[family]
            value = kws.get(keyword)
            if family == "yaml" and value is None and len(node.args) > 1:
                value = node.args[1]
            if any(k.arg is None for k in node.keywords):
                label = "insufficient"
            elif family == "yaml":
                if value is None or ast.unparse(value) == "yaml.Loader":
                    label = "violated"
                elif ast.unparse(value) in ("SafeLoader", "CSafeLoader", "yaml.SafeLoader", "yaml.CSafeLoader"):
                    label = "satisfied"
                else:
                    label = "insufficient"
            elif value is None:
                label = "violated" if family == "timeout" else "satisfied"
            else:
                try:
                    literal = ast.literal_eval(value)
                    if family == "tls":
                        good = literal is not False
                    elif family == "timeout":
                        good = type(literal) in (int, float) and 0 < literal < float("inf")
                    else:
                        good = not bool(literal)
                    label = "satisfied" if good else "violated"
                except (ValueError, TypeError):
                    label = "insufficient"
            # All calls for a (policy, API) stay together; true/false near-duplicates
            # cannot straddle splits. This is not a whole-repository safety label.
            yield {"request": {"rule": BANDIT_RULES[family], "evidence": imports + "\n" + ast.unparse(node)},
                   "label": label, "group_id": "bandit:"+family+":"+name,
                   "source": source["id"], "family": family,
                   "provenance": [provenance(source, entry, line=node.lineno,
                                            label_method="narrow_AST_policy_oracle_v1",
                                            transformation="imports_and_call_only_comments_removed")]}


def clinc_rows(source,cache,skipped):
    import random
    entry = next(e for e in source["files"] if e["path"]=="data/data_full.json")
    data = json.loads((Path(cache)/source["id"]/entry["path"]).read_text())
    intents = sorted({y for _,y in data["train"]})
    # Use each official split, without the optional Wikipedia augmentation.
    # Sample one bounded candidate set per utterance. This is candidate ranking,
    # not the original full-150-class benchmark; metrics must say so.
    seen = set()
    for original,split in (("test","test"),("val","validation"),("train","train")):
        for index,(utterance,label) in enumerate(data[original]):
            key = digest(utterance.strip().casefold())
            if key in seen:
                skipped["clinc_duplicate_utterance"] += 1; continue
            seen.add(key)
            rng = random.Random(int(key[:16],16))
            count = rng.choice((2,4,8))
            choices = rng.sample([x for x in intents if x != label],count-1)+[label]
            rng.shuffle(choices)
            yield {"request":{"task":"Select the intent that best describes this user's request from the supplied candidates.",
                              "context":utterance,
                              "choices":[{"id":c,"description":c.replace("_"," ")} for c in choices]},
                   "label":label,"group_id":"clinc:"+key,"source":source["id"],"family":"intent_routing",
                   "split":split,"provenance":[provenance(source,entry,original_split=original,index=index,
                        label_method="upstream_human_intent_label",transformation="2_4_8_candidate_ranking_with_sampled_distractors")]}


def emotion_rows(source,cache,skipped):
    import random
    entry = next(e for e in source["files"] if e["path"].endswith("test.tsv"))
    names = (Path(cache)/source["id"]/"goemotions/data/emotions.txt").read_text().splitlines()
    selected = ["anger","fear","gratitude","joy","neutral","sadness","surprise"]
    seen=set()
    for line,raw in enumerate((Path(cache)/source["id"]/entry["path"]).read_text().splitlines(),1):
        utterance,labels,comment_id = raw.split("\t")
        if "," in labels or names[int(labels)] not in selected:
            skipped["emotion_multilabel_or_outside_subset"] += 1; continue
        key = digest(utterance.strip().casefold())
        if key in seen:
            skipped["emotion_duplicate_utterance"] += 1; continue
        seen.add(key)
        choices=selected.copy(); random.Random(int(key[:16],16)).shuffle(choices)
        yield {"request":{"task":"Choose the emotion best expressed by the text. Neutral means no specific emotion is expressed.",
                          "context":utterance,"choices":[{"id":c,"description":c} for c in choices]},
               "label":names[int(labels)],"group_id":"emotion:"+key,"source":source["id"],
               "family":"emotion_recognition","split":"test","evaluation_regime":"held_out_task",
               "provenance":[provenance(source,entry,line=line,label_method="upstream_single_human_emotion_label",
                                        transformation="seven_emotion_single_label_subset_test_only")]}


def build(lock_path, cache, output, offline=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new dataset directory")
    lock = get_sources(lock_path, cache, offline)
    candidates, skipped = [], Counter()
    for source in lock["sources"]:
        adapter = {"json-schema":schema_rows,"bandit":bandit_rows,"clinc":clinc_rows,"goemotions":emotion_rows}[source["id"]]
        candidates.extend(adapter(source, cache, skipped))
    unique = {}
    for row in candidates:
        if "rule" in row["request"]:
            old=row["request"]
            row["request"]={"task":old["rule"],"context":old["evidence"],"choices":[
                {"id":"satisfied","description":"The supplied evidence establishes that the stated rule is satisfied."},
                {"id":"violated","description":"The supplied evidence establishes a violation of the stated rule."},
                {"id":"insufficient","description":"The supplied evidence is insufficient to determine whether the stated rule is satisfied."}]}
        row.setdefault("evaluation_regime","seen_task_new_examples")
        key = digest(row["request"])
        if key in unique:
            if row["label"] != unique[key]["label"]:
                raise ValueError("Conflicting duplicate labels")
            unique[key]["provenance"].extend(row["provenance"])
            skipped["duplicate_input"] += 1
            continue
        row["id"] = key
        bucket = int(digest(row["group_id"])[:8], 16) % 100
        row.setdefault("split","train" if bucket < 70 else "validation" if bucket < 85 else "test")
        unique[key] = row
    rows = sorted(unique.values(), key=lambda r: r["id"])
    output.mkdir(parents=True)
    (output/"sources.lock.json").write_bytes(Path(lock_path).read_bytes())
    (output/"licenses").mkdir()
    for source in lock["sources"]:
        license_path=Path(cache)/source["id"]/"LICENSE"
        if license_path.exists():
            (output/"licenses"/(source["id"]+"-LICENSE")).write_bytes(license_path.read_bytes())
    (output/"ATTRIBUTION.md").write_text(
        "# Derived public data\n\n"
        "JSON Schema Test Suite: Julian Berman and contributors (MIT).\n\n"
        "Bandit: PyCQA / OpenStack contributors (Apache-2.0).\n\n"
        "CLINC150: Stefan Larson et al., 2019 (CC-BY-3.0).\n\n"
        "GoEmotions: Dorottya Demszky et al., 2020 (CC-BY-4.0), "
        "https://creativecommons.org/licenses/by/4.0/ . The pinned Google Research "
        "root README assigns CC-BY-4.0 to datasets.\n\n"
        "Modifications: filtering, candidate ranking, code-call extraction and bounded-ambiguity "
        "examples. Per-row provenance records exact sources and transformations. "
        "Source URLs and commits: sources.lock.json. No upstream endorsement is implied.\n")
    path = output/"cases.jsonl"
    path.write_text("".join(canonical(r)+"\n" for r in rows))
    load_rows(path)
    manifest = {"dataset_sha256": digest(path.read_bytes()), "sources_lock_sha256": digest(Path(lock_path).read_bytes()),
                **summarize_rows(rows), "skipped": dict(skipped),
                "split_method":"schema/policy groups 70/15/15; CLINC official splits; GoEmotions whole task test-only",
                "scope":"public multi-task bootstrap with held-out emotion task; not a universal zero-shot capability claim"}
    write_json(output/"manifest.json", manifest)
    return manifest
