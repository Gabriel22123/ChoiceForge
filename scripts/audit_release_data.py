#!/usr/bin/env python3
"""Audit the exact release datasets for public provenance and redistribution."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from decision_model.core import canonical, digest, load_rows, write_json


EXPECTED_LICENSES = {
    "bandit": {"Apache-2.0"},
    "clinc": {"CC-BY-3.0"},
    "goemotions": {"CC-BY-4.0"},
    "json-schema": {"MIT"},
    "paws-wiki": {"PAWS dataset permission: freely usable for any purpose; attribution appreciated"},
    "snli": {"CC-BY-SA-4.0"},
    "typed-decisions": {"Apache-2.0"},
}
ALLOWED_HOSTS = {"github.com", "huggingface.co", "nlp.stanford.edu"}
NOTICE_FILES = {
    "bandit": "third_party/bandit-LICENSE",
    "clinc": "third_party/clinc-LICENSE",
    "goemotions": "third_party/google-research-DATA-LICENSE-NOTICE",
    "json-schema": "third_party/json-schema-LICENSE",
    "paws-wiki": "third_party/paws-DATA-LICENSE",
    "snli": "THIRD_PARTY.md",
    "typed-decisions": "THIRD_PARTY.md",
}
LOCAL_PATH_PATTERN = re.compile(r"(?:^|[\\\"'])/(?:Users|Volumes|home)/[^/\\\"']+", re.I)


def audit(root):
    data = root / "data/release-candidate-v1"
    manifest = json.loads((data / "manifest.json").read_text())
    arms, reports = {}, {}
    for arm in ("control", "expanded"):
        path = data / f"{arm}.jsonl"
        if digest(path.read_bytes()) != manifest[arm]["sha256"]:
            raise ValueError("Release data hash differs: " + arm)
        rows = load_rows(path)
        arms[arm] = rows
        sources, licenses, hosts = Counter(), Counter(), Counter()
        provenance_items = 0
        for row in rows:
            source = row["source"]
            if source not in EXPECTED_LICENSES:
                raise ValueError("Unexpected release source: " + source)
            if source.startswith("bigbench"):
                raise ValueError("Blind-suite source entered release data")
            serialized = canonical(row).lower()
            if LOCAL_PATH_PATTERN.search(serialized):
                raise ValueError("Local filesystem path entered release data")
            sources[source] += 1
            for item in row["provenance"]:
                provenance_items += 1
                license_name = item.get("license")
                if license_name not in EXPECTED_LICENSES[source]:
                    raise ValueError("Unexpected license for " + source)
                url = item.get("url")
                revision = item.get("revision")
                parsed = urlparse(url) if isinstance(url, str) else None
                if (parsed is None or parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or
                        not isinstance(revision, str) or not revision.strip()):
                    raise ValueError("Incomplete public provenance for " + source)
                licenses[(source, license_name)] += 1
                hosts[parsed.hostname] += 1
        reports[arm] = {
            "rows": len(rows), "sources": dict(sorted(sources.items())),
            "provenance_items": provenance_items,
            "licenses": {f"{source}:{license_name}": count
                         for (source, license_name), count in sorted(licenses.items())},
            "public_url_hosts": dict(sorted(hosts.items())),
            "sha256": digest(path.read_bytes()),
        }
    if [row["id"] for row in arms["control"] if row["split"] != "train"] != [
            row["id"] for row in arms["expanded"] if row["split"] != "train"]:
        raise ValueError("Release evaluation rows differ between arms")
    if (reports["control"]["rows"] != 3008 or reports["expanded"]["rows"] != 5056 or
            manifest.get("selection_is_target_blind") is not True or
            set(manifest.get("typed_selection_key_excludes", [])) !=
            {"gold", "probabilities", "label", "rationale"}):
        raise ValueError("Release data protocol differs")
    missing_notices = [path for path in NOTICE_FILES.values() if not (root / path).is_file()]
    if missing_notices:
        raise ValueError("Missing attribution notice: " + ", ".join(missing_notices))
    return {
        "format_version": 1, "study": "release-candidate-v1",
        "passed": True, "arms": reports,
        "allowed_sources": sorted(EXPECTED_LICENSES),
        "allowed_public_hosts": sorted(ALLOWED_HOSTS),
        "notice_files": dict(sorted(NOTICE_FILES.items())),
        "common_evaluation_rows_identical": True,
        "target_blind_typed_selection": True,
        "blind_suite_rows_included": False,
        "internal_or_local_markers_found": 0,
        "combined_dataset_relicensed_as_apache": False,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = audit(root)
    output = root / "docs/evidence/release-data-redistribution-v1.json"
    write_json(output, evidence)
    lines = [
        "# 发布训练数据再分发审计", "",
        "**结论：通过。** 两个固定训练文件只包含七个登记的公开来源，逐行保留许可证、",
        "固定版本和外部 HTTPS 来源；未发现公司内网标识、本机路径或盲测任务行。", "",
        "| 数据 | 行数 | 来源数 | provenance 条目 | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ("control", "expanded"):
        value = evidence["arms"][arm]
        lines.append(f"| {arm} | {value['rows']} | {len(value['sources'])} | "
                     f"{value['provenance_items']} | `{value['sha256']}` |")
    lines += [
        "", "## 边界", "",
        "- 组合数据集不重新声明为 Apache-2.0；每个来源继续服从其上游许可证。",
        "- SNLI 的 CC-BY-SA、CLINC/GoEmotions 的署名要求和 PAWS 数据许可均随包说明。",
        "- 新任务族盲测案例、标签、logits 和逐案例预测不进入源码或发布包。",
        "- 该审计证明来源和本项目文件边界，不替代各上游作者的法律解释。",
        "", "机器可读证据：`docs/evidence/release-data-redistribution-v1.json`",
    ]
    (root / "docs/RELEASE_DATA_AUDIT.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": True, "evidence": str(output),
                      "rows": {arm: evidence["arms"][arm]["rows"] for arm in evidence["arms"]}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
