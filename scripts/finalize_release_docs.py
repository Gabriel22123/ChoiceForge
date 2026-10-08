#!/usr/bin/env python3
"""Replace pre-result status blocks with the frozen study's final verdict."""
from __future__ import annotations

import json
from pathlib import Path


START = "<!-- release-status:start -->"
END = "<!-- release-status:end -->"


def replace_block(text, body):
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError("Expected exactly one ordered release-status block")
    prefix, rest = text.split(START, 1)
    _, suffix = rest.split(END, 1)
    return prefix + START + "\n" + body.strip() + "\n" + END + suffix


def bodies(evidence, weights_sha256):
    eligible = evidence["eligible"]
    seen = evidence["seen"]["mean_accuracy"]
    blind = evidence["blind"]["mean_accuracy"]
    passed = sum(evidence["checks"].values())
    total = len(evidence["checks"])
    if eligible:
        cn = (f"**Release Candidate v1 已通过全部 {total} 项冻结门槛。** 唯一候选为 "
              f"`seed1701-expanded`；权重 SHA-256 为 `{weights_sha256}`。已见公共任务三种子"
              f"平均准确率从 {seen['control']:.2%} 提高到 {seen['expanded']:.2%}，新任务族"
              f"盲测从 {blind['control']:.2%} 提高到 {blind['expanded']:.2%}。复制种子没有"
              "替换预先指定的发布种子。完整门槛、概率损失和限制见"
              "[`最终报告`](docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md)。")
        en = (f"**Release Candidate v1 passes all {total} frozen gates.** The sole candidate is "
              f"`seed1701-expanded`, weights SHA-256 `{weights_sha256}`. Mean seen-task accuracy "
              f"moves from {seen['control']:.2%} to {seen['expanded']:.2%}; fresh task-family "
              f"accuracy moves from {blind['control']:.2%} to {blind['expanded']:.2%}. Replication "
              "seeds did not replace the predesignated release seed. See the "
              "[final report](docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md).")
        card = (f"- Final release checkpoint: `seed1701-expanded` (weights SHA-256 "
                f"`{weights_sha256}`).\n- Frozen advancement gates: {passed}/{total} passed. Mean seen-task "
                f"accuracy {seen['control']:.2%} → {seen['expanded']:.2%}; fresh task-family "
                f"accuracy {blind['control']:.2%} → {blind['expanded']:.2%}.\n- Full evidence and "
                "limitations: `docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md`.")
        readiness = (f"Release Candidate v1 已通过全部 {total} 项冻结门槛，唯一候选为 "
                     f"`seed1701-expanded`，权重 SHA-256 为 `{weights_sha256}`。最终包仍不"
                     "包含 2B 底座和盲测 canary；完整结论见 "
                     "[`RELEASE_CANDIDATE_RESULTS.zh-CN.md`](RELEASE_CANDIDATE_RESULTS.zh-CN.md)。")
    else:
        cn = (f"**Release Candidate v1 已完成，但只通过 {passed}/{total} 项冻结门槛，"
              "因此不产生发布 checkpoint。** 不允许从复制种子中事后替换候选。失败条件和"
              "完整指标见[`最终报告`](docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md)。")
        en = (f"**Release Candidate v1 completed but passes only {passed}/{total} frozen gates, so "
              "no release checkpoint is produced.** Replication seeds cannot be substituted after "
              "the result. See the [final report](docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md).")
        card = (f"- Final release checkpoint: none. Release Candidate v1 passed {passed}/{total} "
                "frozen gates and is ineligible.\n- Replication seeds cannot replace the "
                "predesignated seed. See `docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md`.")
        readiness = (f"Release Candidate v1 只通过 {passed}/{total} 项冻结门槛，不产生最终"
                     "候选，也不允许从复制种子中事后挑选。完整结论见 "
                     "[`RELEASE_CANDIDATE_RESULTS.zh-CN.md`](RELEASE_CANDIDATE_RESULTS.zh-CN.md)。")
    return {"README.zh-CN.md": cn, "README.md": en, "MODEL_CARD.md": card,
            "docs/RELEASE_READINESS.zh-CN.md": readiness}


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "docs/evidence/release-candidate-v1.json").read_text())
    locked = json.loads((root / "runs/release-candidate-v1/locked-endpoints.json").read_text())
    endpoint = evidence.get("release_endpoint") or locked["canonical_release_endpoint"]
    weights = locked["endpoints"][endpoint]["weights_sha256"]
    for name, body in bodies(evidence, weights).items():
        path = root / name
        path.write_text(replace_block(path.read_text(), body))
    print(json.dumps({"eligible": evidence["eligible"], "updated": 4,
                      "weights_sha256": weights}, sort_keys=True))


if __name__ == "__main__":
    main()
