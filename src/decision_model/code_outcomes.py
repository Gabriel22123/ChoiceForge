"""Deterministic program mutations and isolated public-code outcome evaluation.

This module turns executable tests into an auditable reward table.  It does not
pretend that a test pass is a differentiable teacher probability: downstream
experiments may expose either the complete table or only the sampled action's
reward while keeping the underlying examples identical.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ALLOWED_IMPORT_ROOTS = frozenset({
    "bisect", "collections", "copy", "datetime", "decimal", "fractions",
    "functools", "heapq", "itertools", "math", "operator", "random", "re",
    "statistics", "string", "typing",
})
MBPP_REPOSITORY = "https://github.com/google-research/google-research"
MBPP_REVISION = "f46ca8374b4cddef97ca4208ad986049d74d296a"
MBPP_PATH = "mbpp/mbpp.jsonl"
FORBIDDEN_NAMES = frozenset({
    "breakpoint", "compile", "delattr", "eval", "exec", "exit", "getattr",
    "globals", "help", "input", "locals", "open", "quit", "setattr", "vars",
    "__import__",
})
FORBIDDEN_ATTRIBUTE_ROOTS = frozenset({
    "ctypes", "http", "multiprocessing", "os", "pathlib", "requests", "shutil",
    "signal", "socket", "subprocess", "sys", "urllib",
})


class UnsafeProgram(ValueError):
    """Raised when candidate or test code exceeds the local execution policy."""


def _attribute_root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def validate_program(source: str, *, allowed_import_roots=ALLOWED_IMPORT_ROOTS) -> ast.Module:
    """Parse source and reject obvious filesystem, process and network access.

    This is a narrow defence-in-depth policy for a local data-preparation tool,
    not a claim that Python AST filtering is a general security sandbox.
    The subprocess boundary and resource limits remain mandatory.
    """
    if not isinstance(source, str) or not source.strip():
        raise UnsafeProgram("program must be nonempty source text")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeProgram("program is not valid Python") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= set(allowed_import_roots):
                raise UnsafeProgram("program imports a non-whitelisted module")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in allowed_import_roots:
                raise UnsafeProgram("program imports a non-whitelisted module")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise UnsafeProgram("program uses a forbidden name")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or _attribute_root(node) in FORBIDDEN_ATTRIBUTE_ROOTS:
                raise UnsafeProgram("program uses a forbidden attribute")
    return tree


@dataclass(frozen=True)
class TestOutcome:
    passed: bool
    category: str
    detail: str = ""


def _resource_limiter(cpu_seconds: int, memory_bytes: int):
    def limit():
        import resource
        def set_limit(name, soft, hard):
            key = getattr(resource, name, None)
            if key is None:
                return
            try:
                _, existing_hard = resource.getrlimit(key)
                if existing_hard != resource.RLIM_INFINITY:
                    hard = min(hard, existing_hard)
                    soft = min(soft, hard)
                resource.setrlimit(key, (soft, hard))
            except (OSError, ValueError):
                # Some macOS limits are declared but cannot be lowered in a
                # subprocess pre-exec hook. Static filtering, isolation and the
                # wall-clock timeout still apply; Linux release workers enforce
                # the full set.
                pass
        set_limit("RLIMIT_CPU", cpu_seconds, cpu_seconds)
        if sys.platform != "darwin":
            set_limit("RLIMIT_AS", memory_bytes, memory_bytes)
            set_limit("RLIMIT_NPROC", 16, 16)
        set_limit("RLIMIT_FSIZE", 0, 0)
        set_limit("RLIMIT_NOFILE", 16, 16)
        os.umask(0o077)
    return limit


def run_test(candidate: str, test: str, *, setup: str = "", timeout_seconds: float = 2.0,
             memory_bytes: int = 512 * 1024 * 1024) -> TestOutcome:
    """Run one candidate/test pair in an isolated interpreter and temp directory."""
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    validate_program(candidate)
    validate_program(test)
    if setup.strip():
        validate_program(setup)
    payload = "\n\n".join(part for part in (setup, candidate, test) if part.strip())
    with tempfile.TemporaryDirectory(prefix="decision-outcome-") as directory:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "TZ": "UTC"}
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", payload],
                cwd=directory,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(timeout_seconds),
                check=False,
                preexec_fn=_resource_limiter(max(1, int(timeout_seconds)), memory_bytes),
            )
        except subprocess.TimeoutExpired:
            return TestOutcome(False, "timeout")
    if result.returncode == 0:
        return TestOutcome(True, "passed")
    stderr = result.stderr[-2000:]
    if "AssertionError" in stderr:
        return TestOutcome(False, "assertion")
    if result.returncode < 0:
        return TestOutcome(False, "resource_or_signal", str(result.returncode))
    last = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
    return TestOutcome(False, "runtime", last[:300])


def evaluate_candidate(candidate: str, tests: list[str], *, setup: str = "",
                       timeout_seconds: float = 2.0) -> list[TestOutcome]:
    if not isinstance(tests, list) or not tests:
        raise ValueError("tests must be a nonempty list")
    return [run_test(candidate, test, setup=setup, timeout_seconds=timeout_seconds) for test in tests]


_COMPARE_SWAPS = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv,
}


class _MutationCollector(ast.NodeVisitor):
    def __init__(self):
        self.specs: list[tuple[str, int, int, int, object]] = []

    def visit_Compare(self, node):
        for index, operator in enumerate(node.ops):
            replacement = _COMPARE_SWAPS.get(type(operator))
            if replacement:
                self.specs.append(("compare", node.lineno, node.col_offset, index, replacement))
        self.generic_visit(node)

    def visit_BinOp(self, node):
        replacement = _BINOP_SWAPS.get(type(node.op))
        if replacement:
            self.specs.append(("binop", node.lineno, node.col_offset, 0, replacement))
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        replacement = ast.Or if isinstance(node.op, ast.And) else ast.And
        self.specs.append(("boolop", node.lineno, node.col_offset, 0, replacement))
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            self.specs.append(("drop_not", node.lineno, node.col_offset, 0, None))
        self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) is int and abs(node.value) <= 10000:
            self.specs.append(("integer", node.lineno, node.col_offset, 0, node.value + 1))
            self.specs.append(("integer", node.lineno, node.col_offset, 0, node.value - 1))


class _ApplyMutation(ast.NodeTransformer):
    def __init__(self, spec):
        self.spec = spec

    def _matches(self, kind, node):
        return self.spec[:3] == (kind, node.lineno, node.col_offset)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if self._matches("compare", node):
            node.ops[self.spec[3]] = self.spec[4]()
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if self._matches("binop", node):
            node.op = self.spec[4]()
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if self._matches("boolop", node):
            node.op = self.spec[4]()
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if self._matches("drop_not", node):
            return ast.copy_location(node.operand, node)
        return node

    def visit_Constant(self, node):
        if self._matches("integer", node):
            return ast.copy_location(ast.Constant(self.spec[4]), node)
        return node


def deterministic_mutants(source: str, *, limit: int = 31) -> list[dict]:
    """Return unique one-edit AST mutants in a stable source-location order."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    tree = validate_program(source)
    collector = _MutationCollector()
    collector.visit(tree)
    specs = sorted(collector.specs, key=lambda value: (value[1], value[2], value[0], value[3], str(value[4])))
    seen = {ast.unparse(tree)}
    mutants = []
    for spec in specs:
        changed = _ApplyMutation(spec).visit(copy.deepcopy(tree))
        ast.fix_missing_locations(changed)
        text = ast.unparse(changed)
        if text in seen:
            continue
        validate_program(text)
        seen.add(text)
        mutants.append({"operator": spec[0], "line": spec[1], "column": spec[2], "code": text})
        if len(mutants) == limit:
            break
    return mutants


def load_mbpp(path: str | Path) -> list[dict]:
    """Load either the upstream JSON array or JSONL representation."""
    raw = Path(path).read_text()
    stripped = raw.lstrip()
    rows = json.loads(raw) if stripped.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    required = {"task_id", "text", "code", "test_list"}
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not required <= set(row) for row in rows):
        raise ValueError("unexpected MBPP data format")
    if len({int(row["task_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate MBPP task_id")
    return rows


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def mbpp_split(task_id: int) -> str:
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError("MBPP task_id is outside the official 1..974 range")


def prepare_mbpp_outcomes(source_path: str | Path, *, max_candidates: int = 8,
                          timeout_seconds: float = 2.0,
                          allowed_splits=("train", "validation")) -> tuple[list[dict], dict]:
    """Build candidate reward tables from official MBPP code and tests.

    Candidate selection and presentation order depend only on source code and
    task ID, never on execution rewards.  The reference and every mutant are
    normalized through ``ast.unparse`` so formatting cannot reveal the source.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or not 2 <= max_candidates <= 32:
        raise ValueError("max_candidates must be in 2..32")
    allowed_splits = tuple(allowed_splits)
    if not allowed_splits or not set(allowed_splits) <= {"train", "validation", "test"}:
        raise ValueError("allowed_splits must use official train/validation/test names")
    source_sha256 = _sha256(Path(source_path).read_bytes())
    rows: list[dict] = []
    skipped: Counter[str] = Counter()
    for source in sorted(load_mbpp(source_path), key=lambda item: int(item["task_id"])):
        task_id = int(source["task_id"])
        split = mbpp_split(task_id)
        if split not in allowed_splits:
            skipped["split_excluded"] += 1
            continue
        try:
            reference = ast.unparse(validate_program(source["code"]))
            setup = source.get("test_setup_code", "") or ""
            tests = source["test_list"]
            if not isinstance(tests, list) or not tests or any(not isinstance(test, str) for test in tests):
                raise UnsafeProgram("invalid test list")
            # Validate before any subprocess starts, including setup and every test.
            if setup.strip():
                validate_program(setup)
            for test in tests:
                validate_program(test)
            mutations = deterministic_mutants(reference, limit=max_candidates - 1)
        except UnsafeProgram:
            skipped["static_policy_or_parse"] += 1
            continue
        if not mutations:
            skipped["no_supported_mutation"] += 1
            continue
        candidates = [{"operator": "reference", "line": None, "column": None, "code": reference}] + mutations
        outcomes = []
        for candidate in candidates:
            result = evaluate_candidate(candidate["code"], tests, setup=setup,
                                        timeout_seconds=timeout_seconds)
            outcomes.append(result)
        if not all(item.passed for item in outcomes[0]):
            skipped["reference_does_not_pass"] += 1
            continue
        rewards = [sum(item.passed for item in result) / len(result) for result in outcomes]
        if len(set(rewards)) < 2:
            skipped["no_observed_reward_difference"] += 1
            continue
        records = []
        for candidate, result, reward in zip(candidates, outcomes, rewards):
            candidate_id = "program-" + _sha256(candidate["code"])[:16]
            records.append({
                "id": candidate_id,
                "description": candidate["code"],
                "mutation": {key: candidate[key] for key in ("operator", "line", "column")},
                "outcome": {
                    "passed": sum(item.passed for item in result),
                    "total": len(result),
                    "reward": reward,
                    "categories": [item.category for item in result],
                },
            })
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("candidate hash collision")
        random.Random(int(_sha256(f"mbpp:{task_id}:candidate-order")[:16], 16)).shuffle(records)
        best_index = max(range(len(records)), key=lambda index: records[index]["outcome"]["reward"])
        choices = [{"id": record["id"], "description": record["description"]} for record in records]
        candidate_outcomes = {record["id"]: record["outcome"] for record in records}
        mutation_provenance = {record["id"]: record["mutation"] for record in records}
        row_id = _sha256(json.dumps({"task_id": task_id, "choices": choices}, sort_keys=True))
        rows.append({
            "id": row_id,
            "group_id": f"mbpp:{task_id}",
            "source": "mbpp",
            "family": "executable_program_selection",
            "split": split,
            "evaluation_regime": "seen_task_new_examples",
            "request": {
                "task": "Choose the Python program most likely to satisfy the problem specification.",
                "context": source["text"].strip(),
                "choices": choices,
            },
            "label": records[best_index]["id"],
            "candidate_outcomes": candidate_outcomes,
            "mutation_provenance": mutation_provenance,
            "provenance": [{
                "dataset": "Mostly Basic Python Problems",
                "task_id": task_id,
                "url": f"{MBPP_REPOSITORY}/blob/{MBPP_REVISION}/{MBPP_PATH}",
                "revision": MBPP_REVISION,
                "file_sha256": source_sha256,
                "license": "CC-BY-4.0",
                "label_method": "isolated_execution_of_upstream_tests",
                "transformation": "reference_plus_deterministic_single_AST_mutants",
            }],
        })
    manifest = {
        "format_version": 1,
        "source_sha256": source_sha256,
        "allowed_splits": list(allowed_splits),
        "max_candidates": max_candidates,
        "timeout_seconds": timeout_seconds,
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "candidate_count_distribution": dict(sorted(Counter(
            len(row["request"]["choices"]) for row in rows).items())),
        "skipped": dict(sorted(skipped.items())),
        "selection_uses_outcomes": False,
        "presentation_order_uses_outcomes": False,
    }
    return rows, manifest
