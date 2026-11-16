"""Checks that keep the README and Raft's files honest."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import agent
import main

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# SHA-256 of Raft's files as provided (the same in the submitted version).
RAFT_FILES = {
    "Instructions.md": "8ac84080be1a18fa4ad77ffa11687e0b24531d897b1c24eea2e27b7c49084f82",
    "dummy_customer_api.py": "90413fa84ed527576f6240455a602383ec664cb6b0ab44d00b5d7fd661c5b128",
}


@pytest.mark.parametrize(("name", "sha256"), RAFT_FILES.items())
def test_rafts_files_are_unchanged(name, sha256):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha256


def test_readme_diagram_matches_the_graph():
    [diagram] = re.findall(r"```mermaid\n(.*?)```", README, re.DOTALL)
    names = {"request": "__start__", "result": "__end__"}
    edges = {
        (names.get(source, source), names.get(target, target))
        for source, target in re.findall(
            r"^\s*(\w+)(?:\[[^\]]*\]|\(\[[^\]]*\]\))?\s*-\.?->(?:\|[^|]*\|)?\s*(\w+)",
            diagram,
            re.MULTILINE,
        )
    }
    graph = agent.build_graph().get_graph()
    assert edges == {(edge.source, edge.target) for edge in graph.edges}


def readme_examples():
    """(arguments, transcript) of each `$ python main.py ...` example in the README."""
    examples = []
    for block in re.findall(r"```console\n(.*?)```", README, re.DOTALL):
        command, _, transcript = block.partition("\n")
        assert command.startswith("$ python main.py ")
        examples.append((shlex.split(command)[3:], transcript))
    return examples


@pytest.mark.parametrize(("args", "transcript"), readme_examples())
def test_readme_examples_are_current(args, transcript):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER_", "ORDERS_"))}
    result = subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == main.EXIT_OK, result.stderr
    # In a terminal the log (standard error) comes first and the JSON (standard output) last.
    assert result.stderr + result.stdout == transcript
