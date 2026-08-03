"""Deterministic verification based on exit codes."""

import subprocess
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    stdout: str = ""
    stderr: str = ""


@dataclass
class VerificationResult:
    checks: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


class DeterministicVerifier:
    """Runs commands and decides pass/fail from exit codes, not model opinions."""

    def __init__(self, checks: list[dict[str, str | int]]):
        self.checks = checks

    def verify(self, workdir: str | None = None) -> VerificationResult:
        results: list[CheckResult] = []
        for check in self.checks:
            name = str(check["name"])
            command = str(check["command"])
            proc = subprocess.run(command, shell=True, cwd=workdir, capture_output=True, text=True)
            results.append(
                CheckResult(
                    name=name,
                    passed=proc.returncode == 0,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
            )
        return VerificationResult(checks=results)

    @classmethod
    def from_workdir(cls, workdir: str | None = None) -> "DeterministicVerifier":
        """Build a verifier for a specific workdir using the default checks."""
        return cls(build_default_checks())


def build_default_checks() -> list[dict[str, str | int]]:
    return [
        {"name": "pytest", "command": "python -m pytest"},
        {"name": "ruff", "command": "ruff check ."},
        {"name": "mypy", "command": "mypy src"},
    ]
