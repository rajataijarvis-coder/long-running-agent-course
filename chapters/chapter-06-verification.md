# Chapter 06: Deterministic Verification

## What We'll Cover
- Why you cannot trust the model to say done
- Deterministic vs stochastic verification
- Exit-code-based gates
- Building a verifier that runs tests/lint/typecheck

## Don't Trust, Verify
Verification must be deterministic. The model's opinion is not evidence.

## Exit Codes as Ground Truth
A task is only done when tests, lint, typecheck, and build pass.

## Exercise
Write a verifier that runs pytest and ruff and returns a JSON result.

## Key Takeaway
A checklist item is only done when exit codes say it's done.
