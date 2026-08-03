# Chapter 07: Building Your First Local Mission

## What We'll Cover
- Running a mission without Temporal
- The local runner
- Inspecting git history and progress
- Cost tracking at $0

## Local Runner
Before adding durable execution, run missions locally.

## Running It
```bash
uv run lra mission --task "Create hello.py and a test for it" --workdir .lra/workspaces/demo
```

## Cost Tracking
Even at $0, track tokens, time, and model invocations.

## Exercise
Run a local mission to create a simple calculator with tests.

## Key Takeaway
Start local. Prove the loop works before adding durability infrastructure.
