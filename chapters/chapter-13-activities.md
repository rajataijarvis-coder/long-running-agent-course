# Chapter 13: Activities, Retries, and Replay-from-Cache

## What We'll Cover
- Writing idempotent activities
- Retry policies and backoff
- How replay-from-cache works
- Activity idempotency keys

## Activities Are Journaled
Every activity result is recorded. On replay, the same result is returned without re-executing.

## Retry Policy
Use exponential backoff with maximum attempts.

## Exercise
Write an activity that runs tests. Kill the worker mid-run, restart it, and observe replay.

## Key Takeaway
Replay-from-cache is what makes long-running missions cheap and reliable.
