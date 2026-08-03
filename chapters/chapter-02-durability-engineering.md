# Chapter 02: Durability as an Engineering Property

## What We'll Cover
- What durable means for software agents
- Designing for interruption from day one
- Idempotency, retries, and exactly-once semantics
- Why durable sleep is a superpower

## Durable ≠ Reliable
A reliable system works correctly under normal conditions. A durable system keeps working correctly even when the process crashes, the network is flaky, the model API is rate-limited, the machine reboots, or a human kills the job.

## Design for Interruption
Every cycle of your agent should assume it might be interrupted. Write the plan to disk, log the intent, do the work, verify the result, and record the outcome.

## Idempotency Keys
Every side effect gets an idempotency key so retries are safe.

## Durable Sleep
A durable sleep is scheduled by the orchestrator (e.g., Temporal) and parks the workflow at zero cost.

## Exercise
Take a small script and add a log line before every side effect, an idempotency check, and a way to resume from the last completed step.

## Key Takeaway
Durability is not a feature you bolt on. It's a property you design into every step.
