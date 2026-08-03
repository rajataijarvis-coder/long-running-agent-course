# Chapter 27: Docker Compose Stack

## What We'll Cover
- Services needed for production-like LRA
- Temporal + Postgres + Langfuse
- Starting the stack
- Connecting the worker

## Services
```yaml
services:
  temporal:     # :7233
  temporal-ui:  # :8080
  appdb:        # :5432
  langfuse:     # :3000
```

## Exercise
Bring up the stack and submit a mission.

## Key Takeaway
The durable spine separates demos from production agents.
