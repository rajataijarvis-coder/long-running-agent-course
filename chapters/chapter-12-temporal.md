# Chapter 12: Temporal Workflows for Long-Running Agents

## What We'll Cover
- What Temporal is and why it matters
- Workflows vs activities
- Deterministic orchestration
- Converting the local runner to a Temporal workflow

## What Is Temporal?
Temporal is a durable execution platform that journals every workflow step and replays from the journal after crashes.

## Workflow vs Activity
Workflows orchestrate. Activities execute side effects.

## Exercise
Install Temporal locally via Docker Compose and define a workflow with one activity.

## Key Takeaway
Temporal turns hope-the-process-stays-up into the-system-resumes-exactly-where-it-left-off.
