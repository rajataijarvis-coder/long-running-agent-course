# Chapter 14: Durable Sleep and Continue-As-New

## What We'll Cover
- Why normal sleep doesn't scale
- Durable timers in Temporal
- Continue-As-New for unbounded missions
- Bounding history size

## Normal Sleep Wastes Money
A normal sleep holds a process and loses state on crash.

## Durable Sleep
`await workflow.sleep(86400)` parks the workflow at $0 cost.

## Continue-As-New
Start a new workflow with the same state when history grows too large.

## Exercise
Build a workflow that sleeps and then continues-as-new.

## Key Takeaway
Durable sleep and Continue-As-New let missions run for months, not minutes.
