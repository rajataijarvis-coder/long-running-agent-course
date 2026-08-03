# Chapter 04: The Gather → Act → Verify → Checkpoint Cycle

## What We'll Cover
- The four phases of one agent cycle
- How each phase fails and recovers
- Writing the loop in Python
- Why verification is the only merge gate

## One Cycle, Four Phases
GATHER → ACT → VERIFY → CHECKPOINT

## Failure Modes Per Phase
| Phase | Can fail by... | Recovery |
|-------|----------------|----------|
| Gather | Missing context | Re-read anchor |
| Act | Bad tool call | Retry with reflection |
| Verify | Test fails | Record attempt, try again |
| Checkpoint | Git conflict | Resolve or branch |

## Exercise
Implement a minimal loop that reads a checklist, picks an item, acts, verifies, and checkpoints.

## Key Takeaway
A cycle is only complete when verification passes and the checkpoint is durable.
