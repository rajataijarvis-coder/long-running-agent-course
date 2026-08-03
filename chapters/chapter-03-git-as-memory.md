# Chapter 03: Externalizing Truth — Git as Memory

## What We'll Cover
- Why the context window is a cache, not a database
- The mission anchor: four files that hold truth
- How git history becomes episodic memory
- Reconstructing state after reboot

## The Context Window Is a Cache
LLMs have limited context. The real state lives outside the model.

## The Mission Anchor
- progress.md
- checklist.json
- decisions.ndjson
- events.ndjson

## Git as Immutable History
Each checkpoint is a git commit. The commit graph is your agent's memory.

## Exercise
Create a directory with the four anchor files, initialize git, and simulate three cycles by updating files and committing.

## Key Takeaway
If you can't reboot your agent and resume in under a minute, your state is too fragile.
