# Chapter 21: Loop Detection

## What We'll Cover
- Detecting repeated failures
- Hashing tool-call sequences
- Escaping oscillation with reflection
- When to ask a human

## Loop Detection
Hash the last N actions. If the hash repeats, you are in a loop.

## Exercise
Build a LoopDetector that raises an alert after 3 repeats.

## Key Takeaway
Detect loops early. Escaping them requires changing strategy, not retrying.
