# Chapter 08: Asymmetric Multi-Agent Design

## What We'll Cover
- Why multi-agent is a cost, not a feature
- The asymmetric organization
- When to fan out (reads and reviews)
- When to stay single-threaded (writes)

## The Cost of Multi-Agent
More agents means more tokens, more coordination complexity, and more inconsistency.

## The Asymmetric Org
One Lead Engineer writes all coupled code. Researchers fan out for parallel reads. Reviewer verifies independently.

## Exercise
List tasks where parallel reads help and tasks where single-threaded writes are better.

## Key Takeaway
Coupled writes must be single-threaded. Only reads and reviews parallelize safely.
