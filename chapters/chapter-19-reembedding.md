# Chapter 19: Re-embedding After Model Changes

## What We'll Cover
- Why embedding versions matter
- Never compare vectors across models
- Migration strategy for re-embedding
- Schema design for safe swaps

## The Embedding Version Problem
If you change embedding models, old vectors become meaningless. Store model and version with every vector.

## Exercise
Write a migration that adds a new vector column and flips validity atomically.

## Key Takeaway
Vectors are model-dependent. Version them.
