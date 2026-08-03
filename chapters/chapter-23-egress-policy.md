# Chapter 23: Egress Policies

## What We'll Cover
- Default-deny network egress
- Allow-listing domains
- Validating tool arguments
- Sandboxing untrusted code

## Default-Deny Egress
The agent cannot call arbitrary URLs. Start with nothing allowed.

## Exercise
Configure an egress policy for pypi.org, api.github.com, and your model provider.

## Key Takeaway
Default-deny is the only sane security posture.
