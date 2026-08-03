# Chapter 05: Tool Dispatching and Sandboxing

## What We'll Cover
- Why agents need tools
- Allow-list vs block-list dispatch
- Sandboxing: local → Docker → cloud
- Argument validation and egress policies

## Tools Are the Agent's Hands
Tools let an LLM read files, run commands, search the web, commit code, and query databases.

## Allow-List Dispatch
Never let the model run arbitrary commands. Use an allow-list of approved tools.

## Sandboxing
| Level | Isolation | Cost |
|-------|-----------|------|
| Local process | Low | $0 |
| Docker | Medium | Low |
| Cloud | High | Higher |

## Exercise
Build an AllowListDispatcher with read_file, write_file, and run_command tools.

## Key Takeaway
Default-deny tool dispatch is the minimum security bar for any autonomous agent.
