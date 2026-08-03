# Long-Running Agent Systems

A step-by-step, build-along course based on Fareed Khan's LRA project and article:
**"Building a Week-Long Running Agentic System"**.

**Course URL:** https://github.com/rajataijarvis-coder/long-running-agent-course  
**Original work:** https://github.com/FareedKhan-dev/long-running-agent

---

## Course Philosophy

Most AI agents die when:
- The process restarts
- Context window fills up  
- One API call fails

**This course teaches you to build agents that run for days or weeks.**

The model isn't smarter — the *system* around it is.

---

## What You'll Build

A durable, self-improving agent organization that:
1. Plans multi-step software missions
2. Executes in short bursts with tools
3. Verifies with **real tests** (never trust the model)
4. Checkpoints to git + structured logs
5. Resumes exactly after crashes
6. Learns from failures across runs
7. Runs at **$0** using Ollama or free-tier cloud models

---

## Table of Contents

### Part 1: Foundations (Why durability matters)
1. [Chapter 01: The Problem with ChatGPT-style Agents](chapters/chapter-01-the-problem.md)
2. [Chapter 02: Durability as an Engineering Property](chapters/chapter-02-durability-engineering.md)
3. [Chapter 03: Externalizing Truth — Git as Memory](chapters/chapter-03-git-as-memory.md)

### Part 2: The Inner Loop (One cycle of work)
4. [Chapter 04: The Gather → Act → Verify → Checkpoint Cycle](chapters/chapter-04-agent-loop.md)
5. [Chapter 05: Tool Dispatching and Sandboxing](chapters/chapter-05-tools-and-sandbox.md)
6. [Chapter 06: Deterministic Verification — Exit Codes as Ground Truth](chapters/chapter-06-verification.md)
7. [Chapter 07: Building Your First Local Mission](chapters/chapter-07-first-mission.md)

### Part 3: The Organization (Multiple agents)
8. [Chapter 08: Asymmetric Multi-Agent Design](chapters/chapter-08-asymmetric-org.md)
9. [Chapter 09: Research Fan-out (Parallel Reading)](chapters/chapter-09-research-fanout.md)
10. [Chapter 10: The Lead Engineer (Single Writer)](chapters/chapter-10-lead-engineer.md)
11. [Chapter 11: Reviewer and Reflection Agents](chapters/chapter-11-review-reflection.md)

### Part 4: The Durable Spine (Running for weeks)
12. [Chapter 12: Temporal Workflows for Long-Running Agents](chapters/chapter-12-temporal.md)
13. [Chapter 13: Activities, Retries, and Replay-from-Cache](chapters/chapter-13-activities.md)
14. [Chapter 14: Durable Sleep and Continue-As-New](chapters/chapter-14-durable-sleep.md)
15. [Chapter 15: Surviving Crashes — Kill and Resume](chapters/chapter-15-crash-resume.md)

### Part 5: State and Memory
16. [Chapter 16: The Mission Anchor (progress.md, checklist.json, decisions.ndjson)](chapters/chapter-16-mission-anchor.md)
17. [Chapter 17: Tiered Memory: Episodic, Semantic, Procedural](chapters/chapter-17-tiered-memory.md)
18. [Chapter 18: Vector Search and Skill Libraries](chapters/chapter-18-vector-memory.md)
19. [Chapter 19: Re-embedding After Model Changes](chapters/chapter-19-reembedding.md)

### Part 6: Safety and Governance
20. [Chapter 20: Budget Governor and Cost Caps](chapters/chapter-20-budget-governor.md)
21. [Chapter 21: Loop Detection and Escaping Oscillation](chapters/chapter-21-loop-detection.md)
22. [Chapter 22: HITL Gates and Irreversible Actions](chapters/chapter-22-hitl-gates.md)
23. [Chapter 23: Egress Policies and Default-Deny Security](chapters/chapter-23-egress-policy.md)

### Part 7: Self-Improvement
24. [Chapter 24: Capturing Failure Traces](chapters/chapter-24-failure-traces.md)
25. [Chapter 25: Offline Evolution — Propose, Evaluate, Promote](chapters/chapter-25-evolution.md)
26. [Chapter 26: Building an Eval Harness](chapters/chapter-26-eval-harness.md)

### Part 8: Production
27. [Chapter 27: Docker Compose Stack (Temporal + Postgres + Langfuse)](chapters/chapter-27-docker-stack.md)
28. [Chapter 28: Deployment and Blue/Green Workers](chapters/chapter-28-deployment.md)
29. [Chapter 29: Observability with OpenTelemetry and Langfuse](chapters/chapter-29-observability.md)
30. [Chapter 30: Capstone — A Week-Long Mission End-to-End](chapters/chapter-30-capstone.md)

---

## How This Course Publishes

Each chapter is released every few hours automatically via a cron job. Every chapter builds on the previous one.

- **Progress tracking:** `PUBLISHED.md`
- **Latest chapter:** see the `chapters/` directory
- **Schedule:** see `.hermes/cron/course-publisher.json`

---

## Repo Structure

```
.
├── README.md                 # This file
├── PUBLISHED.md              # Which chapters are live
├── chapters/                 # All 30 chapters (released incrementally)
├── code/                     # Reference implementations per chapter
├── exercises/                # Hands-on exercises
├── templates/                # Reusable templates
└── .hermes/
    └── cron/
        └── course-publisher.json   # Auto-publishes chapters
```

---

## Quick Start

```bash
git clone https://github.com/rajataijarvis-coder/long-running-agent-course.git
cd long-running-agent-course

# Read chapters in order
open chapters/chapter-01-the-problem.md
```

---

## Expected Time

- 30 chapters
- 15–30 minutes each
- Full course: 10–15 hours spread across a week

---

## Prerequisites

- Python 3.12+
- Basic async Python
- Git
- Docker (for later chapters)
- Optional: Ollama for $0 runs

---

## License

MIT — same as the original LRA project.

## Acknowledgements

Course based on the work of **Fareed Khan** and the LRA contributors.
Original article: https://levelup.gitconnected.com/building-a-week-long-running-agentic-system-2ad79f8190bb
Original repo: https://github.com/FareedKhan-dev/long-running-agent
