# Chapter 30: Capstone — A Week-Long Mission

## What We'll Cover
- Planning a week-long mission
- Running it on the durable spine
- Surviving crashes and loops
- Reviewing traces and improving prompts

## Mission
Pick a real, multi-step task and break it into 10-20 checklist items.

## Run It
```bash
docker compose up -d
uv run lra worker
uv run lra mission start --task-file mission.md
```

## Key Takeaway
A week-long mission proves your system. Everything before was preparation.

Congratulations — you have built a durable, self-improving agent organization.
