# course-publisher.py
"""
Publishes chapters of the long-running-agent course incrementally.

Usage (run by cron):
    python .hermes/cron/course-publisher.py

What it does:
1. Reads PUBLISHED.md to find the next locked chapter
2. Converts 🔒 → ✅ and adds a timestamp
3. Commits and pushes to GitHub
4. Logs to .hermes/cron/course-publisher.log

Schedule: every 4 hours.
"""

import os
import re
import subprocess
from datetime import datetime, timezone

REPO_DIR = "/Users/rajatjarvis/Downloads/projects/long-running-agent-course"
PUBLISHED_FILE = os.path.join(REPO_DIR, "PUBLISHED.md")
CHAPTERS_DIR = os.path.join(REPO_DIR, "chapters")
LOG_FILE = "/Users/rajatjarvis/.hermes/cron/course-publisher.log"


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts}: {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main() -> None:
    os.chdir(REPO_DIR)

    if not os.path.exists(PUBLISHED_FILE):
        log(f"PUBLISHED.md not found at {PUBLISHED_FILE}")
        return

    with open(PUBLISHED_FILE, "r") as f:
        content = f.read()

    # Find first locked chapter
    match = re.search(r"\|\s*(\d+)\s*\|\s*([^|]+)\|\s*🔒 Locked\s*\|[^|]*\|", content)
    if not match:
        log("All chapters already published or no locked rows found.")
        return

    chapter_num = match.group(1).zfill(2)
    chapter_title = match.group(2).strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Update the row: 🔒 Locked → ✅ Published
    old_row = match.group(0)
    new_row = old_row.replace("🔒 Locked", "✅ Published").replace("\n\n", "").replace(
        f"|\n| {chapter_num} |", f"| {timestamp}\n| {chapter_num} |")
    # Safer: rebuild the row
    new_row = f"| {int(chapter_num)} | {chapter_title} | ✅ Published | {timestamp} |"

    content = content.replace(old_row, new_row)

    with open(PUBLISHED_FILE, "w") as f:
        f.write(content)

    log(f"Marked chapter {chapter_num} '{chapter_title}' as published at {timestamp}")

    # Git commit and push
    try:
        subprocess.run(["git", "add", "PUBLISHED.md"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"publish: Chapter {chapter_num} — {chapter_title}"],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
        log(f"Pushed chapter {chapter_num} to GitHub.")
    except subprocess.CalledProcessError as e:
        log(f"Git operation failed: {e}")


if __name__ == "__main__":
    main()
