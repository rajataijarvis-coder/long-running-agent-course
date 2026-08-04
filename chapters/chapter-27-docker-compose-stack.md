# Chapter 27: Docker Compose Stack

## What We'll Cover

- Why `uv run lra mission` on a laptop is enough for a demo but not for a week-long mission
- The four runtime dependencies a durable LRA deployment needs: **Temporal**, **Postgres/pgvector**, **Ollama**, and **Langfuse**
- How to express the whole local runtime as code in `lra-demo/compose.yaml` with Docker Compose **profiles**
- A runnable bootstrap script, `lra-demo/ch27_docker_compose_stack.py`, that starts the stack, waits for every dependency to become healthy, and writes connection config
- How to point a mission at the stack and verify that crashes resume from the durable control plane

---

## The Concept: The Runtime Is Also Source-Controlled

Chapters 12–15 moved the agent loop into Temporal so a host reboot does not kill a mission. Chapters 17–18 moved memory into Postgres and pgvector so the context window does not have to remember everything. Chapter 29 will pipe telemetry into Langfuse. All of that only works if those services are actually running, reachable, and configured consistently.

A week-long mission cannot rely on "I installed Temporal yesterday and I think it is on port 7233." The runtime must be **infrastructure as code** the same way the agent's state is in git. Docker Compose is the cheapest way to get there: one file pins versions, ports, volumes, and health checks, and one command brings the whole world up.

The LRA stack has four concerns:

| Concern | Service | Why it matters |
|---|---|---|
| Durable execution | Temporal Server + Web UI | Workflows survive process death (Chapters 12–15) |
| Tiered memory | Postgres + pgvector | Long-term memory and vector search (Chapters 17–18) |
| Local models | Ollama | Run the agent at $0 without cloud tokens |
| Observability | Langfuse | Trace every LLM call and cost (Chapter 29) |

We use Compose **profiles** so you only pay the complexity you need. The `durable` and `memory` profiles are required for a real long-running mission. `local-model` and `observability` are optional.

---

## The Stack File: `lra-demo/compose.yaml`

```yaml
# lra-demo/compose.yaml
# Profiles: durable | memory | local-model | observability
services:
  temporal:
    image: temporalio/auto-setup:1.25
    profiles: ["durable"]
    ports:
      - "7233:7233"   # gRPC frontend
      - "8233:8233"   # HTTP API / Web UI
    environment:
      - DB=postgres12
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=postgres
      - DYNAMIC_CONFIG_FILE_PATH=config/dynamicconfig/development-sql.yaml
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "tctl", "--address", "temporal:7233", "workflow", "list"]
      interval: 5s
      timeout: 5s
      retries: 30

  postgres:
    image: pgvector/pgvector:pg16
    profiles: ["memory"]
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: lra
      POSTGRES_PASSWORD: lra
      POSTGRES_DB: lra
    volumes:
      - lra_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lra -d lra"]
      interval: 2s
      timeout: 3s
      retries: 20

  ollama:
    image: ollama/ollama:latest
    profiles: ["local-model"]
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 0"]
      interval: 10s
      timeout: 5s
      retries: 6

  langfuse-web:
    image: langfuse/langfuse:latest
    profiles: ["observability"]
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://lra:lra@postgres:5432/lra
      - NEXTAUTH_SECRET=local-dev-secret
      - SALT=local-dev-salt
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  lra_pgdata:
  ollama_data:
```

A few deliberate choices:

- **Temporal uses Postgres** for its own persistence, but LRA's application memory uses a separate logical database `lra` in the same Postgres container. In production you would split them; locally one container is fine.
- **pgvector image** gives us the vector extension for Chapter 18 without any `CREATE EXTENSION` dance.
- **Profiles** mean `docker compose --profile durable --profile memory up` starts only what the durable core needs. Add `--profile local-model` if you want Ollama, and `--profile observability` when you are ready for Chapter 29.

---

## The Bootstrap Script: `lra-demo/ch27_docker_compose_stack.py`

The YAML file is good, but a week-long mission needs more than "it probably started." We want a script that:

1. Starts the selected profiles
2. Polls every service until it is actually healthy
3. Writes a `.env.stack` file with the connection strings the LRA CLI needs
4. Can tear the stack down again

Here is the full script.

```python
#!/usr/bin/env python3
# lra-demo/ch27_docker_compose_stack.py
"""Start, health-check, and stop the LRA Docker Compose stack."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "compose.yaml"
ENV_FILE = ROOT / ".env.stack"
DEFAULT_PROFILES = ["durable", "memory"]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def compose_base(profiles: list[str]) -> list[str]:
    base = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    for p in profiles:
        base.extend(["--profile", p])
    return base


def stack_up(profiles: list[str]) -> None:
    if not COMPOSE_FILE.exists():
        print(f"Missing {COMPOSE_FILE}", file=sys.stderr)
        sys.exit(1)

    # Pull first so the up command does not spend minutes downloading.
    run(compose_base(profiles) + ["pull"], check=False)

    # Bring up services in detached mode.
    run(compose_base(profiles) + ["up", "-d", "--remove-orphans"])

    # Wait for every enabled service to report healthy.
    wait_for_stack(profiles)
    write_env_file(profiles)
    print("\nStack is up. Connection config written to", ENV_FILE)


def stack_down(profiles: list[str]) -> None:
    run(compose_base(profiles) + ["down", "--volumes", "--remove-orphans"])
    if ENV_FILE.exists():
        ENV_FILE.unlink()
    print("Stack is down.")


def service_names_for_profiles(profiles: list[str]) -> list[str]:
    mapping = {
        "durable": ["temporal"],
        "memory": ["postgres"],
        "local-model": ["ollama"],
        "observability": ["langfuse-web"],
    }
    return [svc for p in profiles for svc in mapping.get(p, [])]


def wait_for_stack(profiles: list[str], timeout: int = 120) -> None:
    services = service_names_for_profiles(profiles)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        unhealthy = []
        for svc in services:
            status = container_health(svc)
            print(f"  {svc}: {status}")
            if status != "healthy":
                unhealthy.append(svc)

        if not unhealthy:
            return

        time.sleep(2)

    print(f"Timed out waiting for: {unhealthy}", file=sys.stderr)
    sys.exit(1)


def container_health(service: str) -> str:
    try:
        proc = run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", service],
            check=False,
        )
        return proc.stdout.strip() or "no-healthcheck"
    except subprocess.CalledProcessError:
        return "missing"


def tcp_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def wait_for_ports(profiles: list[str]) -> None:
    """Optional second-level check: confirm ports are reachable from the host."""
    checks = []
    if "durable" in profiles:
        checks.append(("localhost", 7233, "Temporal gRPC"))
    if "memory" in profiles:
        checks.append(("localhost", 5432, "Postgres"))
    if "local-model" in profiles:
        checks.append(("localhost", 11434, "Ollama"))
    if "observability" in profiles:
        checks.append(("localhost", 3000, "Langfuse"))

    for host, port, name in checks:
        print(f"Waiting for {name} on {host}:{port} ...")
        for _ in range(60):
            if tcp_open(host, port):
                break
            time.sleep(1)
        else:
            print(f"{name} never became reachable", file=sys.stderr)
            sys.exit(1)


def write_env_file(profiles: list[str]) -> None:
    lines = [
        "# Auto-generated by ch27_docker_compose_stack.py",
        f"LRA_TEMPORAL_HOST=localhost:7233",
        f"LRA_TEMPORAL_NAMESPACE=default",
        f"LRA_POSTGRES_URL=postgresql://lra:lra@localhost:5432/lra",
    ]
    if "local-model" in profiles:
        lines.append("LRA_MODEL_PROVIDER=ollama")
        lines.append("LRA_MODEL_NAME=codellama")
    if "observability" in profiles:
        lines.append("LANGFUSE_HOST=http://localhost:3000")

    ENV_FILE.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the LRA local stack")
    parser.add_argument("action", choices=["up", "down", "health"])
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Compose profile to enable (can repeat)",
    )
    args = parser.parse_args()

    profiles = args.profile or DEFAULT_PROFILES

    if args.action == "up":
        stack_up(profiles)
    elif args.action == "down":
        stack_down(profiles)
    elif args.action == "health":
        wait_for_stack(profiles)
        wait_for_ports(profiles)
        print("All services healthy and reachable.")


if __name__ == "__main__":
    main()
```

### Walkthrough

- `compose_base` builds the `docker compose` command with the right profiles. This keeps the script and the YAML in sync.
- `stack_up` pulls images, starts containers, waits for Docker health checks, and writes `.env.stack`. The `--remove-orphans` flag cleans up services you disabled by switching profiles.
- `container_health` reads the Docker health status directly. We do not trust "container is running" because Temporal can be running but still initializing its schema.
- `wait_for_ports` is a second-level sanity check from the host side. A service can pass its Docker health check and still not be reachable on `localhost` if the port mapping is wrong.
- `write_env_file` emits the exact environment variables the LRA CLI uses. Loading them is one `export $(cat .env.stack | xargs)` away.

---

## Hands-On Exercise

1. **Bring up the durable core:**

   ```bash
   cd lra-demo
   python ch27_docker_compose_stack.py up
   ```

   You should see each service report `healthy`, and `.env.stack` should be created.

2. **Inspect the generated config:**

   ```bash
   cat lra-demo/.env.stack
   ```

3. **Source it and run a real mission against Temporal and Postgres:**

   ```bash
   export $(cat lra-demo/.env.stack | xargs)
   uv run lra mission \
     --task "Create a hello.py that prints hello and a test for it" \
     --workdir .lra/workspaces/demo \
     --temporal-host "$LRA_TEMPORAL_HOST" \
     --memory-url "$LRA_POSTGRES_URL"
   ```

4. **Verify durability.** While the mission is running, kill the Temporal container:

   ```bash
   docker compose -f lra-demo/compose.yaml kill temporal
   ```

   Wait ten seconds, then restart it:

   ```bash
   python lra-demo/ch27_docker_compose_stack.py up
   ```

   The mission workflow should resume exactly where it left off, without re-spending tokens for work already done.

5. **Add Ollama and observability:**

   ```bash
   python lra-demo/ch27_docker_compose_stack.py up \
     --profile durable --profile memory \
     --profile local-model --profile observability
   ```

   Then open the Temporal Web UI at `http://localhost:8233` and Langfuse at `http://localhost:3000`.

6. **Tear everything down:**

   ```bash
   python lra-demo/ch27_docker_compose_stack.py down
   ```

---

> **Key Takeaway:** A week-long agent is only as durable as the runtime underneath it. Docker Compose turns Temporal, Postgres, Ollama, and Langfuse into versioned, reproducible infrastructure, and a small bootstrap script guarantees the stack is actually healthy before the first workflow starts.

---

## Next Chapter

With the local stack proven, Chapter 28 moves to **Deployment and Blue/Green Workers**: how to ship the LRA worker processes themselves, rotate new code without killing in-flight missions, and keep the system running while you upgrade it.