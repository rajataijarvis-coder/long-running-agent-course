# Chapter 28: Deployment and Blue/Green Workers

## What We'll Cover

- Why a durable mission needs **stateless, replaceable workers** rather than one long-lived process
- How **blue/green worker deployment** lets you ship a new worker version without killing running missions
- The role of **Temporal task queues**, **graceful shutdown**, and **worker health checks** in the swap
- How to express the worker service in `lra-demo/compose.yaml` and drive the swap from `lra-demo/ch28_blue_green_workers.py`
- A runnable Python deployer that builds green, drains blue, verifies progress, and rolls back on failure

---

## The Concept: Workers Are Cattle, Not Pets

Chapters 12–15 moved the agent loop into a Temporal workflow. Chapter 27 gave you a Docker Compose stack that can keep Temporal, Postgres, Ollama, and Langfuse running for days. But the **worker** — the process that actually executes workflow and activity code — still has to be updated. If you `kill -9` the only worker mid-flight, in-progress activities are interrupted and retried, which wastes tokens and time. If you update the worker binary in place, you risk mixing old and new code inside the same process.

The fix is to treat workers like cattle:

1. Run the current version as the **blue** fleet.
2. Build and start the new version as the **green** fleet.
3. Let both poll the same Temporal task queue while green proves itself.
4. Gracefully **drain blue** (stop polling, finish in-flight activities).
5. Keep green running. If anything looks wrong, roll back to blue.

Because Temporal holds the real execution state and git holds the real mission state, the workers themselves are stateless. A mission survives the swap as long as at least one healthy worker is connected to the task queue.

---

## The Worker Service in `lra-demo/compose.yaml`

Add a worker service to the Compose stack from Chapter 27. The image tag is parameterized so the same file can launch blue or green depending on the `LRA_WORKER_IMAGE_TAG` environment variable.

```yaml
# lra-demo/compose.yaml (worker excerpt)
services:
  lra-worker:
    profiles: ["workers"]
    build:
      context: ..
      dockerfile: Dockerfile.worker
    image: lra-worker:${LRA_WORKER_IMAGE_TAG:-blue}
    environment:
      - TEMPORAL_HOST=temporal:7233
      - TEMPORAL_NAMESPACE=default
      - TEMPORAL_TASK_QUEUE=lra-missions
      - LRA_STATE_REPO=/data/repo
      - LRA_MODEL_BACKEND=ollama
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - lra-repo:/data/repo
    depends_on:
      temporal:
        condition: service_healthy
      postgres:
        condition: service_healthy
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      replicas: 2
```

The `profiles: ["workers"]` line keeps the worker out of the default `docker compose up` from Chapter 27. You start it explicitly with:

```bash
LRA_WORKER_IMAGE_TAG=blue docker compose -p lra -f lra-demo/compose.yaml --profile workers up -d
```

Each worker exposes a tiny HTTP health endpoint so the deployer can tell when green is alive and when blue has exited.

---

## The Blue/Green Deployer

`lra-demo/ch28_blue_green_workers.py` is a small deployment controller. It does not depend on Kubernetes or a CI platform; it drives Docker Compose directly so you can run it from your laptop or a VM.

```python
# lra-demo/ch28_blue_green_workers.py
"""Blue/green worker deployment for the LRA durable control plane.

Usage:
    LRA_WORKER_IMAGE_TAG=green-v2 uv run lra-demo/ch28_blue_green_workers.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from temporalio.client import Client


@dataclass(frozen=True)
class DeployConfig:
    compose_file: Path = Path(__file__).with_name("compose.yaml")
    project_name: str = "lra"
    worker_service: str = "lra-worker"
    blue_tag: str = "blue"
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    drain_timeout_seconds: int = 120
    verify_timeout_seconds: int = 60


class BlueGreenDeployer:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self._temporal: Client | None = None

    # ------------------------------------------------------------------
    # Docker Compose helpers
    # ------------------------------------------------------------------
    def _compose(self, *args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [
            "docker", "compose",
            "-f", str(self.cfg.compose_file),
            "-p", self.cfg.project_name,
            *args,
        ]
        kwargs: dict[str, object] = {"check": check, "text": True}
        if capture:
            kwargs["capture_output"] = True
        return subprocess.run(cmd, **kwargs)  # type: ignore[arg-type]

    def _with_env(self, extra: dict[str, str]) -> dict[str, str]:
        return {**os.environ, **extra}

    def running_container_ids(self, image_tag: str) -> list[str]:
        image = f"{self.cfg.worker_service}:{image_tag}"
        res = subprocess.run(
            ["docker", "ps", "--filter", f"ancestor={image}", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    def running_replicas(self) -> int:
        res = self._compose("ps", self.cfg.worker_service, "--format", "json", capture=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            return 0
        data = json.loads(res.stdout)
        if isinstance(data, dict):
            data = [data]
        return sum(1 for item in data if item.get("State") == "running")

    # ------------------------------------------------------------------
    # Build and deploy green
    # ------------------------------------------------------------------
    def build_green(self, green_tag: str) -> None:
        print(f"Building green image {self.cfg.worker_service}:{green_tag} ...")
        env = self._with_env({"LRA_WORKER_IMAGE_TAG": green_tag})
        self._compose("build", "--no-cache", self.cfg.worker_service, env=env)

    def deploy_green(self, green_tag: str, replicas: int) -> None:
        print(f"Starting {replicas} green worker(s) with tag {green_tag} ...")
        env = self._with_env({"LRA_WORKER_IMAGE_TAG": green_tag})
        self._compose(
            "up", "-d", "--no-deps",
            "--scale", f"{self.cfg.worker_service}={replicas}",
            self.cfg.worker_service,
            env=env,
        )

        deadline = time.time() + self.cfg.verify_timeout_seconds
        while time.time() < deadline:
            if self.running_replicas() >= replicas:
                print("Green workers are healthy.")
                return
            time.sleep(2)
        raise RuntimeError("green workers did not become healthy")

    # ------------------------------------------------------------------
    # Drain blue and verify
    # ------------------------------------------------------------------
    async def temporal_client(self) -> Client:
        if self._temporal is None:
            self._temporal = await Client.connect(
                self.cfg.temporal_host,
                namespace=self.cfg.temporal_namespace,
            )
        return self._temporal

    async def running_workflow_count(self) -> int:
        client = await self.temporal_client()
        count = 0
        async for _ in client.list_workflows("ExecutionStatus='Running'"):
            count += 1
        return count

    async def drain_blue(self) -> None:
        blue_ids = self.running_container_ids(self.cfg.blue_tag)
        if not blue_ids:
            print("No blue workers running.")
            return

        print(f"Draining {len(blue_ids)} blue worker(s) with SIGTERM ...")
        for cid in blue_ids:
            # SIGTERM asks the Temporal SDK to stop polling and finish in-flight activities.
            subprocess.run(["docker", "kill", "--signal", "SIGTERM", cid], check=False)

        deadline = time.time() + self.cfg.drain_timeout_seconds
        while time.time() < deadline:
            remaining = self.running_container_ids(self.cfg.blue_tag)
            if not remaining:
                print("Blue workers drained.")
                return
            time.sleep(2)
        raise RuntimeError("blue workers did not drain before timeout")

    async def verify_green(self) -> bool:
        print("Verifying that missions are still making progress ...")
        before = await self.running_workflow_count()
        await asyncio.sleep(15)
        after = await self.running_workflow_count()

        # A healthy swap: active workflow count is stable or decreasing.
        # A growing backlog means green is not polling successfully.
        if after > before + 2:
            print(f"Workflow backlog grew from {before} to {after}.")
            return False
        print(f"Workflow count stable: {before} -> {after}.")
        return True

    # ------------------------------------------------------------------
    # Orchestrate the swap
    # ------------------------------------------------------------------
    async def swap(self, green_tag: str, replicas: int = 2) -> None:
        self.build_green(green_tag)
        self.deploy_green(green_tag, replicas)
        await self.drain_blue()
        if not await self.verify_green():
            raise RuntimeError("green verification failed")

    async def rollback(self) -> None:
        print("Rolling back to blue ...")
        green_ids = self.running_container_ids(self.cfg.green_tag)
        for cid in green_ids:
            subprocess.run(["docker", "kill", "--signal", "SIGTERM", cid], check=False)

        env = self._with_env({"LRA_WORKER_IMAGE_TAG": self.cfg.blue_tag})
        self._compose(
            "up", "-d", "--no-deps",
            "--scale", f"{self.cfg.worker_service}=2",
            self.cfg.worker_service,
            env=env,
        )


async def main() -> None:
    green_tag = os.environ.get("LRA_WORKER_IMAGE_TAG", "green-v2")
    deployer = BlueGreenDeployer(DeployConfig())

    try:
        await deployer.swap(green_tag, replicas=2)
        print("Blue/green swap complete.")
    except Exception as exc:
        print(f"Swap failed: {exc}", file=sys.stderr)
        await deployer.rollback()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Code Walkthrough

**`build_green`** rebuilds the worker image with the tag you pass in `LRA_WORKER_IMAGE_TAG`. Because the Compose file uses `image: lra-worker:${LRA_WORKER_IMAGE_TAG:-blue}`, the same service definition produces a distinct image for every tag.

**`deploy_green`** starts green containers without restarting blue. Both fleets now poll the same Temporal task queue (`lra-missions`). Temporal dispatches new tasks to whichever worker asks first, so green starts taking work immediately while blue finishes what it already grabbed.

**`drain_blue`** sends `SIGTERM` to the old containers. The Temporal Python worker catches the signal, stops polling for **new** tasks, waits for in-flight activities to complete (up to the configured graceful shutdown period), and then exits. This is why a mission does not lose progress during the swap.

**`verify_green`** connects to Temporal and counts running workflows before and after the drain. If the count is stable or falling, green is healthy. If the backlog spikes, green is failing to poll or crashing on startup.

**`rollback`** kills green and restarts blue. Because the real state is in Temporal and git, rolling back the worker fleet does not revert mission progress.

---

## Hands-On Exercise

1. Start the durable stack and the blue worker fleet:

```bash
cd lra-demo
docker compose -p lra -f compose.yaml --profile infra up -d
LRA_WORKER_IMAGE_TAG=blue docker compose -p lra -f compose.yaml --profile workers up -d
```

2. Start a long mission so you have in-flight work during the swap:

```bash
uv run lra mission \
  --task "Add a small CLI argument parser to hello.py and verify with pytest" \
  --workdir .lra/workspaces/ch28
```

3. In another terminal, make a tiny worker change — for example, add a log line or bump a dependency — and deploy it as green:

```bash
cd lra-demo
LRA_WORKER_IMAGE_TAG=green-v2 uv run ch28_blue_green_workers.py
```

4. While the script runs, watch the worker containers:

```bash
docker ps --filter "name=lra-worker" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
```

You should see both `blue` and `green` containers briefly, then only `green`.

5. Check the Temporal UI at `http://localhost:8233`. The running mission workflow should still be there, and its history should show no failed activities caused by the deployment.

6. Simulate a bad green build by breaking the worker health endpoint and rerunning the script. Confirm that the deployer rolls back to blue and exits with a non-zero status.

---

> **Key Takeaway:** In a durable agent system, the worker is stateless and replaceable. Blue/green deployment lets you ship new behavior without trusting the new behavior first: green proves itself on the real task queue while blue keeps the mission alive, and Temporal's graceful shutdown makes the handoff safe.

---

## Next Chapter

Chapter 29 wires the whole stack into **OpenTelemetry and Langfuse** so you can trace every LLM call, tool invocation, and workflow step across the blue and green fleets.