# Chapter 29: Observability with OpenTelemetry and Langfuse

## What We'll Cover

- Why a **week-long mission** needs more than `print()` statements to stay debuggable
- How **OpenTelemetry** gives a vendor-neutral execution graph across Temporal, model calls, tools, and verification
- How **Langfuse** adds the LLM-specific lens: generations, token usage, cost, scores, and human feedback
- Wiring the `observability` extra and the OTLP exporter into the Docker Compose stack from Chapter 27
- A runnable `lra-demo/ch29_observability.py` script that emits a trace, logs a generation to Langfuse, and prints a direct link
- What to inspect in the UI after a crash, a loop-detector trip, or a budget-governor scope cut

---

## The Concept: If You Cannot See It, You Cannot Run It for a Week

Chapters 12–15 moved the agent loop into Temporal so a crash does not kill the mission. Chapter 27 gave you a Compose stack that keeps Temporal, Postgres, and Ollama alive. Chapter 28 made the workers replaceable. All of that keeps the mission *running*, but it does not tell you *what is happening* inside.

A long-running mission produces thousands of model calls, tool dispatches, verification attempts, and state transitions. Without observability you cannot answer the questions that matter after a day or two:

- Where did the $178.60 from the sample trace actually go?
- Which cycle tripped the loop detector in Chapter 21?
- Did the reviewer agent from Chapter 11 block progress for a real reason?
- Was the crash in Chapter 15 a worker bug, a model failure, or a sandbox timeout?

OpenTelemetry solves the first half of the problem. It is a standard way to emit **traces** (spans with parent/child relationships), **metrics**, and **logs** to any backend. Because the LRA worker is stateless, every span carries the `mission_id`, `cycle`, `workflow_id`, and `activity_id` so a trace can be reconstructed even if the process that emitted it has been replaced twice.

Langfuse solves the second half. It is built for LLM applications: it knows about **generations**, **token usage**, **cost**, **scores**, and **sessions**. When you pipe OpenTelemetry spans into Langfuse, or use the Langfuse SDK directly, you get a UI where you can click from a high-level mission trace down to the exact prompt that produced a bad plan.

The rule in LRA is: **every token spent and every state change is observable**. The model is a black box; the system around it does not have to be.

---

## Code Walkthrough

### 1. The observability module

The core setup lives in `src/lra/observability/tracing.py`. It configures a `TracerProvider` with an OTLP exporter and a resource that identifies the worker, the mission, and the deployment color from Chapter 28.

```python
# src/lra/observability/tracing.py
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

def configure_tracing(service_name: str = "lra-worker") -> trace.TracerProvider:
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        SERVICE_VERSION: os.getenv("LRA_VERSION", "0.1.0"),
        "deployment.color": os.getenv("LRA_DEPLOY_COLOR", "blue"),
    })

    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider

def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
```

The worker calls `configure_tracing()` once at startup. Every span processor is batched, so a crash mid-batch loses at most a few seconds of telemetry, not the whole mission history.

### 2. Instrumenting a model call

The model layer from Chapter 03 is pluggable. The wrapper below is added around every backend so that cost, tokens, and latency are captured the same way whether the model is `stub`, `ollama`, or `claude`.

```python
# src/lra/model/observed.py
import time
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("lra.model")

@contextmanager
def observed_generate(
    mission_id: str,
    cycle: int,
    model: str,
    prompt: str,
):
    start = time.monotonic()
    with tracer.start_as_current_span("model.generate") as span:
        span.set_attributes({
            "lra.mission_id": mission_id,
            "lra.cycle": cycle,
            "gen_ai.system": "lra",
            "gen_ai.request.model": model,
            "gen_ai.prompt.length": len(prompt),
        })
        try:
            yield span
            elapsed = time.monotonic() - start
            span.set_attribute("gen_ai.latency_ms", elapsed * 1000)
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
```

The caller uses it like this:

```python
with observed_generate(mission_id, cycle, model, prompt) as span:
    response = backend.generate(prompt)
    span.set_attributes({
        "gen_ai.usage.input_tokens": response.input_tokens,
        "gen_ai.usage.output_tokens": response.output_tokens,
        "gen_ai.response.length": len(response.text),
        "lra.cost_usd": response.cost_usd,
    })
```

These attributes are standard enough for any OpenTelemetry backend and detailed enough for Langfuse to render a cost breakdown.

### 3. Langfuse integration

Langfuse is wired directly into the model wrapper so that every generation is also a Langfuse **generation** inside a **trace**. The trace ID is shared with OpenTelemetry, so you can jump from the distributed trace to the LLM trace with the same ID.

```python
# src/lra/observability/langfuse_client.py
import os
from langfuse import Langfuse
from langfuse.decorators import observe

_langfuse: Langfuse | None = None

def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
    return _langfuse

def log_generation(
    trace_id: str,
    name: str,
    model: str,
    prompt: str,
    output: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
):
    lf = get_langfuse()
    trace = lf.trace(id=trace_id, name="agent.cycle")
    trace.generation(
        name=name,
        model=model,
        input=prompt,
        output=output,
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "unit": "TOKENS",
        },
        metadata={"cost_usd": cost_usd},
    )
```

The `trace_id` passed here is the OpenTelemetry trace ID formatted as a 32-character hex string. That is the single link between the two systems.

### 4. The runnable demo

`lra-demo/ch29_observability.py` ties it all together. It runs three fake agent cycles, emits OpenTelemetry spans, logs Langfuse generations, and prints a direct URL. It falls back to console output if the OTLP endpoint or Langfuse keys are not configured, so it is safe to run on a laptop.

```python
# lra-demo/ch29_observability.py
import os
import time
import uuid
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# ---------------------------------------------------------------------------
# OpenTelemetry setup with a console fallback
# ---------------------------------------------------------------------------
resource = Resource(attributes={
    SERVICE_NAME: "lra-demo",
    "lra.version": "0.1.0",
})

provider = TracerProvider(resource=resource)

if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        )
    )
else:
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("lra.demo")

# ---------------------------------------------------------------------------
# Langfuse setup with a no-op fallback
# ---------------------------------------------------------------------------
class _NoopTrace:
    def generation(self, **kwargs): return self
    def span(self, **kwargs): return self
    def update(self, **kwargs): pass

class _NoopLangfuse:
    def trace(self, **kwargs): return _NoopTrace()
    def flush(self): pass

def get_langfuse():
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if pk and sk:
        from langfuse import Langfuse
        return Langfuse(
            public_key=pk,
            secret_key=sk,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
    return _NoopLangfuse()

langfuse = get_langfuse()

# ---------------------------------------------------------------------------
# Fake model + tool to keep the demo self-contained
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str

def fake_model_generate(prompt: str, model: str = "stub") -> ModelResult:
    input_tokens = len(prompt.split())
    text = "I'll add a hello.py and a test for it."
    output_tokens = len(text.split())
    return ModelResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=(input_tokens + output_tokens) * 1e-6,
        model=model,
    )

def run_cycle(mission_id: str, cycle: int):
    with tracer.start_as_current_span("agent.cycle") as cycle_span:
        cycle_span.set_attributes({
            "lra.mission_id": mission_id,
            "lra.cycle": cycle,
        })
        trace_id = format(cycle_span.get_span_context().trace_id, "032x")

        lf_trace = langfuse.trace(
            id=trace_id,
            name="agent.cycle",
            metadata={"mission_id": mission_id, "cycle": cycle},
        )

        with tracer.start_as_current_span("model.generate") as gen_span:
            prompt = f"Mission {mission_id} cycle {cycle}: what should we do next?"
            result = fake_model_generate(prompt)

            gen_span.set_attributes({
                "gen_ai.system": "lra",
                "gen_ai.request.model": result.model,
                "gen_ai.usage.input_tokens": result.input_tokens,
                "gen_ai.usage.output_tokens": result.output_tokens,
                "lra.cost_usd": result.cost_usd,
            })

            lf_trace.generation(
                name="model.generate",
                model=result.model,
                input=prompt,
                output=result.text,
                usage={
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.input_tokens + result.output_tokens,
                    "unit": "TOKENS",
                },
            )

        with tracer.start_as_current_span("tool.dispatch") as tool_span:
            tool_name = "write_file"
            tool_span.set_attribute("lra.tool.name", tool_name)
            lf_trace.span(name="tool.dispatch", input=tool_name)
            time.sleep(0.05)

        lf_trace.update(metadata={"cost_usd": result.cost_usd})
        return trace_id, result.cost_usd

def main():
    mission_id = f"demo-{uuid.uuid4().hex[:8]}"
    total_cost = 0.0
    last_trace_id = ""

    for cycle in range(1, 4):
        trace_id, cost = run_cycle(mission_id, cycle)
        last_trace_id = trace_id
        total_cost += cost
        print(f"cycle={cycle} trace_id={trace_id} cost_usd={cost:.6f}")

    langfuse.flush()
    provider.force_flush()

    print(f"\nmission={mission_id} total_cost_usd={total_cost:.6f}")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    print(f"Open Langfuse: {host}/project/*/traces?traceId={last_trace_id}")

if __name__ == "__main__":
    main()
```

Run it with the observability extra installed:

```bash
uv sync --extra observability
uv run --extra observability python lra-demo/ch29_observability.py
```

If you have the Chapter 27 stack running and the environment variables set, the same trace appears in both the OpenTelemetry collector and the Langfuse UI.

### 5. Wiring it into the Compose stack

The `lra-demo/compose.yaml` from Chapter 27 already includes Temporal and Postgres. To add observability, add the Langfuse server and an OpenTelemetry collector. Here is the minimal addition:

```yaml
# lra-demo/compose.yaml (observability section)
services:
  langfuse-server:
    image: langfuse/langfuse:latest
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET:-local}
      SALT: ${LANGFUSE_SALT:-local}
      ENCRYPTION_KEY: ${LANGFUSE_ENCRYPTION_KEY:-0000000000000000000000000000000000000000000000000000000000000000}
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"
```

And a minimal collector config:

```yaml
# lra-demo/otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  otlp/langfuse:
    endpoint: langfuse-server:4318
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp/langfuse]
```

Start the profile from Chapter 27 with observability enabled:

```bash
docker compose --profile observability up -d
```

Then set the environment variables the worker and demo script expect:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000
```

### 6. Connecting to the rest of the system

Observability is not decoration; it is the feedback loop for several earlier chapters:

- **Chapter 20 — Budget Governor:** the governor reads `lra.cost_usd` from each span and stops the mission when the ceiling is hit.
- **Chapter 21 — Loop Detection:** when the loop detector fires, it emits a span event named `loop.detected` with the cycle count and the diff hash that triggered it.
- **Chapter 24 — Failure Traces:** a failed verification writes a span with `status=ERROR` and `exception.message`, which becomes the seed for the eval harness in Chapter 26.
- **Chapter 28 — Blue/Green Workers:** every span carries `deployment.color`, so after a swap you can compare latency and error rates between blue and green.

---

## Hands-on Exercise

1. Start the Chapter 27 stack with the observability profile:

   ```bash
   docker compose --profile observability up -d
   ```

2. Install the observability extra:

   ```bash
   uv sync --extra observability
   ```

3. Run the demo script and capture the trace ID:

   ```bash
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_SECRET_KEY=sk-lf-...
   export LANGFUSE_HOST=http://localhost:3000
   uv run --extra observability python lra-demo/ch29_observability.py
   ```

4. Open the Langfuse URL printed at the end. Verify that:
   - The trace has three `agent.cycle` spans.
   - Each cycle contains a `model.generate` generation with token counts.
   - The total cost matches the sum printed in the terminal.

5. Add a new span around the fake tool dispatch that records the file path and size, rerun, and confirm it appears in the trace.

6. Optional: run a real `uv run lra mission` task against the stack, kill the worker container mid-mission, and verify in Langfuse that the trace continues with the same `mission_id` after the worker restarts.

---

> **Key Takeaway:** A week-long mission is not a prompt; it is a distributed system. OpenTelemetry gives you the execution graph, Langfuse gives you the LLM lens, and together they make the invisible visible.

---

## Next Chapter

In **Chapter 30: Capstone — A Week-Long Mission End-to-End**, we put every piece from the last twenty-nine chapters together: a single `lra mission` command that plans, writes, verifies, crashes, resumes, learns, and finishes a real software task across multiple days.