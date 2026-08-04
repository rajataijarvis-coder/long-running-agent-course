# Chapter 23: Egress Policies and Default-Deny Security

## What We'll Cover

- Why **egress control** is a system-level safety property, not a prompt-engineering trick
- How LRA applies a **default-deny** rule to every outbound request from a tool or sandbox
- The three policy actions — **ALLOW**, **DENY**, and **HITL** — and when each is appropriate
- How the egress engine integrates with the **Tool Dispatcher** from Chapter 5 and the **HITL gates** from Chapter 22
- Why every egress decision is written to a durable **audit log** tied to the Mission Anchor
- A runnable demo in `lra-demo/ch23_egress_policies.py` that blocks, approves, and gates network calls without relying on the model to "behave"

---

## The Concept: Network Is an Irreversible Action

A long-running agent that builds software for a week will inevitably need to touch the network. It installs packages from PyPI, clones repositories, calls APIs, pushes branches, and maybe reports telemetry. Every one of those requests is an **irreversible action**: once data leaves the sandbox, you cannot un-send it.

In Chapter 22 we said that irreversible actions need a durable HITL gate. Egress is the most common class of irreversible action, and it needs more than a gate — it needs a **policy engine** that decides, for every single outbound request, whether the request is allowed, denied, or escalated to a human.

The model must not be trusted to make that decision. A prompt that says "only call safe APIs" is not a security boundary. The boundary belongs in the **Tool Dispatcher + Sandbox** layer from Chapter 5, where every request can be inspected, matched against rules, and blocked before it leaves the process.

LRA's safety rule is simple:

> **Default deny.** If a request does not match an explicit allow rule, it is denied. No exceptions, no "the model thought it was fine."

This is the same engineering discipline we applied to verification in Chapter 6: the model proposes, the system enforces. The model can ask to call `https://api.github.com`, but the policy engine decides whether that call is permitted.

---

## Why Default-Deny Matters for Long-Horizon Agents

A chat-style agent runs for minutes. A week-long mission runs for thousands of cycles. Over that horizon:

- A confused model can hallucinate a URL and leak context to it.
- A tool can follow a redirect to a domain the model never named.
- A dependency install can pull a package that phones home.
- A reviewer or evolver agent can try to fetch "helpful" documentation from an untrusted site.

If the system relies on the model to "do the right thing," any of these becomes a data-loss incident. If the system relies on a **default-deny policy**, the incident becomes a blocked request and an audit entry.

The policy also protects the **budget governor** from Chapter 20. An uncontrolled egress call to a paid embedding API or a cloud service can burn the mission's token ceiling. The governor caps LLM spend; the egress policy caps **network spend and exfiltration surface**.

---

## The Policy Engine

LRA's egress layer lives in `src/lra/safety/egress.py` and is invoked by `src/lra/execution/dispatcher.py` before any tool talks to the network. The core abstraction is:

- `EgressRequest` — one outbound call: URL, method, tool name, metadata.
- `EgressRule` — a matching rule: domains, IPs, ports, schemes, and an action.
- `EgressPolicy` — an ordered list of rules plus a default action (always `DENY` in production).
- `EgressAuditor` — a durable audit log of every decision.

The matching logic supports:

- exact domains (`api.github.com`)
- wildcard domains (`*.pypi.org`)
- CIDR blocks (`10.0.0.0/8`)
- port and scheme restrictions (`443`, `https`)

The action can be:

- `ALLOW` — execute immediately.
- `DENY` — block and raise `EgressDenied`.
- `HITL` — pause the workflow and wait for durable human approval (Chapter 22).

---

## Demo: `lra-demo/ch23_egress_policies.py`

The demo below is a self-contained Python script. It does not make real network calls; it uses a mock transport so you can run it safely and see the policy decisions.

```python
#!/usr/bin/env python3
"""
lra-demo/ch23_egress_policies.py

Default-deny egress policy for long-running agents.

Run:
    uv run python lra-demo/ch23_egress_policies.py
"""
from __future__ import annotations

import fnmatch
import ipaddress
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Sequence
from urllib.parse import urlparse


class Action(Enum):
    ALLOW = auto()
    DENY = auto()
    HITL = auto()


@dataclass(frozen=True)
class EgressRule:
    """One rule in the egress policy. Empty fields mean 'match any'."""
    name: str
    action: Action
    domains: tuple[str, ...] = ()
    ips: tuple[str, ...] = ()          # CIDR or single IP
    ports: tuple[int, ...] = ()
    schemes: tuple[str, ...] = ()
    comment: str = ""

    def __post_init__(self):
        # Normalize so matching is case-insensitive.
        object.__setattr__(self, "domains", tuple(d.lower() for d in self.domains))
        object.__setattr__(self, "schemes", tuple(s.lower() for s in self.schemes))


@dataclass(frozen=True)
class EgressRequest:
    url: str
    method: str = "GET"
    tool_name: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class EgressEvent:
    timestamp: str
    request: EgressRequest
    action: Action
    rule: str | None
    executed: bool
    detail: str


class EgressPolicy:
    """Ordered rule list with a default action. First match wins."""

    def __init__(self, rules: Sequence[EgressRule], default: Action = Action.DENY):
        self.rules = list(rules)
        self.default = default

    def evaluate(self, request: EgressRequest) -> tuple[Action, EgressRule | None]:
        parsed = urlparse(request.url)
        hostname = parsed.hostname or ""
        port = parsed.port
        scheme = parsed.scheme.lower()

        for rule in self.rules:
            if self._matches(rule, hostname, port, scheme):
                return rule.action, rule
        return self.default, None

    @staticmethod
    def _matches(rule: EgressRule, hostname: str, port: int | None, scheme: str) -> bool:
        if rule.schemes and scheme not in rule.schemes:
            return False
        if rule.ports and port not in rule.ports:
            return False

        if rule.domains:
            host_lower = hostname.lower()
            if any(EgressPolicy._host_matches(host_lower, pat) for pat in rule.domains):
                return True

        if rule.ips:
            try:
                addr = ipaddress.ip_address(hostname)
            except ValueError:
                addr = None
            if addr:
                for net in rule.ips:
                    if addr in ipaddress.ip_network(net, strict=False):
                        return True
        return False

    @staticmethod
    def _host_matches(host: str, pattern: str) -> bool:
        # Wildcard like *.pypi.org
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if host == suffix or host.endswith("." + suffix):
                return True
        if fnmatch.fnmatchcase(host, pattern):
            return True
        # Exact or suffix match
        if host == pattern:
            return True
        if host.endswith("." + pattern):
            return True
        return False


class EgressAuditor:
    """Durable audit log. In production this writes to the Mission Anchor (Chapter 16)."""

    def __init__(self):
        self.events: list[EgressEvent] = []

    def log(
        self,
        request: EgressRequest,
        action: Action,
        rule: EgressRule | None,
        executed: bool,
        detail: str,
    ) -> None:
        self.events.append(
            EgressEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                request=request,
                action=action,
                rule=rule.name if rule else None,
                executed=executed,
                detail=detail,
            )
        )

    def to_json(self) -> str:
        payload = [
            {
                "ts": e.timestamp,
                "tool": e.request.tool_name,
                "method": e.request.method,
                "url": e.request.url,
                "action": e.action.name,
                "rule": e.rule,
                "executed": e.executed,
                "detail": e.detail,
            }
            for e in self.events
        ]
        return json.dumps(payload, indent=2)


class EgressDenied(Exception):
    pass


class ToolDispatcher:
    """
    The dispatcher from Chapter 5, with egress enforcement wired in.
    Every tool call that touches the network is evaluated before it runs.
    """

    def __init__(
        self,
        policy: EgressPolicy,
        auditor: EgressAuditor,
        hitl: Callable[[EgressRequest, EgressRule], bool],
        transport: Callable[[EgressRequest], dict] | None = None,
    ):
        self.policy = policy
        self.auditor = auditor
        self.hitl = hitl
        self.transport = transport or self._default_transport

    @staticmethod
    def _default_transport(req: EgressRequest) -> dict:
        # In production this is the sandboxed HTTP client (httpx, docker, e2b).
        # Here we simulate a successful call so the demo needs no network.
        return {"status": 200, "body": f"mock-response-for-{req.url}"}

    def call(
        self,
        tool_name: str,
        url: str,
        method: str = "GET",
        metadata: dict | None = None,
    ) -> dict:
        req = EgressRequest(
            url=url,
            method=method,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        action, rule = self.policy.evaluate(req)

        if action is Action.DENY:
            self.auditor.log(
                req, action, rule, executed=False, detail="default-deny or explicit deny rule"
            )
            label = f"rule {rule.name}" if rule else "default deny"
            raise EgressDenied(f"{tool_name} -> {url} blocked by {label}")

        if action is Action.HITL:
            approved = self.hitl(req, rule) if rule else False
            if not approved:
                self.auditor.log(req, action, rule, executed=False, detail="HITL rejected")
                raise EgressDenied(f"{tool_name} -> {url} rejected by HITL")
            self.auditor.log(req, action, rule, executed=True, detail="HITL approved")
            return self.transport(req)

        # ALLOW
        self.auditor.log(req, action, rule, executed=True, detail="allowed")
        return self.transport(req)


def main() -> int:
    policy = EgressPolicy(
        rules=[
            EgressRule(
                name="pypi",
                action=Action.ALLOW,
                domains=("pypi.org", "*.pypi.org"),
                ports=(443,),
                schemes=("https",),
                comment="install dependencies",
            ),
            EgressRule(
                name="github-api",
                action=Action.ALLOW,
                domains=("api.github.com",),
                ports=(443,),
                schemes=("https",),
                comment="read issues/PRs",
            ),
            EgressRule(
                name="internal-git",
                action=Action.ALLOW,
                domains=("git.local",),
                ports=(443, 22),
                schemes=("https", "ssh"),
                comment="push to internal origin",
            ),
            EgressRule(
                name="unknown-cloud",
                action=Action.HITL,
                domains=("*.example.io",),
                ports=(443,),
                schemes=("https",),
                comment="unvetted cloud API",
            ),
            EgressRule(
                name="telemetry",
                action=Action.DENY,
                domains=("telemetry.megacorp.com",),
                ports=(443,),
                schemes=("https",),
                comment="no exfil",
            ),
        ],
        default=Action.DENY,
    )

    auditor = EgressAuditor()

    # Deterministic HITL stub: approve one known request, reject everything else.
    approvals = {("researcher", "https://data.example.io/v1/search"): True}

    def hitl_resolver(req: EgressRequest, rule: EgressRule) -> bool:
        return approvals.get((req.tool_name, req.url), False)

    dispatcher = ToolDispatcher(policy, auditor, hitl_resolver)

    results: list[tuple[str, str, object]] = []

    # 1. ALLOW: matches pypi rule.
    try:
        r = dispatcher.call(
            "pip_install",
            "https://files.pythonhosted.org/packages/.../requests.whl",
        )
        results.append(("pypi", "ALLOWED", r["status"]))
    except EgressDenied as e:
        results.append(("pypi", "DENIED", str(e)))

    # 2. HITL approved: matches *.example.io and is in approvals map.
    try:
        r = dispatcher.call("researcher", "https://data.example.io/v1/search")
        results.append(("researcher", "HITL-ALLOWED", r["status"]))
    except EgressDenied as e:
        results.append(("researcher", "DENIED", str(e)))

    # 3. HITL rejected: matches *.example.io but not approved.
    try:
        dispatcher.call("researcher", "https://other.example.io/v1/fetch")
        results.append(("researcher-other", "HITL-ALLOWED", None))
    except EgressDenied as e:
        results.append(("researcher-other", "HITL-REJECTED", str(e)))

    # 4. Explicit DENY: telemetry rule.
    try:
        dispatcher.call("metrics", "https://telemetry.megacorp.com/batch")
        results.append(("telemetry", "ALLOWED", None))
    except EgressDenied as e:
        results.append(("telemetry", "DENIED", str(e)))

    # 5. Default DENY: no rule matches.
    try:
        dispatcher.call("browser", "https://evil.com/exfil")
        results.append(("browser", "ALLOWED", None))
    except EgressDenied as e:
        results.append(("browser", "DEFAULT-DENIED", str(e)))

    print("=== Egress Results ===")
    for name, outcome, detail in results:
        print(f"{name:20} {outcome:15} {detail}")

    print("\n=== Audit Log ===")
    print(auditor.to_json())

    # Sanity checks
    assert results[0][1] == "ALLOWED"
    assert results[1][1] == "HITL-ALLOWED"
    assert results[2][1] == "HITL-REJECTED"
    assert results[3][1] == "DENIED"
    assert results[4][1] == "DEFAULT-DENIED"
    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Code Walkthrough

### `Action`, `EgressRule`, and `EgressRequest`

`Action` is the tristate the policy returns. `EgressRule` is immutable and normalized in `__post_init__` so domain and scheme matching is case-insensitive. `EgressRequest` carries everything the dispatcher knows about the call: the URL, the HTTP method, the tool that made it, and any metadata.

### `EgressPolicy.evaluate`

The policy parses the URL, then walks the rule list in order. The first matching rule wins. If nothing matches, the default action is returned. In production LRA the default is always `DENY`.

The `_matches` helper checks scheme, port, domain, and IP. Domain matching supports exact, suffix, and `*.` wildcard patterns. IP matching supports CIDR blocks. Empty fields in a rule mean "do not filter on this dimension," which lets you write broad or narrow rules.

### `EgressAuditor`

Every decision — allow, deny, or HITL — is recorded with a timestamp, the request, the matched rule, whether the call was actually executed, and a human-readable detail. In the full LRA system this audit stream is written to the **Mission Anchor** from Chapter 16 so it survives crashes and can be replayed during incident review.

### `ToolDispatcher.call`

This is where the policy meets the tool loop from Chapter 4. Before any network call runs:

1. Build an `EgressRequest`.
2. Ask the policy for an action.
3. If `DENY`, log and raise `EgressDenied`.
4. If `HITL`, invoke the durable HITL resolver. If rejected, raise. If approved, execute and log.
5. If `ALLOW`, execute and log.

The `transport` parameter is a seam for testing and for sandboxing. In production it is replaced by the real sandboxed client — local subprocess, Docker container, or e2b sandbox from Chapter 5.

### The HITL stub

The demo uses a deterministic `hitl_resolver` so it can run without a human. In the real system, the resolver sends a durable signal to the Temporal workflow from Chapter 12 and waits for `approve`, `steer`, or `reject`. The request is frozen in the workflow history, so a crash mid-approval resumes exactly at the same gate.

---

## Integration with the Rest of LRA

| Layer | Role in egress control |
|---|---|
| **Tool Dispatcher** (Chapter 5) | Intercepts every tool call and builds the `EgressRequest`. |
| **Sandbox** (Chapter 5) | Actually performs the network call inside an isolated environment. |
| **HITL gates** (Chapter 22) | Handles `Action.HITL` with durable human approval. |
| **Mission Anchor** (Chapter 16) | Stores the audit log durably across reboots. |
| **Budget Governor** (Chapter 20) | Uses egress audit events to detect unexpected spend or repeated blocked attempts. |
| **Think → Act → Verify → Checkpoint** (Chapter 4) | The dispatcher is invoked in the **Act** phase; a denial becomes a verifiable failure. |

---

## Hands-On Exercise

1. Open `lra-demo/ch23_egress_policies.py`.
2. Add a new `EgressRule` named `github-raw` that allows `GET` requests to `raw.githubusercontent.com` on port `443` over `https`.
3. Add a test call to `dispatcher.call("reader", "https://raw.githubusercontent.com/FareedKhan-dev/long-running-agent/main/README.md")` and assert it is `ALLOWED`.
4. Add a CIDR-based rule that allows any host in `192.168.0.0/16` on port `443`, then test with `https://192.168.1.50/health`.
5. Run the script and confirm that:
   - allowed calls return status `200`,
   - denied calls raise `EgressDenied`,
   - the audit log contains one entry per call,
   - the default-deny case still blocks `https://evil.com/exfil`.

6. (Stretch) Replace the mock transport with `httpx` and point it at `https://httpbin.org/get` under an allow rule. Verify that the policy still blocks the call if you remove the rule.

---

> **Key Takeaway:** In a long-running agent, the network is an irreversible action. Default-deny egress turns "please don't leak data" from a prompt into an engineering guarantee: every outbound request is inspected, matched against explicit policy, and either allowed, escalated to a human, or blocked — and every decision is journaled.

---

## Next Chapter Teaser

**Chapter 24: Capturing Failure Traces** — When a tool call, verification, or egress decision fails, the model needs more than an error message. We will build a failure-trace system that captures the full context of a failure, stores it durably, and feeds it back into the next cycle so the agent learns from mistakes instead of repeating them.