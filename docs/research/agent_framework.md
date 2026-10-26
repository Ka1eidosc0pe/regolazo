# Microsoft Agent Framework (Python) on Azure OpenAI - developer reference

Researched 2026-10-26. Verification legend: **[V]** = executed locally against the installed
packages (no Azure calls; the OpenAI SDK call was stubbed) or read in the installed source;
**[D]** = taken from official Microsoft Learn docs; **[U]** = not verified, needs a live check.

Sources: PyPI; installed source of `agent-framework-core/openai/orchestrations`; Learn pages
`agents/providers/openai`, `agents/providers/azure-openai`, `agents/observability`;
https://github.com/microsoft/agent-framework. **Beware: most blog posts and pre-2026 samples are
stale** (`ChatAgent`, `AzureOpenAIChatClient`, `AgentThread`, `get_new_thread`, `ai_function`,
`model_id`/`deployment_name` no longer exist or were renamed) [V][D].

## 1. Install

| Package (PyPI) | Version | Role |
|---|---|---|
| `agent-framework-core` | **1.23.0** (2026-10-22) | `Agent`, `@tool`, workflows, observability, sessions |
| `agent-framework-openai` | **1.17.0** | `OpenAIChatCompletionClient`, `OpenAIChatClient` (Responses); also does Azure OpenAI |
| `agent-framework-orchestrations` | **1.4.0** | Sequential / Concurrent / GroupChat / Handoff / Magentic builders |
| `azure-monitor-opentelemetry` | 1.8.11 | App Insights export (docs want `>=1.8.10,<2`) |

All require Python `>=3.10`; tested here on 3.12.9 [V]. Imports are `agent_framework`,
`agent_framework.openai`, `agent_framework.orchestrations`, `agent_framework.observability`.

```bash
uv pip install agent-framework-core agent-framework-openai agent-framework-orchestrations azure-monitor-opentelemetry
```

Do NOT depend on the umbrella `agent-framework` (1.23.0): it pins `agent-framework-core[all]`,
which drags in ~30 connectors, several still pre-release; `uv` refuses to resolve it without
`--prerelease=allow` [V]. The `agent-framework-azure-ai` package is stale (1.0.0rc6) - ignore.
`openai` is constrained `>=2.25,<4`; the resolver picked **openai 3.39.0**, which uses `httpx2`
(not `httpx`) as its transport [V]. If you write tests that mock HTTP, do not assume `httpx`.

## 2. Chat agent on Azure OpenAI with an API key

Class: `OpenAIChatCompletionClient` (Chat Completions API). Docs recommend `OpenAIChatClient`
(Responses API) for new code, but Chat Completions is fully supported and is what we use [D].

Routing rules (source [V], docs [D]):
- Azure is selected when `azure_endpoint=` or `credential=` is passed explicitly. Otherwise, if
  `OPENAI_API_KEY` is set the client silently goes to **api.openai.com**. Always pass
  `azure_endpoint` explicitly.
- Env vars read for Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_CHAT_COMPLETION_MODEL` (fallback `AZURE_OPENAI_MODEL`),
  `AZURE_OPENAI_BASE_URL`. Our key variable is **`AZURE_AI_KEY`**, which the framework does not read,
  so pass `api_key=` ourselves. Default api-version if unset: `2024-12-01-preview`.
- `model=` is the **deployment name**; the SDK URL becomes `.../openai/deployments/<model>/chat/completions`.
  One client instance per deployment; an Agent takes exactly one client.
- Passing `credential=` (azure-identity) is the "preferred" Azure auth surface per docs, but `api_key=` works.

```python
import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

def make_client(deployment: str) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=deployment,                                   # "gpt-5-mini" | "gpt-4.1-mini"
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_AI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

storyteller = Agent(client=make_client("gpt-5-mini"), name="storyteller",
                    instructions="You write vivid, factual match stories.",
                    default_options={"max_tokens": 4000})
checker = Agent(client=make_client("gpt-4.1-mini"), name="fact_checker",
                instructions="Verify every claim against the provided facts.")
result = await storyteller.run("Tell the story of the 2005 final.")   # AgentResponse
print(result.text, result.usage_details)   # {'input_token_count':..,'output_token_count':..,'total_token_count':..}
```

Endpoint form: pass the resource root (`https://<res>.cognitiveservices.azure.com/` or
`https://<res>.openai.azure.com`), **not** a URL ending in `/openai/v1` (use `base_url=` for that;
the client even prints a hint if you get this wrong) [V]. Construction with a `.cognitiveservices.azure.com`
endpoint and API key works offline [V]; a real call against our Foundry/AIServices resource,
including a `.services.ai.azure.com` host, is **[U]**.

Per-call overrides: `agent.run(msg, options={...}, tools=[...])`. Constructor `default_options` are merged
under run-time `options`. Custom timeout/retries: build the SDK client yourself and inject it (the
public ctor has no `timeout` argument):

```python
from openai import AsyncAzureOpenAI
ac = AsyncAzureOpenAI(azure_endpoint=..., api_key=..., api_version=..., timeout=60.0, max_retries=5)
client = OpenAIChatCompletionClient(model="gpt-5-mini", async_client=ac)   # [V] base_url=.../openai/, retries=5
```

Docs note: a single `OpenAIChatCompletionClient` is **not** guaranteed safe for concurrent calls (the Responses
client is); use one client + one `Agent` (+ one `AgentSession`) per concurrent run [D].

## 3. Function tools

`@tool` (from `agent_framework`) builds the JSON schema from type hints; descriptions come from the
docstring and `Annotated[T, "desc"]` or `pydantic.Field(description=...)` [V]. Sync and async functions both work.
Pass `tools=fn` or `tools=[fn1, fn2]` to `Agent(...)` or `agent.run(...)`.

```python
from typing import Annotated
from agent_framework import tool

@tool                      # or @tool(name="get_stat", description="...", approval_mode="never_require", max_invocations=5)
def get_stat(player: Annotated[str, "Player name"], season: Annotated[str, "e.g. 2024-25"] = "2024-25") -> dict:
    """Look up a player's goals for a season."""
    return {"player": player, "goals": 27}

agent = Agent(client=make_client("gpt-5-mini"), name="analyst", tools=[get_stat])
```

Verified request/flow (`tool_choice:"auto"`, schema sent as `tools=[{"type":"function",...}]`) [V]:
1. model returns `tool_calls`; 2. framework validates args with Pydantic, runs the function (parallel calls run
concurrently by default); 3. result is serialized (dict/BaseModel -> JSON string) into a `role:"tool"` message;
4. model is called again; loop ends on a normal text answer. `response.messages` contains the whole trail
(`function_call`, `function_result`, `text` contents). A raised exception is returned to the model as an error
result (details hidden unless `include_detailed_errors`).

**Limiting iterations** - settings live on the *client*, via `FunctionInvocationConfiguration` [V]:

```python
client = make_client("gpt-5-mini")
client.function_invocation_configuration["max_iterations"] = 4        # LLM round-trips (default 40)
client.function_invocation_configuration["max_function_calls"] = 8    # total tool executions (default None)
client.function_invocation_configuration["max_duration_seconds"] = 30.0
# also: max_consecutive_errors_per_request, allow_concurrent_invocation, terminate_on_unknown_calls
```
When a limit trips, the framework sends the last request with `tool_choice:"none"` and returns the model's
text (in my stub test it returned the literal fallback "Function invocation limit reached before a final
answer could be produced." because the stub kept asking for tools) [V]. `@tool(max_invocations=N)` is a
*lifetime* cap per tool object, not per request. `approval_mode="always_require"` pauses for human approval.

## 4. Structured output

Pass a Pydantic class (or a JSON-schema dict) as `response_format`; read `response.value` [V].

```python
from pydantic import BaseModel, Field

class Verdict(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)

checker = Agent(client=make_client("gpt-4.1-mini"), name="fact_checker", instructions="...",
                default_options={"response_format": Verdict})        # or per call: options={"response_format": Verdict}
v = (await checker.run(draft)).value          # -> Verdict instance (None if parsing failed); raw JSON in .text
```
Wire format sent: `response_format={"type":"json_schema","json_schema":{"name":"Verdict","strict":true,"schema":{...,
"additionalProperties":false}}}` [V]. `strict` requires every field in `required` (the framework did this even for the
defaulted `issues`). Needs an Azure api-version that supports `json_schema` (>= 2024-08-01-preview) and a model that
supports it - gpt-4.1-mini and gpt-5-mini both should, **[U]** live. Combining `response_format` with tools in one agent
works at the API level but gets flaky on small models; keep the checker tool-free.

## 5. Multi-agent orchestration

Agents are workflow nodes directly (`WorkflowBuilder` auto-wraps them in `AgentExecutor`). Inside workflows, agents are
**always run in streaming mode** under the hood [V], so any test stub must support `stream=True`. Run results:
`result = await wf.run(input)`; `result.get_outputs()`; or `async for ev in wf.run(input, stream=True)` where
`ev.type` in `output | intermediate | executor_invoked | executor_completed | superstep_* | status | ...` and `ev.data`.

Prebuilt builders (`from agent_framework.orchestrations import ...`, all keyword-only ctor, `.build()` returns `Workflow`) [V]:

```python
SequentialBuilder(participants=[a, b, c]).build()              # a -> b -> c, full conversation passed along
ConcurrentBuilder(participants=[a, b]).build()                 # fan-out/fan-in; optional .with_aggregator(fn)
GroupChatBuilder(participants=[a, b], selection_func=pick, termination_condition=done,
                 max_rounds=6).build()                         # star topology; or orchestrator_agent=Agent(...)
HandoffBuilder(participants=[triage, billing], termination_condition=...).with_start_agent(triage) \
    .add_handoff(triage, [billing]).add_handoff(billing, [triage]).build()
# Handoff: every participant Agent must be created with require_per_service_call_history_persistence=True,
# otherwise build() raises ValueError. Handoff agents get auto-injected handoff tools (local tool calling required).
```
Output shape (v1.4.0, observed): Sequential returns one `AgentResponse` for the **last** participant only; Concurrent returns one
`AgentResponse` whose `.messages` has one message per participant; GroupChat returns a final `AgentResponse` containing only the
orchestrator's "terminated"/"max_rounds" notice. To see the actual drafts use `output_from="all"` (or `intermediate_output_from=`)
and stream events. Magentic and the experimental `@workflow/@step` functional API (`FunctionalWorkflow`, marked experimental) also exist - skip for now.

Low-level graph API: `WorkflowBuilder(start_executor=x, max_iterations=N).add_edge(a, b, condition=...)`, `.add_fan_out_edges`,
`.add_fan_in_edges`, `.add_switch_case_edge_group`, `.add_chain([...])`. Custom node = `Executor` subclass with `@handler`
methods typed `WorkflowContext[SendType, OutputType]`; `ctx.send_message(x)` routes along edges, `ctx.yield_output(x)` emits a
workflow output. `max_iterations` (default 100 supersteps) is a hard stop on graph cycles (a `WorkflowConvergenceException` class exists for this; exact failure mode not exercised).

### Bounded Storyteller <-> Fact-Checker loop (max 3)

Recommended: **plain Python loop** - simplest, trivially bounded, easy to test and trace (each `agent.run` still emits spans) [V]:

```python
async def tell_story(brief: str, max_iter: int = 3):
    prompt = brief
    for i in range(1, max_iter + 1):
        draft = (await storyteller.run(prompt)).text
        v = (await checker.run(f"Facts:\n{facts}\n\nDraft:\n{draft}")).value   # Verdict
        if v.approved:
            return draft, i, True
        prompt = f"{brief}\n\nRevise the draft, fixing: {v.issues}\n\nDraft:\n{draft}"
    return draft, max_iter, False            # ship last draft, flagged unapproved
```

If a visible workflow graph is wanted (cycle with counter carried in the message) - also [V] (returned `iterations=2, approved=True`
when the checker rejected once, and `iterations=3, approved=False` when it always rejected):

```python
from dataclasses import dataclass, field
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

@dataclass
class Task:  brief: str; issues: list[str] = field(default_factory=list); prior: str = ""; iteration: int = 1
@dataclass
class Draft: text: str; iteration: int; brief: str
@dataclass
class Result: text: str; iterations: int; approved: bool; issues: list[str]

class StorytellerExec(Executor):
    def __init__(self, agent): super().__init__(id="storyteller"); self.agent = agent
    @handler
    async def write(self, t: Task, ctx: WorkflowContext[Draft]) -> None:
        p = t.brief if not t.issues else f"{t.brief}\nFix: {t.issues}\n{t.prior}"
        await ctx.send_message(Draft((await self.agent.run(p)).text, t.iteration, t.brief))

class CheckerExec(Executor):
    def __init__(self, agent, max_iter=3): super().__init__(id="checker"); self.agent, self.max = agent, max_iter
    @handler
    async def check(self, d: Draft, ctx: WorkflowContext[Task, Result]) -> None:
        v = (await self.agent.run(d.text, options={"response_format": Verdict})).value
        if v.approved or d.iteration >= self.max:
            await ctx.yield_output(Result(d.text, d.iteration, v.approved, v.issues))
        else:
            await ctx.send_message(Task(d.brief, v.issues, d.text, d.iteration + 1))

s, c = StorytellerExec(storyteller), CheckerExec(checker)
wf = WorkflowBuilder(start_executor=s, max_iterations=20).add_edge(s, c).add_edge(c, s).build()
out = (await wf.run(Task("Write the story"))).get_outputs()   # [Result(...)]
```
GroupChat variant (`selection_func=lambda st: "storyteller" if st.current_round % 2 == 0 else "fact_checker"`,
`termination_condition=lambda conv: conv[-1].author_name == "fact_checker" and "approved" in conv[-1].text`,
`max_rounds=6`) also terminates correctly and logs "reached max_rounds" [V], but you have to parse JSON verdicts out of text
and dig drafts out of events; prefer the two options above.

## 6. Sessions (state), async, streaming

- Everything is async: `await agent.run(...)`. Concurrency: `await asyncio.gather(a1.run(..), a2.run(..))` with separate
  clients/agents [V].
- Stateless by default. Multi-turn = `session = agent.create_session()` then `agent.run(msg, session=session)`; the default
  in-memory history provider replays prior messages (verified: 2nd request had system,user,assistant,user) [V].
  `AgentSession` (formerly `AgentThread`) serialises with `session.to_dict()` / `AgentSession.from_dict(d)` [V]; persistence
  providers exist (`FileHistoryProvider`, Cosmos, Redis, ...). For one-shot calls inside a loop, omit `session` and pass the
  full prompt each time (cheaper, deterministic).
- Streaming: `agent.run(msg, stream=True)` returns a `ResponseStream`; `async for u in stream: u.text` then
  `final = await stream.get_final_response()` (full `AgentResponse`, includes usage) [V]. Chat Completions reports
  usage in streaming via `stream_options.include_usage` (framework sets it).
- `agent.as_tool(name=..., description=...)` turns an agent into a `FunctionTool` for a supervisor-agent pattern [V].
  `workflow.as_agent()` exists [V] to expose a workflow as an agent.

## 7. Observability (OpenTelemetry -> Application Insights)

Instrumentation is **on by default** in code paths (`ENABLE_INSTRUMENTATION=false` to disable); you only need to install
providers/exporters. Sensitive payloads (prompts, responses, tool args/results) are **off** unless `ENABLE_SENSITIVE_DATA=true`
or `enable_instrumentation(enable_sensitive_data=True)` [D][V]. Call setup once at startup, before the first agent run.

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework.observability import create_resource, enable_instrumentation

configure_azure_monitor(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
    resource=create_resource(),            # honours OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES
    enable_live_metrics=False,
)
enable_instrumentation(enable_sensitive_data=False)   # optional if env vars already set; True only in dev
```
[V] imports + construction + one agent run with a bogus local ingestion endpoint (no errors other than the expected export
warning); real ingestion into App Insights is **[U]**. Alternatives: `configure_otel_providers()` (reads
`OTEL_EXPORTER_OTLP_*`, `ENABLE_CONSOLE_EXPORTERS=true` for stdout; needs `opentelemetry-exporter-otlp-proto-*` installed),
or the Aspire dashboard via OTLP `http://localhost:4317`. Agent Framework does not load `.env` itself (`load_dotenv()` yourself).

Spans observed (names / key attributes) [V]:
- `invoke_agent <agent_name>`: `gen_ai.operation.name`, `gen_ai.agent.id/name/description`, `gen_ai.request.model`,
  `gen_ai.provider.name=microsoft.agent_framework`, `gen_ai.usage.input_tokens/output_tokens` (summed across the tool loop),
  `gen_ai.tool.definitions`, `gen_ai.response.finish_reasons`.
- `chat <deployment>` (child): `gen_ai.request.model`, `gen_ai.provider.name=azure.ai.openai`, `gen_ai.response.id`,
  `gen_ai.response.model`, `gen_ai.usage.*`, `server.address`. (Message content as attributes/events only with sensitive data on.)
- `execute_tool <fn>`: `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.description`, `agent_framework.function.invocation.duration`.
- Workflows: `workflow.build`, `workflow.run` (`workflow.id/name`), `executor.process <id>` (`executor.id`, `executor.type`),
  `edge_group.process`, `message.send`. Name your builder (`WorkflowBuilder(name="story_loop", ...)`) and agents for readable traces.
- Metrics: `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `agent_framework.function.invocation.duration`.
- Wrap a whole request in your own parent span: `with agent_framework.observability.get_tracer().start_as_current_span("match_story"):`.
In App Insights these land in `dependencies`/`requests` (span name in `name`, attributes in `customDimensions`); Foundry's
Tracing UI additionally requires registering the agent id (docs: "register custom agent") - optional **[U]**.

## 8. Gotchas

**gpt-5-mini is a reasoning model** (behaviour from OpenAI/Azure model docs, **[U]** against our deployment):
- Framework behaviour [V]: `options["max_tokens"]` is translated to `max_completion_tokens` (never sends `max_tokens`).
  Other keys are forwarded **unfiltered**: `temperature`, `top_p`, `seed`, `reasoning_effort`, `verbosity`
  (`"low"|"medium"|"high"`, in `OpenAIChatCompletionOptions`) all reached the SDK call. So the framework will happily send
  `temperature=0.2` and Azure will reject it with a 400 on gpt-5-mini (only default temperature=1 is supported). **Do not set
  temperature/top_p/presence/frequency penalties on gpt-5-mini**; set them only on gpt-4.1-mini. Have two option presets.
- `max_completion_tokens` includes hidden reasoning tokens. Too small a cap yields empty text with `finish_reason="length"`.
  Use >= 2000-4000 for short outputs, or lower `reasoning_effort` (`"minimal"|"low"|"medium"|"high"`) for speed. Reasoning-token counts
  appear in `usage_details` (`completion/reasoning_tokens`) [V in source].
- The system prompt goes out as role `system` (the client has `instruction_role=` if a model needs `developer`) [V].
- Requires a recent `api_version` (e.g. `2025-04-01-preview` or newer) for gpt-5-family params [U]; keep it in
  `AZURE_OPENAI_API_VERSION`. Azure v1 route (`base_url=".../openai/v1/"`) needs no api-version [D].

**Rate limits, retries, timeouts:**
- No retry logic in the framework (grep of core + openai client: none). Retries come from the OpenAI SDK: default
  `max_retries=2` with backoff and `Retry-After` on 408/409/429/5xx, default timeout 600 s (connect 5 s) [V in openai 3.39].
  For a demo with many parallel agents raise `max_retries` (5-6) and lower `timeout` (30-60 s) via the injected `AsyncAzureOpenAI`.
- Errors surface as `agent_framework.exceptions.ChatClientException`; the SDK error (`openai.RateLimitError`, etc.) is
  `exc.__cause__` [V]. Content-filter hits raise `OpenAIContentFilterException`. Wrap loop iterations in try/except and degrade
  gracefully (ship the last good draft).
- Limit concurrency with an `asyncio.Semaphore` around `agent.run`; Azure TPM/RPM quotas are per deployment.
- Tool loop cost guard: set `max_iterations`/`max_function_calls` (section 3). Workflow guard: `max_iterations`, `max_rounds`.

**Env/key gotchas:** `OPENAI_API_KEY` in the environment hijacks routing to OpenAI unless `azure_endpoint=` is passed;
`.env` is not auto-loaded; `AZURE_AI_KEY` is not a framework variable (pass `api_key=`). Never log the key; `SecretString` is used internally.

**Deterministic tests / record-replay:** there is **no built-in LLM record/replay or cassette support** in the framework (searched
source). Patterns, in order of preference:
1. *Stub the SDK call* (what I used to verify everything above, works for tool loops, structured output, sessions, workflows; ~25 lines):
   `client.client.chat.completions.create = fake_create` where `fake_create(**kw)` returns
   `openai.types.chat.ChatCompletion.model_validate({...})` (non-stream) or an async-context-manager yielding
   `ChatCompletionChunk`s when `kw["stream"]` is true (needed when agents run inside workflows). `kw` is the exact request
   (messages, `max_completion_tokens`, `response_format`, tools), so tests can assert on prompts.
2. *Record/replay at the `agent.run` boundary*: wrap calls with a JSON cache keyed by `sha256([agent.name, prompt, options])`;
   modes `record` (call live, store `.text`) and `replay` (raise on miss). Re-parse structured output with
   `Verdict.model_validate_json(text)`. Verified as a ~12-line helper [V]. Commit the JSON fixtures; make live-vs-replay an env switch.
3. Subclass `BaseChatClient` (implement `_inner_get_response(self, *, messages, stream, options, **kw)`) for a fully fake client, or use
   `@chat_middleware` and set `context.result` to short-circuit (more fiddly for streaming).
4. For eval-style checks the core package ships `LocalEvaluator`, `evaluate_agent`, `keyword_check`, `tool_called_check` (not exercised here).
Also keep the loop in plain Python (section 5) so tests do not need a workflow runtime.

**Other:** `Agent` instances are cheap; build per request if you need isolation. `Message(role=..., contents=[...])` for manual messages;
`agent.run` accepts `str | Message | list[Message]`. Class names that changed: `OpenAIChatClient` is now **Responses**, and the
old chat-completions client is `OpenAIChatCompletionClient`. Pre-release connectors (`1.0.0bYYMMDD`) churn weekly; pin
`==` for core/openai/orchestrations in `pyproject.toml`.

## Not verified (needs one live smoke test)
1. Real calls to our Foundry/AIServices endpoint host with `AsyncAzureOpenAI` + `api_key` for `gpt-5-mini` and `gpt-4.1-mini`,
   including the required minimum `api_version`.
2. That Azure rejects `temperature` on gpt-5-mini and the exact `max_completion_tokens` sizing/`reasoning_effort` accepted values.
3. `json_schema` strict structured output on both deployments with our api-version.
4. Actual telemetry arrival in Application Insights (only constructed against a dummy endpoint) and Foundry Tracing UI registration.
5. Streaming behaviour against real Azure (verified only with a stubbed chunk stream).

## Live verification (2026-10-26, our Azure resource)

Run by the lead engineer with `agent-framework-core==1.23.0`, `agent-framework-openai==1.17.0` against
`regolazo-ai-j76s9`, using `OpenAIChatCompletionClient(model=<deployment>, azure_endpoint=AZURE_OPENAI_ENDPOINT,
api_key=AZURE_AI_KEY, api_version=AZURE_OPENAI_API_VERSION)`:

| Check | gpt-5-mini | gpt-4.1-mini |
|---|---|---|
| `@tool` function call, number taken from the tool | works (5.2 s) | works (1.5 s) |
| `response_format=<Pydantic model>` → `response.value` | works (2.8 s) | works (0.9 s) |
| `default_options={"max_tokens": 2000}` (sent as `max_completion_tokens`) | accepted | accepted |
| `temperature=0.2` | **rejected** (`ChatClientException`) | not tested |

Implications for Step 8: never pass `temperature` to gpt-5-mini; use gpt-4.1-mini for latency-critical live
hooks (~1 s) and gpt-5-mini for narratives and fact-checking (~3–5 s). Tools must return display names, not
player ids (both models echoed "H09" when the tool only exposed ids).
