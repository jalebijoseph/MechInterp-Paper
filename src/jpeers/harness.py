"""Multi-agent conversation harness with per-turn J-space recording.

Each :class:`Agent` is bound to an endpoint (local GPU, Azure, Vertex, ...). A
:class:`Conversation` keeps one shared transcript and renders it from each agent's
point of view: the agent's own turns become ``assistant`` messages, everybody else's
become ``user`` messages prefixed with the speaker's name. Every turn is one
:class:`~jpeers.schemas.GenerateRequest`; the returned J-space trace is saved next to
the transcript by :class:`RunWriter`.

Span labels recorded on every turn (usable in metrics and in intervention selectors):

    speaker:<name>   every message by <name> in this agent's view
    peer_last        the most recent message by someone other than this agent
    self             this agent's own earlier messages
    msg:<i>, role:<r>  per rendered chat message (from the server)
    prompt           the whole prompt
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Literal

from jpeers.client import EndpointPool
from jpeers.schemas import GenerateRequest, InterventionSpec, RecordSpec, SpanQuery
from jpeers.trace import JSpaceTrace

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    name: str
    endpoint: str
    system_prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.0
    seed: int = 0
    watch: list[str] = field(default_factory=list)
    interventions: list[InterventionSpec] = field(default_factory=list)
    record: bool = True


@dataclass
class Turn:
    speaker: str
    content: str
    private_to: list[str] | None = None  # None = visible to everyone


@dataclass
class TurnRecord:
    index: int
    agent: str
    endpoint: str
    model: str
    text: str
    prompt_len: int
    n_generated: int
    timings: dict
    interventions: list[dict]
    trace_file: str | None
    tag: str = ""
    messages: list[dict] | None = None
    extra: dict = field(default_factory=dict)


class RunWriter:
    """``runs/<experiment>/<run_id>/{config.json, turns.jsonl, traces/*.npz}``."""

    def __init__(self, root: str | Path, experiment: str, run_id: str | None = None,
                 config: dict | None = None, save_messages: bool = True) -> None:
        run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / experiment / run_id
        n = 2
        while (self.dir / "turns.jsonl").exists():  # never mix two runs in one directory
            self.dir = Path(root) / experiment / f"{run_id}-{n}"
            n += 1
        (self.dir / "traces").mkdir(parents=True, exist_ok=True)
        self.save_messages = save_messages
        self._n = 0
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str),
                                                  encoding="utf-8")

    def write(self, rec: TurnRecord, trace: JSpaceTrace | None) -> TurnRecord:
        rec.index = self._n
        if trace is not None:
            rec.trace_file = f"traces/{self._n:05d}.npz"
            trace.save(str(self.dir / rec.trace_file))
        if not self.save_messages:
            rec.messages = None
        with open(self.dir / "turns.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), default=str) + "\n")
        self._n += 1
        return rec


def iter_turns(run_dir: str | Path, *, load_traces: bool = True
               ) -> Iterator[tuple[dict, JSpaceTrace | None]]:
    run_dir = Path(run_dir)
    with open(run_dir / "turns.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            trace = None
            if load_traces and rec.get("trace_file"):
                trace = JSpaceTrace.load(str(run_dir / rec["trace_file"]))
            yield rec, trace


class Conversation:
    """A shared transcript and the agents that take turns on it.

    Args:
        agents: Participants.
        pool: Endpoint pool used to resolve ``agent.endpoint``.
        writer: Where turns are saved (optional).
        peer_framing: ``"agent"`` shows other speakers as named agents; ``"document"``
            shows their text as anonymous shared notes (the no-peer control).
        record_defaults: Base :class:`RecordSpec`; each agent's ``watch`` is appended.
        tag: Free-form label stored on every turn (condition name, task id, ...).
    """

    def __init__(self, agents: list[Agent], pool: EndpointPool, *, writer: RunWriter | None = None,
                 peer_framing: Literal["agent", "document"] = "agent",
                 record_defaults: RecordSpec | None = None, tag: str = "",
                 extra: dict | None = None) -> None:
        self.agents = {a.name: a for a in agents}
        self.pool = pool
        self.writer = writer
        self.peer_framing = peer_framing
        self.record_defaults = record_defaults or RecordSpec()
        self.tag = tag
        self.extra = extra or {}
        self.transcript: list[Turn] = []
        self.records: list[TurnRecord] = []
        self.traces: list[JSpaceTrace | None] = []

    # ------------------------------------------------------------------ transcript

    def say(self, speaker: str, content: str, private_to: list[str] | None = None) -> None:
        """Inject a message (task brief, coordinator note, scripted peer message)."""
        self.transcript.append(Turn(speaker, content, private_to))

    def _label(self, speaker: str) -> str:
        if self.peer_framing == "document" and speaker in self.agents:
            return "Shared notes"
        return speaker

    def view_for(self, agent: Agent) -> tuple[list[dict], list[SpanQuery]]:
        msgs: list[dict] = [{"role": "system", "content": agent.system_prompt}]
        queries: list[SpanQuery] = []
        last_peer: str | None = None
        for turn in self.transcript:
            if turn.private_to is not None and agent.name not in turn.private_to \
                    and turn.speaker != agent.name:
                continue
            if turn.speaker == agent.name:
                msgs.append({"role": "assistant", "content": turn.content})
                queries.append(SpanQuery(label="self", text=turn.content, occurrence="last"))
            else:
                msgs.append({"role": "user",
                             "content": f"[{self._label(turn.speaker)}]: {turn.content}"})
                queries.append(SpanQuery(label=f"speaker:{turn.speaker}", text=turn.content,
                                         occurrence="last"))
                if turn.speaker in self.agents:
                    last_peer = turn.content
        if last_peer is not None:
            queries.append(SpanQuery(label="peer_last", text=last_peer, occurrence="last"))
        return _merge_same_role(msgs), queries

    # ------------------------------------------------------------------ turns

    def step(self, name: str, *, nudge: str | None = None, prefill: str = "",
             interventions: list[InterventionSpec] | None = None, max_new_tokens: int | None = None,
             record: RecordSpec | None = None, append: bool = True, extra: dict | None = None
             ) -> TurnRecord:
        """Let agent ``name`` speak once.

        Args:
            nudge: Extra user message for this turn only (e.g. "Assign the task now.").
            prefill: Teacher-forced start of the reply (e.g. ``"Assign to:"``).
            interventions: Override the agent's interventions for this turn.
            append: Add the reply to the shared transcript.
        """
        agent = self.agents[name]
        messages, queries = self.view_for(agent)
        if nudge:
            messages = _merge_same_role(messages + [{"role": "user", "content": nudge}])
        rec_spec = (record or self.record_defaults).model_copy(deep=True)
        rec_spec.enabled = rec_spec.enabled and agent.record
        rec_spec.watch = list(dict.fromkeys([*rec_spec.watch, *agent.watch]))
        ivs = agent.interventions if interventions is None else interventions
        req = GenerateRequest(messages=messages, assistant_prefill=prefill,
                              max_new_tokens=max_new_tokens or agent.max_new_tokens,
                              temperature=agent.temperature, seed=agent.seed,
                              record=rec_spec, interventions=ivs, spans=queries)
        endpoint = self.pool.get(agent.endpoint)
        resp = endpoint.generate(req)
        text = (prefill + resp.text).strip()
        trace = JSpaceTrace.from_response(resp)
        rec = TurnRecord(index=len(self.records), agent=name, endpoint=agent.endpoint,
                         model=resp.model, text=text, prompt_len=resp.prompt_len,
                         n_generated=resp.n_generated, timings=resp.timings,
                         interventions=[iv.model_dump() for iv in ivs], trace_file=None,
                         tag=self.tag, messages=messages, extra={**self.extra, **(extra or {})})
        if append:
            self.transcript.append(Turn(name, text))
        if self.writer is not None:
            self.writer.write(rec, trace)
        self.records.append(rec)
        self.traces.append(trace)
        logger.info("[%s] %s: %s", self.tag, name, text[:160].replace("\n", " "))
        return rec

    def run(self, order: list[str], rounds: int = 1, **step_kw) -> list[TurnRecord]:
        out = []
        for _ in range(rounds):
            for name in order:
                out.append(self.step(name, **step_kw))
        return out


def _merge_same_role(msgs: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role (strict templates require
    user/assistant alternation) and make sure the first non-system turn is a user turn."""
    out: list[dict] = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    first = 1 if out and out[0]["role"] == "system" else 0
    if len(out) > first and out[first]["role"] == "assistant":
        out.insert(first, {"role": "user", "content": "(conversation start)"})
    return out
