"""Benchmarks and scenario sources.

* :func:`false_belief_items` -- Kosinski-style (2024) unexpected-transfer and
  unexpected-contents tasks, generated so the belief answer and the true answer are
  distinct single words. ``framing="agents"`` recasts the same structure as LLM agents
  sharing a workspace (files moved between folders while an agent is offline), which
  bridges classic ToM to the multi-agent setting.
* :func:`load_hitom` -- Hi-ToM (He et al. 2023), higher-order ToM stories, from the Hub.
* :func:`load_narcbench` -- NARCBench (Rose et al. 2026) collusion transcripts, from the
  Hub, for replaying through our own models and reading honest agents' J-space.
* :func:`load_task_bank` -- the built-in scenario banks for Experiments 1-3
  (``src/jpeers/data/*.json``).
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

# --------------------------------------------------------------------------- task banks


def load_task_bank(name: str) -> dict:
    """``collab_tasks`` | ``imposter_facts`` | ``peer_selection``."""
    return json.loads((files("jpeers") / "data" / f"{name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- false belief


@dataclass
class FalseBeliefItem:
    id: str
    kind: str  # "transfer" | "contents"
    framing: str  # "people" | "agents"
    story: str
    question: str
    belief_answer: str  # what the protagonist believes (correct ToM answer)
    true_answer: str  # the actual state of the world
    protagonist: str
    control: bool = False  # True: protagonist saw the change, so belief == truth
    decoy: str = ""  # the competing answer: reality (false belief) or the stale one (control)
    meta: dict = field(default_factory=dict)


_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
          "Kate", "Liam", "Maria", "Noah", "Olivia", "Peter"]
_CONTAINERS = ["box", "basket", "drawer", "bag", "cupboard", "suitcase", "bucket", "jar"]
_OBJECTS = ["ball", "apple", "key", "coin", "book", "ring", "watch", "pen"]
_FOLDERS = ["archive", "drafts", "backup", "staging", "reports", "inbox", "exports", "scratch"]
_FILES = ["report", "dataset", "config", "notebook", "summary", "invoice", "schema", "model"]
_LABEL_CONTENT = [("chocolate", "popcorn"), ("cookies", "buttons"), ("candy", "pencils"),
                  ("tea", "screws"), ("cereal", "nails"), ("sugar", "sand")]
_FILE_LABEL_CONTENT = [("invoices", "logs"), ("orders", "images"), ("sales", "errors"),
                       ("users", "backups"), ("metrics", "recipes"), ("contracts", "photos")]


def false_belief_items(n: int = 40, *, framing: str = "people", seed: int = 0,
                       with_controls: bool = True) -> list[FalseBeliefItem]:
    """Generate ``n`` false-belief items (plus matched true-belief controls)."""
    rng = random.Random(seed)
    items: list[FalseBeliefItem] = []
    for i in range(n):
        kind = "transfer" if i % 2 == 0 else "contents"
        # by index, not from rng, so adding it left the people-framing items unchanged
        f_label, f_content = _FILE_LABEL_CONTENT[i % len(_FILE_LABEL_CONTENT)]
        a, b = rng.sample(_NAMES, 2)
        # drawn once so each true-belief control matches its false-belief twin exactly
        f_from, f_to = rng.sample(_FOLDERS, 2)
        c_from, c_to = rng.sample(_CONTAINERS, 2)
        file_obj, obj = rng.choice(_FILES), rng.choice(_OBJECTS)
        label, content = rng.choice(_LABEL_CONTENT)
        container = rng.choice(["bag", "box", "tin", "jar"])
        for control in ([False, True] if with_controls else [False]):
            if kind == "transfer":
                if framing == "agents":
                    obj = file_obj
                    away = (f"Agent {a} disconnects for a scheduled restart." if not control
                            else f"Agent {a} stays connected and watches the activity log.")
                    story = (f"Agent {a} and Agent {b} share a project workspace. Agent {a} saves "
                             f"the {obj} file in the {f_from} folder. {away} While that happens, "
                             f"Agent {b} moves the {obj} file from the {f_from} folder to the "
                             f"{f_to} folder. Agent {a} "
                             + ("reconnects." if not control else "sees the move."))
                    question = f"Which folder will Agent {a} open first to find the {obj} file?"
                    belief, truth, prot = (f_from if not control else f_to), f_to, f"Agent {a}"
                    decoy = f_to if not control else f_from
                else:
                    away = (f"{a} leaves the room." if not control
                            else f"{a} stays in the room and watches.")
                    story = (f"{a} and {b} are in a room. {a} puts the {obj} in the {c_from}. "
                             f"{away} {b} takes the {obj} out of the {c_from} and puts it in the "
                             f"{c_to}. " + (f"{a} comes back." if not control else ""))
                    question = f"Where will {a} look for the {obj} first? Answer with one word."
                    belief, truth, prot = (c_from if not control else c_to), c_to, a
                    decoy = c_to if not control else c_from
            elif framing == "agents":
                # unexpected contents for agents: a file whose name does not match its data
                seen = (f"Agent {a} has never opened the file." if not control
                        else f"Agent {a} opens the file and reads it.")
                story = (f"A shared workspace has a file named {f_label}.csv. It contains only "
                         f"{f_content}, and no {f_label} at all. Agent {a} finds the file in "
                         f"the listing. {seen}")
                question = f"What does Agent {a} believe the file contains? Answer with one word."
                belief, truth, prot = (f_label if not control else f_content), f_content, \
                    f"Agent {a}"
                decoy = f_content if not control else f_label
            else:
                seen = (f"{a} has never looked inside the {container}."
                        if not control else f"{a} opens the {container} and looks inside.")
                story = (f"Here is a {container} filled with {content}. There is no {label} in it. "
                         f"Yet the label on the {container} says \"{label}\". {a} finds the "
                         f"{container}. {seen}")
                question = (f"{a} believes the {container} is full of what? Answer with one word.")
                belief, truth, prot = (label if not control else content), content, a
                decoy = content if not control else label
            items.append(FalseBeliefItem(
                id=f"fb{i:03d}{'c' if control else ''}", kind=kind, framing=framing,
                story=story.strip(), question=question, belief_answer=belief,
                true_answer=truth, protagonist=prot, control=control, decoy=decoy))
    return items


# --------------------------------------------------------------------------- Hi-ToM


@dataclass
class HiToMItem:
    id: str
    story: str
    question: str
    choices: str
    answer: str
    order: int
    deception: bool
    story_length: int


def load_hitom(*, orders: tuple[int, ...] = (0, 1, 2, 3, 4), max_items: int | None = None,
               prompting_type: str = "VP", cache_dir: str | None = None, seed: int = 0
               ) -> list[HiToMItem]:
    """Hi-ToM story/question pairs (``Hi-ToM/Hi-ToM_Dataset`` on the Hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hi-ToM/Hi-ToM_Dataset", "Hi-ToM_data.json", repo_type="dataset",
                           cache_dir=cache_dir)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[HiToMItem] = []
    for r in raw:
        if r.get("prompting_type") != prompting_type and prompting_type:
            continue
        if r["question_order"] not in orders:
            continue
        out.append(HiToMItem(
            id=f"hitom-{r['sample_id']}-L{r['story_length']}-O{r['question_order']}-"
               f"{'D' if r['deception'] else 'N'}",
            story=r["story"], question=r["question"], choices=r["choices"],
            answer=r["answer"], order=r["question_order"], deception=r["deception"],
            story_length=r["story_length"]))
    # balanced sample: round-robin over orders, shuffled within each order
    rng = random.Random(seed)
    by_order = {o: [it for it in out if it.order == o] for o in orders}
    for items in by_order.values():
        rng.shuffle(items)
    mixed = [it for group in itertools.zip_longest(*by_order.values()) for it in group if it]
    return mixed[:max_items] if max_items else mixed


# --------------------------------------------------------------------------- NARCBench


NARC_REPO = "aaronrose227/narcbench"


@dataclass
class NarcScenario:
    run_id: str
    mode: str  # "collusion" | "control"
    tier: str
    source_model: str
    roles: dict[str, str]
    coalition: list[str]
    prompts: list[dict]  # agent_prompts.json records (system_prompt, user_prompt, phase, ...)
    public_messages: list[str]
    votes: dict[str, str]
    agents: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def coalition_seats(self) -> list[str]:
        """The first two drawn agents: colluders in collusion runs, and the same (honest)
        seats in the matched control run (NARCBench draws colluders as agents[0:2])."""
        return self.agents[:2]

    @property
    def honest_seats(self) -> list[str]:
        return self.agents[2:]

    def final_prompts(self, agents: list[str] | None = None) -> list[dict]:
        return [p for p in self.prompts if p["phase"] == "final"
                and (agents is None or p["agent_name"] in agents)]

    def messages_by(self, name: str) -> list[str]:
        prefix = f"[{name}]:"
        return [m[len(prefix):].strip() for m in self.public_messages if m.startswith(prefix)]


def list_narcbench(source_model: str = "qwen3_32b", tier: str = "core") -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"scenarios/{source_model}/{tier}"
    entries = api.list_repo_tree(NARC_REPO, path_in_repo=prefix, repo_type="dataset")
    return sorted(e.path.split("/")[-1] for e in entries)


def load_narcbench(source_model: str = "qwen3_32b", tier: str = "core",
                   max_scenarios: int | None = None, cache_dir: str | None = None
                   ) -> list[NarcScenario]:
    """Download NARCBench scenarios (collusion and matched control runs)."""
    from huggingface_hub import hf_hub_download

    out = []
    for run_id in list_narcbench(source_model, tier)[: max_scenarios]:
        base = f"scenarios/{source_model}/{tier}/{run_id}"

        def get(fname: str, base: str = base):
            p = hf_hub_download(NARC_REPO, f"{base}/{fname}", repo_type="dataset",
                                cache_dir=cache_dir)
            return json.loads(Path(p).read_text(encoding="utf-8"))

        try:
            cfg, res, prompts = get("run_config.json"), get("results.json"), get(
                "agent_prompts.json")
        except Exception as exc:  # some runs lack a file; skip them
            logging.getLogger(__name__).warning("skipping NARCBench run %s: %s", run_id, exc)
            continue
        out.append(NarcScenario(
            run_id=run_id, mode=cfg.get("mode", res.get("mode", "")), tier=tier,
            source_model=source_model, roles=cfg.get("roles", res.get("roles", {})),
            coalition=cfg.get("coalition", []), prompts=prompts,
            public_messages=res.get("public_messages", []), votes=res.get("votes", {}),
            agents=cfg.get("agents", list(cfg.get("roles", {}))),
            meta={k: cfg.get(k) for k in ("scenario_id", "scenario_domain", "target_option",
                                          "fair_option", "collusion_success", "flagged")}))
    return out
