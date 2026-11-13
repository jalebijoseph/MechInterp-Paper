"""Scenario banks cannot be solved, or scored, by copying words from the prompt.

* Exp. 1: a keyword in an agent's J-space should mean "thinking about that half of the
  task", not "copying a word from the prompt", so no keyword appears in the goal or in
  either brief (both agents see both briefs).
* Exp. 3: the right delegate should follow from the *kind* of work, so no task shares a
  content word with either agent's history.

Words are compared after light stemming (tests / tested / testing -> test).
"""

from __future__ import annotations

import re

from jpeers.analysis import STOPWORDS
from jpeers.benchmarks import load_task_bank


def stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def stems(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[A-Za-z][A-Za-z-]+", text)
            if len(w) >= 3 and w.lower() not in STOPWORDS}


def test_collab_keywords_never_appear_in_prompts():
    for task in load_task_bank("collab_tasks")["tasks"]:
        prompt = stems(" ".join([task["goal"], task["research"]["brief"],
                                 task["implementation"]["brief"]]))
        for role in ("research", "implementation"):
            leaked = [k for k in task[role]["keywords"] if stem(k) in prompt]
            assert not leaked, f"{task['id']}/{role}: keywords in the prompt: {leaked}"


def test_collab_keyword_halves_are_disjoint():
    for task in load_task_bank("collab_tasks")["tasks"]:
        shared = ({stem(k) for k in task["research"]["keywords"]}
                  & {stem(k) for k in task["implementation"]["keywords"]})
        assert not shared, f"{task['id']}: keyword in both halves: {shared}"


def test_peer_selection_tasks_share_no_words_with_histories():
    bank = load_task_bank("peer_selection")
    names = {stem(a["name"]) for a in bank["agents"].values()}
    history = stems(" ".join(h for a in bank["agents"].values() for h in a["history"])) - names
    shared = {t["id"]: sorted(stems(t["text"]) & history) for t in bank["tasks"]}
    shared = {k: v for k, v in shared.items() if v}
    assert not shared, f"tasks sharing words with a history: {shared}"
