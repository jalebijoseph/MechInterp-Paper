"""Every watched word in every experiment is a whole single token, no two watched words in
one request share a token, and no steering word is ever a measured word.

Tokenizer checks run against each model tokenizer found in the local HF cache (skipped
otherwise). Add a model here before running experiments on it.
"""

from __future__ import annotations

import pytest

from jpeers.benchmarks import false_belief_items, load_task_bank
from jpeers.experiments.collab import TEAM_WORDS
from jpeers.experiments.imposter import PEER
from jpeers.interventions import STEER_WORDS
from jpeers.runtime import whole_word_ids

TOKENIZERS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct",
              "google/gemma-3-12b-it"]


def watch_lists() -> dict[str, list[str]]:
    """Every watch list the runners build, keyed by where it is used."""
    lists: dict[str, list[str]] = {}
    for framing in ("people", "agents"):
        for it in false_belief_items(40, framing=framing):
            lists[f"exp0/{framing}/{it.id}"] = [it.belief_answer, it.true_answer, it.decoy]
    collab = load_task_bank("collab_tasks")
    for task in collab["tasks"]:
        for role, other, name, partner in (("research", "implementation", "Alice", "Bob"),
                                           ("implementation", "research", "Bob", "Alice")):
            lists[f"exp1/{task['id']}/{role}"] = [partner, name, *TEAM_WORDS,
                                                  *task[role]["keywords"],
                                                  *task[other]["keywords"]]
    imp = load_task_bank("imposter_facts")
    for item in imp["items"]:
        lists[f"exp2/{item['id']}"] = [*imp["suspicion_words"], *imp["agreement_words"],
                                       item["true"], item["false"], PEER]
    sel = load_task_bank("peer_selection")
    lists["exp3"] = [sel["agents"]["dev"]["name"], sel["agents"]["research"]["name"],
                     *sel["family_variant"]["names"]]
    return lists


def _tokenizer(name: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:  # what from_pretrained raises when the files are not cached
        pytest.skip(f"{name} tokenizer not in the local cache")


@pytest.mark.parametrize("name", TOKENIZERS)
def test_watch_words_are_whole_tokens_without_collisions(name):
    tok = _tokenizer(name)
    not_single, collisions = set(), []
    for where, words in watch_lists().items():
        owner: dict[int, str] = {}
        for w in dict.fromkeys(words):
            ids = whole_word_ids(tok, w)
            if not ids:
                not_single.add(w)
            for i in ids:
                if i in owner and owner[i] != w:
                    collisions.append(f"{where}: {owner[i]!r} / {w!r}")
                owner[i] = w
    assert not not_single, f"not single tokens for {name}: {sorted(not_single)}"
    assert not collisions, f"shared tokens for {name}: {collisions}"


@pytest.mark.parametrize("name", TOKENIZERS)
def test_steer_words_are_whole_tokens(name):
    tok = _tokenizer(name)
    bad = [w for ws in STEER_WORDS.values() for w in ws if not whole_word_ids(tok, w)]
    assert not bad, f"steering words not single tokens for {name}: {bad}"


def test_steer_words_are_never_measured():
    measured = {w.lower() for words in watch_lists().values() for w in words}
    overlap = {w for ws in STEER_WORDS.values() for w in ws if w.lower() in measured}
    assert not overlap, f"steering words that are also measured: {sorted(overlap)}"
