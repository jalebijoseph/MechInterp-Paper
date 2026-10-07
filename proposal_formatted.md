# Proposal: Internal Representations of Theory of Mind Across Agents via the Jacobian Lens

## Table of Contents

1. [Related Work](#related-work)
2. [Motivation](#motivation)
3. [Contributions](#contributions)
4. [Methods](#methods)
5. [Experimental Design](#experimental-design)
6. [Benchmarks](#benchmarks)

---

## Related Work

*Goal: identify gaps in the most recent papers.*

### 1. J-space: Verbalizable Representations Form a Global Workspace

**Gurnee, Sofroniew, Lindsey et al., Anthropic, July 2026** — [anthropic.com/research/global-workspace](https://www.anthropic.com/research/global-workspace)

The Jacobian lens (J-lens) paper introduces a new form of mechanistic interpretability inspired by global workspace theory, in which humans have an unconscious mind and organization of ideas that assists in the generation of new ideas.

- **The Jacobian** measures how a small change to an intermediate activation would change what a model is likely to say.
- **The J-space** built by Anthropic is a representation of the patterns in a model's thought process that play a role in its output. This patterned thinking resembles unconscious thinking in humans, where we actively think about one idea while unconsciously pulling in separate, related topics to form a complete idea.
- This internal thought process **emerged as a byproduct of training** Claude models; it was not explicitly built in.

**Gap:** This direct representation of internal beliefs is what we hope to retrieve from an orchestrator when it interacts with other agents. The J-lens has been tested on a single LLM with its own responses, but it has not explored how LLMs think when interacting with *other* LLM agents.

### 2. Evaluating LLMs in Theory of Mind Tasks

**Kosinski, PNAS 2024** — [arXiv:2302.02083](https://arxiv.org/abs/2302.02083)

This paper defines Theory of Mind (ToM) and shows that, as of 2024, LLMs had become able to solve these kinds of tasks.

- Eleven LLMs of varying capability were evaluated using bespoke false-belief tasks (the gold standard for testing ToM in humans).
- Between GPT-3 and GPT-4, performance went from solving **0%** of tasks to **75%** of tasks.

ToM is a critical part of human communication, so this finding is impactful: it suggests LLMs are becoming more socially skilled at understanding key context.

**Gap:** The paper only shows that LLMs can reason about ToM when asked; it does not indicate whether LLMs *use* ToM when interacting with one another.

### 3. NARCBench

[arXiv:2604.01151](https://arxiv.org/abs/2604.01151)

NARCBench is a benchmark for evaluating whether LLMs working together can detect when a model is deceiving the others it is supposed to be collaborating with.

- A group of four models is asked to work together.
- Two of the four are given a secret "deception" task: appear to participate in the group while pursuing their own collusion.
- A key finding is that detecting collusion across multiple agents requires more than any single probing technique, though detection can be determined conclusively from model performance.

**Gap:** NARCBench's interpretation relies on an outside model analyzing the conversation transcripts. Our approach instead focuses on the *internal* representations of a model's thoughts, and on whether we can show that models understand how other models think.

---

## Motivation

A newly sought-after direction in AI is letting models work together to further automate tasks. This is prominent in Claude Code workflows, where sub-agents communicate and fix separate issues in a codebase while an orchestrator agent manages their work.

- **ToM in agent communication.** Kosinski showed that LLMs acquired an understanding of ToM tasks through training alone. We hypothesize that this conceptual understanding of ToM drives sub-agent communication. If it does not, demonstrating that would itself be a major step toward understanding how AI communication works.
- **A new lens on internal thought.** With J-space, researchers can begin formally analyzing the internal thought structures of Claude and other models. The Jacobian associates intermediate-layer activations with single-token concepts and captures the conceptual meaning of tokens. This tool can help us analyze how LLMs think within sub-agent-driven workflows.
- **Model-family bias.** If models are biased toward their own model family relative to competitors, the Jacobian lens could let us represent that bias formally.

> **Broader goal:** move from asking *whether* a model can perform ToM to investigating *how* it internally represents the information needed to perform ToM.

---

## Contributions

Listed by priority:

1. Show that LLMs have internal representations of ToM across agents.
2. Study how LLMs treat other agents in multi-agent workflows.
3. Determine how agents interact without a Jacobian lens guiding thought.
4. *(If necessary)* Design a new agent workflow that directly incorporates the Jacobian lens and other tools to improve ToM reasoning.
5. Provide a mechanistic interpretability analysis of the findings in MACE — specifically, why peer selection locks onto a single peer without exploration.

---

## Methods

| Method | Description |
| --- | --- |
| **Scenario building** | Design multiple peer-selection and multi-agent scenarios in which an agent must select a peer, communicate an idea to a peer, or pair-work with a peer toward an end goal. |
| **J-lens** | The primary mechanistic interpretability tool, used to analyze the internal thought structure of LLM agents. |
| **SAEs / other interpretability tools** | Used if necessary to build a fuller picture of the LLM's thought process. |

---

## Experimental Design

The study is split into several mini-experiments, each isolating a particular condition.

### Experiment 1: General Collaboration

**Goal:** Analyze LLM internal representations of ToM across agents in a multi-agent workflow.

**Setup:**
- Two or more models (same or different) are given a task and split up to perform sub-tasks within it — e.g., one LLM researches a concept while another researches implementation details.
- The agents are told they are working together.

**Measurement:**
- Store J-lens snapshots from each agent throughout the task to determine whether the agents are conceptually working together or thinking about the problem separately.

**Key signals:**
- Whether the agents are analyzing the same results.
- Whether there is no indication of collaboration or teamwork within their J-space.

**Best case:** Within the J-space, we see the models actively thinking about what the other agent is working on.

### Experiment 2: Imposter

**Goal:** Detect internal signs of suspicion toward a deceptive peer (analogous to NARCBench).

**Setup:**
- Two or more models are given a task, as in Experiment 1.
- One model's system prompt instructs it to provide incorrect information.

**Measurement:**
- Search the J-lens output for concepts indicating suspicion — e.g., tokens such as "suspicious" or "invalid" — suggesting a model recognizes that a peer is giving incorrect information.

### Experiment 3: Peer Selection

**Goal:** Inspired by MACE, determine whether an orchestrator delegates tasks based on its internal model of each peer.

**Setup:**
- An orchestrator agent is given two or more agents to which it can delegate tasks.
- Apply mechanistic interpretability to the orchestrator to see whether it assigns tasks to the agents best correlated with their prior work.

**Example:** Agent A focuses on development; Agent B focuses on research. Does the orchestrator correctly route development tasks to Agent A because it knows that agent has development context?

**Model-family variant:** If the orchestrator is given agents from two different model families, does it consistently prefer one family regardless of model capability (e.g., parameter count)?

### Experiment 4: With vs. Without the J-Lens

**Goal:** Confirm that the J-space directly affects how LLMs communicate with each other.

**Setup:**
- Re-run the above experiments with and without the J-space intact and compare how the models interact.

**Rationale:** Anthropic has already shown that removing the J-space causes LLMs to lose high-level thinking. We want to test whether this also degrades inter-agent communication.

---

## Benchmarks

Models will be tested across multiple scenarios, drawn either from existing benchmarks or designed by us to mimic real-world interactions.

If we build our own benchmarks, we will validate the workflow by comparing:

- runs **with** vs. **without** the J-lens, or
- a workflow **we design to assist LLM ToM** vs. agents working together naturally.
