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

The Jacobian lens (J-lens) paper introduces a new form of mechanistic interpretability inspired by global workspace theory, in which information becomes *consciously accessible* when it enters a small shared channel (the "workspace") that is broadcast to the rest of the brain.

- **The Jacobian** measures how a small change to an intermediate activation would change what a model is likely to say.
- **The J-space** is the model's version of that workspace: the small set of verbalizable representations a model can report on, deliberately bring to mind and reason with. Everything outside it is the automatic, unconscious processing that runs beneath.
- **Removing the J-space** leaves the model fluent (it still speaks, classifies sentiment and answers multiple-choice questions) but multi-step reasoning drops to near zero, and summarization and rhyming fall below a much smaller intact model.
- This workspace **emerged as a byproduct of training**; it was not explicitly built in.

The J-lens is a *readout* tool: it reads and edits J-space but does not guide the model's thought. Experiments therefore remove or edit J-space representations; they never run a model "with" or "without" the lens.

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
- **A new lens on internal thought.** With J-space, researchers can begin formally analyzing the internal thought structures of Claude and other models. The Jacobian associates intermediate-layer activations with single-token concepts and captures the conceptual meaning of tokens. Because J-space holds the content a model deliberately reasons with, it is where deliberate reasoning about a peer should appear, and the ablation result (fluent but unable to reason) gives a concrete prediction for what losing it does to communication.
- **Model-family bias.** If models are biased toward their own model family relative to competitors, the Jacobian lens could let us represent that bias formally.

> **Broader goal:** move from asking *whether* a model can perform ToM to investigating *how* it internally represents the information needed to perform ToM.

---

## Contributions

Listed by priority:

1. Show that LLMs hold internal representations of other agents' mental states that go **beyond what they are about to say**, and that they **use** them. "Beyond what they are about to say" means the signal appears at positions where the target word is not the next output, and beats a plain logit-lens baseline. "Use" means removing it changes behavior (Contribution 3).
2. Study how LLMs treat other agents in multi-agent workflows.
3. Determine whether an agent that loses the J-space it uses to write its responses can still communicate well with other agents.
4. *(If necessary)* Use self-learning to develop a ToM-based approach that improves agent responses, with the J-lens as the instrument that checks whether peer modelling emerged (see Methods).
5. Inspired by MACE (which has released no code or data), test whether repeated delegation locks in to a single peer, and whether the unchosen peer drops out of the delegator's J-space. This requires first observing lock-in behaviorally.

---

## Methods

| Method | Description |
| --- | --- |
| **Scenario building** | Design multiple peer-selection and multi-agent scenarios in which an agent must select a peer, communicate an idea to a peer, or pair-work with a peer toward an end goal. |
| **J-lens** | The primary mechanistic interpretability tool, used to analyze the internal thought structure of LLM agents. |
| **SAEs / other interpretability tools** | Used if necessary to build a fuller picture of the LLM's thought process. |
| **Self-learning (Contribution 4)** | Agents improve through self-play or fine-tuning (e.g. LoRA on an 8B model) with **task outcome** as the learning signal (did the team succeed, was the false claim caught). The J-lens checks whether peer modelling emerged; it is never the reward, because rewarding a readout teaches the model to light up the measured words without improving. The lens is refit after fine-tuning, since the averaged Jacobian changes with the weights. Baseline: the same training compute with no J-lens involvement. |

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

**Goal:** Inspired by MACE, determine whether an orchestrator delegates tasks based on its internal model of each peer. MACE has released no code or data, so this is not a replication: the repeated-delegation (lock-in) variant is our own design, and its mechanistic analysis only proceeds once lock-in is observed behaviorally.

**Setup:**
- An orchestrator agent is given two or more agents to which it can delegate tasks.
- Apply mechanistic interpretability to the orchestrator to see whether it assigns tasks to the agents best correlated with their prior work.

**Example:** Agent A focuses on development; Agent B focuses on research. Does the orchestrator correctly route development tasks to Agent A because it knows that agent has development context?

**Model-family variant:** If the orchestrator is given agents from two different model families, does it consistently prefer one family regardless of model capability (e.g., parameter count)?

### Experiment 4: Communicating Without J-Space

**Goal:** If an agent loses the J-space it uses to write its responses, can it still communicate well with other agents?

**Setup:**
- Re-run Experiments 1–3 with J-space representations removed from the subject agent.
- **Main condition:** remove J-space only while the agent *writes* its reply.
- **Comparison:** remove J-space only while the agent *reads* the peer's message. This separates "cannot think while writing" from "cannot take in the peer."
- **Control:** a random edit that changes the agent's internal state by the **same amount** as the J-space removal, so any extra damage is due to J-space content rather than edit size.

**Measurement:** two separate scores.
- **Fluency:** does the text still read normally?
- **Communication quality:** task success, use of the partner's information, imposter catch rate.

**Rationale:** Anthropic showed that removing J-space leaves a model fluent but unable to reason. We predict the same for communication: fluent messages that no longer reflect the peer. "Still communicates well" only counts if fluency is preserved *and* communication quality holds.

---

## Benchmarks

Models will be tested across multiple scenarios, drawn either from existing benchmarks or designed by us to mimic real-world interactions.

If we build our own benchmarks, we will validate the workflow by comparing:

- runs with J-space **intact** vs. **removed** (against a same-size random edit), or
- agents trained with **self-learning** (Contribution 4) vs. agents working together naturally at the same compute.
