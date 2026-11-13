
Relevant Research Papers: (identify gaps in most recent papers)

1: J-space: Verbalizable Representations Form a GlobalWorkspace (Gurnee, Sofroniew, Lindsey et al., Anthropic, July 2026): https://www.anthropic.com/research/global-workspace

The Jacobian lens paper introduces a new form of mechanistic interpretation, inspired by the global workspace theory where humans have an unconscious mind and organization of ideas that assists in the generation of new ideas. 

The Jacobian is a mathematical algorithm that measures how a small change to an intermediate activation would change what a model is likely to say. The J-space built by anthropic is a representation of patterns in an AI models’ thought process that play a role in its output. This patterned thinking is a version of unconscious thinking seen in humans, where we actively are thinking of an idea and unconsciously pulling separate, related topics to come up with a formalized idea.

Importantly, this internal thought process seen in models came as a byproduct of training Claude models and was not inherently built out.

This direct representation of internal beliefs is what we hope to retrieve out of an orchestrator when interacting with other agents. J-Lens has been tested on a single LLM with its own responses, but it hasn’t explored how LLMs think when interacting with different LLM agents. 

2: Kosinski,Evaluating LLMs in Theory of Mind Tasks (PNAS 2024; arXiv 2302.02083):

This paper defines what Theory of Mind is and highlights how LLM models had inherently been able to solve these kinds of questions in 2024. 

Eleven different LLM models of varying capabilities were examined using bespoke-false-belief tasks (gold standard in testing ToM in humans). Between GPT-3 and GPT-4, LLM models went from solving 0% of tasks to 75% of tasks

Theory Of Mind is a critical aspect of human communication, making this find impactful because it highlights that LLM models are starting to become more socially skilled in understanding key context. Importantly, this paper has only proved that LLM’s have conscious reasoning for ToM, but it doesn’t indicate that LLM’s use it when interacting with one another.

3.: NARCBench: https://arxiv.org/abs/2604.01151

NARCBench is a benchmark used to evaluate LLM’s that are all working together to detect whether a model is deceiving other models that it should be working with. At a high level, a group of 4 models are being asked to work together, while 2 of the 4 models were given a secret “deception” task to act as if they are participating in the group while designing their own collusions. 

One of the key elements they focused on was that detecting collusions across multiple agents requires more than any single-probing technique, but they can conclusively determine the detection based on the model performance. 

Comparative to the J-space analysis, their version of interpretation is built around an outside model analyzing the conversations provided. Our idea instead focuses deeper on the internal representations of a model's thoughts and if we can prove that models understand how other models think.

Motivation

One of the newly sought after ideas in AI work is being able to let them work together to further automate tasks. This can be seen strongly in Claude Code workflows where sub-agents are being driven to communicate and fix their own separate issues in a code-base while an orchestrator agent manages their work

After reading the paper (Kosinski,Evaluating LLMs in Theory of Mind Tasks (PNAS 2024; arXiv 2302.02083)) where LLM’s autonomously were able to understand ToM tasks sub-consciously through training, we theorize that this conceptual understanding of ToM has driven sub-agent communication. And if it hasn’t then it could be a major leap in understanding how AI communication occurs more thoroughly

Additionally, with the findings from the J-Space pattern, researchers are able to begin formally analyzing the internal thought structures of how Claude or other AI models think. The jacobian directly associates thoughts between layers as single-token concepts and has conceptual understanding of token meaning. This novel mechanistic interpretability model can help us further analyze the way LLM models think across sub-agent driven workflows. If models have a bias towards their own model families comparative to competitor model families, we could also represent that idea formally through the Jacobian Lens.

The broader goal is to move from asking whether the model can perform ToM to investigating how it internally represents the information needed to perform ToM.

Contributions (by priority, ideas for contributions)

1: Proving that LLM’s have internal representations of ToM across agents
2: Studying the way that LLM’s can treat other agents in multi-agent workflows
3: Determining how agents interact together without a jacobian lens guiding thought
4: (IF NECESSARY): Designing a new workflow that agents use that directly implements jacobian lens and other tools to improve ToM thoughts
5: A mech-interp analysis of the findings found in MACE (why does peer selection lock to a single peer without exploration)

Methods

Scenario-building: We will design multiple peer-selection or multi-agent scenarios, where an agent will either need to select a peer, communicate a peer about an idea, or work alongside a peer to pair-work on an end-goal

J-Lens: The Mechanistic Interpretability tool we plan to use to analyze the internal thought structure of LLM agents. 

SAE/Other Mechanistic Interpretability tools: If necessary, could leverage these tools to assist in understanding the full LLM thought process

Experimental Design

This experiment will be split into a multitude of mini-experiments to determine certain conditions

1: General Experiment
The first experiment will analyze the LLM Internal representations of ToM across agents through a multi-agent workflow scenario. Two or more models (same or different) will be given a task, and will be split up to perform certain sub-tasks within this task. For instance, one LLM may research a concept while another LLM will research implementation details. They will be notified they are working together in this scenario. 

As this first experiment occurs, we’ll store snapshots of the J-lens given by both agents to see if these agents are conceptually working together or separately thinking about a given problem. A few key flags would be if the two agents are analyzing the same results, if the LLM’s have no indication of collaboration or teamwork within their J-Space, among other things. The best-case would be if, internally within the j-space, we can see that the models are actively thinking about what the other agent is working on



2: Imposter Experiment
Similar to the first experiment, we are going to provide two or more models a task. In the system prompt of one of the models, we will provide a flag to provide incorrect information. This experiment will act similarly to the findings found in NARCBench. We want to find keywords that indicate a form of suspicion in a model's thought process. A key work we could find within the J-lens is “suspicious” or “invalid” that indicate that it knows a model is giving incorrect information

3: Peer Selection
Inspired by the ideas provided by MACE, we will provide an orchestrator agent two or more agents to delegate tasks. In this scenario, we want to perform mechanistic interpretability on the orchestrator to see if it can correctly provide agents tasks that are more correlated to the work it did before. 

As an example, if we have Agent A, focused on development, and Agent B, focused on research. Will the agent correctly give Agent A the development tasks because it knows the agent has context on the development?

Additionally, if we were to provide an agent with two different models, will an orchestrator always choose a specific model family regardless of LLM model capability (ex: parameters).

4: Experiment with or without J-Lens
Using the experiments, we want to analyze how models will interact without the J-lens assisting their thought processes. According to Anthropic, we already saw that the loss of a J-space makes LLM’s lose high level thinking. We want to confirm that this J-lens is directly affecting the way LLM’s communicate with each other.

Benchmarks

In general, we will be testing the models across multiple scenarios. These scenarios can be provided by certain benchmarks or designed by us to mimic real-world interactions. In the scenario that we make our own benchmarks, we would validate the workflow based on if we had not used J-lens, or, if we design a workflow that assists in LLM ToM actions, against the agents working together naturally.
