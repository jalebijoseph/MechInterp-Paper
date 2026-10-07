"""jpeers: Jacobian-lens (J-space) analysis of theory of mind between LLM agents.

Layers:
    registry      -- which models have pre-fitted lenses and where they live
    runtime       -- load model + lens, generate while recording / editing J-space
    interventions -- J-space ablation, steering and swapping specs
    server        -- HTTP inference server (local, Azure ML / Foundry, Vertex AI, Cloud Run)
    client        -- endpoint backends the harness talks to
    harness       -- multi-agent conversations with per-turn J-space traces
    experiments   -- the four proposal experiments
    benchmarks    -- ToM / collusion benchmark loaders
    analysis      -- metrics over recorded traces
"""

__version__ = "0.1.0"
