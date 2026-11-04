# Audited follow-up v2 implementation plan

> For agentic workers: execute independent modules in parallel, test first, then
> integrate and run a fresh review before deploying the immutable run snapshot.

Goal: implement the accepted audit fixes and execute the requested32B follow-up
and original Exp4 semantic rescore without disturbing original data or other jobs.
Architecture: opt-in jpeers.followup modules, shared recorded request/measurement
helpers, versioned semantic scoring, and an owned two-model lifecycle controller.
Tech stack: existing Python/HF/PyTorch/HTTP/Pydantic/NumPy/pytest.
Spec: docs/superpowers/specs/2026-11-04-audited-followup-v2-design.md.

Global constraints: preserve originals; target model remains pinnedQwen3-32B BF16;
only physical GPUs0–3 idle/UUID selected; no silent fallback or semantic success for
uncertain judgments; fresh outputs must disclose the post-audit protocol.

Review focus: ambiguous/negated/quoted facts, partial outputs and invalid JSON;
accidental target/reference leaks; arbitrary peer-name and plan-position priors;
missing/duplicate condition rows; server cleanup and old-run completion races.

- [ ] Semantic scorer: write regression tests before code; structured grading,
      evidence validation, calibration, durable response-hash cache and abstentions.
- [ ] Exp1/2: test exact prompt matching/private-info boundaries, reward/knowledge
      factorial, code probes, full author-anchored NARC spans and independent units.
- [ ] Exp3: test strict name/choice/tie parsing, role/roster swaps, actual different
      plan families, identical controls and feedback/session metrics.
- [ ] Shared/integration: request writer, fair band-average metrics, endpoint CLI,
      immutable configs, result-count validation and resumable orchestration.
- [ ] Runtime: plain HF OLMo endpoint, pinned source/model downloads, CPU waiting,
      UUID allocation, readiness checks, calibration, model process cleanup.
- [ ] Verification: full CPU suite, independent code review, small live calibration
      smoke, exact input/expected-count manifests, then launch durable follow-up.
- [ ] Rescoring: frozen Exp4 outputs, versioned grader/cache, condition coverage,
     180 paired units, semantic/legacy comparisons and unresolved-review bounds.
- [ ] Preserve code/config/results and trace data for the existing branch/LFS flow.
