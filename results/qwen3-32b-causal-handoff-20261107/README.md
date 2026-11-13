# Causal restart evidence

Attempt 004 applies the JSON-format correction described below. It completed all development baselines and stopped at the decoder discovery gate without testing interventions or held-out contexts. The attempt-003 results and evidence in the earlier sections remain unchanged.

The recording-window repair was tested and the causal study restarted as attempt 003 at **2026-11-07 22:01:14 Eastern time**. It completed all 32 control requests and stopped at the declared discovery gate at **22:02:49**. The model server shut down automatically. No development baseline, decoder fitting, intervention selection, or held-out generation ran in that attempt.

## Repairs and verification

The recording request had incorrectly set `max_positions=1`, restricting capture to the final prompt token. The selected neutral suffix ended nine tokens earlier. Restoring the existing 4096-token default retains that anchor while the unchanged selector still records exactly one prompt position and excludes generated positions.

The first repaired live response exposed a second bug: integer vocabulary keys are converted to strings by JSON, changing their sorted order. Receipt saving now normalizes the JSON representation before calculating its integrity checksum. Loading and tamper detection remain strict.

Regression tests reproduced both failures before the fixes and pass afterward. The frozen cluster suite passed **245 tests and 86 subtests**, with eight optional tokenizer checks skipped. See [preflight.json](attempt-003/preflight.json) and [cpu-tests.log](attempt-003/cpu-tests.log). The local full suite was also attempted, but collection required unavailable `fastapi` in `test_plain_hf_server.py` and `test_stack.py`; the complete cluster suite includes these tests and passed.

Live verification checked all **32 receipts**, with zero missing traces and zero checksum mismatches. The first control records position **134** in a **144-token** prompt, at the end of suffix span `[131,135)`. Residuals have shape `[1,6,5120]`; both decoder-direction arrays have shape `[6,2,5120]`. See [restart-verification.json](attempt-003/restart-verification.json).

## Control outcome

The unchanged strict JSON scorer rejected all 32 replies because each wrapped otherwise valid JSON in a Markdown code fence:

| Control | Recorded passing responses | Required accuracy |
|---|---:|---:|
| No handoff | 0/16 | 0.95 |
| Own-only extraction | 0/16 | 0.95 |

A separate diagnostic found that the JSON inside the fences contained the expected labels in **32/32** cases. This diagnostic changes neither the saved scores nor the gate. There were zero generation-cap hits. See [discovery.json](attempt-003/results/discovery.json) and [format-diagnostic.json](attempt-003/format-diagnostic.json). This stop provides no causal-intervention result.

## Preserved evidence

Attempts 001–003 remain intact on the cluster. Their initial exact-byte export contains 186 saved metadata and result artifacts, including all 32 NPZ traces and 34 raw receipts. [restart-evidence-manifest.json](restart-evidence-manifest.json) records every original path, byte size, and SHA-256. The complete attempt-004 export is described below. Source trees, weights, environments, lock files, and redundant archive files are excluded; source hashes and both repair patches are retained. NPZ traces and raw causal receipts are configured for Git LFS.

- Attempt 001: missing trace from the one-token recording window.
- Attempt 002: valid live trace, followed by the integer-key receipt checksum failure.
- Attempt 003: both repairs verified; control-format gate stopped discovery cleanly.

The first attempt-002 CPU preflight also omitted test fixtures from its copied source tree. Its 18 fixture-related failures are preserved in `cpu-tests-missing-fixtures.log`; restoring the existing fixtures produced a passing complete preflight before launch.

## JSON-format correction: attempt 004

The scorer now declares `json-object-v2` in each score and in run provenance. It accepts one entire JSON object, either bare or inside a single complete triple-backtick block labeled `json` or left unlabeled. The JSON must still have exactly the `own` and `incoming` keys, valid label values, and no duplicate keys. Extra prose, multiple answers, malformed or truncated output, and generation-cap hits remain failures. Prompt text, task inventory, intervention settings, and all scientific gate thresholds are unchanged.

Raw replies remain verbatim in receipts and turn records. `output_format` distinguishes bare and fenced JSON; `raw_strict_format_valid` separately records whether the raw reply was a valid bare-object answer before the cap check. The run cannot resume earlier provenance that lacks this scoring contract.

An offline replay of the 32 saved attempt-003 control responses scores **32/32 joint-correct** under v2. This is parser validation, not a replacement for fresh control generation. All 186 previously exported artifacts were rechecked against their SHA-256 inventory and remain unchanged. See [format-preflight-replay.json](attempt-004/format-preflight-replay.json). New-attempt launch metadata and live verification are saved separately from the completed earlier attempts.

The fresh frozen cluster preflight passed **314 tests and 86 subtests**, with eight optional tokenizer checks skipped. Attempt 004 launched at **2026-11-07 22:18:24 Eastern time** on idle physical GPUs **0–1**. At **22:21:09**, fresh controls passed **16/16 no-handoff** and **16/16 own-only**, deterministic clean replay passed, and **29/256 development baselines** had completed. All **62 saved receipts** checked had valid checksums and nonmissing traces; there were zero cache hits. Raw control replies remain fenced, with their bare-JSON compliance recorded separately. See [restart-verification.json](attempt-004/restart-verification.json), [preflight.json](attempt-004/preflight.json), and [launch.json](attempt-004/launch.json).

Those verification files are the initial live snapshot. The complete terminal attempt-004 outputs are now saved locally, including all 289 raw receipts and 289 traces. The original cluster copy remains at the path in [queue-location.json](queue-location.json).

## Attempt 004 final discovery result

The run stopped at **2026-11-07 22:29:51 Eastern time** with `discovery_failed`. It completed **289 fresh calls**: 32 controls, one clean replay, and all **256 development baselines**, with zero cache hits. Both control groups passed, clean replay passed, and development incoming-label, own-label, and normalized-format accuracy were all **100%**.

None of the **18 decoder fits** reached the declared **0.80 incoming-label cross-validation accuracy**. J-space fits ranged from **48.8% to 54.3%**. The best fit across all decoder families was **62.9%**, a residual-contrast feature at layer 48. No setting qualified for intervention testing; `settings` is empty and no selection was frozen. There were zero intervention and held-out calls. This result does not measure causal necessity.

The worker exited cleanly and the model server stopped automatically. See [final discovery report](attempt-004/results/discovery.json), [final status](attempt-004/results/status.json), and [manager shutdown record](attempt-004/manager-status.json). The earlier launch verification remains preserved as a dated snapshot.

The [decoder/readout audit, updated November 12](../../docs/2026-11-08-causal-decoder-audit.md), now includes the actual 256 development traces. All 18 original fits reproduce, and stable solvers change zero predictions. Fixed-alpha train-only probes improve full-residual decoding to 79.3% but leave the two-label J span at 49.2–53.1%; neither passes the unchanged 80% gate. No production code, original scores, or gates were changed by the audit.

The [complete output manifest](attempt-004/complete-output-manifest-20261113T024104Z-4dbfe001.json) inventories **1,176 original files / 344,258,708 bytes**, including all **289 receipts and 289 NPZ traces**. Every file was verified during extraction; all 14 previously local attempt-004 files were byte-identical. See [complete-output verification](attempt-004/complete-output-verification-20261112.json). The [actual-trace audit directory](decoder-audit-20261112/) includes the CPU audit driver, detailed report, compact development-state dataset, and independent local verification. Trace NPZs and raw causal receipt JSON use Git LFS.
