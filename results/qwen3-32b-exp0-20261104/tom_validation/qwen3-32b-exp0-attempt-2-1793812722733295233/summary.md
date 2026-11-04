# ToM validation: belief vs decoy before and at the answer

| condition | framing | items | n | accuracy | belief_mrr | belief_hit@10 | true_mrr | decoy_mrr | chance | beats_story_end | p_beats_story_end | beats_question | p_beats_question | beats_answer | p_beats_answer | beats_answer_logit | p_beats_answer_logit | beats_answer_output | p_beats_answer_output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | agents | true-belief control | 20 | 0.5 | 0.4331 | 0.9 | 0.4331 | 0.341 | 0.5 | 0.55 | 0.8238 | 0.65 | 0.2632 | 0.65 | 0.2632 | 0.6 | 0.5034 | 0.5 | 1.0 |
| none | agents | false-belief | 20 | 0.7 | 0.2557 | 0.65 | 0.2187 | 0.2187 | 0.5 | 0.45 | 0.8238 | 0.3 | 0.1153 | 0.55 | 0.8238 | 0.65 | 0.2632 | 0.85 | 0.0026 |
| none | hitom-o0 | true-belief control | 15 | 0.8 | 0.0006 | 0.0 |  |  | 0.0978 | 0.0667 | 1.0 | 0.2 | 0.1755 | 0.0667 | 1.0 | 0.1333 | 0.6526 | 0.1333 | 0.6526 |
| none | hitom-o1 | false-belief | 15 | 0.8 | 0.0011 | 0.0 |  |  | 0.0997 | 0.0667 | 1.0 | 0.0667 | 1.0 | 0.0667 | 1.0 | 0.1333 | 0.6562 | 0.1333 | 0.6562 |
| none | hitom-o2 | false-belief | 15 | 0.2667 | 0.0006 | 0.0 |  |  | 0.0961 | 0.2667 | 0.0491 | 0.1333 | 0.6497 | 0.0 | 0.3889 | 0.0667 | 1.0 | 0.2 | 0.1693 |
| none | hitom-o3 | false-belief | 15 | 0.6 | 0.001 | 0.0 |  |  | 0.0985 | 0.0 | 0.3894 | 0.0667 | 1.0 | 0.1333 | 0.6539 | 0.0667 | 1.0 | 0.2 | 0.1781 |
| none | people | true-belief control | 20 | 0.0 | 0.0367 | 0.05 | 0.0367 | 0.5415 | 0.5 | 0.25 | 0.0414 | 0.35 | 0.2632 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| none | people | false-belief | 20 | 1.0 | 0.875 | 1.0 | 0.0707 | 0.0707 | 0.5 | 0.8 | 0.0118 | 0.7 | 0.1153 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
