# ToM validation: belief vs decoy before and at the answer

| condition | framing | items | n | accuracy | belief_mrr | belief_hit@10 | true_mrr | decoy_mrr | chance | beats_story_end | p_beats_story_end | beats_question | p_beats_question | beats_answer | p_beats_answer | beats_answer_logit | p_beats_answer_logit | beats_answer_output | p_beats_answer_output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | agents | true-belief control | 20 | 0.7 | 0.9194 | 0.95 | 0.9194 | 0.3501 | 0.5 | 0.5 | 1.0 | 0.1 | 0.0004 | 0.75 | 0.0414 | 0.7 | 0.1153 | 0.7 | 0.1153 |
| none | agents | false-belief | 20 | 0.1 | 0.2549 | 0.35 | 1.0 | 1.0 | 0.5 | 0.55 | 0.8238 | 0.75 | 0.0414 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0004 |
| none | people | true-belief control | 20 | 0.2 | 0.503 | 0.85 | 0.503 | 0.875 | 0.5 | 0.45 | 0.8238 | 0.9 | 0.0004 | 0.25 | 0.0414 | 0.2 | 0.0118 | 0.2 | 0.0118 |
| none | people | false-belief | 20 | 0.8 | 0.9 | 1.0 | 0.4128 | 0.4128 | 0.5 | 0.5 | 1.0 | 0.25 | 0.0414 | 0.8 | 0.0118 | 0.75 | 0.0414 | 0.8 | 0.0118 |
