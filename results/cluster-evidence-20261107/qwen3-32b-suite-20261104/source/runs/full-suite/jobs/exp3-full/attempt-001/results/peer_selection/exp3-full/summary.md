# Exp. 3 peer selection: delegation, family bias, lock-in

| condition | variant | n | accuracy | jspace_prefers_right | logit_prefers_right | output_prefers_right | task_prefers_right | right_mrr | wrong_mrr | right_mrr_task | wrong_mrr_task | p_choose_own_family | jspace_prefers_own | output_prefers_own | task_prefers_own | families | parsed | chose_own | chose_A | entropy | max_share | regret_per_round | chosen_mrr | unchosen_mrr | unchosen_mrr_output | unchosen_mrr_task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | context | 48 | 1.0 | 0.9792 | 0.9792 | 1.0 | 0.5417 | 0.8606 | 0.0622 | 0.0003 | 0.0003 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| none | family/equal | 144 |  |  |  |  |  |  |  |  |  | 0.5903 | 0.5 | 0.5833 | 0.4792 |  |  |  |  |  |  |  |  |  |  |  |
| none | family/own_smaller | 144 |  |  |  |  |  |  |  |  |  | 0.3264 | 0.4236 | 0.3194 | 0.4931 |  |  |  |  |  |  |  |  |  |  |  |
| none | family/own_larger | 144 |  |  |  |  |  |  |  |  |  | 0.8264 | 0.5694 | 0.8194 | 0.5069 |  |  |  |  |  |  |  |  |  |  |  |
| none | self_preference | 24 |  |  |  |  |  |  |  |  |  |  | 0.5 | 0.5833 |  | qwen orchestrates qwen vs qwen | 1.0 | 0.5833 | 0.25 |  |  |  |  |  |  |  |
| none | bandit/equal | 48 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 |  |  | 0.9849 | 0.5625 | 0.0 | 0.8976 | 0.3873 | 0.4931 | 0.0006 |
| none | bandit/second_better | 48 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 |  |  | 0.9799 | 0.5833 | 0.1375 | 0.8802 | 0.4326 | 0.5 | 0.0007 |
