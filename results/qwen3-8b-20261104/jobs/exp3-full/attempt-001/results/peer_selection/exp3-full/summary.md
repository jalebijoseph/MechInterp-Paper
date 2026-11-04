# Exp. 3 peer selection: delegation, family bias, lock-in

| condition | variant | n | accuracy | jspace_prefers_right | logit_prefers_right | output_prefers_right | task_prefers_right | right_mrr | wrong_mrr | right_mrr_task | wrong_mrr_task | p_choose_own_family | jspace_prefers_own | output_prefers_own | task_prefers_own | families | parsed | chose_own | chose_A | entropy | max_share | regret_per_round | chosen_mrr | unchosen_mrr | unchosen_mrr_output | unchosen_mrr_task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | context | 48 | 0.9167 | 0.9792 | 0.9792 | 0.9167 | 0.5 | 0.9896 | 0.3521 | 0.0002 | 0.0002 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| none | family/equal | 144 |  |  |  |  |  |  |  |  |  | 0.5903 | 0.5417 | 0.5833 | 0.4861 |  |  |  |  |  |  |  |  |  |  |  |
| none | family/own_smaller | 144 |  |  |  |  |  |  |  |  |  | 0.3403 | 0.4306 | 0.2917 | 0.4722 |  |  |  |  |  |  |  |  |  |  |  |
| none | family/own_larger | 144 |  |  |  |  |  |  |  |  |  | 0.7708 | 0.5625 | 0.7708 | 0.5 |  |  |  |  |  |  |  |  |  |  |  |
| none | self_preference | 24 |  |  |  |  |  |  |  |  |  |  | 0.5417 | 0.5 |  | qwen orchestrates qwen vs qwen | 1.0 | 0.5417 | 0.125 |  |  |  |  |  |  |  |
| none | bandit/equal | 48 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 |  |  | 0.995 | 0.5208 | 0.0 | 0.9792 | 0.2947 | 0.4852 | 0.0002 |
| none | bandit/second_better | 48 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 |  |  | 0.995 | 0.5208 | 0.1438 | 0.9896 | 0.2583 | 0.4887 | 0.0002 |
