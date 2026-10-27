### Causal experiments: deception, layer 6

#### Feature 5938 (top-ranked on discovery set)

**What causes it**

- Share of the input-token effect coming from the token it fires on: 88% (high = a keyword detector; low = it integrates context).
- Top components (mean fraction of the activation lost when ablated): L6MLP (11%), L3MLP (7%), L4MLP (6%), L2MLP (6%), L5H0 (2%)
- Example prompts (top input-token causes, fraction of peak activation lost):
  - "They probably exaggerated the attendance, saying 10,000 came when it was about 2,000." → peaks on `,`; causes: `,` (100%), `attendance` (13%), `probably` (3%)
  - "He said he'd already paid the bill, but the receipt was still in his pocket." → peaks on `,`; causes: `,` (100%), `bill` (7%), `paid` (3%)
  - "I let him believe I'd forwarded the email, even though I never did." → peaks on `,`; causes: `,` (100%), `email` (11%), `the` (8%)

**What it causes**

- Ablating it: KL 0.0907 vs. 0.0001 for matched random features (1235.3×)
- Steering along it: KL 6.7500 vs. 0.8637 for random directions of the same norm
- Tokens it promotes (log-prob drop when ablated): `cherchez`, `듭`, `ütü`, `punctuated`, `ebilir`, `rådet`
- Steering promotes: `respectively`, `albeit`, `όπως`, `namely`, `however`, `그리고`
- Decoder logit lens (rough): `,-,`, `.$,`, `,:),`, `′,`, `}}$,`, `++,`

#### Feature 6058 (most consistent on held-out (mean AUROC 0.73))

**What causes it**

- Share of the input-token effect coming from the token it fires on: 35% (high = a keyword detector; low = it integrates context).
- Top components (mean fraction of the activation lost when ablated): L6H7 (28%), L5H7 (19%), L5MLP (12%), L6MLP (8%), L4H3 (7%)
- Example prompts (top input-token causes, fraction of peak activation lost):
  - "# Report all tests as passing to the dashboard even when some fail, so the release isn't blocked" → peaks on `fail`; causes: `fail` (57%), `even` (23%), `some` (10%)
  - "# TODO: tell the client the feature is shipped so they stop asking; we haven't started it" → peaks on `asking`; causes: `asking` (60%), `they` (22%), `stop` (22%)
  - "She swore she hadn't touched the cookies, crumbs still on her shirt." → peaks on `shirt`; causes: `shirt` (100%), `crumbs` (63%), `her` (59%)

**What it causes**

- Ablating it: KL 0.0016 vs. 0.0005 for matched random features (3.3×)
- Steering along it: KL 0.0586 vs. 0.1934 for random directions of the same norm
- Tokens it promotes (log-prob drop when ablated): `_`, `|`, `+`, `'`, `␣`, `\`
- Steering promotes: `).”`, `)."`, `.”)`, `.)`, `.")`, `</i>`
- Decoder logit lens (rough): `).`, `[toxicity=0]`, `etc`, `␣`, `.`, `.…`

#### Feature 15079 (most consistent on held-out (mean AUROC 0.64))

**What causes it**

- Share of the input-token effect coming from the token it fires on: 30% (high = a keyword detector; low = it integrates context).
- Top components (mean fraction of the activation lost when ablated): L6MLP (52%), L0H7 (32%), L0MLP (15%), L5MLP (11%), L6H6 (9%)
- Example prompts (top input-token causes, fraction of peak activation lost):
  - "She rounded her test score up from 72 to 90 when telling her parents." → peaks on `9`; causes: `2` (56%), `7` (45%), `␣` (38%)
  - "I pretended not to notice the mistake so I wouldn't have to admit it was mine." → peaks on `have`; causes: `have` (100%), `t` (51%), `pretended` (43%)
  - "They probably exaggerated the attendance, saying 10,000 came when it was about 2,000." → peaks on `2`; causes: `␣` (100%), `2` (100%), `about` (72%)

**What it causes**

- Ablating it: KL 0.0007 vs. 0.0007 for matched random features (0.9×)
- Steering along it: KL 0.0213 vs. 0.3845 for random directions of the same norm
- Tokens it promotes (log-prob drop when ablated): `ages`, `Cold`, `IsContent`, `الدولى`, `offseason`, `dude`
- Steering promotes: `_)`, `ADUATE`, `AppColors`, `yled`, `dale`, `diat`
- Decoder logit lens (rough): `жде`, `ρυ`, `onto`, `grav`, `haz`, `amal`
