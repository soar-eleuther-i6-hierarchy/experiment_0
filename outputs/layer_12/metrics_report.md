<a href="../../" title="Back to the experiment_0 index" style="position:fixed;top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;padding:9px 13px;text-decoration:none">&#8592; Back to index</a>

# Exp 0 - metrics report

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  3262 candidate edges

- **Out-degree**: 80 parents, 384 children, 383 multi-parented; top-1 parent holds 11.8% of edges, Gini 0.732, max out-degree 384.
- **Superparents**: 5 (e.g. feature 44 _technical terminology related to programming and coding con…_: 384 children, fires on 98.8% of tokens)
- **Reconstruction**: 533/3262 edges improve reconstruction (16.3%).
- **Frequency control**: mean survival 0.594 over 3211 testable edges; 1407 (43.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.273 over 70 parents; 4 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.789.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 44 -> 139 | 1.00 | 0.00 | 1.43/0.01 | Y | 1.00 | 0.03 | technical terminology related to progra… | mathematical expressions and calculatio… |
| 44 -> 328 | 1.00 | 0.01 | 0.18/-0.00 | n | 1.00 | 0.03 | technical terminology related to progra… | references to mathematical labels or eq… |
| 86 -> 162 | 1.00 | 0.02 | -0.00/0.00 | n | - | 0.22 | items related to beach or water sports … | instances of asynchronous processing or… |
| 82 -> 162 | 1.00 | 0.03 | -0.00/0.00 | n | - | 0.15 | technical terms related to scientific m… | instances of asynchronous processing or… |
| 44 -> 272 | 1.00 | 0.02 | 0.18/-0.00 | n | 1.00 | 0.03 | technical terminology related to progra… | expressions of encouragement and suppor… |
| 18 -> 162 | 1.00 | 0.20 | -0.01/0.00 | n | - | 0.39 | references to legal context or terminol… | instances of asynchronous processing or… |
| 44 -> 162 | 1.00 | 0.01 | 0.18/0.00 | n | - | 0.03 | technical terminology related to progra… | instances of asynchronous processing or… |
| 123 -> 162 | 1.00 | 0.16 | -0.00/0.00 | n | - | 0.42 | lines of code or programming-related st… | instances of asynchronous processing or… |

## Block pair 1->2  -  34142 candidate edges

- **Out-degree**: 190 parents, 662 children, 403 multi-parented; top-1 parent holds 0.9% of edges, Gini 0.664, max out-degree 303.
- **Superparents**: 0
- **Reconstruction**: 161/34142 edges improve reconstruction (0.5%).
- **Frequency control**: mean survival 0.069 over 33614 testable edges; 33095 (98.5%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.472 over 154 parents; 120 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.732.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 436 -> 601 | 0.99 | 0.14 | -0.00/-0.00 | n | - | 0.52 | terms related to semiconductor devices … | mathematical expressions and their eval… |
| 505 -> 601 | 0.99 | 0.06 | 0.00/-0.00 | n | - | 0.50 | phrases related to medical studies and … | mathematical expressions and their eval… |
| 292 -> 1307 | 0.99 | 0.22 | -0.00/-0.01 | n | - | 0.52 | terms related to account verification a… | chemical terms and their properties |
| 259 -> 1307 | 0.99 | 0.05 | -0.00/-0.01 | n | - | 0.48 | phrases related to communication and un… | chemical terms and their properties |
| 424 -> 1307 | 0.99 | 0.07 | -0.00/-0.01 | n | - | 0.48 | expressions of goodness and love in var… | chemical terms and their properties |
| 487 -> 1307 | 0.99 | 0.14 | -0.00/-0.01 | n | - | 0.50 | mathematical notation and expressions r… | chemical terms and their properties |
| 416 -> 601 | 0.99 | 0.38 | -0.01/-0.00 | n | - | 0.54 | punctuation and specific formatting cha… | mathematical expressions and their eval… |
| 486 -> 601 | 0.99 | 0.14 | -0.00/-0.00 | n | - | 0.53 | the presence of technical terms related… | mathematical expressions and their eval… |

## Block pair 2->3  -  431127 candidate edges

- **Out-degree**: 736 parents, 1767 children, 1327 multi-parented; top-1 parent holds 0.3% of edges, Gini 0.710, max out-degree 1159.
- **Superparents**: 0
- **Reconstruction**: 423/431127 edges improve reconstruction (0.1%).
- **Frequency control**: mean survival 0.024 over 401676 testable edges; 400673 (99.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.409 over 582 parents; 41 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.703.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 1134 -> 7820 | 1.00 | 0.30 | -0.00/-0.00 | n | - | 0.46 | the presence of the word "on" in variou… | numerical patterns or sequences |
| 1233 -> 7820 | 1.00 | 0.19 | 0.00/-0.00 | n | - | 0.45 | terms associated with control and compa… | numerical patterns or sequences |
| 1681 -> 5544 | 1.00 | 0.27 | -0.00/-0.00 | n | - | 0.46 | phrases related to subscriptions and us… | characters or sequences indicative of n… |
| 1685 -> 7820 | 1.00 | 0.71 | 0.00/-0.00 | n | - | 0.47 | configuration settings and commands rel… | numerical patterns or sequences |
| 1539 -> 5750 | 1.00 | 0.14 | 0.33/0.30 | Y | 1.00 | 0.01 | legal disclaimers and liability stateme… | legal and liability-related terms assoc… |
| 956 -> 7820 | 1.00 | 0.55 | 0.00/-0.00 | n | - | 0.47 | text related to software licensing and … | numerical patterns or sequences |
| 1345 -> 7820 | 1.00 | 0.05 | -0.00/-0.00 | n | - | 0.37 | specific terms related to experimental … | numerical patterns or sequences |
| 1343 -> 6486 | 1.00 | 0.20 | -0.00/0.01 | n | - | 0.63 | instances of the word "when" in various… | mathematical symbols and expressions, p… |
