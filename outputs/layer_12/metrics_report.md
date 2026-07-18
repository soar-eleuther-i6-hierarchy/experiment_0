# Exp 0 - metrics report

- model / SAE: `gemma-2-2b-res-matryoshka-dc` @ `blocks.12.hook_resid_post`
- tokens sampled: **48971** over 400 docs
- edge criterion: reverse coverage >= 0.5, both endpoints fire >= 20

## Block pair 0->1  -  3262 candidate edges

- **Out-degree**: 80 parents, 384 children, 383 multi-parented; top-1 parent holds 11.8% of edges, Gini 0.732, max out-degree 384.
- **Superparents**: 5 (e.g. feature 44 _technical terminology related to programming and coding con…_: 384 children, fires on 98.8% of tokens)
- **Reconstruction**: 533/3262 edges improve reconstruction (16.3%).
- **Frequency control**: mean survival 0.594 over 3211 testable edges; 1407 (43.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.273 over 70 parents; 4 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.789.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 86 -> 162 | 1.00 | 0.02 | -0.00/0.00 | n | - | 0.22 | items related to beach or water sports … | instances of asynchronous processing or… |
| 91 -> 162 | 1.00 | 0.36 | -0.00/0.00 | n | - | 0.46 | numerical expressions, particularly tho… | instances of asynchronous processing or… |
| 123 -> 162 | 1.00 | 0.16 | -0.00/0.00 | n | - | 0.42 | lines of code or programming-related st… | instances of asynchronous processing or… |
| 44 -> 318 | 1.00 | 0.03 | 0.18/-0.02 | n | 1.00 | 0.03 | technical terminology related to progra… | code-related constructs and syntax |
| 44 -> 283 | 1.00 | 0.00 | 1.69/0.05 | Y | 1.00 | 0.03 | technical terminology related to progra… | terms related to cell biology, specific… |
| 18 -> 162 | 1.00 | 0.20 | -0.01/0.00 | n | - | 0.39 | references to legal context or terminol… | instances of asynchronous processing or… |
| 44 -> 272 | 1.00 | 0.02 | 0.18/-0.00 | n | 1.00 | 0.03 | technical terminology related to progra… | expressions of encouragement and suppor… |
| 44 -> 162 | 1.00 | 0.01 | 0.18/0.00 | n | - | 0.03 | technical terminology related to progra… | instances of asynchronous processing or… |

## Block pair 1->2  -  34142 candidate edges

- **Out-degree**: 190 parents, 662 children, 403 multi-parented; top-1 parent holds 0.9% of edges, Gini 0.664, max out-degree 303.
- **Superparents**: 0
- **Reconstruction**: 161/34142 edges improve reconstruction (0.5%).
- **Frequency control**: mean survival 0.069 over 33614 testable edges; 33095 (98.5%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.472 over 154 parents; 120 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.732.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 505 -> 601 | 0.99 | 0.06 | 0.00/-0.00 | n | - | 0.50 | phrases related to medical studies and … | mathematical expressions and their eval… |
| 436 -> 601 | 0.99 | 0.14 | -0.00/-0.00 | n | - | 0.52 | terms related to semiconductor devices … | mathematical expressions and their eval… |
| 292 -> 1307 | 0.99 | 0.22 | -0.00/-0.01 | n | - | 0.52 | terms related to account verification a… | chemical terms and their properties |
| 259 -> 1307 | 0.99 | 0.05 | -0.00/-0.01 | n | - | 0.48 | phrases related to communication and un… | chemical terms and their properties |
| 487 -> 1307 | 0.99 | 0.14 | -0.00/-0.01 | n | - | 0.50 | mathematical notation and expressions r… | chemical terms and their properties |
| 424 -> 1307 | 0.99 | 0.07 | -0.00/-0.01 | n | - | 0.48 | expressions of goodness and love in var… | chemical terms and their properties |
| 506 -> 601 | 0.99 | 0.12 | -0.00/-0.00 | n | - | 0.51 | phrases related to academic structure a… | mathematical expressions and their eval… |
| 289 -> 601 | 0.99 | 0.31 | -0.00/-0.00 | n | - | 0.54 | specific numeric versioning information… | mathematical expressions and their eval… |

## Block pair 2->3  -  431127 candidate edges

- **Out-degree**: 736 parents, 1767 children, 1327 multi-parented; top-1 parent holds 0.3% of edges, Gini 0.710, max out-degree 1159.
- **Superparents**: 0
- **Reconstruction**: 423/431127 edges improve reconstruction (0.1%).
- **Frequency control**: mean survival 0.024 over 401676 testable edges; 400673 (99.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.409 over 582 parents; 41 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.703.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 1572 -> 5875 | 1.00 | 0.18 | 0.24/0.06 | Y | 1.00 | - | ellipses and incomplete thoughts | ellipses or dramatic pauses in text |
| 1348 -> 6486 | 1.00 | 0.10 | 0.00/0.01 | n | - | 0.47 | documents related to legal proceedings … | mathematical symbols and expressions, p… |
| 1450 -> 5508 | 1.00 | 0.02 | 0.16/0.13 | Y | 1.00 | - | terms and phrases associated with sport… | terms related to football and other spo… |
| 1233 -> 7820 | 1.00 | 0.19 | 0.00/-0.00 | n | - | 0.45 | terms associated with control and compa… | numerical patterns or sequences |
| 1658 -> 7820 | 1.00 | 0.85 | 0.00/-0.00 | n | - | 0.47 | references to mathematical results or t… | numerical patterns or sequences |
| 956 -> 7820 | 1.00 | 0.55 | 0.00/-0.00 | n | - | 0.47 | text related to software licensing and … | numerical patterns or sequences |
| 1539 -> 3338 | 1.00 | 0.14 | 0.27/0.36 | Y | 1.00 | 0.01 | legal disclaimers and liability stateme… | phrases related to legal or liability i… |
| 1539 -> 5750 | 1.00 | 0.14 | 0.33/0.30 | Y | 1.00 | 0.01 | legal disclaimers and liability stateme… | legal and liability-related terms assoc… |
