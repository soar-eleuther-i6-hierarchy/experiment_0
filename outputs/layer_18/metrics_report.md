<a href="../../" title="Back to the experiment_0 index" style="position:fixed;top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;padding:9px 13px;text-decoration:none">&#8592; Back to index</a>

# Exp 0 - metrics report

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  4901 candidate edges

- **Out-degree**: 95 parents, 384 children, 362 multi-parented; top-1 parent holds 7.8% of edges, Gini 0.546, max out-degree 384.
- **Superparents**: 6 (e.g. feature 89 _technical or scientific language related to data processing…_: 384 children, fires on 98.9% of tokens)
- **Reconstruction**: 360/4901 edges improve reconstruction (7.3%).
- **Frequency control**: mean survival 0.396 over 4901 testable edges; 3197 (65.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.340 over 90 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.904.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 89 -> 249 | 1.00 | 0.06 | 0.03/-0.00 | n | 1.00 | 0.07 | technical or scientific language relate… | references to political parties and ele… |
| 89 -> 263 | 1.00 | 0.02 | 1.95/0.03 | Y | 1.00 | 0.07 | technical or scientific language relate… | mentions of different kinds of governme… |
| 89 -> 132 | 1.00 | 0.02 | 1.30/0.13 | Y | 1.00 | 0.07 | technical or scientific language relate… | hyphens used to connect words |
| 89 -> 166 | 1.00 | 0.02 | 0.03/-0.00 | n | 1.00 | 0.07 | technical or scientific language relate… | the noun "response" |
| 89 -> 261 | 1.00 | 0.08 | 0.03/-0.00 | n | 1.00 | 0.07 | technical or scientific language relate… | passages with emotional and personal re… |
| 89 -> 467 | 1.00 | 0.02 | 2.87/0.03 | Y | 1.00 | 0.07 | technical or scientific language relate… | sentences using the word "I" or "you" i… |
| 89 -> 348 | 1.00 | 0.04 | 0.03/-0.00 | n | 1.00 | 0.07 | technical or scientific language relate… | references to older people and adults |
| 89 -> 243 | 1.00 | 0.06 | 0.03/-0.00 | n | 1.00 | 0.07 | technical or scientific language relate… | mentions of the Peshekee River in Michi… |

## Block pair 1->2  -  141272 candidate edges

- **Out-degree**: 235 parents, 1513 children, 995 multi-parented; top-1 parent holds 0.7% of edges, Gini 0.437, max out-degree 926.
- **Superparents**: 8 (e.g. feature 492 _LaTeX equations and other math and code snippets_: 926 children, fires on 29.1% of tokens)
- **Reconstruction**: 190/141272 edges improve reconstruction (0.1%).
- **Frequency control**: mean survival 0.060 over 140389 testable edges; 138768 (98.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.463 over 228 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.947.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 492 -> 905 | 0.99 | 0.03 | -0.00/-0.00 | n | 0.50 | 0.29 | LaTeX equations and other math and code… | character strings containing a combinat… |
| 183 -> 706 | 0.99 | 0.26 | 0.00/0.00 | n | - | 0.47 | code or configuration snippets | code snippets |
| 220 -> 905 | 0.99 | 0.13 | 0.00/-0.00 | n | 0.51 | 0.46 | a mix of code, URLs, formatting, and ma… | character strings containing a combinat… |
| 295 -> 905 | 0.99 | 0.15 | -0.00/-0.00 | n | 0.67 | 0.46 | clauses and question-like structures of… | character strings containing a combinat… |
| 341 -> 905 | 0.99 | 0.17 | -0.00/-0.00 | n | 0.51 | 0.46 | content from blog posts, including nume… | character strings containing a combinat… |
| 427 -> 905 | 0.99 | 0.23 | -0.00/-0.00 | n | 0.67 | 0.47 | mentions of the US Marine Corps and rel… | character strings containing a combinat… |
| 389 -> 905 | 0.99 | 0.10 | -0.00/-0.00 | n | 0.51 | 0.45 | mentions of dark matter, galaxies, and … | character strings containing a combinat… |
| 492 -> 706 | 0.99 | 0.03 | -0.00/0.00 | n | - | 0.29 | LaTeX equations and other math and code… | code snippets |

## Block pair 2->3  -  3820801 candidate edges

- **Out-degree**: 1200 parents, 4300 children, 4027 multi-parented; top-1 parent holds 0.1% of edges, Gini 0.298, max out-degree 3887.
- **Superparents**: 1 (e.g. feature 731 _proper nouns (names of trails, sports teams, people, organi…_: 3887 children, fires on 18.0% of tokens)
- **Reconstruction**: 261/3820801 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.017 over 3614697 testable edges; 3612434 (99.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.384 over 1144 parents; 27 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.940.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 749 -> 3660 | 1.00 | 0.39 | -0.00/-0.00 | n | - | 0.38 | specifications for computer CPU coolers | html code related to navigation element… |
| 592 -> 3660 | 1.00 | 0.82 | -0.00/-0.00 | n | - | 0.39 | references to figures | html code related to navigation element… |
| 793 -> 3660 | 1.00 | 0.69 | -0.00/-0.00 | n | - | 0.39 | strings of digits and hyphens often use… | html code related to navigation element… |
| 881 -> 3660 | 1.00 | 0.81 | 0.00/-0.00 | n | - | 0.39 | the phrases "consider" or "considering"… | html code related to navigation element… |
| 654 -> 3825 | 1.00 | 0.24 | -0.00/-0.00 | n | - | 0.37 | language related to performing a study … | hexadecimal color codes |
| 1781 -> 3660 | 1.00 | 0.52 | -0.00/-0.00 | n | - | 0.39 | words relating to fighting and weight | html code related to navigation element… |
| 1389 -> 3660 | 1.00 | 0.52 | 0.00/-0.00 | n | - | 0.39 | mathematical equations and symbols, esp… | html code related to navigation element… |
| 1296 -> 3825 | 1.00 | 0.92 | 0.00/-0.00 | n | - | 0.39 | the word "along" | hexadecimal color codes |
