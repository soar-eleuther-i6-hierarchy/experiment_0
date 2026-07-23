<a href="../../" title="Back to the experiment_0 index" style="position:fixed;top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;padding:9px 13px;text-decoration:none">&#8592; Back to index</a>

# Exp 0 - metrics report

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  3067 candidate edges

- **Out-degree**: 77 parents, 383 children, 382 multi-parented; top-1 parent holds 12.5% of edges, Gini 0.788, max out-degree 383.
- **Superparents**: 6 (e.g. feature 70 _proper nouns that have mixed upper and lowercase letters or…_: 383 children, fires on 99.3% of tokens)
- **Reconstruction**: 1304/3067 edges improve reconstruction (42.5%).
- **Frequency control**: mean survival 0.758 over 3067 testable edges; 870 (28.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.229 over 69 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.767.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 70 -> 459 | 1.00 | 0.01 | 1.50/0.10 | Y | 1.00 | 0.02 | proper nouns that have mixed upper and … | the word "work" and any words that coll… |
| 70 -> 238 | 1.00 | 0.01 | 1.01/0.03 | Y | 1.00 | 0.02 | proper nouns that have mixed upper and … | code terms related to calculating maxim… |
| 70 -> 148 | 1.00 | 0.01 | 0.88/0.00 | n | 1.00 | 0.02 | proper nouns that have mixed upper and … | trademarked product names |
| 70 -> 158 | 1.00 | 0.00 | 0.62/-0.00 | n | 1.00 | 0.02 | proper nouns that have mixed upper and … | the string "pone" which appears to be p… |
| 70 -> 441 | 1.00 | 0.00 | 1.00/0.48 | Y | 1.00 | 0.02 | proper nouns that have mixed upper and … | the word "such" |
| 70 -> 439 | 1.00 | 0.00 | 0.60/1.02 | Y | 1.00 | 0.02 | proper nouns that have mixed upper and … | statistical data from scientific public… |
| 70 -> 436 | 1.00 | 0.02 | 1.04/-0.00 | n | 1.00 | 0.02 | proper nouns that have mixed upper and … | LaTeX mathematical notation. |
| 70 -> 409 | 1.00 | 0.01 | 2.01/0.19 | Y | 1.00 | 0.02 | proper nouns that have mixed upper and … | the word "patients" and words immediate… |

## Block pair 1->2  -  28588 candidate edges

- **Out-degree**: 212 parents, 1078 children, 621 multi-parented; top-1 parent holds 3.6% of edges, Gini 0.725, max out-degree 1026.
- **Superparents**: 1 (e.g. feature 448 _a grab bag of proper nouns including names, places, and cod…_: 1026 children, fires on 48.4% of tokens)
- **Reconstruction**: 586/28588 edges improve reconstruction (2.0%).
- **Frequency control**: mean survival 0.132 over 27860 testable edges; 25769 (92.5%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.423 over 174 parents; 111 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.588.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 448 -> 1577 | 0.98 | 0.02 | -0.00/0.00 | n | - | 0.11 | a grab bag of proper nouns including na… | mentions of "CD3" or "CD4", as in CD34+ |
| 448 -> 882 | 0.98 | 0.02 | -0.00/-0.00 | n | 0.90 | 0.11 | a grab bag of proper nouns including na… | LaTeX equation labels |
| 448 -> 1387 | 0.98 | 0.02 | -0.00/0.00 | n | 0.80 | 0.11 | a grab bag of proper nouns including na… | mentions of DNA |
| 263 -> 1013 | 0.98 | 0.03 | -0.00/-0.00 | n | 0.68 | 0.37 | a variety of words, often nouns, someti… | citations and references |
| 448 -> 512 | 0.98 | 0.02 | -0.00/-0.00 | n | 0.74 | 0.11 | a grab bag of proper nouns including na… | the words "vector" with a trailing word… |
| 205 -> 1958 | 0.98 | 0.18 | -0.00/-0.00 | n | - | 0.57 | first-person pronouns alongside auxilia… | the symbol omega used in mathematical a… |
| 128 -> 1741 | 0.98 | 0.04 | -0.00/-0.00 | n | - | 0.46 | descriptions of foreign political confl… | the "fs" string, possibly related to fo… |
| 448 -> 1303 | 0.98 | 0.02 | -0.00/-0.00 | n | 0.89 | 0.11 | a grab bag of proper nouns including na… | the word "profile," often in a scientif… |

## Block pair 2->3  -  274313 candidate edges

- **Out-degree**: 806 parents, 3645 children, 1651 multi-parented; top-1 parent holds 1.1% of edges, Gini 0.744, max out-degree 2906.
- **Superparents**: 1 (e.g. feature 1457 _words used in official documents and scientific publications_: 2906 children, fires on 87.1% of tokens)
- **Reconstruction**: 2459/274313 edges improve reconstruction (0.9%).
- **Frequency control**: mean survival 0.050 over 247809 testable edges; 242239 (97.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.423 over 642 parents; 410 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.602.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 1457 -> 8146 | 1.00 | 0.00 | 0.02/0.41 | Y | 1.00 | 0.04 | words used in official documents and sc… | the word "unique" within code snippets |
| 1457 -> 4507 | 1.00 | 0.00 | 0.02/0.09 | Y | 1.00 | 0.04 | words used in official documents and sc… | the word "excited" or other forms of th… |
| 1457 -> 7343 | 1.00 | 0.00 | 0.03/0.11 | Y | - | 0.04 | words used in official documents and sc… | the word "to", often preceding a variab… |
| 923 -> 6997 | 1.00 | 0.37 | -0.00/-0.00 | n | - | 0.67 | mathematical equations and expressions | the word "spatial" |
| 1258 -> 6997 | 1.00 | 0.40 | -0.00/-0.00 | n | - | 0.67 | the terms "get" and "set" that appear i… | the word "spatial" |
| 875 -> 6997 | 1.00 | 0.33 | -0.01/-0.00 | n | - | 0.67 | LaTeX math symbols | the word "spatial" |
| 916 -> 6952 | 1.00 | 0.09 | -0.00/-0.00 | n | - | 0.54 | biological or technological research te… | backslashes following by numbers |
| 1457 -> 4512 | 1.00 | 0.00 | 0.03/0.21 | Y | - | 0.04 | words used in official documents and sc… | hyphens |
