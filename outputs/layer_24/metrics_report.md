# Exp 0 - metrics report

- model / SAE: `gemma-2-2b-res-matryoshka-dc` @ `blocks.24.hook_resid_post`
- tokens sampled: **48971** over 400 docs
- edge criterion: reverse coverage >= 0.5, both endpoints fire >= 20

## Block pair 0->1  -  4940 candidate edges

- **Out-degree**: 97 parents, 384 children, 384 multi-parented; top-1 parent holds 7.8% of edges, Gini 0.570, max out-degree 384.
- **Superparents**: 6 (e.g. feature 32 _numbers within a document_: 384 children, fires on 99.5% of tokens)
- **Reconstruction**: 879/4940 edges improve reconstruction (17.8%).
- **Frequency control**: mean survival 0.517 over 4940 testable edges; 2635 (53.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.350 over 92 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.944.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 32 -> 421 | 1.00 | 0.02 | 0.00/-0.00 | n | 1.00 | 0.05 | numbers within a document | text related to the game World of Warcr… |
| 32 -> 365 | 1.00 | 0.01 | -0.00/-0.00 | n | 1.00 | 0.05 | numbers within a document | citations in research papers |
| 32 -> 336 | 1.00 | 0.03 | 1.24/0.16 | Y | 1.00 | 0.05 | numbers within a document | numbers, especially as part of referenc… |
| 32 -> 387 | 1.00 | 0.02 | 0.00/0.00 | n | 1.00 | 0.05 | numbers within a document | contractions with the word "not." |
| 32 -> 193 | 1.00 | 0.00 | 1.10/0.00 | n | 1.00 | 0.05 | numbers within a document | language common to legal opinions. |
| 32 -> 201 | 1.00 | 0.03 | 1.15/0.02 | Y | 1.00 | 0.05 | numbers within a document | names of people and titles or abbreviat… |
| 32 -> 502 | 1.00 | 0.11 | 1.88/0.02 | Y | 1.00 | 0.05 | numbers within a document | words and phrases associated with polit… |
| 32 -> 162 | 1.00 | 0.07 | 0.00/0.00 | n | 1.00 | 0.05 | numbers within a document | terminology used in scientific publicat… |

## Block pair 1->2  -  108810 candidate edges

- **Out-degree**: 233 parents, 989 children, 816 multi-parented; top-1 parent holds 0.8% of edges, Gini 0.516, max out-degree 891.
- **Superparents**: 7 (e.g. feature 355 _words related to medical and business processes or programs_: 891 children, fires on 31.8% of tokens)
- **Reconstruction**: 83/108810 edges improve reconstruction (0.1%).
- **Frequency control**: mean survival 0.056 over 105552 testable edges; 104570 (99.1%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.485 over 206 parents; 163 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.841.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 281 -> 791 | 1.00 | 0.27 | -0.00/-0.00 | n | - | 0.52 | words associated with cells and molecul… | LaTeX labels within a mathematical docu… |
| 390 -> 791 | 1.00 | 0.21 | -0.00/-0.00 | n | - | 0.53 | glowing reviews of role playing games | LaTeX labels within a mathematical docu… |
| 367 -> 791 | 1.00 | 0.43 | -0.00/-0.00 | n | - | 0.53 | mathematical expressions and notation d… | LaTeX labels within a mathematical docu… |
| 448 -> 791 | 1.00 | 0.71 | -0.00/-0.00 | n | - | 0.54 | HTML tags, particularly paragraph begin… | LaTeX labels within a mathematical docu… |
| 396 -> 791 | 1.00 | 0.29 | -0.00/-0.00 | n | - | 0.52 | names of chemical structures and analys… | LaTeX labels within a mathematical docu… |
| 476 -> 791 | 1.00 | 0.30 | -0.00/-0.00 | n | - | 0.53 | code samples and mathematical equations | LaTeX labels within a mathematical docu… |
| 132 -> 791 | 1.00 | 0.34 | -0.00/-0.00 | n | - | 0.53 | numerical values involving measurements… | LaTeX labels within a mathematical docu… |
| 190 -> 791 | 1.00 | 0.25 | 0.00/-0.00 | n | - | 0.53 | phrases referring to political concepts… | LaTeX labels within a mathematical docu… |

## Block pair 2->3  -  3695288 candidate edges

- **Out-degree**: 1105 parents, 5343 children, 4450 multi-parented; top-1 parent holds 0.1% of edges, Gini 0.379, max out-degree 4289.
- **Superparents**: 2 (e.g. feature 2038 _first-person narratives, especially where opinions or perso…_: 4289 children, fires on 21.9% of tokens)
- **Reconstruction**: 247/3695288 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.017 over 3248273 testable edges; 3243688 (99.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.379 over 1005 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.886.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 1478 -> 4983 | 1.00 | 0.38 | -0.00/-0.00 | n | - | 0.39 | first-person pronouns such as "I" and "… | academic research, specifically related… |
| 1347 -> 7954 | 1.00 | 0.70 | -0.00/-0.00 | n | - | 0.39 | sentences that begin with "There are" | legal citations |
| 1877 -> 8084 | 1.00 | 0.21 | -0.00/-0.00 | n | - | 0.38 | words and phrases related to user authe… | equations and references in academic pa… |
| 921 -> 3019 | 1.00 | 0.16 | 0.00/-0.00 | n | - | 0.37 | concepts of duality and co-existence | parentheses |
| 1877 -> 8130 | 1.00 | 0.21 | -0.00/-0.00 | n | - | 0.38 | words and phrases related to user authe… | LaTeX commands for typesetting mathemat… |
| 1191 -> 6898 | 1.00 | 0.43 | -0.00/-0.00 | n | - | 0.39 | technical language related to online se… | HTML input tags |
| 921 -> 3089 | 1.00 | 0.16 | 0.00/0.00 | n | - | 0.37 | concepts of duality and co-existence | LaTeX code snippets |
| 1110 -> 3361 | 1.00 | 0.18 | -0.00/-0.00 | n | - | 0.37 | content related to customer experience … | LaTeX commands related to commutative d… |
