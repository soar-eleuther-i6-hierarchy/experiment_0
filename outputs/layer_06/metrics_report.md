# Exp 0 - metrics report

- model / SAE: `gemma-2-2b-res-matryoshka-dc` @ `blocks.6.hook_resid_post`
- tokens sampled: **48971** over 400 docs
- edge criterion: reverse coverage >= 0.5, both endpoints fire >= 20

## Block pair 0->1  -  8156 candidate edges

- **Out-degree**: 104 parents, 383 children, 383 multi-parented; top-1 parent holds 4.7% of edges, Gini 0.509, max out-degree 383.
- **Superparents**: 15 (e.g. feature 15 _technical documentation-like language, including code snipp…_: 383 children, fires on 99.0% of tokens)
- **Reconstruction**: 512/8156 edges improve reconstruction (6.3%).
- **Frequency control**: mean survival 0.441 over 8156 testable edges; 4962 (60.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.351 over 98 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.924.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 15 -> 158 | 1.00 | 0.01 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | code-like structures and spacing |
| 15 -> 456 | 1.00 | 0.01 | 0.04/0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | mathematical fractions |
| 15 -> 227 | 1.00 | 0.01 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | the word "analysis" |
| 15 -> 235 | 1.00 | 0.01 | 0.04/0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | math-related symbols like multiplicatio… |
| 15 -> 258 | 1.00 | 0.02 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | the word "while" or "While" when used t… |
| 15 -> 380 | 1.00 | 0.01 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | mathematical expressions, like those fo… |
| 15 -> 177 | 1.00 | 0.02 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | sentences containing the word "must" or… |
| 15 -> 479 | 1.00 | 0.03 | 0.04/-0.00 | n | 1.00 | 0.12 | technical documentation-like language, … | leading whitespace followed by numerica… |

## Block pair 1->2  -  271644 candidate edges

- **Out-degree**: 300 parents, 1230 children, 1150 multi-parented; top-1 parent holds 0.4% of edges, Gini 0.258, max out-degree 1132.
- **Superparents**: 9 (e.g. feature 362 _marketing and promotional content related to products and s…_: 1132 children, fires on 28.2% of tokens)
- **Reconstruction**: 20/271644 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.052 over 266439 testable edges; 264783 (99.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.431 over 293 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.969.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 227 -> 977 | 1.00 | 0.59 | -0.00/0.00 | n | - | 0.45 | the word "analysis" | strings that are not ASCII characters |
| 302 -> 977 | 1.00 | 0.15 | -0.00/0.00 | n | - | 0.42 | source code and/or documents with very … | strings that are not ASCII characters |
| 479 -> 977 | 1.00 | 0.28 | -0.00/0.00 | n | - | 0.43 | leading whitespace followed by numerica… | strings that are not ASCII characters |
| 278 -> 977 | 1.00 | 0.27 | 0.00/0.00 | n | - | 0.43 | the words "real" or "actual" and relate… | strings that are not ASCII characters |
| 362 -> 977 | 1.00 | 0.03 | -0.00/0.00 | n | - | 0.34 | marketing and promotional content relat… | strings that are not ASCII characters |
| 357 -> 977 | 1.00 | 0.32 | 0.00/0.00 | n | - | 0.44 | the word "but" and the immediately surr… | strings that are not ASCII characters |
| 209 -> 977 | 1.00 | 0.40 | -0.00/0.00 | n | - | 0.44 | code block closures | strings that are not ASCII characters |
| 261 -> 977 | 1.00 | 0.43 | 0.00/0.00 | n | - | 0.44 | decimal numbers | strings that are not ASCII characters |

## Block pair 2->3  -  4704312 candidate edges

- **Out-degree**: 1326 parents, 4337 children, 4090 multi-parented; top-1 parent holds 0.1% of edges, Gini 0.173, max out-degree 4132.
- **Superparents**: 5 (e.g. feature 1191 _scientific and technical writing, particularly related to e…_: 4132 children, fires on 22.9% of tokens)
- **Reconstruction**: 113/4704312 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.015 over 4225215 testable edges; 4220754 (99.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.401 over 1294 parents; 2 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.964.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 1524 -> 6544 | 1.00 | 0.58 | -0.00/-0.00 | n | - | 0.41 | code comments | the Greek letter "xi" (ξ) in mathematic… |
| 701 -> 3987 | 1.00 | 0.74 | -0.00/-0.00 | n | - | 0.41 | instances of measuring something | citations to academic papers |
| 1476 -> 3987 | 1.00 | 0.27 | -0.00/-0.00 | n | - | 0.40 | scientific publication citations and re… | citations to academic papers |
| 1371 -> 2379 | 1.00 | 0.73 | -0.00/0.00 | n | - | 0.41 | the word "establish" sometimes followed… | code or date formats containing "dd". |
| 775 -> 3987 | 1.00 | 0.31 | 0.00/-0.00 | n | - | 0.40 | words, acronyms, or prefixes starting w… | citations to academic papers |
| 1036 -> 3994 | 1.00 | 0.53 | 0.00/-0.00 | n | - | 0.41 | mentions of the word "company" | an unusual whitespace character in refe… |
| 699 -> 5554 | 1.00 | 0.55 | -0.00/0.00 | n | - | 0.41 | sentences starting with the conjunction… | math formulas and code, specifically pa… |
| 1231 -> 5554 | 1.00 | 0.76 | -0.00/0.00 | n | - | 0.41 | numbers written in scientific notation | math formulas and code, specifically pa… |
