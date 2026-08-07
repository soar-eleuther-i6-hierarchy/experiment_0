<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;margin-left:auto;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# Exp 0 - metrics report

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

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
- **Frequency control**: mean survival 0.052 over 266439 testable edges; 264778 (99.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.431 over 293 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.969.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 302 -> 977 | 1.00 | 0.15 | -0.00/0.00 | n | - | 0.42 | source code and/or documents with very … | strings that are not ASCII characters |
| 278 -> 977 | 1.00 | 0.27 | 0.00/0.00 | n | - | 0.43 | the words "real" or "actual" and relate… | strings that are not ASCII characters |
| 431 -> 977 | 1.00 | 0.13 | 0.00/0.00 | n | - | 0.43 | words and phrases used in formal or abs… | strings that are not ASCII characters |
| 261 -> 977 | 1.00 | 0.43 | 0.00/0.00 | n | - | 0.44 | decimal numbers | strings that are not ASCII characters |
| 357 -> 977 | 1.00 | 0.32 | 0.00/0.00 | n | - | 0.44 | the word "but" and the immediately surr… | strings that are not ASCII characters |
| 260 -> 977 | 1.00 | 0.22 | -0.00/0.00 | n | - | 0.44 | sentences that use first person pronouns | strings that are not ASCII characters |
| 479 -> 977 | 1.00 | 0.28 | -0.00/0.00 | n | - | 0.43 | leading whitespace followed by numerica… | strings that are not ASCII characters |
| 311 -> 977 | 1.00 | 0.21 | 0.00/0.00 | n | - | 0.43 | something, but it's too difficult to de… | strings that are not ASCII characters |

## Block pair 2->3  -  4704312 candidate edges

- **Out-degree**: 1326 parents, 4337 children, 4090 multi-parented; top-1 parent holds 0.1% of edges, Gini 0.173, max out-degree 4132.
- **Superparents**: 5 (e.g. feature 1191 _scientific and technical writing, particularly related to e…_: 4132 children, fires on 22.9% of tokens)
- **Reconstruction**: 113/4704312 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.015 over 4222667 testable edges; 4218195 (99.9%) are frequency-driven (survival < 0.5).
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
