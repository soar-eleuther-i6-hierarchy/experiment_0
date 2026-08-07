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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma2_2b/layer_03/in_block_edges.html">3</a><a class="pill" href="../../../outputs/gemma2_2b/layer_06/in_block_edges.html">6</a><a class="pill" href="../../../outputs/gemma2_2b/layer_12/in_block_edges.html">12</a><a class="pill on" href="../../../outputs/gemma2_2b/layer_18/in_block_edges.html">18</a><a class="pill" href="../../../outputs/gemma2_2b/layer_24/in_block_edges.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma2_2b/layer_18/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma2_2b/layer_18/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma2_2b/layer_18/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma2_2b/layer_18/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma2_2b/layer_18/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma2_2b/layer_18/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma2_2b/layer_18/qualitative_check.html">Qualitative report</a></div></nav>

# In-block (same-level) directed edges

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **345** directed edges, **1** duplicate pairs, 270 survive PMI>0; PolyFrac 86%, Gini 0.949.
- In-block superparents: 3 (e.g. F89 _technical or scientific language related to d…_: 126 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 89 → 116 | 1.00 | 0.01 | technical or scientific language related to d… | words related to international treaties and c… |
| 89 → 113 | 1.00 | 0.01 | technical or scientific language related to d… | phrases about media, especially film and tele… |
| 89 → 16 | 1.00 | 0.01 | technical or scientific language related to d… | first-person singular pronouns and auxiliary … |
| 89 → 98 | 1.00 | 0.01 | technical or scientific language related to d… | terms from scientific papers about cellular a… |
| 89 → 73 | 1.00 | 0.01 | technical or scientific language related to d… | descriptions of events and observations, ofte… |
| 89 → 40 | 1.00 | 0.01 | technical or scientific language related to d… | code snippets and sequences |
| 89 → 95 | 1.00 | 0.01 | technical or scientific language related to d… | text referring to people's biographies, inclu… |
| 89 → 78 | 1.00 | 0.01 | technical or scientific language related to d… | phrases in scientific writing about cell and … |

_Duplicate pairs (rename/split candidates):_ 9≈89 (mentions of political, …)

## Block B1  (384 features)
- **375** directed edges, **0** duplicate pairs, 163 survive PMI>0; PolyFrac 1%, Gini 0.997.
- In-block superparents: 1 (e.g. F304 _words or phrases related to technical manuals…_: 369 children, fires 81%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 304 → 295 | 0.91 | 0.12 | words or phrases related to technical manuals… | clauses and question-like structures often in… |
| 304 → 135 | 0.90 | 0.11 | words or phrases related to technical manuals… | words and phrases related to scientific and t… |
| 304 → 229 | 0.90 | 0.11 | words or phrases related to technical manuals… | discussions of abstract existential problems … |
| 304 → 373 | 0.90 | 0.10 | words or phrases related to technical manuals… | arguments containing a series of related poin… |
| 304 → 363 | 0.90 | 0.10 | words or phrases related to technical manuals… | the word "be" in various forms |
| 304 → 507 | 0.90 | 0.10 | words or phrases related to technical manuals… | a mix of different things, including UI eleme… |
| 304 → 351 | 0.89 | 0.10 | words or phrases related to technical manuals… | words or phrases indicating proof or quality … |
| 304 → 453 | 0.89 | 0.10 | words or phrases related to technical manuals… | words that relate to societal values and recr… |

## Block B2  (1536 features)
- **385** directed edges, **5** duplicate pairs, 385 survive PMI>0; PolyFrac 32%, Gini 0.937.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 555 → 796 | 0.90 | 4.81 | phrases that include the word "state" or refe… | LaTeX commands |
| 1508 → 796 | 0.88 | 5.06 | mathematical notation and symbols | LaTeX commands |
| 1526 → 796 | 0.88 | 4.11 | words and phrases related to space missions a… | LaTeX commands |
| 2022 → 796 | 0.88 | 3.65 | a hodgepodge of mostly technical or scientifi… | LaTeX commands |
| 517 → 796 | 0.88 | 4.33 | LaTeX and mathematical notation | LaTeX commands |
| 729 → 796 | 0.88 | 5.71 | lines of code or data with identifying charac… | LaTeX commands |
| 522 → 796 | 0.88 | 5.06 | code blocks and possibly code syntax | LaTeX commands |
| 1484 → 796 | 0.88 | 3.80 | descriptions of genetic syndromes and mutatio… | LaTeX commands |

_Duplicate pairs (rename/split candidates):_ 592≈964 (references to figures); 592≈1746 (references to figures); 796≈900 (LaTeX commands); 900≈1524 (JavaDoc-style comments …); 964≈1746 (legal citations)

## Block B3  (6144 features)
- **6091** directed edges, **833** duplicate pairs, 6091 survive PMI>0; PolyFrac 9%, Gini 0.981.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 6165 → 3712 | 0.98 | 5.69 | hyphenated sequences of letters and numbers | code documentation tags like "summary" |
| 5300 → 5454 | 0.97 | 5.16 | words related to antimicrobials and antibioti… | substrings that match the format "[@B##-#ijer… |
| 5884 → 8087 | 0.97 | 3.60 | code snippets and tags like html, css, javasc… | HTML document class declarations |
| 3563 → 5454 | 0.97 | 5.41 | intensifiers such as "very", "more" and "most… | substrings that match the format "[@B##-#ijer… |
| 7353 → 5454 | 0.97 | 4.34 | words related to literature, specifically boo… | substrings that match the format "[@B##-#ijer… |
| 4655 → 5490 | 0.97 | 4.45 | comparisons between groups. | LaTeX commands |
| 5805 → 5454 | 0.97 | 4.61 | content discussing making work or life more f… | substrings that match the format "[@B##-#ijer… |
| 3637 → 5454 | 0.97 | 3.89 | source code syntax | substrings that match the format "[@B##-#ijer… |

_Duplicate pairs (rename/split candidates):_ 2103≈2288 (the phrase "Challenge M…); 2103≈2291 (the phrase "Challenge M…); 2103≈2658 (the phrase "Challenge M…); 2103≈3067 (the phrase "Challenge M…); 2103≈3421 (the phrase "Challenge M…); 2103≈3712 (the phrase "Challenge M…)
