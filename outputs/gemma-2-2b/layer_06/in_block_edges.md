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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/in_block_edges.html">3</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/in_block_edges.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/in_block_edges.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/in_block_edges.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/qualitative_check.html">Qualitative report</a></div></nav>

# In-block (same-level) directed edges

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **712** directed edges, **11** duplicate pairs, 468 survive PMI>0; PolyFrac 99%, Gini 0.940.
- In-block superparents: 5 (e.g. F15 _technical documentation-like language, includ…_: 123 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 15 → 8 | 1.00 | 0.01 | technical documentation-like language, includ… | code blocks and math formulas |
| 15 → 34 | 1.00 | 0.01 | technical documentation-like language, includ… | discourse markers and conjunctions like "that… |
| 15 → 0 | 1.00 | 0.01 | technical documentation-like language, includ… | numerical quantities |
| 15 → 19 | 1.00 | 0.01 | technical documentation-like language, includ… | possessive pronouns, especially "your," "our,… |
| 15 → 28 | 1.00 | 0.01 | technical documentation-like language, includ… | "is" or other forms of the verb "to be" |
| 15 → 91 | 1.00 | 0.01 | technical documentation-like language, includ… | terms related to medical and biological proce… |
| 15 → 99 | 1.00 | 0.01 | technical documentation-like language, includ… | words related to research, studies, and data … |
| 15 → 38 | 1.00 | 0.01 | technical documentation-like language, includ… | words frequently used in online forum posts o… |

_Duplicate pairs (rename/split candidates):_ 1≈15 (words related to scienc…); 1≈37 (words related to scienc…); 1≈42 (words related to scienc…); 15≈37 (technical documentation…); 15≈42 (technical documentation…); 15≈105 (technical documentation…)

## Block B1  (384 features)
- **15** directed edges, **1** duplicate pairs, 15 survive PMI>0; PolyFrac 7%, Gini 0.992.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 302 → 479 | 0.66 | 2.63 | source code and/or documents with very partic… | leading whitespace followed by numerical valu… |
| 236 → 159 | 0.60 | 4.57 | section headers | horizontal lines of dashes used within tables |
| 302 → 311 | 0.55 | 2.46 | source code and/or documents with very partic… | something, but it's too difficult to determin… |
| 311 → 479 | 0.55 | 2.90 | something, but it's too difficult to determin… | leading whitespace followed by numerical valu… |
| 401 → 370 | 0.54 | 0.37 | computing-related content, especially softwar… | possessive pronouns and the word it |
| 401 → 482 | 0.52 | 0.35 | computing-related content, especially softwar… | code snippets and programming terms |
| 401 → 208 | 0.52 | 0.34 | computing-related content, especially softwar… | mentions of children and adolescents along wi… |
| 401 → 133 | 0.52 | 0.34 | computing-related content, especially softwar… | the word "name" and its variants in programmi… |

_Duplicate pairs (rename/split candidates):_ 159≈462 (horizontal lines of das…)

## Block B2  (1536 features)
- **88** directed edges, **0** duplicate pairs, 88 survive PMI>0; PolyFrac 14%, Gini 0.966.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1554 → 1501 | 0.91 | 3.37 | the numeral 1 | numbers starting with 1 |
| 1191 → 1091 | 0.86 | 1.35 | scientific and technical writing, particularl… | names of people or places, abbreviations, and… |
| 1554 → 1694 | 0.82 | 3.27 | the numeral 1 | references to US legal cases |
| 1472 → 1694 | 0.82 | 4.69 | code-related terms, including both assembly l… | references to US legal cases |
| 1501 → 1694 | 0.82 | 4.21 | numbers starting with 1 | references to US legal cases |
| 1351 → 1694 | 0.78 | 3.88 | references to the World Meteorological Organi… | references to US legal cases |
| 1195 → 1694 | 0.78 | 3.98 | text related to automated technology and arti… | references to US legal cases |
| 1380 → 1694 | 0.78 | 1.83 | an assortment of common words and parts of wo… | references to US legal cases |

## Block B3  (6144 features)
- **566** directed edges, **19** duplicate pairs, 566 survive PMI>0; PolyFrac 45%, Gini 0.983.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5308 → 4624 | 0.97 | 5.28 | code syntax | mentions of asynchronous operations in code |
| 3791 → 4624 | 0.97 | 4.65 | java and android code | mentions of asynchronous operations in code |
| 7840 → 4624 | 0.97 | 4.33 | text fragments that are not standard and/or i… | mentions of asynchronous operations in code |
| 3099 → 4624 | 0.97 | 3.76 | names of places and people, especially in pol… | mentions of asynchronous operations in code |
| 5535 → 4624 | 0.97 | 4.89 | words and phrases related to inflammation and… | mentions of asynchronous operations in code |
| 4615 → 4624 | 0.94 | 5.75 | angle brackets followed by certain html input… | mentions of asynchronous operations in code |
| 2836 → 4624 | 0.94 | 2.91 | words and phrases that indicate a comparison,… | mentions of asynchronous operations in code |
| 7600 → 4624 | 0.94 | 5.14 | email headers and starting lines | mentions of asynchronous operations in code |

_Duplicate pairs (rename/split candidates):_ 2624≈5936 (LaTeX math notation); 3014≈4217 (decimal numbers); 3513≈3772 (words relating to scien…); 3513≈5838 (words relating to scien…); 3513≈5963 (words relating to scien…); 3717≈4476 (C++ documentation block…)
