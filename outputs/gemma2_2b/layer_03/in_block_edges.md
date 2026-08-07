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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma2_2b/layer_03/in_block_edges.html">3</a><a class="pill" href="../../../outputs/gemma2_2b/layer_06/in_block_edges.html">6</a><a class="pill" href="../../../outputs/gemma2_2b/layer_12/in_block_edges.html">12</a><a class="pill" href="../../../outputs/gemma2_2b/layer_18/in_block_edges.html">18</a><a class="pill" href="../../../outputs/gemma2_2b/layer_24/in_block_edges.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma2_2b/layer_03/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma2_2b/layer_03/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma2_2b/layer_03/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma2_2b/layer_03/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma2_2b/layer_03/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma2_2b/layer_03/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma2_2b/layer_03/qualitative_check.html">Qualitative report</a></div></nav>

# In-block (same-level) directed edges

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **557** directed edges, **16** duplicate pairs, 411 survive PMI>0; PolyFrac 98%, Gini 0.946.
- In-block superparents: 4 (e.g. F70 _proper nouns that have mixed upper and lowerc…_: 124 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 70 → 77 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | LaTeX math environments |
| 70 → 13 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the ends of sentences or phrases that include… |
| 70 → 36 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the word "to" |
| 70 → 26 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | first and third person pronouns along with re… |
| 70 → 10 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | symbols utilized in complex scientific public… |
| 70 → 5 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | equal signs followed by a space and a hexadec… |
| 70 → 121 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | possessive pronouns and adjectives. |
| 70 → 57 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the article "a" |

_Duplicate pairs (rename/split candidates):_ 1≈17 (words related to groups…); 17≈99 (words or phrases relate…); 17≈111 (words or phrases relate…); 39≈75 (scientific terms, espec…); 39≈81 (scientific terms, espec…); 52≈70 (sections from legal and…)

## Block B1  (384 features)
- **275** directed edges, **1** duplicate pairs, 275 survive PMI>0; PolyFrac 12%, Gini 0.997.
- In-block superparents: 1 (e.g. F448 _a grab bag of proper nouns including names, p…_: 244 children, fires 48%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 505 → 439 | 0.74 | 5.70 | legal and scientific research article formatt… | statistical data from scientific publications |
| 249 → 439 | 0.74 | 4.15 | law references, including numbers, footnotes,… | statistical data from scientific publications |
| 448 → 330 | 0.71 | 0.39 | a grab bag of proper nouns including names, p… | exponents |
| 448 → 208 | 0.70 | 0.38 | a grab bag of proper nouns including names, p… | scientific and technical texts, especially th… |
| 448 → 496 | 0.70 | 0.38 | a grab bag of proper nouns including names, p… | terms related to animal models in neuroscienc… |
| 448 → 188 | 0.70 | 0.37 | a grab bag of proper nouns including names, p… | LaTeX code |
| 448 → 280 | 0.70 | 0.37 | a grab bag of proper nouns including names, p… | citations to published research and words ind… |
| 448 → 217 | 0.69 | 0.37 | a grab bag of proper nouns including names, p… | mathematical formulas and notation |

_Duplicate pairs (rename/split candidates):_ 276≈342 (strings of numbers)

## Block B2  (1536 features)
- **1707** directed edges, **7** duplicate pairs, 858 survive PMI>0; PolyFrac 3%, Gini 0.991.
- In-block superparents: 1 (e.g. F1457 _words used in official documents and scientif…_: 1403 children, fires 88%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1457 → 1281 | 1.00 | 0.13 | words used in official documents and scientif… | the word "indicated" and related terms like "… |
| 1457 → 1259 | 0.99 | 0.12 | words used in official documents and scientif… | uses of the word "whether" |
| 1457 → 1032 | 0.98 | 0.11 | words used in official documents and scientif… | the `aligned` environment tag in LaTeX. |
| 1457 → 1837 | 0.98 | 0.11 | words used in official documents and scientif… | medical articles talking about probabilities … |
| 1457 → 1265 | 0.98 | 0.11 | words used in official documents and scientif… | mathematical symbols, especially gamma |
| 1457 → 1816 | 0.98 | 0.11 | words used in official documents and scientif… | the digit 5 |
| 1457 → 1274 | 0.98 | 0.11 | words used in official documents and scientif… | the word "stress" |
| 1457 → 1061 | 0.97 | 0.10 | words used in official documents and scientif… | the word "video", sometimes followed by a hyp… |

_Duplicate pairs (rename/split candidates):_ 655≈1032 (mathematical or physics…); 655≈1533 (mathematical or physics…); 954≈1593 (the phrase "required by"); 954≈2021 (the phrase "required by"); 1032≈1533 (the `aligned` environme…); 1233≈1765 (references to US Suprem…)

## Block B3  (6144 features)
- **1640** directed edges, **142** duplicate pairs, 1640 survive PMI>0; PolyFrac 42%, Gini 0.980.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5692 → 6997 | 1.00 | 5.11 | code documentation containing parameter decla… | the word "spatial" |
| 7291 → 6997 | 1.00 | 3.91 | the names of cities, municipalities, and rela… | the word "spatial" |
| 7147 → 6997 | 1.00 | 5.34 | mentions of disagreement and debate | the word "spatial" |
| 7141 → 6997 | 1.00 | 3.74 | prepositions | the word "spatial" |
| 7721 → 6997 | 1.00 | 5.61 | DNA sequences with their directionality. | the word "spatial" |
| 5562 → 6997 | 1.00 | 3.93 | proper nouns referring to people, places, org… | the word "spatial" |
| 6542 → 6997 | 1.00 | 5.49 | occurrences of the word "work" in code-relate… | the word "spatial" |
| 4300 → 6997 | 1.00 | 3.46 | technical details in product descriptions, me… | the word "spatial" |

_Duplicate pairs (rename/split candidates):_ 2334≈4691 (the character "t" at th…); 2346≈2358 (sequences of numbers, e…); 2346≈2556 (sequences of numbers, e…); 2346≈2915 (sequences of numbers, e…); 2346≈3948 (sequences of numbers, e…); 2346≈4532 (sequences of numbers, e…)
