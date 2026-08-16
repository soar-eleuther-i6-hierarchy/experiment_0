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
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/in_block_edges.html">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/in_block_edges.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/in_block_edges.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/in_block_edges.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_01/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_check.html">Qualitative report</a></div></nav>

# In-block (same-level) directed edges

**Layer 1**　·　gemma-2-2b / 1-res-matryoshka-dc　·　blocks.1.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **875** directed edges, **22** duplicate pairs, 451 survive PMI>0; PolyFrac 100%, Gini 0.930.
- In-block superparents: 8 (e.g. F53 _linguistic expressions related to organizing,…_: 123 children, fires 100%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 53 → 17 | 1.00 | 0.00 | linguistic expressions related to organizing,… | sequences of special characters and formatted… |
| 53 → 35 | 1.00 | 0.00 | linguistic expressions related to organizing,… | references to specific "this" or "these," ind… |
| 86 → 57 | 1.00 | 0.01 | descriptions of technological or mechanical m… | references to specific time, place, and event… |
| 53 → 50 | 1.00 | 0.00 | linguistic expressions related to organizing,… | references to the computer system or technolo… |
| 53 → 57 | 1.00 | 0.00 | linguistic expressions related to organizing,… | references to specific time, place, and event… |
| 86 → 48 | 1.00 | 0.01 | descriptions of technological or mechanical m… | indicators of complex mathematical expression… |
| 86 → 109 | 1.00 | 0.00 | descriptions of technological or mechanical m… | references to the definite article "the" and … |
| 53 → 16 | 1.00 | 0.00 | linguistic expressions related to organizing,… | references to legal descriptions and obligati… |

_Duplicate pairs (rename/split candidates):_ 6≈8 (references to fuel inje…); 6≈13 (references to fuel inje…); 6≈70 (references to fuel inje…); 6≈95 (references to fuel inje…); 6≈105 (references to fuel inje…); 6≈114 (references to fuel inje…)

## Block B1  (384 features)
- **225** directed edges, **2** duplicate pairs, 225 survive PMI>0; PolyFrac 34%, Gini 0.979.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 190 → 237 | 0.94 | 5.06 | questions related to identifying specific inf… | questions or prompts asking for specific info… |
| 134 → 148 | 0.81 | 0.74 | references or citations in a text | complex numerical relationships and patterns … |
| 361 → 148 | 0.75 | 0.91 | references to specific technical and statisti… | complex numerical relationships and patterns … |
| 134 → 400 | 0.73 | 0.64 | references or citations in a text | occurrences of code snippets involving output… |
| 186 → 148 | 0.73 | 3.82 | references to scientific or academic contexts | complex numerical relationships and patterns … |
| 285 → 148 | 0.73 | 1.91 | neural responses related to the perception an… | complex numerical relationships and patterns … |
| 276 → 148 | 0.73 | 4.54 | patterns indicating code structures or syntax… | complex numerical relationships and patterns … |
| 420 → 148 | 0.73 | 3.33 | references to legal case citations and relate… | complex numerical relationships and patterns … |

_Duplicate pairs (rename/split candidates):_ 371≈403 (detected regions or fea…); 375≈395 (instructions for updati…)

## Block B2  (1536 features)
- **3481** directed edges, **184** duplicate pairs, 3481 survive PMI>0; PolyFrac 75%, Gini 0.906.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1478 → 1846 | 0.93 | 4.87 | references to energy, energy differences, or … | judgments and related legal final decisions |
| 646 → 1846 | 0.92 | 4.68 | references to laboratory tests and medical co… | judgments and related legal final decisions |
| 979 → 1846 | 0.92 | 4.67 | mentions of molecules or structures containin… | judgments and related legal final decisions |
| 1006 → 1846 | 0.92 | 3.86 | a pattern involving complex mathematical form… | judgments and related legal final decisions |
| 790 → 1846 | 0.92 | 3.94 | references to specific levels of a biomarker … | judgments and related legal final decisions |
| 1082 → 1846 | 0.92 | 2.70 | expressions of nostalgia and longing for the … | judgments and related legal final decisions |
| 517 → 1791 | 0.92 | 4.76 | patterns related to capturing or directing co… | structures related to tensor products in math… |
| 868 → 1846 | 0.92 | 3.31 | references to investigations or inquiries | judgments and related legal final decisions |

_Duplicate pairs (rename/split candidates):_ 526≈676 (the sequence of the let…); 526≈1791 (the sequence of the let…); 526≈1845 (the sequence of the let…); 562≈724 (references to rules, gu…); 562≈766 (references to rules, gu…); 562≈1163 (references to rules, gu…)

## Block B3  (6144 features)
- **6718** directed edges, **4885** duplicate pairs, 6695 survive PMI>0; PolyFrac 17%, Gini 0.991.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 7353 → 5659 | 1.00 | 4.11 | patterns related to structured data or code s… | representations of byte size information |
| 5088 → 5659 | 1.00 | 5.77 | references to formal agreements or alliances | representations of byte size information |
| 3482 → 5659 | 1.00 | 4.30 | instances of the word "then" | representations of byte size information |
| 3482 → 6191 | 1.00 | 4.30 | instances of the word "then" | references to diagrammatic or visual represen… |
| 5839 → 6191 | 1.00 | 5.38 | HTML tags and dropdown menu elements in docum… | references to diagrammatic or visual represen… |
| 6402 → 5659 | 1.00 | 3.64 | records of timing and synchronization codes | representations of byte size information |
| 2906 → 5659 | 1.00 | 5.62 | references to legal remand actions | representations of byte size information |
| 3443 → 5659 | 1.00 | 3.30 | statements or phrases that express personal o… | representations of byte size information |

_Duplicate pairs (rename/split candidates):_ 2050≈2295 (recurring visual or tex…); 2050≈4600 (recurring visual or tex…); 2050≈8064 (recurring visual or tex…); 2092≈2179 (references to property-…); 2092≈2261 (references to property-…); 2092≈2310 (references to property-…)

## Block B4  (24576 features)
- **7403** directed edges, **5578** duplicate pairs, 7403 survive PMI>0; PolyFrac 45%, Gini 0.997.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 18377 → 12852 | 1.00 | 5.90 | references to specific years or dates | references to empty objects or placeholders i… |
| 11540 → 26243 | 1.00 | 0.90 | mentions of specific clothing accessory compo… | explicit numerical identifiers and versions i… |
| 23821 → 30462 | 1.00 | 5.53 | references to monetary amounts, financial tra… | mathematical expressions related to specific … |
| 18377 → 12928 | 1.00 | 5.90 | references to specific years or dates | references to bacteria |
| 23821 → 14181 | 1.00 | 5.53 | references to monetary amounts, financial tra… | references to specific individuals named Garc… |
| 8785 → 9120 | 1.00 | 5.00 | references to specific laboratory assay types… | references to a specific chemical process or … |
| 16895 → 12928 | 1.00 | 5.65 | the names of individuals mentioned in the text | references to bacteria |
| 25602 → 9120 | 1.00 | 5.77 | references to the sensory perception of the f… | references to a specific chemical process or … |

_Duplicate pairs (rename/split candidates):_ 8289≈8695 (references to the Earth…); 8289≈9120 (references to the Earth…); 8289≈9621 (references to the Earth…); 8289≈9712 (references to the Earth…); 8289≈9762 (references to the Earth…); 8289≈9955 (references to the Earth…)
