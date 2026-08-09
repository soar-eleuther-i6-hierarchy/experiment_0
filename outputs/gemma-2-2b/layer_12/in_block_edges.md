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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/in_block_edges.html">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">6</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_12/in_block_edges.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/in_block_edges.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/in_block_edges.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_12/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma-2-2b/layer_12/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/qualitative_check.html">Qualitative report</a></div></nav>

# In-block (same-level) directed edges

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **417** directed edges, **3** duplicate pairs, 302 survive PMI>0; PolyFrac 99%, Gini 0.951.
- In-block superparents: 2 (e.g. F44 _technical terminology related to programming …_: 125 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 44 → 25 | 1.00 | 0.01 | technical terminology related to programming … | letters and symbols, particularly those assoc… |
| 44 → 106 | 1.00 | 0.01 | technical terminology related to programming … | biological processes and mechanisms related t… |
| 44 → 113 | 1.00 | 0.01 | technical terminology related to programming … | technical terms related to electrical and mec… |
| 44 → 111 | 1.00 | 0.01 | technical terminology related to programming … | scientific terminology related to genetic ana… |
| 44 → 30 | 1.00 | 0.01 | technical terminology related to programming … | phrases related to healthcare and training in… |
| 44 → 40 | 1.00 | 0.01 | technical terminology related to programming … | expressions of personal experiences and emoti… |
| 44 → 92 | 1.00 | 0.01 | technical terminology related to programming … | mentions of legal proceedings and implication… |
| 44 → 101 | 1.00 | 0.01 | technical terminology related to programming … | terms related to health and healthcare resear… |

_Duplicate pairs (rename/split candidates):_ 27≈75 (mathematical expression…); 44≈114 (technical terminology r…); 82≈101 (technical terms related…)

## Block B1  (384 features)
- **77** directed edges, **0** duplicate pairs, 77 survive PMI>0; PolyFrac 9%, Gini 0.995.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 383 → 487 | 0.86 | 0.75 | terms related to mathematical proofs and stru… | mathematical notation and expressions related… |
| 383 → 350 | 0.81 | 0.69 | terms related to mathematical proofs and stru… | syntactical structures and special characters… |
| 383 → 499 | 0.80 | 0.67 | terms related to mathematical proofs and stru… | references to legal proceedings and outcomes |
| 383 → 133 | 0.77 | 0.63 | terms related to mathematical proofs and stru… | technical terms related to computer systems a… |
| 383 → 488 | 0.74 | 0.59 | terms related to mathematical proofs and stru… | topics related to indigenous peoples and thei… |
| 383 → 398 | 0.73 | 0.58 | terms related to mathematical proofs and stru… | incomplete or interrupted sentences and their… |
| 383 → 249 | 0.67 | 0.50 | terms related to mathematical proofs and stru… | references to legal cases and jurisdictions |
| 383 → 188 | 0.66 | 0.48 | terms related to mathematical proofs and stru… | references to legal terminology and court cas… |

## Block B2  (1536 features)
- **268** directed edges, **10** duplicate pairs, 268 survive PMI>0; PolyFrac 21%, Gini 0.963.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1011 → 1001 | 0.75 | 2.05 | technical descriptions of computer systems an… | mathematical expressions and notations relate… |
| 1925 → 1001 | 0.74 | 3.54 | CSS-related issues, particularly those involv… | mathematical expressions and notations relate… |
| 887 → 1001 | 0.73 | 3.02 | technical terms related to programming and so… | mathematical expressions and notations relate… |
| 2002 → 1001 | 0.73 | 5.17 | JSON-like structure elements and properties | mathematical expressions and notations relate… |
| 1916 → 1001 | 0.73 | 3.18 | legal and medical terms related to injuries a… | mathematical expressions and notations relate… |
| 761 → 1001 | 0.73 | 3.43 | references to data structures or programming … | mathematical expressions and notations relate… |
| 1110 → 1001 | 0.73 | 2.88 | key terms related to healthcare professionals… | mathematical expressions and notations relate… |
| 1075 → 1001 | 0.73 | 4.78 | JSON-like structured data elements and key-va… | mathematical expressions and notations relate… |

_Duplicate pairs (rename/split candidates):_ 1001≈1052 (mathematical expression…); 1001≈1466 (mathematical expression…); 1001≈1659 (mathematical expression…); 1001≈1883 (mathematical expression…); 1052≈1466 (mathematical symbols an…); 1052≈1659 (mathematical symbols an…)

## Block B3  (6144 features)
- **1844** directed edges, **258** duplicate pairs, 1844 survive PMI>0; PolyFrac 37%, Gini 0.986.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5636 → 6486 | 1.00 | 4.71 | mathematical notations and symbols | mathematical symbols and expressions, particu… |
| 3394 → 6486 | 1.00 | 4.68 | discussions related to cryptography and encry… | mathematical symbols and expressions, particu… |
| 6216 → 6486 | 0.98 | 5.32 | numerical data related to counts, measurement… | mathematical symbols and expressions, particu… |
| 3873 → 6486 | 0.98 | 3.80 | phrases related to medical guidelines and pro… | mathematical symbols and expressions, particu… |
| 7244 → 6486 | 0.98 | 4.84 | references to academic work and acknowledgmen… | mathematical symbols and expressions, particu… |
| 3145 → 6486 | 0.98 | 5.37 | medical terminologies related to diagnostic m… | mathematical symbols and expressions, particu… |
| 3179 → 6486 | 0.98 | 4.71 | phrases related to political campaigns and th… | mathematical symbols and expressions, particu… |
| 4609 → 6486 | 0.98 | 2.98 | terms related to magnetic properties and perm… | mathematical symbols and expressions, particu… |

_Duplicate pairs (rename/split candidates):_ 2168≈2209 (structured programming …); 2168≈3519 (structured programming …); 2168≈3541 (structured programming …); 2168≈3661 (structured programming …); 2168≈3671 (structured programming …); 2168≈3788 (structured programming …)
