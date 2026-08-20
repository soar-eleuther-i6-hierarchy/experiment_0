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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# In-block (same-level) directed edges

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **640** directed edges, **13** duplicate pairs, 411 survive PMI>0; PolyFrac 100%, Gini 0.944.
- In-block superparents: 6 (e.g. F32 _numbers within a document_: 123 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 32 → 84 | 1.00 | 0.01 | numbers within a document | periods at the end of sentences, especially i… |
| 32 → 72 | 1.00 | 0.01 | numbers within a document | snippets of mathematical proofs |
| 32 → 58 | 1.00 | 0.01 | numbers within a document | LaTeX mathematical notation |
| 32 → 83 | 1.00 | 0.00 | numbers within a document | detailed storytelling of dramatic events invo… |
| 32 → 104 | 1.00 | 0.00 | numbers within a document | scientific jargon related to biological proce… |
| 32 → 126 | 1.00 | 0.00 | numbers within a document | dates with months and years |
| 32 → 88 | 1.00 | 0.00 | numbers within a document | words and symbols related to measurements in … |
| 32 → 51 | 1.00 | 0.00 | numbers within a document | words related to online sales and marketing |

_Duplicate pairs (rename/split candidates):_ 19≈32 (Japanese names or media…); 19≈49 (Japanese names or media…); 19≈85 (Japanese names or media…); 19≈92 (Japanese names or media…); 32≈49 (numbers within a docume…); 32≈85 (numbers within a docume…)

## Block B1  (384 features)
- **26** directed edges, **0** duplicate pairs, 26 survive PMI>0; PolyFrac 14%, Gini 0.980.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 471 → 484 | 0.91 | 1.96 | code snippets and related terms like "diff," … | words pertaining to social structures and org… |
| 226 → 186 | 0.64 | 1.63 | equations | characters and symbols associated with mathem… |
| 336 → 498 | 0.64 | 3.20 | numbers, especially as part of references, co… | single quotes and apostrophes |
| 207 → 310 | 0.63 | 1.90 | words related to biological research and medi… | mentions of antibodies and animal types used … |
| 207 → 426 | 0.61 | 1.87 | words related to biological research and medi… | text from scientific research papers about te… |
| 500 → 186 | 0.59 | 2.56 | words related to knowing, wondering, and tell… | characters and symbols associated with mathem… |
| 207 → 169 | 0.58 | 1.80 | words related to biological research and medi… | terms related to biological experiments and s… |
| 207 → 396 | 0.57 | 1.80 | words related to biological research and medi… | names of chemical structures and analysis |

## Block B2  (1536 features)
- **1155** directed edges, **6** duplicate pairs, 859 survive PMI>0; PolyFrac 7%, Gini 0.990.
- In-block superparents: 1 (e.g. F2042 _words related to court cases, delays in proje…_: 934 children, fires 59%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1234 → 887 | 0.91 | 5.04 | Lithuanian words within a historical context,… | words in a Scandinavian language, possibly Da… |
| 1830 → 1323 | 0.86 | 4.98 | non-English words such as those used to descr… | code or programming discussion in Spanish |
| 730 → 1234 | 0.85 | 3.83 | names and occupations within a biographical c… | Lithuanian words within a historical context,… |
| 2042 → 1096 | 0.85 | 0.36 | words related to court cases, delays in proje… | complex clauses involving multiple people, re… |
| 2042 → 1313 | 0.84 | 0.35 | words related to court cases, delays in proje… | words and numbers somewhat randomly |
| 1475 → 545 | 0.84 | 3.83 | medical research papers and articles | authors' last names, publication years and ot… |
| 730 → 887 | 0.83 | 3.80 | names and occupations within a biographical c… | words in a Scandinavian language, possibly Da… |
| 2042 → 1904 | 0.82 | 0.33 | words related to court cases, delays in proje… | technical descriptions of power inverters |

_Duplicate pairs (rename/split candidates):_ 582≈1831 (punctuation and discour…); 1032≈1234 (French text where peopl…); 1114≈1170 (math problems involving…); 1117≈1660 (russian words, especial…); 1210≈1810 (code snippets and Japan…); 1823≈1890 (number comparisons and …)

## Block B3  (6144 features)
- **969** directed edges, **54** duplicate pairs, 969 survive PMI>0; PolyFrac 28%, Gini 0.980.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 3416 → 7729 | 0.96 | 5.83 | a seemingly random collection of abbreviation… | characters with diacritics or less common let… |
| 4210 → 3416 | 0.94 | 4.63 | mentions of funding and research institutions | a seemingly random collection of abbreviation… |
| 2533 → 3200 | 0.94 | 4.58 | chemical formulas, numbers, units, and codes | sentences that describe the authors of a study |
| 2056 → 3200 | 0.94 | 5.02 | research papers describing studies, especiall… | sentences that describe the authors of a study |
| 4210 → 7729 | 0.94 | 4.62 | mentions of funding and research institutions | characters with diacritics or less common let… |
| 5528 → 5354 | 0.93 | 4.58 | segments of a song lyric that expresses relat… | instances of confusion, failure and a general… |
| 5971 → 6106 | 0.93 | 4.31 | code snippets, particularly those defining re… | Java code related to executing queries using … |
| 6882 → 3200 | 0.92 | 4.99 | hexadecimal color codes and date formats | sentences that describe the authors of a study |

_Duplicate pairs (rename/split candidates):_ 2481≈2758 (text related to car spe…); 2481≈6767 (text related to car spe…); 2572≈5553 (words related to softwa…); 2597≈3113 (the description of base…); 2597≈3308 (the description of base…); 2597≈3411 (the description of base…)
