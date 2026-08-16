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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/qualitative_check.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/qualitative_check.html">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/qualitative_check.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/qualitative_check.html">12</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_18/qualitative_check.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/qualitative_check.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_18/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/in_block_edges.html">In-block report</a><a class="on" href="../../../outputs/gemma-2-2b/layer_18/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **107 -> 223**  `R=0.78 F=0.13 recon_gain=0.369 recon=Y surv=0.95 p_fires=10%`
    - parent [107](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/107): _text from patents or technical documents._
    - child  [223](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/223): _the beginning sections of technical documentation for inventions_
- **18 -> 281**  `R=0.69 F=0.43 recon_gain=0.309 recon=Y surv=0.67 p_fires=5%`
    - parent [18](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/18): _sentences that end with citations or numerical values in parentheses or brackets_
    - child  [281](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/281): _source code syntax_
- **82 -> 499**  `R=0.85 F=0.14 recon_gain=0.276 recon=Y surv=0.65 p_fires=13%`
    - parent [82](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/82): _small, common words such as prepositions and conjunctions_
    - child  [499](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/499): _the preposition "in"._
- **27 -> 420**  `R=0.94 F=0.04 recon_gain=0.248 recon=Y surv=1.00 p_fires=34%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/27): _snippets of code, mathematical formulas, and citation-like text_
    - child  [420](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/420): _blank lines_
- **53 -> 309**  `R=0.64 F=0.25 recon_gain=0.220 recon=Y surv=0.73 p_fires=5%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/53): _the verb "is" along with other auxiliary verbs ("are", "was", "be", "been")_
    - child  [309](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/309): _uses of the verb "to be" in different tenses._
- **27 -> 475**  `R=0.64 F=0.01 recon_gain=0.210 recon=Y surv=0.90 p_fires=34%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/27): _snippets of code, mathematical formulas, and citation-like text_
    - child  [475](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/475): _math formulas including derivatives and functions with variables_
- **83 -> 317**  `R=0.60 F=0.13 recon_gain=0.196 recon=Y surv=1.04 p_fires=4%`
    - parent [83](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/83): _months and years_
    - child  [317](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/317): _identifies months of the year, especially when followed by a number_
- **56 -> 458**  `R=0.88 F=0.11 recon_gain=0.191 recon=Y surv=1.07 p_fires=32%`
    - parent [56](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/56): _words and phrases related to documentation, data, and specifications in technical fields like programming, engineering, or science._
    - child  [458](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/458): _words related to academic, sporting, governmental, technological, and musical professions and endeavors._

### reject:superparent  (4)

- **89 -> 263**  `R=1.00 F=0.02 recon_gain=1.949 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [263](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/263): _mentions of different kinds of government and municipal departments_
- **89 -> 132**  `R=1.00 F=0.02 recon_gain=1.302 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [132](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/132): _hyphens used to connect words_
- **89 -> 249**  `R=1.00 F=0.05 recon_gain=2.136 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [249](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/249): _references to political parties and elections, especially regarding the Democratic and Green parties._
- **89 -> 261**  `R=1.00 F=0.07 recon_gain=2.682 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [261](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/261): _passages with emotional and personal reflections about poignant life events_

### reject:freq-driven  (4)

- **61 -> 397**  `R=0.72 F=0.23 recon_gain=0.086 recon=Y surv=0.38 p_fires=4%`
    - parent [61](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/61): _commas and various punctuation marks._
    - child  [397](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/397): _commas within longer sentences and paragraphs._
- **70 -> 183**  `R=0.69 F=0.33 recon_gain=0.173 recon=Y surv=0.44 p_fires=5%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/70): _indented code snippets and other code-related artifacts_
    - child  [183](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/183): _code or configuration snippets_
- **82 -> 246**  `R=0.68 F=0.17 recon_gain=0.148 recon=Y surv=0.28 p_fires=13%`
    - parent [82](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/82): _small, common words such as prepositions and conjunctions_
    - child  [246](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/246): _the word "of" and the phrase "of the."_
- **21 -> 456**  `R=0.65 F=0.26 recon_gain=0.049 recon=Y surv=0.28 p_fires=5%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/21): _numbers and related symbols in tables_
    - child  [456](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/456): _the number 4_

### reject:no-recon  (4)

- **56 -> 152**  `R=0.75 F=0.13 recon_gain=0.095 recon=n surv=1.14 p_fires=32%`
    - parent [56](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/56): _words and phrases related to documentation, data, and specifications in technical fields like programming, engineering, or science._
    - child  [152](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/152): _medical or scientific terms, experiments, and study results, often also flagging numbers and percentages._
- **95 -> 411**  `R=0.71 F=0.08 recon_gain=0.052 recon=n surv=0.91 p_fires=28%`
    - parent [95](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/95): _text referring to people's biographies, including names, locations, organizations and timeline data_
    - child  [411](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/411): _clauses in historical narrative sentences_
- **60 -> 142**  `R=0.69 F=0.07 recon_gain=0.033 recon=n surv=1.03 p_fires=39%`
    - parent [60](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/60): _technical descriptions of equipment used in scientific experiments._
    - child  [142](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/142): _words and phrases related to liquids and gasses flowing through pipes/tanks/reservoirs._
- **60 -> 335**  `R=0.68 F=0.02 recon_gain=0.017 recon=n surv=1.00 p_fires=39%`
    - parent [60](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/60): _technical descriptions of equipment used in scientific experiments._
    - child  [335](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/335): _code snippets from various different programming languages_
