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
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../cross_depth_comparison__v1__2026-08-06T17-47.html">Cross-depth</a><a class="" href="../kill_rates__v1__2026-08-06T17-47.html">Kill rates</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/layer_03/qualitative_check.html">3</a><a class="pill" href="../../outputs/layer_06/qualitative_check.html">6</a><a class="pill" href="../../outputs/layer_12/qualitative_check.html">12</a><a class="pill" href="../../outputs/layer_18/qualitative_check.html">18</a><a class="pill on" href="../../outputs/layer_24/qualitative_check.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../outputs/layer_24/metrics_dashboard.html">Dashboard</a><a class="" href="../../outputs/layer_24/superparent_sankey.html">Superparents</a><a class="" href="../../outputs/layer_24/qualitative_dashboard.html">Qualitative</a><a class="" href="../../outputs/layer_24/metrics_report.html">Metrics report</a><a class="on" href="../../outputs/layer_24/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **14 -> 308**  `R=0.92 F=0.03 recon_gain=0.719 recon=Y surv=0.72 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [308](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/308): _hyphens_
- **14 -> 467**  `R=0.92 F=0.04 recon_gain=0.684 recon=Y surv=0.83 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [467](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/467): _symbols used in scientific publications, particularly in articles about chemistry or physics_
- **14 -> 191**  `R=0.94 F=0.09 recon_gain=0.489 recon=Y surv=0.96 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [191](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/191): _place names in Argentina_
- **14 -> 230**  `R=0.83 F=0.01 recon_gain=0.457 recon=Y surv=1.06 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [230](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/230): _references to figures, tables, equations, or supplementary material_
- **14 -> 345**  `R=0.78 F=0.01 recon_gain=0.450 recon=Y surv=0.98 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [345](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/345): _HTML closing tags_
- **103 -> 192**  `R=0.73 F=0.07 recon_gain=0.428 recon=Y surv=1.05 p_fires=5%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/103): _Danish words or phrases, especially those that might be movie titles, as well as parenthetical year dates_
    - child  [192](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/192): _code snippets and questions about programming, some of which are in Russian_
- **103 -> 493**  `R=0.91 F=0.12 recon_gain=0.393 recon=Y surv=1.04 p_fires=5%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/103): _Danish words or phrases, especially those that might be movie titles, as well as parenthetical year dates_
    - child  [493](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/493): _mixed English and Russian text related to website coding_
- **14 -> 256**  `R=0.76 F=0.02 recon_gain=0.326 recon=Y surv=0.90 p_fires=42%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/14): _code and related symbols, alongside references and math within research papers_
    - child  [256](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/256): _section headings and figure/table/equation references in research articles_

### reject:superparent  (4)

- **32 -> 201**  `R=1.00 F=0.03 recon_gain=1.146 recon=Y surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [201](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/201): _names of people and titles or abbreviations referring to people_
- **32 -> 387**  `R=1.00 F=0.02 recon_gain=0.000 recon=n surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [387](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/387): _contractions with the word "not."_
- **32 -> 193**  `R=1.00 F=0.00 recon_gain=1.102 recon=n surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [193](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/193): _language common to legal opinions._
- **32 -> 365**  `R=1.00 F=0.01 recon_gain=-0.000 recon=n surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [365](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/365): _citations in research papers_

### reject:freq-driven  (4)

- **57 -> 292**  `R=0.94 F=0.04 recon_gain=-0.001 recon=n surv=0.41 p_fires=22%`
    - parent [57](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/57): _currency symbols from various countries_
    - child  [292](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/292): _conditional statements and array iteration in code_
- **115 -> 292**  `R=0.94 F=0.06 recon_gain=-0.001 recon=n surv=0.46 p_fires=14%`
    - parent [115](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/115): _code snippets_
    - child  [292](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/292): _conditional statements and array iteration in code_
- **76 -> 292**  `R=0.94 F=0.09 recon_gain=-0.000 recon=n surv=0.46 p_fires=9%`
    - parent [76](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/76): _code debugging related terms and outputs, focused on parallel processing_
    - child  [292](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/292): _conditional statements and array iteration in code_
- **80 -> 292**  `R=0.94 F=0.05 recon_gain=-0.000 recon=n surv=0.37 p_fires=15%`
    - parent [80](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/80): _capitalized words and phrases that appear to be commands or strong assertions_
    - child  [292](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/292): _conditional statements and array iteration in code_

### reject:no-recon  (4)

- **34 -> 465**  `R=0.98 F=0.16 recon_gain=-0.000 recon=n surv=0.83 p_fires=6%`
    - parent [34](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/34): _code-related keywords within a code-heavy document._
    - child  [465](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/465): _references to figures and tables_
- **116 -> 465**  `R=0.98 F=0.11 recon_gain=0.000 recon=n surv=0.94 p_fires=8%`
    - parent [116](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/116): _code blocks_
    - child  [465](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/465): _references to figures and tables_
- **25 -> 465**  `R=0.98 F=0.13 recon_gain=-0.001 recon=n surv=0.79 p_fires=7%`
    - parent [25](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/25): _CSS or styling code_
    - child  [465](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/465): _references to figures and tables_
- **73 -> 446**  `R=0.96 F=0.19 recon_gain=0.000 recon=n surv=1.00 p_fires=14%`
    - parent [73](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/73): _technical academic language_
    - child  [446](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/446): _words related to events, changes, difficulties, or limitations_
