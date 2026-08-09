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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/qualitative_check.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/qualitative_check.html">3</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_06/qualitative_check.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/qualitative_check.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/qualitative_check.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/qualitative_check.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">In-block report</a><a class="on" href="../../../outputs/gemma-2-2b/layer_06/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **29 -> 283**  `R=0.65 F=0.04 recon_gain=0.688 recon=Y surv=1.12 p_fires=1%`
    - parent [29](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/29): _citations to legal cases_
    - child  [283](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/283): _legal citations_
- **2 -> 197**  `R=0.57 F=0.18 recon_gain=0.189 recon=Y surv=1.13 p_fires=8%`
    - parent [2](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/2): _proper names and associated titles_
    - child  [197](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/197): _authors' last names_
- **2 -> 439**  `R=0.66 F=0.20 recon_gain=0.174 recon=Y surv=1.08 p_fires=8%`
    - parent [2](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/2): _proper names and associated titles_
    - child  [439](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/439): _names_
- **88 -> 504**  `R=0.64 F=0.13 recon_gain=0.172 recon=Y surv=0.61 p_fires=11%`
    - parent [88](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/88): _the words "in", "to", "from", "as", "on", "with", "for", "by", "between", "than", "while", "above", "during", "containing", "like", "of", "when", "onto", "through", "_
    - child  [504](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/504): _the word "at"_
- **35 -> 141**  `R=0.60 F=0.15 recon_gain=0.163 recon=Y surv=0.84 p_fires=12%`
    - parent [35](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/35): _mentions of mathematical derivations and existing research_
    - child  [141](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/141): _the word "the," and pronouns_
- **91 -> 444**  `R=0.60 F=0.07 recon_gain=0.147 recon=Y surv=0.97 p_fires=14%`
    - parent [91](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/91): _terms related to medical and biological processes, particularly those involving molecular biology, biochemistry, and cellular mechanisms._
    - child  [444](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/444): _mentions of specific proteins and experimental shorthand in a scientific paper_
- **72 -> 159**  `R=0.79 F=0.10 recon_gain=0.146 recon=Y surv=1.07 p_fires=2%`
    - parent [72](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/72): _various elements and separators within tables and structured data, especially statistical or scientific tables_
    - child  [159](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/159): _horizontal lines of dashes used within tables_
- **43 -> 343**  `R=0.66 F=0.06 recon_gain=0.143 recon=Y surv=1.15 p_fires=12%`
    - parent [43](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/43): _discourse markers indicating argumentation, comparison/contrast, or qualification._
    - child  [343](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/343): _the word "some" followed by a determiner or adjective._

### reject:superparent  (4)

- **15 -> 433**  `R=1.00 F=0.01 recon_gain=1.690 recon=Y surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [433](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/433): _occurrences of the word "then" and nearby words_
- **15 -> 344**  `R=1.00 F=0.01 recon_gain=2.184 recon=Y surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [344](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/344): _LaTeX math mode formatting_
- **15 -> 133**  `R=1.00 F=0.01 recon_gain=1.075 recon=Y surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [133](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/133): _the word "name" and its variants in programming contexts_
- **15 -> 283**  `R=1.00 F=0.00 recon_gain=0.724 recon=Y surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [283](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/283): _legal citations_

### reject:freq-driven  (4)

- **8 -> 479**  `R=0.79 F=0.36 recon_gain=0.410 recon=Y surv=0.39 p_fires=5%`
    - parent [8](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/8): _code blocks and math formulas_
    - child  [479](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/479): _leading whitespace followed by numerical values_
- **124 -> 333**  `R=0.73 F=0.10 recon_gain=0.325 recon=Y surv=0.39 p_fires=5%`
    - parent [124](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/124): _floating point numbers_
    - child  [333](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/333): _hexadecimal-like codes_
- **88 -> 480**  `R=0.67 F=0.30 recon_gain=0.196 recon=Y surv=0.47 p_fires=11%`
    - parent [88](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/88): _the words "in", "to", "from", "as", "on", "with", "for", "by", "between", "than", "while", "above", "during", "containing", "like", "of", "when", "onto", "through", "_
    - child  [480](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/480): _prepositions_
- **23 -> 365**  `R=0.66 F=0.21 recon_gain=0.228 recon=Y surv=0.20 p_fires=4%`
    - parent [23](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/23): _occurrences of numbers, likely data points extracted from medical studies_
    - child  [365](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/365): _decimal numbers greater than 10_

### reject:no-recon  (4)

- **62 -> 238**  `R=0.79 F=0.04 recon_gain=0.040 recon=n surv=0.97 p_fires=17%`
    - parent [62](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/62): _proper nouns relating to places, people, organizations, events, or works of art._
    - child  [238](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/238): _LaTeX math symbols_
- **89 -> 275**  `R=0.77 F=0.21 recon_gain=0.101 recon=n surv=1.03 p_fires=21%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/89): _abbreviations and identifiers like codes or chemical formulae including "cli", "acr", "task", "PTRH2", "KO", "MDR", "ATD", "DEA", "AdoHcyase", "ORM"_
    - child  [275](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/275): _markup, code, or other structured text_
- **103 -> 238**  `R=0.76 F=0.08 recon_gain=0.146 recon=n surv=0.79 p_fires=8%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/103): _code containing hexadecimal values preceded by "x"_
    - child  [238](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/238): _LaTeX math symbols_
- **116 -> 264**  `R=0.75 F=0.04 recon_gain=0.032 recon=n surv=1.00 p_fires=36%`
    - parent [116](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/116): _C programming language code_
    - child  [264](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/264): _code-related documents_
