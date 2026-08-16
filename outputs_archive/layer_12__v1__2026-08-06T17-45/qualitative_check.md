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
.x0nav details.dens{display:flex;flex-wrap:wrap;align-items:center;gap:13px;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
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

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **27 -> 410**  `R=0.53 F=0.04 recon_gain=0.288 recon=Y surv=0.61 p_fires=7%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/27): _mathematical expressions or equations related to inequalities_
    - child  [410](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/410): _references to financial transactions or economic concepts_
- **69 -> 174**  `R=0.62 F=0.18 recon_gain=0.205 recon=Y surv=0.52 p_fires=11%`
    - parent [69](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/69): _phrases that indicate relationships, particularly in the context of implementation and effect_
    - child  [174](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/174): _the presence of the word "in" across different contexts_
- **10 -> 370**  `R=0.74 F=0.17 recon_gain=0.164 recon=Y surv=1.16 p_fires=7%`
    - parent [10](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/10): _names of individuals in various contexts_
    - child  [370](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/370): _proper names and specific identifiers within text_
- **22 -> 426**  `R=0.55 F=0.27 recon_gain=0.143 recon=Y surv=0.61 p_fires=4%`
    - parent [22](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/22): _mentions of specific metrics or numerical data related to studies or evaluations_
    - child  [426](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/426): _quantitative data regarding measurements and statistics_
- **69 -> 263**  `R=0.52 F=0.12 recon_gain=0.142 recon=Y surv=0.53 p_fires=11%`
    - parent [69](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/69): _phrases that indicate relationships, particularly in the context of implementation and effect_
    - child  [263](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/263): _references to concerns or issues regarding societal topics and discussions_
- **27 -> 511**  `R=0.59 F=0.08 recon_gain=0.140 recon=Y surv=0.67 p_fires=7%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/27): _mathematical expressions or equations related to inequalities_
    - child  [511](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/511): _specific formatting elements or symbols used in code or documentation_
- **3 -> 282**  `R=0.51 F=0.16 recon_gain=0.134 recon=Y surv=0.85 p_fires=5%`
    - parent [3](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/3): _instances of the verb "to be" in various forms_
    - child  [282](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/282): _phrases related to existence or presence, particularly focusing on the verb "be."_
- **106 -> 444**  `R=0.65 F=0.06 recon_gain=0.125 recon=Y surv=0.96 p_fires=15%`
    - parent [106](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/106): _biological processes and mechanisms related to gene expression and cellular responses_
    - child  [444](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/444): _biological terms related to cancer and apoptosis processes_

### reject:superparent  (4)

- **44 -> 272**  `R=1.00 F=0.02 recon_gain=0.179 recon=n surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [272](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/272): _expressions of encouragement and support in conversations_
- **44 -> 318**  `R=1.00 F=0.03 recon_gain=0.180 recon=n surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [318](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/318): _code-related constructs and syntax_
- **44 -> 139**  `R=1.00 F=0.00 recon_gain=1.427 recon=Y surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [139](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/139): _mathematical expressions and calculations_
- **44 -> 162**  `R=1.00 F=0.01 recon_gain=0.179 recon=n surv=- p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [162](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/162): _instances of asynchronous processing or communication_

### reject:freq-driven  (4)

- **100 -> 328**  `R=0.96 F=0.06 recon_gain=-0.000 recon=n surv=0.15 p_fires=14%`
    - parent [100](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/100): _statistical data and numerical values_
    - child  [328](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/328): _references to mathematical labels or equations_
- **19 -> 328**  `R=0.96 F=0.10 recon_gain=-0.001 recon=n surv=0.15 p_fires=8%`
    - parent [19](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/19): _function definitions and calls in programming contexts_
    - child  [328](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/328): _references to mathematical labels or equations_
- **123 -> 328**  `R=0.96 F=0.16 recon_gain=-0.000 recon=n surv=0.00 p_fires=5%`
    - parent [123](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/123): _lines of code or programming-related structures_
    - child  [328](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/328): _references to mathematical labels or equations_
- **106 -> 328**  `R=0.96 F=0.05 recon_gain=-0.000 recon=n surv=0.00 p_fires=15%`
    - parent [106](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/106): _biological processes and mechanisms related to gene expression and cellular responses_
    - child  [328](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/328): _references to mathematical labels or equations_

### reject:no-recon  (4)

- **77 -> 249**  `R=0.87 F=0.05 recon_gain=-0.015 recon=n surv=0.91 p_fires=25%`
    - parent [77](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/77): _technical and scientific terms related to typesetting and printing processes_
    - child  [249](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/249): _references to legal cases and jurisdictions_
- **86 -> 254**  `R=0.87 F=0.03 recon_gain=-0.001 recon=n surv=0.73 p_fires=34%`
    - parent [86](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/86): _items related to beach or water sports equipment_
    - child  [254](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/254): _legal and regulatory references, particularly related to the Consumer Credit Protection Act and equal protection statutes_
- **86 -> 399**  `R=0.84 F=0.06 recon_gain=-0.001 recon=n surv=0.93 p_fires=34%`
    - parent [86](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/86): _items related to beach or water sports equipment_
    - child  [399](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/399): _chemical compounds and their classifications_
- **106 -> 399**  `R=0.83 F=0.13 recon_gain=-0.000 recon=n surv=0.94 p_fires=15%`
    - parent [106](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/106): _biological processes and mechanisms related to gene expression and cellular responses_
    - child  [399](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/399): _chemical compounds and their classifications_
