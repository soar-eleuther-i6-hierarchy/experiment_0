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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma2_2b/layer_03/qualitative_check.html">3</a><a class="pill" href="../../../outputs/gemma2_2b/layer_06/qualitative_check.html">6</a><a class="pill" href="../../../outputs/gemma2_2b/layer_12/qualitative_check.html">12</a><a class="pill" href="../../../outputs/gemma2_2b/layer_18/qualitative_check.html">18</a><a class="pill" href="../../../outputs/gemma2_2b/layer_24/qualitative_check.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma2_2b/layer_03/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma2_2b/layer_03/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma2_2b/layer_03/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma2_2b/layer_03/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma2_2b/layer_03/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/gemma2_2b/layer_03/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **112 -> 436**  `R=0.78 F=0.14 recon_gain=1.979 recon=Y surv=0.82 p_fires=5%`
    - parent [112](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/112): _parenthetical statements_
    - child  [436](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/436): _LaTeX mathematical notation._
- **84 -> 273**  `R=0.64 F=0.13 recon_gain=1.488 recon=Y surv=0.60 p_fires=3%`
    - parent [84](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/84): _the word "this" used as a pronoun to make connections between topics_
    - child  [273](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/273): _the word "This"_
- **84 -> 284**  `R=0.71 F=0.15 recon_gain=0.976 recon=Y surv=0.84 p_fires=3%`
    - parent [84](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/84): _the word "this" used as a pronoun to make connections between topics_
    - child  [284](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/284): _the word "this", sometimes followed by "is", "can", or "that"_
- **26 -> 140**  `R=0.77 F=0.13 recon_gain=0.590 recon=Y surv=1.00 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [140](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/140): _mentions of a male person as "he" or with "him" possessive pronouns._
- **26 -> 306**  `R=0.73 F=0.15 recon_gain=0.505 recon=Y surv=0.88 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [306](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/306): _first and second person pronouns_
- **26 -> 299**  `R=0.56 F=0.15 recon_gain=0.356 recon=Y surv=1.03 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [299](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/299): _"it" or other third person pronouns_
- **48 -> 284**  `R=0.62 F=0.03 recon_gain=0.280 recon=Y surv=1.13 p_fires=13%`
    - parent [48](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/48): _various discourse markers or conjunctions_
    - child  [284](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/284): _the word "this", sometimes followed by "is", "can", or "that"_
- **31 -> 505**  `R=0.71 F=0.10 recon_gain=0.258 recon=Y surv=1.15 p_fires=2%`
    - parent [31](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/31): _horizontal lines of dashes_
    - child  [505](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/505): _legal and scientific research article formatting like spacing, section dividers, values, and symbols_

### reject:superparent  (4)

- **70 -> 394**  `R=1.00 F=0.00 recon_gain=1.146 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [394](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/394): _various fraction representations_
- **70 -> 148**  `R=1.00 F=0.01 recon_gain=0.876 recon=n surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [148](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/148): _trademarked product names_
- **70 -> 302**  `R=1.00 F=0.00 recon_gain=0.859 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [302](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/302): _citations_
- **70 -> 439**  `R=1.00 F=0.00 recon_gain=0.601 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [439](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/439): _statistical data from scientific publications_

### reject:freq-driven  (4)

- **53 -> 276**  `R=0.87 F=0.21 recon_gain=0.400 recon=Y surv=0.18 p_fires=5%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/53): _numbers, especially large numbers or sequences of digits_
    - child  [276](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/276): _strings of numbers_
- **95 -> 276**  `R=0.76 F=0.26 recon_gain=0.437 recon=Y surv=0.45 p_fires=3%`
    - parent [95](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/95): _references to outside literature_
    - child  [276](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/276): _strings of numbers_
- **95 -> 342**  `R=0.72 F=0.22 recon_gain=0.267 recon=Y surv=0.39 p_fires=3%`
    - parent [95](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/95): _references to outside literature_
    - child  [342](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/342): _legal citations_
- **53 -> 342**  `R=0.67 F=0.14 recon_gain=0.238 recon=Y surv=0.38 p_fires=5%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/53): _numbers, especially large numbers or sequences of digits_
    - child  [342](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/342): _legal citations_

### reject:no-recon  (4)

- **111 -> 188**  `R=0.78 F=0.03 recon_gain=0.044 recon=n surv=1.08 p_fires=30%`
    - parent [111](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/111): _code, formatting and markup syntax, and specialized terminology_
    - child  [188](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/188): _LaTeX code_
- **111 -> 280**  `R=0.76 F=0.25 recon_gain=0.054 recon=n surv=1.06 p_fires=30%`
    - parent [111](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/111): _code, formatting and markup syntax, and specialized terminology_
    - child  [280](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/280): _citations to published research and words indicating a study or library_
- **111 -> 494**  `R=0.76 F=0.10 recon_gain=0.060 recon=n surv=1.08 p_fires=30%`
    - parent [111](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/111): _code, formatting and markup syntax, and specialized terminology_
    - child  [494](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/494): _legal citations to the Federal Circuit court_
- **99 -> 188**  `R=0.69 F=0.02 recon_gain=0.024 recon=n surv=1.05 p_fires=34%`
    - parent [99](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/99): _words and phrases in scientific documents related to physics and quantum mechanics_
    - child  [188](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/188): _LaTeX code_
