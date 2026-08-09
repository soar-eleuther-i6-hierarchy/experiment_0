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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **107 -> 223**  `R=0.78 F=0.12 recon_gain=0.369 recon=Y surv=0.97 p_fires=10%`
    - parent [107](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/107): _text from patents or technical documents._
    - child  [223](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/223): _the beginning sections of technical documentation for inventions_
- **18 -> 281**  `R=0.69 F=0.43 recon_gain=0.309 recon=Y surv=0.67 p_fires=5%`
    - parent [18](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/18): _sentences that end with citations or numerical values in parentheses or brackets_
    - child  [281](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/281): _source code syntax_
- **53 -> 363**  `R=0.62 F=0.14 recon_gain=0.190 recon=Y surv=0.97 p_fires=5%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/53): _the verb "is" along with other auxiliary verbs ("are", "was", "be", "been")_
    - child  [363](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/363): _the word "be" in various forms_
- **82 -> 502**  `R=0.56 F=0.08 recon_gain=0.139 recon=Y surv=0.60 p_fires=13%`
    - parent [82](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/82): _small, common words such as prepositions and conjunctions_
    - child  [502](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/502): _the word "for" used as an introductory word._
- **93 -> 397**  `R=0.62 F=0.16 recon_gain=0.133 recon=Y surv=0.98 p_fires=5%`
    - parent [93](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/93): _sentences that are providing context or a transition between ideas_
    - child  [397](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/397): _commas within longer sentences and paragraphs._
- **63 -> 179**  `R=0.66 F=0.20 recon_gain=0.113 recon=Y surv=1.16 p_fires=18%`
    - parent [63](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/63): _terms, concepts, and processes from scientific journals, with a bias towards biology and medicine_
    - child  [179](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/179): _words related to data processing, parameters, and setting limits on processes_
- **63 -> 250**  `R=0.61 F=0.27 recon_gain=0.112 recon=Y surv=1.03 p_fires=18%`
    - parent [63](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/63): _terms, concepts, and processes from scientific journals, with a bias towards biology and medicine_
    - child  [250](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/250): _words and phrases dealing with system functions, technical processes, and problem-solving_
- **8 -> 272**  `R=0.64 F=0.11 recon_gain=0.111 recon=Y surv=1.21 p_fires=10%`
    - parent [8](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/8): _words related to scientific or technical jargon, especially related to medicine, physics, and chemistry._
    - child  [272](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/272): _instances of words that start with the prefix "sub-", and a few prefixes consisting of two letters_

### reject:superparent  (4)

- **89 -> 249**  `R=1.00 F=0.06 recon_gain=0.026 recon=n surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [249](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/249): _references to political parties and elections, especially regarding the Democratic and Green parties._
- **89 -> 263**  `R=1.00 F=0.02 recon_gain=1.949 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [263](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/263): _mentions of different kinds of government and municipal departments_
- **89 -> 132**  `R=1.00 F=0.02 recon_gain=1.302 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [132](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/132): _hyphens used to connect words_
- **89 -> 166**  `R=1.00 F=0.02 recon_gain=0.026 recon=n surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [166](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/166): _the noun "response"_

### reject:freq-driven  (4)

- **13 -> 471**  `R=0.92 F=0.15 recon_gain=-0.000 recon=n surv=0.16 p_fires=6%`
    - parent [13](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/13): _statistical data with numerical values_
    - child  [471](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/471): _random alphanumeric strings_
- **43 -> 435**  `R=0.91 F=0.03 recon_gain=0.000 recon=n surv=0.49 p_fires=29%`
    - parent [43](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/43): _code, financial and political jargon_
    - child  [435](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/435): _French characters with accents_
- **21 -> 471**  `R=0.91 F=0.16 recon_gain=-0.003 recon=n surv=0.12 p_fires=5%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/21): _numbers and related symbols in tables_
    - child  [471](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/471): _random alphanumeric strings_
- **42 -> 471**  `R=0.91 F=0.04 recon_gain=-0.002 recon=n surv=0.39 p_fires=21%`
    - parent [42](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/42): _technical and medical language_
    - child  [471](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/471): _random alphanumeric strings_

### reject:no-recon  (4)

- **8 -> 435**  `R=0.93 F=0.09 recon_gain=-0.000 recon=n surv=0.70 p_fires=10%`
    - parent [8](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/8): _words related to scientific or technical jargon, especially related to medicine, physics, and chemistry._
    - child  [435](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/435): _French characters with accents_
- **91 -> 435**  `R=0.92 F=0.05 recon_gain=-0.000 recon=n surv=0.56 p_fires=18%`
    - parent [91](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/91): _source code or programming-related text, including numbers, names, and attributes._
    - child  [435](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/435): _French characters with accents_
- **8 -> 471**  `R=0.91 F=0.08 recon_gain=-0.000 recon=n surv=0.63 p_fires=10%`
    - parent [8](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/8): _words related to scientific or technical jargon, especially related to medicine, physics, and chemistry._
    - child  [471](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/471): _random alphanumeric strings_
- **43 -> 471**  `R=0.90 F=0.03 recon_gain=0.000 recon=n surv=0.51 p_fires=29%`
    - parent [43](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/43): _code, financial and political jargon_
    - child  [471](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/471): _random alphanumeric strings_
