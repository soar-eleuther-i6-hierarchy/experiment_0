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

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **29 -> 283**  `R=0.65 F=0.03 recon_gain=0.688 recon=Y surv=1.12 p_fires=2%`
    - parent [29](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/29): _citations to legal cases_
    - child  [283](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/283): _legal citations_
- **2 -> 197**  `R=0.57 F=0.16 recon_gain=0.189 recon=Y surv=1.13 p_fires=8%`
    - parent [2](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/2): _proper names and associated titles_
    - child  [197](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/197): _authors' last names_
- **35 -> 141**  `R=0.60 F=0.14 recon_gain=0.163 recon=Y surv=0.84 p_fires=12%`
    - parent [35](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/35): _mentions of mathematical derivations and existing research_
    - child  [141](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/141): _the word "the," and pronouns_
- **72 -> 159**  `R=0.79 F=0.06 recon_gain=0.146 recon=Y surv=1.07 p_fires=2%`
    - parent [72](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/72): _various elements and separators within tables and structured data, especially statistical or scientific tables_
    - child  [159](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/159): _horizontal lines of dashes used within tables_
- **43 -> 383**  `R=0.66 F=0.08 recon_gain=0.133 recon=Y surv=1.04 p_fires=12%`
    - parent [43](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/43): _discourse markers indicating argumentation, comparison/contrast, or qualification._
    - child  [383](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/383): _words or phrases used to transition to the next thought or idea_
- **72 -> 462**  `R=0.59 F=0.05 recon_gain=0.132 recon=Y surv=1.32 p_fires=2%`
    - parent [72](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/72): _various elements and separators within tables and structured data, especially statistical or scientific tables_
    - child  [462](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/462): _section dividers in research papers_
- **117 -> 476**  `R=0.59 F=0.09 recon_gain=0.126 recon=Y surv=1.14 p_fires=18%`
    - parent [117](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/117): _words that indicate something is related to academia, number, or computer programming_
    - child  [476](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/476): _phrases containing the word 'large' or words meaning 'large'_
- **89 -> 455**  `R=0.69 F=0.08 recon_gain=0.111 recon=Y surv=1.06 p_fires=20%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/89): _abbreviations and identifiers like codes or chemical formulae including "cli", "acr", "task", "PTRH2", "KO", "MDR", "ATD", "DEA", "AdoHcyase", "ORM"_
    - child  [455](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/455): _mentions of people and organizations involved in legal cases_

### reject:superparent  (4)

- **15 -> 395**  `R=1.00 F=0.02 recon_gain=0.038 recon=n surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [395](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/395): _percentages_
- **15 -> 258**  `R=1.00 F=0.02 recon_gain=0.038 recon=n surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [258](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/258): _the word "while" or "While" when used to introduce a subordinate clause_
- **15 -> 344**  `R=1.00 F=0.01 recon_gain=0.038 recon=n surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [344](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/344): _LaTeX math mode formatting_
- **15 -> 133**  `R=1.00 F=0.02 recon_gain=0.038 recon=n surv=1.00 p_fires=99%`
    - parent [15](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/15): _technical documentation-like language, including code snippets, software names, specifications, and error messages._
    - child  [133](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/133): _the word "name" and its variants in programming contexts_

### reject:freq-driven  (4)

- **123 -> 235**  `R=0.95 F=0.07 recon_gain=-0.000 recon=n surv=0.28 p_fires=11%`
    - parent [123](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/123): _numerical data and special characters often used in code or data representation_
    - child  [235](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/235): _math-related symbols like multiplication signs and exponents._
- **13 -> 235**  `R=0.95 F=0.15 recon_gain=-0.000 recon=n surv=0.14 p_fires=6%`
    - parent [13](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/13): _scientific citations and related formatting_
    - child  [235](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/235): _math-related symbols like multiplication signs and exponents._
- **14 -> 235**  `R=0.95 F=0.09 recon_gain=0.000 recon=n surv=0.07 p_fires=9%`
    - parent [14](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/14): _html code snippets, specifically those including href tags pointing to java files_
    - child  [235](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/235): _math-related symbols like multiplication signs and exponents._
- **40 -> 235**  `R=0.95 F=0.25 recon_gain=-0.000 recon=n surv=0.07 p_fires=3%`
    - parent [40](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/40): _the slash character, which appears to indicate mathematical equations written in LaTeX_
    - child  [235](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/235): _math-related symbols like multiplication signs and exponents._

### reject:no-recon  (4)

- **62 -> 238**  `R=0.90 F=0.08 recon_gain=-0.001 recon=n surv=0.85 p_fires=18%`
    - parent [62](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/62): _proper nouns relating to places, people, organizations, events, or works of art._
    - child  [238](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/238): _LaTeX math symbols_
- **123 -> 410**  `R=0.83 F=0.16 recon_gain=-0.000 recon=n surv=0.73 p_fires=11%`
    - parent [123](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/123): _numerical data and special characters often used in code or data representation_
    - child  [410](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/410): _mathematical symbols, formulas, and units_
- **13 -> 385**  `R=0.82 F=0.19 recon_gain=-0.000 recon=n surv=0.69 p_fires=6%`
    - parent [13](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/13): _scientific citations and related formatting_
    - child  [385](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/385): _section headers_
- **77 -> 151**  `R=0.81 F=0.11 recon_gain=-0.002 recon=n surv=0.90 p_fires=14%`
    - parent [77](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/77): _parts of a computer program, identifying elements like file paths, library names, and versions_
    - child  [151](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/151): _URLs_
