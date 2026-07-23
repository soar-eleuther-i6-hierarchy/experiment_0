<a href="../../" title="Back to the experiment_0 index" style="position:fixed;top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;padding:9px 13px;text-decoration:none">&#8592; Back to index</a>

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
