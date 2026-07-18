# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

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
