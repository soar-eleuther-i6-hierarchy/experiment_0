<a href="../../" title="Back to the experiment_0 index" style="position:fixed;top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;padding:9px 13px;text-decoration:none">&#8592; Back to index</a>

# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **84 -> 273**  `R=0.64 F=0.13 recon_gain=1.488 recon=Y surv=0.60 p_fires=3%`
    - parent [84](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/84): _the word "this" used as a pronoun to make connections between topics_
    - child  [273](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/273): _the word "This"_
- **84 -> 284**  `R=0.71 F=0.15 recon_gain=0.976 recon=Y surv=0.83 p_fires=3%`
    - parent [84](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/84): _the word "this" used as a pronoun to make connections between topics_
    - child  [284](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/284): _the word "this", sometimes followed by "is", "can", or "that"_
- **60 -> 384**  `R=0.53 F=0.02 recon_gain=0.593 recon=Y surv=1.31 p_fires=2%`
    - parent [60](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/60): _LaTeX mathematical expressions, especially those involving symbols and equations._
    - child  [384](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/384): _LaTeX math delimiters and special characters_
- **26 -> 140**  `R=0.77 F=0.13 recon_gain=0.590 recon=Y surv=1.00 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [140](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/140): _mentions of a male person as "he" or with "him" possessive pronouns._
- **106 -> 439**  `R=0.52 F=0.05 recon_gain=0.541 recon=Y surv=1.50 p_fires=1%`
    - parent [106](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/106): _code and related keywords_
    - child  [439](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/439): _statistical data from scientific publications_
- **26 -> 306**  `R=0.73 F=0.15 recon_gain=0.505 recon=Y surv=0.88 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [306](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/306): _first and second person pronouns_
- **26 -> 299**  `R=0.56 F=0.15 recon_gain=0.356 recon=Y surv=1.03 p_fires=6%`
    - parent [26](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/26): _first and third person pronouns along with references to other people._
    - child  [299](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/299): _"it" or other third person pronouns_
- **48 -> 284**  `R=0.62 F=0.03 recon_gain=0.280 recon=Y surv=1.12 p_fires=13%`
    - parent [48](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/48): _various discourse markers or conjunctions_
    - child  [284](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/284): _the word "this", sometimes followed by "is", "can", or "that"_

### reject:superparent  (4)

- **70 -> 310**  `R=1.00 F=0.01 recon_gain=1.266 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [310](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/310): _the keyword "public" in code_
- **70 -> 238**  `R=1.00 F=0.01 recon_gain=1.007 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [238](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/238): _code terms related to calculating maximum and minimum values_
- **70 -> 476**  `R=1.00 F=0.01 recon_gain=1.039 recon=n surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [476](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/476): _the word "most" followed by "of"_
- **70 -> 258**  `R=1.00 F=0.00 recon_gain=1.247 recon=Y surv=1.00 p_fires=99%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/70): _proper nouns that have mixed upper and lowercase letters or consist of uppercase letters followed by lowercase letters_
    - child  [258](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/258): _lines that start with a plus (+) sign, which may represent code blocks or other structured data_

### reject:freq-driven  (4)

- **58 -> 461**  `R=0.94 F=0.02 recon_gain=-0.003 recon=n surv=0.30 p_fires=42%`
    - parent [58](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/58): _words and abbreviations involved in scientific research and reporting of clinical trials._
    - child  [461](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/461): _mathematical formulas with exponents, and superscripts_
- **13 -> 461**  `R=0.93 F=0.15 recon_gain=-0.001 recon=n surv=0.00 p_fires=6%`
    - parent [13](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/13): _the ends of sentences or phrases that include numbers or symbols_
    - child  [461](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/461): _mathematical formulas with exponents, and superscripts_
- **92 -> 461**  `R=0.93 F=0.12 recon_gain=-0.011 recon=n surv=0.36 p_fires=7%`
    - parent [92](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/92): _scientific results including numbers, standard deviations, units, and sometimes variable names_
    - child  [461](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/461): _mathematical formulas with exponents, and superscripts_
- **103 -> 461**  `R=0.93 F=0.15 recon_gain=0.001 recon=n surv=0.24 p_fires=6%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/103): _a combination of asterisks, brackets, and quotes often used in mathematical typography_
    - child  [461](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/461): _mathematical formulas with exponents, and superscripts_

### reject:no-recon  (4)

- **58 -> 302**  `R=0.94 F=0.02 recon_gain=-0.003 recon=n surv=0.56 p_fires=42%`
    - parent [58](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/58): _words and abbreviations involved in scientific research and reporting of clinical trials._
    - child  [302](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/302): _citations_
- **101 -> 461**  `R=0.93 F=0.04 recon_gain=-0.001 recon=n surv=0.60 p_fires=22%`
    - parent [101](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/101): _legal jargon and names of people and institutions_
    - child  [461](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/461): _mathematical formulas with exponents, and superscripts_
- **1 -> 302**  `R=0.90 F=0.03 recon_gain=-0.014 recon=n surv=0.53 p_fires=30%`
    - parent [1](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/1): _words related to groups of people, institutions, or legal concepts_
    - child  [302](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/302): _citations_
- **62 -> 177**  `R=0.87 F=0.09 recon_gain=-0.021 recon=n surv=0.68 p_fires=12%`
    - parent [62](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/62): _scientifically technical words_
    - child  [177](https://www.neuronpedia.org/gemma-2-2b/3-res-matryoshka-dc/177): _legal citations_
