# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

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
- **43 -> 383**  `R=0.66 F=0.08 recon_gain=0.133 recon=Y surv=1.02 p_fires=12%`
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

## Block pair 1->2

### survivor  (8)

- **406 -> 1900**  `R=0.81 F=0.01 recon_gain=0.479 recon=Y surv=1.05 p_fires=5%`
    - parent [406](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/406): _code licenses_
    - child  [1900](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1900): _legal disclaimers related to conditions and warranties_
- **216 -> 1554**  `R=0.85 F=0.49 recon_gain=0.099 recon=Y surv=1.07 p_fires=5%`
    - parent [216](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/216): _sentences starting with "the" followed by a noun_
    - child  [1554](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1554): _the numeral 1_
- **139 -> 1939**  `R=0.84 F=0.06 recon_gain=0.095 recon=Y surv=0.79 p_fires=5%`
    - parent [139](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/139): _code containing `import` statements, `@` symbols and code relating to `android`_
    - child  [1939](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1939): _terms in Android Java import statements_
- **352 -> 1900**  `R=0.68 F=0.01 recon_gain=0.061 recon=Y surv=1.02 p_fires=4%`
    - parent [352](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/352): _proper nouns and/or legal language_
    - child  [1900](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1900): _legal disclaimers related to conditions and warranties_
- **139 -> 1900**  `R=0.71 F=0.01 recon_gain=0.056 recon=Y surv=1.03 p_fires=5%`
    - parent [139](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/139): _code containing `import` statements, `@` symbols and code relating to `android`_
    - child  [1900](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1900): _legal disclaimers related to conditions and warranties_
- **223 -> 734**  `R=0.78 F=0.22 recon_gain=0.048 recon=Y surv=0.94 p_fires=18%`
    - parent [223](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/223): _words and phrases related to Javascript, CSS, and other web programming languages_
    - child  [734](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/734): _programming-related text including "I am trying to"._
- **134 -> 919**  `R=0.55 F=0.05 recon_gain=0.047 recon=Y surv=1.10 p_fires=8%`
    - parent [134](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/134): _mentions of departments within universities or institutes and related items like professors and accreditation_
    - child  [919](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/919): _names of universities_
- **252 -> 1887**  `R=0.54 F=0.05 recon_gain=0.047 recon=Y surv=1.03 p_fires=3%`
    - parent [252](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/252): _HTML code snippets_
    - child  [1887](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1887): _CSS properties_

### reject:superparent  (4)

- **362 -> 977**  `R=1.00 F=0.03 recon_gain=-0.000 recon=n surv=- p_fires=28%`
    - parent [362](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/362): _marketing and promotional content related to products and services_
    - child  [977](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/977): _strings that are not ASCII characters_
- **204 -> 1404**  `R=1.00 F=0.06 recon_gain=-0.003 recon=n surv=- p_fires=14%`
    - parent [204](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/204): _the legal term "res ipsa loquitur"_
    - child  [1404](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1404): _the keyword "struct" followed by a name_
- **245 -> 977**  `R=1.00 F=0.08 recon_gain=-0.000 recon=n surv=- p_fires=10%`
    - parent [245](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/245): _words related to directions, roads, or physical location_
    - child  [977](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/977): _strings that are not ASCII characters_
- **207 -> 977**  `R=1.00 F=0.05 recon_gain=-0.000 recon=n surv=- p_fires=17%`
    - parent [207](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/207): _words about evaluating or influencing creative or academic work_
    - child  [977](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/977): _strings that are not ASCII characters_

### reject:freq-driven  (4)

- **408 -> 1226**  `R=0.99 F=0.14 recon_gain=-0.000 recon=n surv=0.40 p_fires=6%`
    - parent [408](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/408): _uses of the auxiliary verbs "have", "has", and "been"._
    - child  [1226](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1226): _references to code sections_
- **298 -> 1226**  `R=0.99 F=0.58 recon_gain=0.000 recon=n surv=0.41 p_fires=1%`
    - parent [298](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/298): _mathematical formulas_
    - child  [1226](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1226): _references to code sections_
- **308 -> 1226**  `R=0.99 F=0.24 recon_gain=-0.001 recon=n surv=0.41 p_fires=3%`
    - parent [308](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/308): _citations (Author et al., Journal, year) and other components of academic references_
    - child  [1226](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1226): _references to code sections_
- **389 -> 1226**  `R=0.99 F=0.18 recon_gain=-0.000 recon=n surv=0.41 p_fires=5%`
    - parent [389](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/389): _mentions of money and finances_
    - child  [1226](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1226): _references to code sections_

### reject:no-recon  (4)

- **192 -> 1068**  `R=0.99 F=0.09 recon_gain=-0.000 recon=n surv=0.94 p_fires=10%`
    - parent [192](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/192): _words that have a high correlation to a particular topic, with the most activated topic being characters in a science-fiction Japanese show_
    - child  [1068](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1068): _the word "et" when used as part of the phrase "et al"_
- **151 -> 1589**  `R=0.99 F=0.45 recon_gain=-0.001 recon=n surv=0.88 p_fires=2%`
    - parent [151](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/151): _URLs_
    - child  [1589](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1589): _URLs_
- **308 -> 1589**  `R=0.99 F=0.24 recon_gain=-0.001 recon=n surv=0.88 p_fires=3%`
    - parent [308](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/308): _citations (Author et al., Journal, year) and other components of academic references_
    - child  [1589](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1589): _URLs_
- **145 -> 1586**  `R=0.99 F=0.53 recon_gain=0.000 recon=n surv=0.87 p_fires=2%`
    - parent [145](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/145): _mentions of months_
    - child  [1586](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1586): _months of the year_

## Block pair 2->3

### survivor  (8)

- **1554 -> 7399**  `R=0.99 F=0.14 recon_gain=1.030 recon=Y surv=1.00 p_fires=3%`
    - parent [1554](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1554): _the numeral 1_
    - child  [7399](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7399): _scientific and technical terminology_
- **1501 -> 7399**  `R=0.95 F=0.21 recon_gain=0.879 recon=Y surv=1.00 p_fires=2%`
    - parent [1501](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1501): _numbers starting with 1_
    - child  [7399](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7399): _scientific and technical terminology_
- **1970 -> 6005**  `R=0.60 F=0.02 recon_gain=0.865 recon=Y surv=1.50 p_fires=2%`
    - parent [1970](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1970): _the word "following" preceding an enumeration or list of items_
    - child  [6005](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/6005): _the word "following"_
- **1136 -> 7708**  `R=0.87 F=0.02 recon_gain=0.738 recon=Y surv=1.15 p_fires=2%`
    - parent [1136](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1136): _mentions of police_
    - child  [7708](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7708): _the word "police"_
- **1554 -> 7252**  `R=0.64 F=0.03 recon_gain=0.527 recon=Y surv=1.41 p_fires=3%`
    - parent [1554](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1554): _the numeral 1_
    - child  [7252](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7252): _code listings or structured data with indentation_
- **1554 -> 5167**  `R=0.86 F=0.16 recon_gain=0.360 recon=Y surv=1.00 p_fires=3%`
    - parent [1554](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1554): _the numeral 1_
    - child  [5167](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/5167): _lines containing only whitespace_
- **1475 -> 3889**  `R=0.56 F=0.05 recon_gain=0.352 recon=Y surv=0.59 p_fires=1%`
    - parent [1475](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1475): _the word "and', often in the context of a list-like structure_
    - child  [3889](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3889): _the word "and" or the phrase "can be"_
- **1242 -> 3788**  `R=0.86 F=0.03 recon_gain=0.278 recon=Y surv=1.11 p_fires=3%`
    - parent [1242](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1242): _words related to inventions and embodiments of those inventions_
    - child  [3788](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3788): _the word "relates," particularly in the context of describing an invention_

### reject:superparent  (4)

- **1780 -> 3994**  `R=1.00 F=0.05 recon_gain=-0.001 recon=n surv=- p_fires=16%`
    - parent [1780](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1780): _words that have some kind of evaluative meaning, either positive or negative_
    - child  [3994](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3994): _an unusual whitespace character in references_
- **1309 -> 5554**  `R=1.00 F=0.06 recon_gain=-0.001 recon=n surv=- p_fires=13%`
    - parent [1309](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1309): _research article abstracts or scientific content_
    - child  [5554](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/5554): _math formulas and code, specifically parentheses containing numbers and letters_
- **863 -> 6544**  `R=1.00 F=0.06 recon_gain=-0.001 recon=n surv=- p_fires=14%`
    - parent [863](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/863): _mentions of dates in the text_
    - child  [6544](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/6544): _the Greek letter "xi" (ξ) in mathematical equations and expressions_
- **1191 -> 3987**  `R=1.00 F=0.04 recon_gain=-0.000 recon=n surv=- p_fires=23%`
    - parent [1191](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1191): _scientific and technical writing, particularly related to experiments and data._
    - child  [3987](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3987): _citations to academic papers_

### reject:freq-driven  (4)

- **1072 -> 2953**  `R=0.99 F=0.62 recon_gain=-0.000 recon=n surv=0.40 p_fires=1%`
    - parent [1072](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1072): _instances of multiplication, especially in scientific or technical contexts_
    - child  [2953](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/2953): _the non-breaking space character used in scientific and medical texts_
- **1335 -> 7777**  `R=0.99 F=0.54 recon_gain=-0.000 recon=n surv=0.40 p_fires=2%`
    - parent [1335](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1335): _error messages and exception handling in code_
    - child  [7777](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7777): _strings representing memory addresses_
- **930 -> 3035**  `R=0.99 F=0.47 recon_gain=-0.000 recon=n surv=0.40 p_fires=2%`
    - parent [930](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/930): _phrases comparing things_
    - child  [3035](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3035): _the word "whereas," which introduces a comparison or contrast_
- **1236 -> 7777**  `R=0.99 F=0.87 recon_gain=-0.000 recon=n surv=0.40 p_fires=1%`
    - parent [1236](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1236): _mathematical formulas involving exponential functions_
    - child  [7777](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7777): _strings representing memory addresses_

### reject:no-recon  (4)

- **757 -> 3035**  `R=1.00 F=0.66 recon_gain=0.000 recon=n surv=1.00 p_fires=1%`
    - parent [757](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/757): _phrases using the word "compared"_
    - child  [3035](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3035): _the word "whereas," which introduces a comparison or contrast_
- **1076 -> 3035**  `R=1.00 F=0.55 recon_gain=-0.000 recon=n surv=1.00 p_fires=2%`
    - parent [1076](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/1076): _the word "than" followed by various pronouns or nouns_
    - child  [3035](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3035): _the word "whereas," which introduces a comparison or contrast_
- **2003 -> 7215**  `R=1.00 F=0.14 recon_gain=-0.000 recon=n surv=1.00 p_fires=6%`
    - parent [2003](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/2003): _words and abbreviations in a technical or scientific context_
    - child  [7215](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/7215): _the word "bond" in the context of finance or scientific/mathematical study of material properties_
- **753 -> 3035**  `R=1.00 F=0.75 recon_gain=0.000 recon=n surv=1.00 p_fires=1%`
    - parent [753](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/753): _the word "where" in mathematical writing_
    - child  [3035](https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc/3035): _the word "whereas," which introduces a comparison or contrast_
