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
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../outputs/">Results</a><a class="on" href="../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# Synthetic toy calibration — every metric against a known tree

Each metric is graded on the pathology it is meant to catch, using the production thresholds in `config.py`. Margin = how decisively the metric separated the two classes (higher is better).

| rank | metric | job | verdict | margin | detail |
|---|---|---|---|---|---|
| 1 | 5. frequency control | reject frequency-coincidence edge, keep genuine | PASS | >1000x | 1/1 freq edges rejected (survival=[0.0]); 21/21 genuine survive (min genuine survival=1.00, thr=0.5) |
| 2 | 3'. parent-conditioned redundancy | flag the split parent inside its own firing set, spare a genuine one | PASS | >1000x | split parent=1.00, genuine parent=0.00 (thr=0.5); the conditioned form the global Jaccard defers to (covers parent_conditioned_redundancy) |
| 3 | — absorption (negative control) | confirm coverage cannot propose an absorbed edge at all | PASS | >1000x | true edge (8->21) has R=0.00 < tau=0.5 and is absent from the candidate set, so metrics 2-9 never see it. Absorption is not measurable in this design — a blind spot that is now demonstrated rather than argued. Fixing it needs a different candidate generator, not a better grader |
| 4 | 2. reconstruction | reject superparent edges, keep genuine | PASS | 987.7x | 33/33 superparent edges rejected, 21/21 genuine kept (parent-gain: genuine>=3.26, superparent<=0.0033, thr=0.01) |
| 5 | 3. sibling redundancy | flag feature-split parent, spare healthy | PASS | 119.0x | split parent redundancy=1.00 (flagged); healthy parents max=0.01 (thr=0.5) |
| 6 | 6. independence null (PMI) | rank genuine edges above base-rate/superparent co-firing | PASS | 39.7x | min genuine PMI=2.80 > max superparent PMI=0.07; base-rate confound only - topical co-occurrence is not in this toy (needs a model-based null) |
| 7 | 4. out-degree / superparent | identify superparent, spare genuine parents | PASS | 1.5x | detected superparents [7] (truth [7]); removing the superparent collapses Gini 0.620->0.409 and top-1 share 52%->13% (covers degree_stats, gini) |
| 8 | 9. energy concentration (share_energy) | flag feature-split parent (a child holds >=90% of its energy) | PASS | 1.5x | split parent max child share=1.00 (thr=0.9); genuine parents max=0.66 |
| 9 | — multi-parenting (flag) | intruding parent passes every gate; the child's in-degree reports it | PASS | 1.5x | intruder edge (13->31) survives coverage, reconstruction and the frequency control; child 31 has in-degree 3 in the candidate set (true parent + intruder + the superparent overlay), the symptom only the out-degree metric reads. Detection is a report, not a cut |
| 10 | 7. joint-child coverage (support) | children cover a genuine parent's firing, not a superparent's | PASS | 1.4x | genuine R_supp>=0.67 vs superparent=0.46; r_supp and joint_child_coverage_exact agree (True) - a same-formula drift guard, not independent grading; upper>=exact for all parents: True (covers r_supp, joint_child_coverage_upper; joint_child_coverage_exact is proved numerically in tests/test_metric_math.py) |
| 11 | 8. joint-child coverage (energy) | energy-weighted child coverage separates genuine from superparent | PASS | 1.4x | genuine R_mass>=0.66 vs superparent=0.46 |
| 12 | — cross-block siblings (flag) | co-extensive pair proposed as an edge; energy share ~ 1 flags the rename | PASS | 1.1x | identical firing sets straddling the block boundary: coverage keeps (14->32) with R=1.00, and the child holds 1.00 of the parent's energy (thr=0.9) — the rename/duplicate signature from joint-child coverage |
| 13 | 1. coverage (edge set) | recover genuine tree edges | PASS | 1.0x | 21/21 genuine edges kept; edge set also holds 42 non-genuine (that is what metrics 2-5 must prune) |
| 14 | 2b. probe S_res (rank rule) | accept every true parent; carry no signal against an unrelated one | PASS | 1.0x | 21/21 genuine edges pass the top-5 rank rule (median true-parent rank 1); superparent 1/21 (median rank 23) against 2.2 expected by chance at k/D=10.4%. The superparent holds 91% of the negative class, so it cannot enter the probe direction; what passes is coincidence, not detection. **The rule's strictness is set by dictionary size** — the same k is 0.015% on gemma's 32768 and 0.28% on PCFG's 1792 (covers train_probe, sres_rank_check, negative_parent_composition) |
| 15 | 7. in-block directed edges | direct the containment pair, call the co-extensive pair a duplicate | PASS | 1.0x | containment (27, 28) recovered as a directed edge; co-extensive (29, 30) reported as a duplicate and not an edge; parent_of antisymmetric: True (so the in-block graph is acyclic); the 3 split-child pairs also surface as duplicates, which is the same pathology seen from inside one block (covers directed_coverage, duplicate_pairs) |
| 16 | — topical co-occurrence (negative control) | confirm no metric here rejects a shared-topic pair | PASS | 1.0x | a non-edge (conditionally independent given a shared topic) survives coverage, reconstruction, frequency, PMI — R=1.00, PMI=3.78. This is the open column in the properties matrix; closing it needs a model-based topic null, not another threshold |
| 17 | — composition (negative control) | confirm no metric rejects a component -> composed-child edge | PASS | 1.0x | the composed child fires exactly where both components fire, so each component contains it; (10->23) survives, (11->23) survives. Nothing in the battery tests atomicity — the third open column; closing it needs a decomposition test, not another threshold |

**17/17 metrics calibrated.** Every scorecard row recovers the genuine tree and rejects its injected pathology on this toy.

These rows grade every metric function on the pathology it targets, including the four that read per-token residuals and masks (`train_probe`, `sres_rank_check`, `negative_parent_composition`, `parent_conditioned_redundancy`) and the two within-block ones (`directed_coverage`, `duplicate_pairs`). Two pure-formula helpers are proved against their definitions in `tests/test_metric_math.py` rather than by class separation here: `joint_child_coverage_exact` (identical to `r_supp`, kept as a drift guard) and `per_token_ablation_gain` (the closed-form ablation identity).

Until 7 August this page claimed the four per-token functions were *calibrated in Tier 2*. They were not: Tier 2 imports coverage, reconstruction and the frequency control and nothing else, so the strict test — the one that rejects most surviving edges on gemma — had no ground-truth calibration anywhere. The toy now carries the per-token view (`resid`, `fired`, `W_dec`) that made it testable.

The dash-named rows are **negative controls and flags**: the controls pass when the battery does *not* do something (absorption is unreachable because coverage gates the candidate set; a shared-topic pair and both composition edges survive every filter — the open columns of the properties matrix, and a claim that nothing catches them is worth what a demonstration is worth), and the flags check the two structures whose edges survive on purpose: the multi-parenting intruder is reported by the child's in-degree, the cross-block sibling pair by its energy share.