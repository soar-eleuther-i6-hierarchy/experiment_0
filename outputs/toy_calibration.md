# Exp 0 - metric calibration on synthetic ground truth

Each metric is graded on the pathology it is meant to catch, using the production thresholds in `config.py`. Margin = how decisively the metric separated the two classes (higher is better).

| rank | metric | job | verdict | margin | detail |
|---|---|---|---|---|---|
| 1 | 5. frequency control | reject frequency-coincidence edge, keep genuine | PASS | >1000x | 1/1 freq edges rejected (survival=[0.0]); 20/20 genuine survive (min genuine survival=1.00, thr=0.5) |
| 2 | 2. reconstruction | reject superparent edges, keep genuine | PASS | 661.4x | 24/24 superparent edges rejected, 20/20 genuine kept (parent-gain: genuine>=3.05, superparent<=0.0046, thr=0.01) |
| 3 | 3. sibling redundancy | flag feature-split parent, spare healthy | PASS | 92.0x | split parent redundancy=1.00 (flagged); healthy parents max=0.01 (thr=0.5) |
| 4 | 1. coverage (edge set) | recover genuine tree edges | PASS | 1.0x | 20/20 genuine edges kept; edge set also holds 28 non-genuine (that is what metrics 2-5 must prune) |
| 5 | 4. out-degree / superparent | identify superparent, spare genuine parents | PASS | 1.0x | detected superparents [7] (truth [7]); Gini=0.432, top-1 share=50% |

**5/5 metrics calibrated.** All five recover the genuine tree and reject their injected pathology on this toy.