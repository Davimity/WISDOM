# WISDOM model roadmap

WISDOM studies whether chemical structure inside a protein can identify localized phenomena on its
molecular surface from protein-level labels alone. Each version changes one principal scientific
hypothesis so improvements remain attributable rather than mixing several architectural changes.

## WISDOM v1 — Atom-to-Surface Baseline

Implemented now. Small learned element and residue embeddings enter a relation-aware atomic GCN.
Incident atomic embeddings are averaged at each fixed surface point and concatenated with invariant
multiscale curvature. A surface GCN produces one local logit per point, and a surface-area-weighted
mean converts those weakly supervised local predictions into one protein logit.

The question is: **does one-way atom-to-surface information support an end-to-end protein
classifier while exposing meaningful local scores?**

## WISDOM v2 — Better weak supervision

Implemented and technically verified, but not yet trained or evaluated as scientific evidence. The
v1 atom encoder, atom-to-surface transfer, curvature projection, surface encoder, and local head
remain fixed. A LambdaForge sweep compares the exact v1 area-weighted mean against normalized
log-sum-exp, softmax pooling, a top-10-percent mean, and learned gated attention. Noisy-OR is omitted
because treating thousands of correlated surface points as independent events can saturate its
probabilistic product without providing a defensible physical interpretation.

The model returns the protein logit, point logits and probabilities in original NPZ order, plus an
area-aware localization distribution, its normalized entropy, the area fraction above probability
0.5, and the maximum point probability. These maps and diagnostics are save-ready tensors, not
point-level labels or additional training losses. The question is whether a different
multiple-instance pooling rule can preserve small localized signals that the mean may dilute.

## WISDOM v3 — Rich atomic chemistry

Documentation only. Compare the atomic R-GCN with suitable LambdaForge PNA, GraphTransformer, or
EGNN components and progressively consume reliable edge distances, bond orders, or covalent
attributes. Spatial-only, covalent-only, and combined ablations must isolate the value of each
relation. The question is whether richer internal chemical-edge information improves prediction.

## WISDOM v4 — Bidirectional structure–surface communication

Documentation only. Introduce one or two local rounds of atoms→surface→atoms communication over the
precomputed bipartite graph, without global attention. The question is whether feedback from surface
context improves the internal representation beyond v1's one-way projection.

## WISDOM v5 — Geometric surface operator

Documentation only. Replace the generic surface GCN with an independently licensed and validated
dMaSIF-like quasi-geodesic operator over WISDOM's lightweight surface. The question is whether a
surface-specific convolution improves on ordinary sparse graph propagation.

## WISDOM v6 — Robust or dynamic surface sampling

Documentation only. Test small jitter with reprojection, subsampling, and consistency between
discretizations while reusing candidate neighborhoods. The physical hypothesis is that equivalent
samplings of the same molecular boundary should produce stable predictions.

## WISDOM v7 — Advanced representation and pretraining

Documentation only and intentionally deferred until earlier hypotheses are supported. Possible
experiments include self-supervision, contrastive atom/surface objectives, complementary protein
language embeddings, multitask learning, richer equivariant encoders, or continuous fields.

Future comparisons must use LambdaForge seeds, ablations, aggregation and paired statistics. No
code for v3–v7 belongs in the implemented v1/v2 scope.
