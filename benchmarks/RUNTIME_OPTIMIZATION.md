# WISDOM exact runtime optimization report

## Scope and equivalence

This refactor changes execution order and tensor storage only. It does not change model parameters,
semantic gates, graph membership, atom-to-surface scores, DiffusionNet, pooling, loss, precision, or
the HPO search space. The production `state_dict` keys and shapes are unchanged.

The main identity applies to each bias-free atomic message matrix `W`:

$$
\frac{1}{d_i}\sum_j a_{ji}Wh_j
=
W\left(\frac{1}{d_i}\sum_j a_{ji}h_j\right).
$$

The optimized encoder therefore multiplies and aggregates hidden vectors per edge, then applies
`W` once per atom. The former path applied the same `H × H` matrix once per directed edge. Tests
compare outputs, input gradients, every parameter gradient, and one AdamW step against a preserved
reference implementation with strict FP32 tolerances.

Persisted atomic pairs already satisfy `src < dst`; their distance, bond, residue, chain, and
sequence-separation attributes are symmetric. The collator now keeps one copy of each pair, while
the encoder adds `u → v` and `v → u` contributions directly. Schema validation excludes self-edges,
so no former self-edge multiplicity can be lost.

Atom-to-surface transfer reuses the three normalized geometric fields, evaluates the orientation
origin once per forward, and contracts weights with gathered atom embeddings through `bmm`. The
masked softmax, invalid-entry zeroing, renormalization, scores, and gates are unchanged.

DiffusionNet constructs the exact stacked operator `[Gx; Gy]` once per protein and forward. Every
block obtains both tangent derivatives with one sparse multiplication and applies the shared
bias-free gradient mixer once to the stacked result. V3 DeltaConv reuses the same operator for its
gradient and transposed-divergence operations. In both cases, splitting the stacked result recovers
the former x/y tensors exactly up to floating-point reduction order.

## Reproducible benchmark

Run:

```bash
python benchmarks/benchmark_runtime.py \
  --device cuda \
  --iterations 10 \
  --output runtime-benchmark.json \
  --profile runtime-profile.md
```

The script includes the previous operations as benchmark-only references, performs warmup, calls
`torch.cuda.synchronize()` around timing regions, measures allocated/reserved CUDA peaks, exercises
`H = 64, 128, 256`, and emits a `torch.profiler` comparison. The following representative results
were measured with PyTorch 2.13.0/CUDA 13.0 on an NVIDIA GeForce RTX 3050 Ti Laptop GPU. They are
evidence for the implementation, not claimed H100 timings.

| Component | Workload | Before | After | Speedup | Allocated MiB before → after |
|---|---:|---:|---:|---:|---:|
| Atomic forward + backward | large, H=64 | 6.75 ms | 5.78 ms | 1.17× | 92.3 → 63.4 |
| Atomic forward + backward | large, H=128 | 8.52 ms | 6.19 ms | 1.38× | 156.5 → 104.1 |
| Atomic forward + backward | large, H=256 | 16.26 ms | 9.80 ms | 1.66× | 286.1 → 189.8 |
| Atom-to-surface forward + backward | medium, H=64 | 3.89 ms | 3.16 ms | 1.23× | 128.7 → 110.4 |
| Atom-to-surface forward + backward | medium, H=128 | 5.51 ms | 4.40 ms | 1.25× | 195.0 → 175.7 |
| Atom-to-surface forward + backward | medium, H=256 | 9.23 ms | 7.38 ms | 1.25× | 327.5 → 306.2 |
| DiffusionNet forward + backward | medium, H=64 | 1.66 ms | 1.45 ms | 1.14× | 25.5 → 25.7 |
| DiffusionNet forward + backward | medium, H=128 | 1.73 ms | 1.53 ms | 1.14× | 33.8 → 33.3 |
| DiffusionNet forward + backward | medium, H=256 | 2.86 ms | 2.73 ms | 1.05× | 49.1 → 49.3 |
| DeltaConv forward + backward | medium, H=64 | 6.61 ms | 4.43 ms | 1.49× | 32.0 → 33.9 |
| DeltaConv forward + backward | medium, H=128 | 6.81 ms | 4.93 ms | 1.38× | 47.3 → 49.0 |
| DeltaConv forward + backward | medium, H=256 | 10.23 ms | 9.09 ms | 1.12× | 78.8 → 84.5 |
| Complete V1 optimizer step | 4 proteins, H=64 | 41.99 ms | 36.33 ms | 1.16× | 175.1 → 150.1 |
| Complete V1 optimizer step | 4 proteins, H=128 | 43.49 ms | 38.46 ms | 1.13× | 284.9 → 243.8 |
| Complete V1 optimizer step | 4 proteins, H=256 | 55.71 ms | 46.87 ms | 1.19× | 505.8 → 441.5 |

The complete-model maximum absolute logit difference was between `2.98e-8` and `5.96e-8`, depending
on width. Compact atomic edge tensors use exactly half the previous host and transfer bytes. The
profiler's atomic `aten::mm` CUDA time fell from about 1.05 ms to 0.34 ms in the representative
H=128 profile; total self CUDA time fell from about 2.99 ms to 2.20 ms.

## Additional decisions

- The stacked `[Gx; Gy]` derivative is enabled because local benchmarks show a repeatable benefit
  and strict tests prove equal outputs, input gradients, and all parameter gradients. Its H100
  benchmark remains pending: the attempted LambdaForge submission never started because the
  configured cluster SSH connection timed out before submission.
- `torch.compile` remains disabled. Variable protein loops, sparse tensors, stochastic gates, and
  short HPO candidates make compile latency and graph breaks likely to dominate without
  target-hardware evidence.
- Exact-zero inference pruning remains disabled. Checking device gate values per batch would add a
  synchronization, while caching an inference plan would complicate the public model lifecycle for
  a benefit that does not apply to training.
- Existing evaluation already uses `torch.inference_mode()`. Existing recursive transfers already
  use `non_blocking=True`, and CUDA loaders already pin host memory, so no duplicate mechanism was
  added.
- Repeated `.float()` calls on already-FP32 operator tensors are no-op views; no permanent duplicate
  operator packs were introduced.

## Runtime-budget recommendation

Keep `surface_chunk_size=8192` and `atomic_message_chunk_size=65536` for now. The optimized atomic
path interprets the latter as a directional-message budget: a compact pair has two directions, so
each internal pair chunk uses at most half the configured value. Lowering either value would add
launch overhead; increasing them has no demonstrated whole-model benefit and weakens peak-memory
bounds. Retune only from an H100 benchmark using the target dataset and candidate widths.

V2 and V3 inherit the shared atomic and transfer changes. V2 receives the same benefit for every
pooling rule. V3 receives it for every surface encoder, while its DiffusionNet and DeltaConv choices
also use the exact stacked differential operators.
