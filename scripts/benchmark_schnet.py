"""Measure isolated synthetic SchNet forward/backward, never a scientific training epoch."""

# ruff: noqa: I001

import json
import time
import torch
import argparse

from pathlib import Path
from statistics import median
from wisdom.models.atomic.AtomicGraphEncoder import AtomicGraphEncoder


def benchmark(mode: str, count: int, width: int, nodes: int, steps: int,
              device: str) -> dict[str, object]:
    """Time a fixed sparse two-relation interaction with synchronized phase boundaries.

    Args:
        mode: Gaussian or direct radial representation.
        count: Gaussian input count, inactive in direct mode.
        width: Internal atomic width; output remains 128.
        nodes: Synthetic atoms; each connects to at most eight later atoms.
        steps: Measured forward/backward repetitions, after two warmups.
        device: Explicit cpu or CUDA device; no implicit device fallback.

    Returns:
        Hardware-labelled timings, parameter count and isolated CUDA peaks. The synthetic topology,
        16-dimensional chemical input and artificial distances do not predict full WISDOM cost.
    """
    torch.manual_seed(173)
    target = torch.device(device)
    encoder = AtomicGraphEncoder(
        16, 128, 2, 0.2, 6.0, 65536, "schnet", radial_count=count,
        internal_dim=width, radial_encoding=mode).to(target).train()
    pairs = torch.tensor([(i, i + delta) for i in range(nodes)
                          for delta in range(1, min(9, nodes - i))],
                         dtype=torch.long, device=target).T
    covalent = pairs[1] - pairs[0] == 1
    spatial = torch.ones(pairs.shape[1], dtype=torch.bool, device=target)
    distances = torch.linspace(1.4, 5.9, pairs.shape[1], device=target)
    zeros = torch.zeros_like(distances)
    ones = torch.ones_like(distances)
    features = torch.randn(nodes, 16, device=target)
    gates = {f"edge.{relation}.{field}": ones.new_tensor(1.0)
             for relation, fields in (("covalent", ("distance", "bond_order", "same_residue")),
                                      ("spatial", ("distance", "same_residue", "same_chain",
                                                   "residue_separation"))) for field in fields}
    gates.update({f"atomic_graph.{relation}": ones.new_tensor(1.0)
                  for relation in ("covalent", "spatial")})
    forward, backward = [], []
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    for step in range(steps + 2):
        encoder.zero_grad(set_to_none=True)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        output = encoder(features, pairs, spatial, covalent, distances, ones, zeros, ones,
                         zeros, gates)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        output.square().mean().backward()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        if step >= 2:
            forward.append(forward_seconds)
            backward.append(time.perf_counter() - started)
    return {
        "scope": "isolated synthetic encoder; FP32; no optimizer, transfer or surface network",
        "torch": torch.__version__, "device": str(target), "threads": torch.get_num_threads(),
        "hardware": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
        "radial_encoding": mode, "gaussian_count": count if mode == "gaussian" else None,
        "internal_width": width, "output_width": 128, "input_width": 16, "layers": 2,
        "nodes": nodes, "undirected_edges": pairs.shape[1], "seed": 173,
        "forward_seconds": forward, "backward_seconds": backward,
        "median_forward_seconds": median(forward), "median_backward_seconds": median(backward),
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(target) if target.type == "cuda" else None),
    }


def main() -> None:
    """Run only the explicitly requested short synthetic benchmark and write its raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "gaussian"), default="gaussian")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = benchmark(args.mode, args.count, args.width, args.nodes, args.steps, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
