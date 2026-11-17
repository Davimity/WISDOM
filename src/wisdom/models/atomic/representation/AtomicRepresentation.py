"""Opt-in chemical input blocks and frozen external context; no task labels are accepted."""

import json
import torch

from torch import Tensor, nn
from typing import Any, ClassVar
from collections.abc import Mapping, Iterable
from wisdom.models.residue.ResidueGraphEncoder import ResidueGraphEncoder
from wisdom.preprocessing.common.structure.chemistry.context_chemistry import (
    ATOM_NAMES, EXACT_ATOMS,
)
from wisdom.models.atomic.representation.AtomicInputSource import AtomicInputSource
from wisdom.models.atomic.representation.ExternalIntegration import ExternalIntegration


class AtomicRepresentation(nn.Module):
    """Construct V10 inputs with masks, training-only statistics and independent source gates."""

    WIDTHS: ClassVar = {
        "backbone_torsions": 9,
        "sidechain_torsions": 12,
        "covalent_statistics": 14,
        "functional_groups": 12,
        "atomic_sasa": 2,
    }
    normalization_fitted: Tensor
    external_context_mean: Tensor
    external_context_scale: Tensor
    external_atom_mean: Tensor
    external_atom_scale: Tensor

    def __init__(
        self,
        sources: Iterable[str] = (),
        identity_dim: int = 8,
        external_model: str | None = None,
        external_mode: str = "none",
        context_dim: int = 32,
        standard_deviation_floor: float = 1e-6,
        force_sources: str = "learned",
        residue_graph: Mapping[str, Any] | None = None,
        context_control: str = "aligned",
        context_seed: int = 2026,
        drop_sources: Iterable[str] = (),
        atomic_external_model: str | None = None,
        atomic_context_dim: int = 32,
        atomic_message_control: bool = False,
    ) -> None:
        """Declare input blocks without reading a dataset or fitting normalization.

        Args:
            sources: Unique canonical source blocks, empty preserves historical inputs.
            identity_dim: Width of each global atom-name/residue-atom table, default eight.
            external_model: esm2/esmc/saprot; one frozen extractor, never a concatenation.
            external_mode: none/static/context/both/replace, corresponding to E0-E4.
            context_dim: Trainable per-residue context projection width, default 32.
            standard_deviation_floor: Positive train-fitted standard deviation floor.
            force_sources: learned/open/closed for NEW sources only; historical gates stay intact.
            residue_graph: Optional independent ResidueGraphEncoder configuration; None preserves
                the original context path. Missing CA nodes always retain that original path.
            context_control: aligned, mean or permuted within valid residues of each protein.
            context_seed: Fixed permutation seed shared across train/evaluation, default 2026.
            drop_sources: Nominal-width input ablations, including element and historical gate
                names without atom.; graph construction still consumes physical chemistry.
            atomic_external_model: None or slae; independent frozen per-atom scalar features.
            atomic_context_dim: Learned SLAE projection width, default 32.
            atomic_message_control: Remove atom-to-atom messages but retain per-row capacity;
                default false. Pretrained contextual neighborhoods are not thereby removed.

        Raises:
            ValueError: Unsupported, duplicated or incompatible source/integration choices.
        """
        super().__init__()
        self.sources = tuple(AtomicInputSource(source).value for source in sources)
        self.external_mode = ExternalIntegration(external_mode)
        if len(set(self.sources)) != len(self.sources) or force_sources not in {
            "learned",
            "open",
            "closed",
        }:
            raise ValueError("atomic sources must be unique; force_sources is learned/open/closed")
        if external_model not in {None, "esm2", "esmc", "saprot"} or (
            (external_model is None) != (self.external_mode is ExternalIntegration.NONE)
        ):
            raise ValueError("one external_model and a non-none external_mode must be paired")
        self.external_model = external_model
        self.floor = standard_deviation_floor
        self.force_sources = force_sources
        self.tables = nn.ModuleDict()
        for name, count in (
            ("atom_name", len(ATOM_NAMES) + 1),
            ("residue_atom", len(EXACT_ATOMS) + 1),
        ):
            if name in self.sources:
                self.tables[name] = nn.Embedding(count, identity_dim)
        self.contextual = self.external_mode in {
            ExternalIntegration.CONTEXT,
            ExternalIntegration.BOTH,
            ExternalIntegration.REPLACE,
        }
        width = 960 if external_model == "esmc" else 480
        self.context_projection = nn.Linear(width, context_dim) if self.contextual else None
        if atomic_external_model not in {None, "slae"}:
            raise ValueError("atomic_external_model must be the verified slae interface or null")
        self.atomic_external_model = atomic_external_model
        self.atomic_message_control = atomic_message_control
        self.atomic_context_projection = (nn.Linear(128, atomic_context_dim)
                                           if atomic_external_model else None)
        self.width = sum(
            identity_dim if source in self.tables else self.WIDTHS[source]
            for source in self.sources
        ) + (context_dim if self.contextual else 0) + (
            atomic_context_dim if atomic_external_model else 0)
        for name, size in (
            ("covalent_statistics", 11),
            ("atomic_sasa", 1),
            ("external_context", width),
            ("external_atom", 128),
        ):
            if name in self.sources or (name == "external_context" and self.contextual) or (
                name == "external_atom" and atomic_external_model
            ):
                self.register_buffer(name + "_mean", torch.zeros(size))
                self.register_buffer(name + "_scale", torch.ones(size))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.initialization_audit: dict[str, Any] = {}
        if context_control not in {"aligned", "mean", "permuted"}:
            raise ValueError("context_control must be aligned, mean or permuted")
        self.context_control, self.context_seed = context_control, context_seed
        self.drop_sources = tuple(drop_sources)
        allowed = {"element", "residue_type", "atom_role", "hybridization", "formal_charge",
                   "aromaticity", "hbond_donor", "hbond_acceptor", "residue_hydropathy",
                   "residue_polarity", "external_context", "external_atom", "residue_graph",
                   *self.sources}
        if set(self.drop_sources) - allowed:
            raise ValueError("unknown atom input source ablation")
        if residue_graph and not self.contextual:
            raise ValueError("residue communication requires an explicitly aligned residue context")
        self.residue_encoder: ResidueGraphEncoder | None = None
        if residue_graph:
            # Optional branch allocation must not move the common atomic input projection's
            # random draws. Exact alpha=0/zero-output recovery needs genuinely paired weights.

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() ^ 0x563134)
                self.residue_encoder = ResidueGraphEncoder(context_dim, **residue_graph)

    @property
    def gate_definitions(self) -> tuple[tuple[str, str], ...]:
        """Return direct descendants of the existing atomic-context gate, without family parents.

        Returns:
            Stable source names; E1 has no extra gate because it changes the existing table only.
        """
        names = (*self.sources, *(("external_context",) if self.contextual else ()),
                 *(("external_atom",) if self.atomic_external_model else ()))
        return tuple(("atom." + name, "surface.atom_context") for name in names)

    @staticmethod
    def required_assets(options: Mapping[str, Any] | None) -> tuple[str, ...]:
        """Resolve declared immutable dependencies before loading or fitting any dataset.

        Args:
            options: Exact constructor options or None for the historical route.

        Returns:
            Ordered required families; a missing family is an error, never silently disabled.
        """
        options = options or {}
        sources = options.get("sources", ())
        mode = ExternalIntegration(options.get("external_mode", "none"))
        model = options.get("external_model")
        required = {"atomic_context"} if sources or mode is not ExternalIntegration.NONE else set()
        if "atomic_sasa" in sources:
            required.add("atomic_sasa")
        if mode in {ExternalIntegration.STATIC, ExternalIntegration.BOTH}:
            required.add(f"static_embedding_table_{model}")
        if mode in {
            ExternalIntegration.CONTEXT,
            ExternalIntegration.BOTH,
            ExternalIntegration.REPLACE,
        }:
            required.add(f"external_embeddings_{model}")
        if options.get("residue_graph"):
            required.add("residue_graph")
        if options.get("atomic_external_model"):
            required.update(("atomic_context", "external_atomic_slae"))
        return tuple(sorted(required))

    @staticmethod
    def covalent_fields(data: Mapping[str, Tensor]) -> Tensor:
        """Derive fourteen C1 channels from positive-order represented covalent pairs only.

        Args:
            data: One disjoint atomic batch including structural edges and per-atom context fields.

        Returns:
            Float32 [N,11] counts: heavy degree, C/N/O/S/P/other, order 1/1.5/2/3.
            Ring/support/disulfide indicators are joined separately from the verified context.
        """
        numbers = data["atomic_numbers"]
        active = data["atom_edge_is_covalent"].bool() & (data["atom_edge_bond_order"] > 0)
        edges = data["atom_edge_index"][:, active]
        order = data["atom_edge_bond_order"][active].repeat(2)
        receiver, sender = torch.cat((edges, edges.flip(0)), dim=1)
        heavy = numbers[sender] > 1
        receiver, sender, order = receiver[heavy], sender[heavy], order[heavy]
        fields = numbers.new_zeros((len(numbers), 11), dtype=torch.float32)
        element = numbers[sender]
        recognized = torch.zeros_like(element, dtype=torch.bool)
        fields[:, 0].index_add_(0, receiver, torch.ones_like(order, dtype=torch.float32))
        for column, number in enumerate((6, 7, 8, 16, 15), 1):
            mask = element == number
            recognized |= mask
            fields[:, column].index_add_(0, receiver, mask.float())
        fields[:, 6].index_add_(0, receiver, (~recognized).float())
        for column, value in enumerate((1.0, 1.5, 2.0, 3.0), 7):
            fields[:, column].index_add_(
                0, receiver, torch.isclose(order, order.new_tensor(value)).float()
            )
        return fields

    def raw_blocks(self, data: Mapping[str, Any]) -> dict[str, Tensor]:
        """Assemble unnormalized atom inputs, preserving local residue indexing in every member.

        Args:
            data: Collated structural_assets, atom ownership, local residue indices and base graph.

        Returns:
            Block-to-[N,F] tensors plus external_context at [R_total,d_F] and residue lookup [N].
            No quality, identifier, target, source path or experimental metadata becomes a feature.
        """
        assets = data["structural_assets"]
        parent = data["atom_residue_indices"]
        owners = data["atom_batch"]
        parts: dict[str, list[Tensor]] = {name: [] for name in self.sources}
        context, validity, residue_lookup = [], [], []
        residue_offset = 0
        for index, atom_context in enumerate(assets["atomic_context"]["members"]):
            local_parent = parent[owners == index]
            for name in self.sources:
                if name in self.tables:
                    key = "atom_name_id" if name == "atom_name" else "atom_identity_id"
                    values = atom_context[key].long()
                elif name in {"backbone_torsions", "sidechain_torsions"}:
                    section = slice(0, 3) if name == "backbone_torsions" else slice(3, 7)
                    angle = atom_context["torsion"][:, section]
                    valid = atom_context["torsion_valid"][:, section].float()
                    trig = torch.stack((angle.sin() * valid, angle.cos() * valid), dim=-1)
                    values = torch.cat((trig.flatten(1), valid), dim=1)[local_parent]
                elif name == "functional_groups":
                    bits = atom_context["functional_group_bits"].long()
                    known = atom_context["chemistry_known"].float()
                    flags = ((bits[:, None] >> torch.arange(11, device=bits.device)) & 1).float()
                    values = torch.cat((flags * known[:, None], known[:, None]), dim=1)
                elif name == "atomic_sasa":
                    member = assets["atomic_sasa"]["members"][index]
                    values = torch.stack((member["sasa"], member["valid"].float()), dim=1)
                else:
                    known = atom_context["chemistry_known"].float()
                    disulfide = ((atom_context["functional_group_bits"].long() >> 10) & 1).float()
                    values = torch.stack(
                        (atom_context["ring_member"].float() * known, known, disulfide), dim=1
                    )
                parts[name].append(values)
            if self.contextual:
                member = assets[f"external_embeddings_{self.external_model}"]["members"][index]
                embedding = member["embedding"]
                if self.context_control != "aligned":
                    embedding = embedding.clone()
                    valid_rows = member["valid"].bool()
                    if valid_rows.any():
                        if self.context_control == "mean":
                            embedding[valid_rows] = embedding[valid_rows].mean(0)
                        else:
                            generator = torch.Generator().manual_seed(self.context_seed)
                            order = torch.randperm(int(valid_rows.sum()), generator=generator)
                            embedding[valid_rows] = embedding[valid_rows][
                                order.to(embedding.device)]
                context.append(embedding)
                validity.append(member["valid"])
                residue_lookup.append(local_parent + residue_offset)
                residue_offset += len(member["valid"])
        result = {name: torch.cat(values) for name, values in parts.items()}
        if "covalent_statistics" in result:
            result["covalent_statistics"] = torch.cat(
                (self.covalent_fields(data), result["covalent_statistics"]), dim=1
            )
        if self.contextual:
            result.update(
                external_context=torch.cat(context),
                external_valid=torch.cat(validity),
                external_lookup=torch.cat(residue_lookup),
            )
        if self.atomic_external_model:
            members = assets["external_atomic_slae"]["members"]
            result["external_atom"] = torch.cat([member["embedding"] for member in members])
            result["external_atom_valid"] = torch.cat([member["valid"] for member in members])
        return result

    def fit(self, batches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Fit streaming FP64 population moments solely on the selected training members.

        Args:
            batches: One traversal of the authorized training view, never validation/test.

        Returns:
            Serializable per-source support, mean and population standard deviation. Residue
            context contributes once per valid residue, not once per atom. No label is consulted.

        Raises:
            ValueError: A required continuous source has zero valid training support.
        """
        moments: dict[str, tuple[int, Tensor, Tensor]] = {}
        for data in batches:
            blocks = self.raw_blocks(data)
            for name in ("covalent_statistics", "atomic_sasa", "external_context", "external_atom"):
                if name not in blocks:
                    continue
                values = blocks[name]
                if name == "atomic_sasa":
                    values = values[values[:, 1].bool(), :1]
                elif name == "external_context":
                    values = values[blocks["external_valid"].bool()]
                elif name == "external_atom":
                    values = values[blocks["external_atom_valid"].bool()]
                else:
                    values = values[:, :11]
                values = values.detach().double().cpu()
                if not len(values):
                    continue
                n, mean, m2 = moments.get(
                    name,
                    (
                        0,
                        torch.zeros(values.shape[1], dtype=torch.float64),
                        torch.zeros(values.shape[1], dtype=torch.float64),
                    ),
                )
                new_mean = values.mean(0)
                delta = new_mean - mean
                total = n + len(values)
                m2 = (
                    m2
                    + (values - new_mean).square().sum(0)
                    + delta.square() * n * len(values) / total
                )
                moments[name] = (total, mean + delta * len(values) / total, m2)
        report: dict[str, Any] = {}
        for name in ("covalent_statistics", "atomic_sasa", "external_context", "external_atom"):
            if not hasattr(self, name + "_mean"):
                continue
            if name not in moments:
                raise ValueError(f"no valid training support for {name}")
            count, mean, m2 = moments[name]
            scale = (m2 / count).sqrt().clamp_min(self.floor)
            getattr(self, name + "_mean").copy_(mean.float())
            getattr(self, name + "_scale").copy_(scale.float())
            report[name] = {"count": count, "mean": mean.tolist(), "scale": scale.tolist()}
        self.normalization_fitted.fill_(True)
        return report

    def forward(self, data: Mapping[str, Any], gates: Mapping[str, Tensor]) -> Tensor:
        """Project context once per residue, then concatenate independently gated atom blocks.

        Args:
            data: The same structural-only batch contract as raw_blocks.
            gates: One-forward source values, shared across all atoms and layers.

        Returns:
            Additional float [N,F] descriptors. Masks are gated with values. External projection
            bias is multiplied by residue validity AFTER projection, so missingness cannot leak it.

        Raises:
            RuntimeError: A continuous source was used without training-fitted statistics.
        """
        if not self.width:
            return data["atomic_numbers"].new_empty(
                (len(data["atomic_numbers"]), 0), dtype=torch.float32
            )
        raw = self.raw_blocks(data)
        parts = []
        for name in self.sources:
            values = raw[name]
            if name in self.tables:
                values = self.tables[name](values)
            elif hasattr(self, name + "_mean"):
                if not self.normalization_fitted:
                    raise RuntimeError("atomic representation normalization has not been fitted")
                size = 11 if name == "covalent_statistics" else 1
                normalized = (values[:, :size] - getattr(self, name + "_mean")) / getattr(
                    self, name + "_scale"
                )
                if name == "atomic_sasa":
                    normalized = normalized * values[:, 1:2]
                values = torch.cat((normalized, values[:, size:]), dim=1)
            parts.append(values * gates["atom." + name])
        if self.context_projection is not None:
            if not self.normalization_fitted:
                raise RuntimeError("external context normalization has not been fitted")
            values = (
                raw["external_context"] - self.external_context_mean
            ) / self.external_context_scale
            values = self.context_projection(values) * raw["external_valid"][:, None]
            if "external_context" in self.drop_sources:
                values = torch.zeros_like(values)
            projected = values * gates["atom.external_context"]
            if self.residue_encoder is not None and (
                self.residue_encoder.topology.value == "descending"
            ) and "residue_graph" not in self.drop_sources:
                offset = 0
                increments = []
                for member in data["structural_assets"]["residue_graph"]["members"]:
                    count = len(member["backbone_valid"])
                    increments.append(self.residue_encoder(values[offset:offset + count],
                        raw["external_valid"][offset:offset + count], member))
                    offset += count
                if offset != len(values):
                    raise ValueError("residue graph and physical context coverage differ")
                projected = projected + torch.cat(increments)
            parts.append(projected[raw["external_lookup"]])
        if self.atomic_context_projection is not None:
            if not self.normalization_fitted:
                raise RuntimeError("external atomic normalization has not been fitted")
            values = (raw["external_atom"] - self.external_atom_mean) / self.external_atom_scale
            values = self.atomic_context_projection(values) * raw["external_atom_valid"][:, None]
            parts.append(values * gates["atom.external_atom"])
        return (
            torch.cat(parts, dim=1)
            if parts
            else data["atomic_numbers"].new_empty(
                (len(data["atomic_numbers"]), 0), dtype=torch.float32
            )
        )

    def communicate(self, data: Mapping[str, Any], atoms: Tensor,
                    surface_ptr: Tensor) -> tuple[Tensor, Tensor | None]:
        """Apply one declared post-atomic roundtrip or a parallel residue-to-surface branch.

        Args:
            data: Original collated structural-only inputs, with protein-local residue assets.
            atoms: Encoded atomic states [N,H]; never raw targets or pooled protein labels.
            surface_ptr: CPU disjoint point offsets [B+1] defining point ownership.

        Returns:
            Possibly corrected atomic states and optional [M,H] parallel surface context. No
            second atomic GNN or DiffusionNet is hidden in the roundtrip adapter.
        """
        encoder = self.residue_encoder
        if encoder is None or encoder.topology.value == "descending" or (
            "residue_graph" in self.drop_sources
        ):
            return atoms, None
        raw = self.raw_blocks(data)
        values = (raw["external_context"] - self.external_context_mean
                  ) / self.external_context_scale
        assert self.context_projection is not None and encoder.atomic_output is not None
        context = self.context_projection(values) * raw["external_valid"][:, None]
        if "external_context" in self.drop_sources:
            context = torch.zeros_like(context)
        parts = []
        surface_parts = []
        offset = 0
        for index, member in enumerate(data["structural_assets"]["residue_graph"]["members"]):
            count = len(member["backbone_valid"])
            owned = data["atom_batch"] == index
            parent = data["atom_residue_indices"][owned].long()
            roundtrip = encoder.topology.value == "roundtrip"
            correction = encoder(context[offset:offset + count],
                raw["external_valid"][offset:offset + count], member,
                atoms[owned] if roundtrip else None, parent if roundtrip else None)
            message = encoder.atomic_output(correction)
            if roundtrip:
                parts.append(atoms[owned] + message[parent])
            else:
                # Point candidates were deduplicated from ALL nearby heavy atoms at preprocessing,
                # not from V11's short active atom prefix. Missing support contributes exactly zero.

                size = int(surface_ptr[index + 1] - surface_ptr[index])
                pointer = member["surface_ptr"].long()
                if len(pointer) != size + 1:
                    raise ValueError("residue surface CSR belongs to a different point domain")
                metadata = json.loads(member["metadata_json"])["parameters"]
                if encoder.surface_radius > metadata["surface_radius"]:
                    raise ValueError("parallel residue radius exceeds preprocessed support")
                rows = torch.repeat_interleave(torch.arange(size, device=atoms.device),
                                                pointer[1:] - pointer[:-1])
                rank = torch.arange(len(rows), device=atoms.device) - torch.repeat_interleave(
                    pointer[:-1], pointer[1:] - pointer[:-1])
                ids = member["surface_residue"].long()
                distance = member["surface_distance"].float()
                active = (distance <= encoder.surface_radius) & (rank < encoder.surface_k)
                active &= member["backbone_valid"][ids, 1].bool()
                rows, ids, distance = rows[active], ids[active], distance[active]
                weights = torch.exp(-distance / encoder.surface_radius)
                normalizer = weights.new_zeros(size).index_add(0, rows, weights)
                output = atoms.new_zeros((size, atoms.shape[1]))
                output.index_add_(0, rows, message[ids] * (
                    weights / normalizer[rows].clamp_min(1e-12))[:, None])
                surface_parts.append(output)
            offset += count
        return (torch.cat(parts), None) if parts else (atoms, torch.cat(surface_parts))

    def initialize_residue_table(self, table: nn.Embedding, member: Mapping[str, Any]) -> None:
        """Map twenty true pretrained input rows to a signed, padded, norm-matched residue table.

        Args:
            table: Trainable [21,d_r] UNK/ALA/ARG/.../VAL table already paired with the control.
            member: Validated model-qualified static table, independent of contextual embeddings.

        Raises:
            ValueError: Unsupported vocabulary, degenerate rank or non-finite adaptation.
        """
        rows = member["weights"].detach().double().cpu()
        alphabet = json.loads(member["alphabet"])
        centered = rows - rows.mean(0)
        _, singular, right = torch.linalg.svd(centered, full_matrices=False)
        rank = min(table.embedding_dim, int((singular > singular[0] * 1e-7).sum()))
        if rank == 0:
            raise ValueError("external static table has zero centered rank")
        directions = right[:rank].T
        pivots = directions.abs().argmax(0)
        directions = directions * directions[pivots, torch.arange(rank)].sign()
        projected = centered @ directions
        scale = (table.embedding_dim / projected.square().sum(1).mean()).sqrt()
        names = "ARNDCQEGHILKMFPSTWYV"
        result = torch.zeros_like(table.weight)
        result[1:21, :rank] = (projected[[alphabet.index(name) for name in names]] * scale).to(
            result
        )
        with torch.no_grad():
            table.weight.copy_(result)
        self.initialization_audit = {
            "rank": rank,
            "scale": float(scale),
            "mean": rows.mean(0).tolist(),
            "directions": directions.tolist(),
            "singular_values": singular.tolist(),
            "unknown": "zero",
            "trainable": True,
        }
