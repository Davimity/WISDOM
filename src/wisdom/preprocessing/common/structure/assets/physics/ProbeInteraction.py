"""Exact neutral Lennard--Jones probes with a real OpenMM interaction-group reference."""

import importlib
import numpy as np

from typing import Any
from scipy.spatial import cKDTree


class ProbeInteraction:
    """Evaluate protein--probe interactions only; prepared hydrogens are physical sources too."""

    def __init__(self, positions: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray) -> None:
        """Bind fixed prepared atoms without interpreting their formal charges as partial ones.

        Args:
            positions: Prepared atom centers [N,3] in Å, including hydrogens.
            sigma: Force-field LJ diameters [N] in nm.
            epsilon: Force-field LJ well depths [N] in kJ/mol.
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.sigma     = np.asarray(sigma, dtype=np.float64)
        self.epsilon   = np.asarray(epsilon, dtype=np.float64)
        self.tree      = cKDTree(self.positions)

    def evaluate(
        self, queries: np.ndarray, probe: tuple[float, float],
        overlap_fraction: float = 0.65, cutoff: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum 4*sqrt(e_i*e_s)*[(sigma_is/r)^12-(sigma_is/r)^6] in bounded query batches.

        Args:
            queries: Fixed query coordinates [Q,3] in Å.
            probe: Actual force-field (sigma_nm, epsilon_kJ_per_mol), with charge fixed at zero.
            overlap_fraction: Reject distances below this fraction of mixed sigma; these are
                unavailable measurements, not clipped repulsive energies.
            cutoff: Optional exact hard cutoff in Å; zero computes the complete finite protein.

        Returns:
            Float64 energies [Q] in kJ/mol and Boolean per-probe overlap validity [Q]. Invalid
            entries are zero only as masked storage. A finite repulsive band remains unaltered.
        """
        mixed_sigma = 0.5 * (self.sigma + probe[0])
        mixed_depth = np.sqrt(self.epsilon * probe[1])
        values = np.zeros(len(queries), dtype=np.float64)
        valid  = np.ones(len(queries), dtype=bool)

        # Blocking over 32 queries is bounded physical reference evaluation, not a persistent
        # dense graph or a model query--atom attention matrix. No approximation is made at r=0.

        if not cutoff:
            active = mixed_depth > 0
            positions = self.positions[active]
            diameter  = mixed_sigma[active]
            depth     = mixed_depth[active]
            for start in range(0, len(queries), 32):
                points = np.asarray(queries[start:start + 32], dtype=np.float64)
                distance = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2) * .1
                admitted = ~np.any(distance < overlap_fraction * diameter, axis=1)
                valid[start:start + len(points)] = admitted
                ratio6 = (diameter / distance[admitted]) ** 6
                energies = np.zeros(len(points), dtype=np.float64)
                energies[admitted] = np.sum(4 * depth * (ratio6 * ratio6 - ratio6), axis=1)
                values[start:start + len(points)] = energies
            return values, valid

        # Radius-limited neighborhoods have different lengths and retain their exact per-query
        # summation. The full-protein default above removes Python work per individual voxel.

        for start in range(0, len(queries), 32):
            points = np.asarray(queries[start:start + 32], dtype=np.float64)
            for local, point in enumerate(points):
                ids = (np.asarray(self.tree.query_ball_point(point, cutoff), dtype=np.intp)
                       if cutoff else np.arange(len(self.positions)))
                distance = np.linalg.norm(point - self.positions[ids], axis=1) * 0.1
                active = mixed_depth[ids] > 0
                if np.any(distance[active] < overlap_fraction * mixed_sigma[ids][active]):
                    valid[start + local] = False
                    continue
                ratio6 = (mixed_sigma[ids][active] / distance[active]) ** 6
                values[start + local] = np.sum(
                    4 * mixed_depth[ids][active] * (ratio6 * ratio6 - ratio6)
                )
        return values, valid

    def reference(self, queries: np.ndarray, probe: tuple[float, float],
                  cutoff: float = 0.0) -> np.ndarray:
        """Evaluate the same test particle with OpenMM, never protein self-interactions.

        Args:
            queries: Physically admissible coordinates [Q,3], Å.
            probe: Actual neutral (sigma_nm, epsilon_kJ_per_mol).
            cutoff: Zero for NoCutoff, otherwise hard cutoff in Å with no switching/correction.

        Returns:
            Reference-platform energies [Q], kJ/mol. Exceptions between protein atoms cannot
            affect a nonbonded test particle: it has no bonds or exclusions to the protein.

        Raises:
            ImportError: OpenMM was not provisioned; no approximate reference replaces it.
        """
        mm: Any = importlib.import_module("openmm")
        system = mm.System()
        force = mm.CustomNonbondedForce(
            "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            "sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
        )
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon")
        for sigma, epsilon in zip(self.sigma, self.epsilon, strict=True):
            system.addParticle(1)
            force.addParticle([float(sigma), float(epsilon)])
        system.addParticle(1)
        force.addParticle(list(probe))
        force.addInteractionGroup(set(range(len(self.positions))), {len(self.positions)})
        if cutoff:
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
            force.setCutoffDistance(cutoff * 0.1)
        force.setUseLongRangeCorrection(False)
        system.addForce(force)
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        result = []
        try:
            for point in queries:
                context.setPositions(np.vstack((self.positions, point)) * 0.1)
                state = context.getState(getEnergy=True)
                result.append(state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole))
        finally:
            del context, integrator
        return np.asarray(result, dtype=np.float64)
