"""Native geometry, acoustic constraints and mass-coordinate units."""

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.defects.constants import HBAR_EV, AMU2KG, ANGS2M, EV2J

KAPPA = HBAR_EV**2 * EV2J / (AMU2KG * ANGS2M**2)


def native_geometry(
    initial, final, force_initial, force_final, w0, gradient, quadratic, masses=None
):
    if not isinstance(initial, Structure) or not isinstance(final, Structure):
        raise ValueError("Native pymatgen Structure objects are required")
    if (
        len(initial) != len(final)
        or len(initial) < 2
        or not np.allclose(
            initial.lattice.matrix, final.lattice.matrix, atol=1e-12, rtol=0
        )
    ):
        raise ValueError("The native periodic cells must have matching fixed lattices")
    ids = initial.site_properties.get("atom_id")
    other = final.site_properties.get("atom_id")
    if (
        ids is None
        or other is None
        or not all(isinstance(x, str) for x in [*ids, *other])
        or len(set(ids)) != len(ids)
        or set(ids) != set(other)
    ):
        raise ValueError("Unique matching native atom_id properties are required")
    order = np.array([other.index(key) for key in ids])
    axes = (3 * order[:, None] + np.arange(3)).ravel()
    if any(initial[i].specie != final[j].specie for i, j in enumerate(order)):
        raise ValueError("Matched sites must have the same chemical species")
    n = len(initial)
    size = 3 * n
    mass = (
        np.array([float(site.specie.atomic_mass) for site in initial])
        if masses is None
        else np.asarray(masses, float)
    )
    if mass.shape != (n,) or mass.min() <= 0 or not np.all(np.isfinite(mass)):
        raise ValueError("Positive native per-atom masses are required")
    ki = np.asarray(force_initial, float)
    kf = np.asarray(force_final, float)
    gradient = np.asarray(gradient, float)
    quadratic = np.asarray(quadratic, float)
    if (
        ki.shape != (size, size)
        or kf.shape != (size, size)
        or gradient.shape != (size,)
        or quadratic.shape != (size, size)
    ):
        raise ValueError("Invalid Cartesian force/coupling shape")
    if not all(np.all(np.isfinite(x)) for x in (ki, kf, gradient, quadratic)):
        raise ValueError("Nonfinite Cartesian data")
    kf = kf[np.ix_(axes, axes)]
    translation = np.tile(np.eye(3), (n, 1))
    for name, k in [("initial", ki), ("final", kf)]:
        if np.max(np.abs(k - k.T)) > 1e-10 or np.max(np.abs(k @ translation)) > 1e-9:
            raise ValueError(
                name + " force constants violate Hessian symmetry/acoustic sum rule"
            )
    if (
        np.max(np.abs(translation.T @ gradient)) > 1e-10
        or np.max(np.abs(quadratic @ translation)) > 1e-10
    ):
        raise ValueError("The electronic coupling must be translationally invariant")
    if np.max(np.abs(quadratic - quadratic.T)) > 1e-10:
        raise ValueError("Quadratic electronic coupling must be symmetric")
    # The native lattice selects the shortest periodic branch for each matched atom.
    delta = []
    for i, j in enumerate(order):
        _, image = initial.lattice.get_distance_and_image(
            initial[i].frac_coords, final[j].frac_coords
        )
        delta.append(
            initial.lattice.get_cartesian_coords(
                final[j].frac_coords + image - initial[i].frac_coords
            )
        )
    delta = np.asarray(delta).ravel()
    return {
        "matched_final_indices": order.tolist(),
        "periodic_displacement": delta.reshape((n, 3)),
        "atom_ids": list(ids),
    }
