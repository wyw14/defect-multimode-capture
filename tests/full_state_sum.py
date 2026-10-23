"""Positive state sum for fully mixed fifteen-mode harmonic surfaces.

Low quantum states are integrated analytically as Gaussian polynomials. The
omitted positive spectral mass is bounded by the exact thermal expectation of
W squared, independently of energy. No time correlation or propagator is used.
"""

import itertools
import numpy as np
from scipy import linalg
from pymatgen.core import Lattice, Structure
from independent import KAPPA, KB, normal_coordinates


def make_case():
    atoms = 6
    rng = np.random.default_rng(250111)
    initial = Structure(
        Lattice.cubic(14.0),
        ["C", "O", "Si", "S", "Fe", "Ce"],
        [[1.0 + 1.5 * j, 1.2 + 0.13 * j, 2.0 + 0.17 * (j % 3)] for j in range(atoms)],
        coords_are_cartesian=True,
        site_properties={"atom_id": [f"site-{j}" for j in range(atoms)]},
    )
    mass = np.array([float(site.specie.atomic_mass) for site in initial])
    root_mass = np.repeat(np.sqrt(mass), 3)
    internal = linalg.null_space(
        (root_mass[:, None] * np.tile(np.eye(3), (atoms, 1))).T
    )
    rotate, _ = np.linalg.qr(rng.normal(size=(15, 15)))
    wi = np.linspace(0.168, 0.178, 15)
    wf = np.linspace(0.1682, 0.1778, 15)
    force_i = np.diag(wi**2)
    force_f = rotate.T @ np.diag(wf**2) @ rotate
    shift = rng.normal(size=15) * 0.009
    linear = rng.normal(size=15) * 2e-6
    quadratic = rng.normal(size=(15, 15)) * 1e-7
    quadratic = (quadratic + quadratic.T) / 2
    transform = root_mass[:, None] * internal
    displacement = np.sqrt(KAPPA) / root_mass * (internal @ shift)
    final = Structure(
        initial.lattice,
        [site.specie for site in initial],
        initial.cart_coords + displacement.reshape((atoms, 3)),
        coords_are_cartesian=True,
        site_properties=initial.site_properties,
    )
    return {
        "initial": initial,
        "final": final,
        "ki": transform @ force_i @ transform.T / KAPPA,
        "kf": transform @ force_f @ transform.T / KAPPA,
        "coupling": {
            "constant": 0.01,
            "linear": transform @ linear / np.sqrt(KAPPA),
            "quadratic": transform @ quadratic @ transform.T / KAPPA,
        },
        "temperature": 200.0,
        "energies": np.array([-0.17, -0.02, 0.0, 0.02, 0.17, 0.32]),
        "width": 0.03,
        "masses": None,
        "queries": [],
        "full_state_sum": True,
    }


def state_polynomials(frequencies, rotation, shift, maximum):
    """Normalized oscillator polynomials of total degree zero, one and two."""
    size = len(frequencies)
    rows = [(0.0, np.zeros(size, dtype=int))]
    polynomials = [(1.0, np.zeros(size), np.zeros((size, size)))]
    linear = np.sqrt(2 * frequencies)[:, None] * rotation
    offsets = -np.sqrt(2 * frequencies) * shift
    if maximum >= 1:
        for j in range(size):
            quantum = np.zeros(size, dtype=int)
            quantum[j] = 1
            rows.append((frequencies[j], quantum))
            polynomials.append((offsets[j], linear[j], np.zeros((size, size))))
    if maximum >= 2:
        for j in range(size):
            quantum = np.zeros(size, dtype=int)
            quantum[j] = 2
            rows.append((2 * frequencies[j], quantum))
            polynomials.append(
                (
                    (offsets[j] ** 2 - 1) / np.sqrt(2),
                    np.sqrt(2) * offsets[j] * linear[j],
                    np.sqrt(2) * np.outer(linear[j], linear[j]),
                )
            )
        for j, k in itertools.combinations(range(size), 2):
            quantum = np.zeros(size, dtype=int)
            quantum[[j, k]] = 1
            rows.append((frequencies[j] + frequencies[k], quantum))
            polynomials.append(
                (
                    offsets[j] * offsets[k],
                    offsets[j] * linear[k] + offsets[k] * linear[j],
                    np.outer(linear[j], linear[k]) + np.outer(linear[k], linear[j]),
                )
            )
    return np.array([row[0] for row in rows]), polynomials


def positive_state_sum(case):
    p = normal_coordinates(case)
    wi, wf, rotation, shift = p["wi"], p["wf"], p["rotation"], p["shift"]
    size = len(wi)
    ai = np.diag(wi)
    af = rotation.T @ np.diag(wf) @ rotation
    center_f = rotation.T @ shift
    stiffness = ai + af
    covariance = linalg.inv(stiffness)
    center = covariance @ af @ center_f
    log_overlap = (
        0.25 * np.log(wi * wf).sum()
        + size * 0.5 * np.log(2)
        - 0.5 * np.linalg.slogdet(stiffness)[1]
        - 0.5 * (center_f @ af @ center_f - center @ stiffness @ center)
    )
    overlap = np.exp(log_overlap)
    initial_energy, initial = state_polynomials(wi, np.eye(size), np.zeros(size), 1)
    final_energy, final = state_polynomials(wf, rotation, shift, 2)
    operator = (p["w0"], p["linear"], p["quadratic"])

    def centered(poly):
        constant, linear, quadratic = poly
        slope = linear + quadratic @ center
        mean = (
            constant
            + linear @ center
            + 0.5 * center @ quadratic @ center
            + 0.5 * np.trace(quadratic @ covariance)
        )
        return mean, slope, quadratic

    operator_mean, operator_slope, operator_matrix = centered(operator)
    amplitudes = np.empty((len(final), len(initial)))
    for a, poly_i in enumerate(initial):
        mi, li, bi = centered(poly_i)
        for b, poly_f in enumerate(final):
            mf, lf, bf = centered(poly_f)
            # Isserlis' theorem for three real quadratic polynomials.
            value = mi * mf * operator_mean
            value += mi * (
                lf @ covariance @ operator_slope
                + 0.5 * np.trace(bf @ covariance @ operator_matrix @ covariance)
            )
            value += mf * (
                li @ covariance @ operator_slope
                + 0.5 * np.trace(bi @ covariance @ operator_matrix @ covariance)
            )
            value += operator_mean * (
                li @ covariance @ lf + 0.5 * np.trace(bi @ covariance @ bf @ covariance)
            )
            value += li @ covariance @ operator_matrix @ covariance @ lf
            value += li @ covariance @ bf @ covariance @ operator_slope
            value += lf @ covariance @ bi @ covariance @ operator_slope
            value += np.trace(
                bi @ covariance @ bf @ covariance @ operator_matrix @ covariance
            )
            amplitudes[b, a] = overlap * value
    beta = 1 / (KB * case["temperature"])
    population = np.prod(-np.expm1(-beta * wi)) * np.exp(-beta * initial_energy)
    weight = amplitudes**2 * population[None, :]
    transitions = (
        final_energy[:, None] - initial_energy[None, :] + 0.5 * (wf.sum() - wi.sum())
    )
    width = case["width"]
    kernel = np.array(
        [
            np.sum(weight * np.exp(-0.5 * ((energy - transitions) / width) ** 2))
            / (np.sqrt(2 * np.pi) * width)
            for energy in case["energies"]
        ]
    )
    thermal = np.diag(0.5 / wi / np.tanh(beta * wi / 2))
    mean = p["w0"] + 0.5 * np.trace(p["quadratic"] @ thermal)
    total = (
        mean**2
        + p["linear"] @ thermal @ p["linear"]
        + 0.5 * np.trace(p["quadratic"] @ thermal @ p["quadratic"] @ thermal)
    )
    missing = float(total - weight.sum())
    if missing < -1e-15:
        raise AssertionError(
            "Partial positive state sum exceeds complete operator norm"
        )
    bound = max(missing, 0.0) / (np.sqrt(2 * np.pi) * width)
    return {
        "kernel": kernel,
        "omega_initial": wi,
        "omega_final": wf,
        "log_Z_initial": float(-np.log(2 * np.sinh(beta * wi / 2)).sum()),
        "log_Z_final": float(-np.log(2 * np.sinh(beta * wf / 2)).sum()),
        "uniform_truncation_bound": bound,
        "full_second_moment": float(total),
        "retained_spectral_mass": float(weight.sum()),
        "force_commutator_norm": float(linalg.norm(ai @ af - af @ ai)),
        "final_initial_overlap": rotation,
    }
