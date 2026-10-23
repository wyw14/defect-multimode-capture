"""Independent native normal modes, oscillator wavefunctions and closed thermal lines.

No author/candidate Gaussian propagator imports. Explicit quantum-state sums are
used on coupled three-mode cases; exact Cartesian invariants give larger
isotropic cases without summing an exponential basis.
"""

from copy import deepcopy
import itertools
import numpy as np
from scipy import linalg
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.defects.constants import HBAR_EV, EV2J, AMU2KG, ANGS2M, KB

KAPPA = HBAR_EV**2 * EV2J / (AMU2KG * ANGS2M**2)


def pair_laplacian(matrix):
    return np.block([[matrix, -matrix], [-matrix, matrix]])


def coupled_case():
    lattice = Lattice.cubic(8.0)
    initial = Structure(
        lattice,
        ["Si", "O"],
        [[0.8, 1.6, 1.6], [3.2, 2.4, 2.8]],
        coords_are_cartesian=True,
        site_properties={"atom_id": ["Si-a", "O-a"]},
    )
    delta = np.array([[0.02, -0.01, 0.03], [-0.01, 0.02, 0.01]])
    final = Structure(
        lattice,
        ["Si", "O"],
        initial.cart_coords + delta,
        coords_are_cartesian=True,
        site_properties={"atom_id": ["Si-a", "O-a"]},
    )
    a = np.array([[8.0, 1.0, 0.4], [1.0, 12.0, 0.8], [0.4, 0.8, 17.0]])
    rotate = linalg.expm(
        np.array([[0.0, -0.4, 0.2], [0.4, 0.0, -0.3], [-0.2, 0.3, 0.0]])
    )
    b = rotate @ np.diag([9.0, 14.0, 19.0]) @ rotate.T
    g = np.array([0.10, -0.15, 0.07])
    tensor = np.array(
        [[0.12, 0.035, -0.02], [0.035, -0.08, 0.015], [-0.02, 0.015, 0.15]]
    )
    case = {
        "initial": initial,
        "final": final,
        "ki": pair_laplacian(a),
        "kf": pair_laplacian(b),
        "coupling": {
            "constant": 0.002,
            "linear": np.r_[g, -g],
            "quadratic": pair_laplacian(tensor),
        },
        "temperature": 300.0,
        "energies": np.array([-0.10, -0.05, -0.015, 0.025, 0.055, 0.10, 0.15]),
        "width": 0.018,
        "masses": None,
        "queries": [],
    }
    return case


def normal_coordinates(case):
    initial = case["initial"]
    final = case["final"]
    ids = initial.site_properties["atom_id"]
    other = final.site_properties["atom_id"]
    order = np.array([other.index(key) for key in ids])
    n = len(ids)
    axes = (3 * order[:, None] + np.arange(3)).ravel()
    mass = (
        np.array([float(s.specie.atomic_mass) for s in initial])
        if case["masses"] is None
        else np.asarray(case["masses"])
    )
    rm = np.repeat(np.sqrt(mass), 3)
    invm = 1 / rm
    li, vi = np.linalg.eigh(invm[:, None] * case["ki"] * invm[None, :])
    kf = case["kf"][np.ix_(axes, axes)]
    lf, vf = np.linalg.eigh(invm[:, None] * kf * invm[None, :])
    vi = vi[:, 3:]
    vf = vf[:, 3:]
    wi = np.sqrt(KAPPA * li[3:])
    wf = np.sqrt(KAPPA * lf[3:])
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    delta = []
    for i, j in enumerate(order):
        raw = final[j].frac_coords - initial[i].frac_coords
        images = shifts - np.rint(raw)
        vectors = (raw[None, :] + images) @ initial.lattice.matrix
        delta.append(vectors[np.argmin(np.sum(vectors * vectors, axis=1))])
    delta = np.asarray(delta).ravel()
    g = case["coupling"]["linear"]
    b = case["coupling"]["quadratic"]
    return {
        "wi": wi,
        "wf": wf,
        "rotation": vf.T @ vi,
        "shift": vf.T @ (rm * delta) / np.sqrt(KAPPA),
        "w0": case["coupling"]["constant"],
        "linear": np.sqrt(KAPPA) * vi.T @ (invm * g),
        "quadratic": KAPPA * vi.T @ (invm[:, None] * b * invm[None, :]) @ vi,
    }


def explicit_quantum(case, cut=8):
    p = normal_coordinates(case)
    wi = p["wi"]
    wf = p["wf"]
    j = p["rotation"]
    d = p["shift"]
    n = len(wi)
    nodes = int(np.ceil((2 * n * (cut - 1) + 3) / 2))
    x, w = np.polynomial.hermite.hermgauss(nodes)
    stiffness = np.diag(wi) + j.T @ np.diag(wf) @ j
    source = j.T @ (wf * d)
    center = np.linalg.solve(stiffness, source)
    values, vectors = np.linalg.eigh(stiffness)
    transform = (vectors / np.sqrt(values)) @ vectors.T * np.sqrt(2)
    grid = np.array(list(itertools.product(range(nodes), repeat=n)))
    points = center + x[grid] @ transform.T
    weights = np.prod(w[grid], axis=1)
    final = points @ j.T - d

    def oscillator_polynomials(arguments):
        h = np.empty((n, cut, len(points)))
        h[:, 0] = 1.0
        h[:, 1] = np.sqrt(2) * arguments.T
        for k in range(1, cut - 1):
            h[:, k + 1] = (
                np.sqrt(2 / (k + 1)) * arguments.T * h[:, k]
                - np.sqrt(k / (k + 1)) * h[:, k - 1]
            )
        states = np.array(list(itertools.product(range(cut), repeat=n)))
        wave = np.ones((len(states), len(points)))
        for axis in range(n):
            wave *= h[axis, states[:, axis]]
        return states, wave

    ni, ai = oscillator_polynomials(points * np.sqrt(wi))
    nf, af = oscillator_polynomials(final * np.sqrt(wf))
    op = (
        p["w0"]
        + points @ p["linear"]
        + 0.5 * np.einsum("pi,ij,pj->p", points, p["quadratic"], points)
    )
    pref = (
        np.prod((wi * wf) ** 0.25 / np.sqrt(np.pi))
        * np.sqrt(2) ** n
        / np.sqrt(np.linalg.det(stiffness))
        * np.exp(-0.5 * (d @ (wf * d) - source @ center))
    )
    matrix = pref * (af * (weights * op)[None, :]) @ ai.T
    ei = (ni + 0.5) @ wi
    ef = (nf + 0.5) @ wf
    beta = 1 / (KB * case["temperature"])
    log_zi = float(-np.log(2 * np.sinh(beta * wi / 2)).sum())
    log_zf = float(-np.log(2 * np.sinh(beta * wf / 2)).sum())
    transition = ef[:, None] - ei[None, :]
    probability = np.exp(-beta * ei - log_zi)[None, :] * (matrix * matrix)
    width = case["width"]
    result = np.array(
        [
            np.sum(probability * np.exp(-0.5 * ((e - transition) / width) ** 2))
            / (np.sqrt(2 * np.pi) * width)
            for e in case["energies"]
        ]
    )
    return {
        "kernel": result,
        "omega_initial": wi,
        "omega_final": wf,
        "log_Z_initial": log_zi,
        "log_Z_final": log_zf,
        "initial_population_mass": float(np.exp(-beta * ei - log_zi).sum()),
        "basis_states": len(ei),
        "quadrature_nodes": nodes,
    }


def isotope_case(atoms=6, temperature=850.0, omega=0.07):
    species = ["Si" if i % 2 == 0 else "O" for i in range(atoms)]
    coords = np.array(
        [
            [0.7 + 1.1 * i, 1.0 + 0.2 * (i % 3), 1.4 + 0.3 * (i % 2)]
            for i in range(atoms)
        ]
    )
    initial = Structure(
        Lattice.cubic(12.0),
        species,
        coords,
        coords_are_cartesian=True,
        site_properties={"atom_id": ["atom-" + str(i) for i in range(atoms)]},
    )
    mass = np.array([float(s.specie.atomic_mass) for s in initial])
    rm = np.repeat(np.sqrt(mass), 3)
    trans = rm[:, None] * np.tile(np.eye(3), (atoms, 1))
    internal = linalg.null_space(trans.T)
    ki = (omega * omega / KAPPA) * (rm[:, None] * (internal @ internal.T) * rm[None, :])
    rng = np.random.default_rng(7027 + atoms)
    dim = 3 * atoms - 3
    v = rng.normal(size=dim) * 0.0008
    b = rng.normal(size=(dim, dim)) * 0.00008
    b = (b + b.T) / 2
    g = rm * (internal @ v) / np.sqrt(KAPPA)
    tensor = rm[:, None] * (internal @ b @ internal.T) * rm[None, :] / KAPPA
    return {
        "initial": initial,
        "final": initial.copy(),
        "ki": ki,
        "kf": ki.copy(),
        "coupling": {"constant": 0.008, "linear": g, "quadratic": tensor},
        "temperature": temperature,
        "energies": np.array([-0.15, -0.07, -0.02, 0.0, 0.05, 0.09, 0.14]),
        "width": 0.02,
        "masses": None,
        "queries": [],
        "closed_mode_quantum": omega,
        "base_masses": mass,
    }


def closed_isotropic(case):
    mass = np.array(case["base_masses"])
    actual = mass if case["masses"] is None else np.asarray(case["masses"])
    ratios = actual / mass
    if np.max(ratios) - np.min(ratios) > 1e-10:
        raise ValueError("Closed isotope oracle requires a uniform isotope factor")
    omega = case["closed_mode_quantum"] / np.sqrt(ratios[0])
    beta = 1 / (KB * case["temperature"])
    occupation = 1 / np.expm1(beta * omega)
    invmass = np.repeat(1 / actual, 3)
    g = case["coupling"]["linear"]
    b = case["coupling"]["quadratic"]
    w0 = case["coupling"]["constant"]
    normv = KAPPA * np.sum(g * invmass * g)
    traceb = KAPPA * np.sum(invmass * np.diag(b))
    traceb2 = KAPPA * KAPPA * np.einsum("ij,ji,i,j->", b, b, invmass, invmass)
    mean = w0 + (2 * occupation + 1) * traceb / (4 * omega)
    strengths = np.array(
        [
            occupation**2 * traceb2 / (8 * omega**2),
            occupation * normv / (2 * omega),
            mean**2 + occupation * (occupation + 1) * traceb2 / (4 * omega**2),
            (occupation + 1) * normv / (2 * omega),
            (occupation + 1) ** 2 * traceb2 / (8 * omega**2),
        ]
    )
    lines = omega * np.arange(-2, 3)
    width = case["width"]
    result = np.array(
        [
            np.sum(strengths * np.exp(-0.5 * ((e - lines) / width) ** 2))
            / (np.sqrt(2 * np.pi) * width)
            for e in case["energies"]
        ]
    )
    dim = 3 * len(mass) - 3
    logz = float(-dim * np.log(2 * np.sinh(beta * omega / 2)))
    return {
        "kernel": result,
        "omega_initial": np.full(dim, omega),
        "omega_final": np.full(dim, omega),
        "log_Z_initial": logz,
        "log_Z_final": logz,
        "line_strengths": strengths,
    }


def arguments(case):
    args = [
        case["initial"].as_dict(),
        case["final"].as_dict(),
        case["ki"],
        case["kf"],
        case["coupling"],
        case["temperature"],
        case["energies"],
        case["width"],
    ]
    return args, {"masses": case["masses"], "queries": case["queries"]}


def perturb(case, query, step):
    changed = {**case, "coupling": dict(case["coupling"]), "queries": []}
    for name, key in [("initial", "initial_cartesian"), ("final", "final_cartesian")]:
        struct = case[name]
        changed[name] = Structure(
            struct.lattice,
            [s.specie for s in struct],
            struct.cart_coords + step * query.get(key, 0),
            coords_are_cartesian=True,
            site_properties=struct.site_properties,
        )
    for key, qkey in [
        ("ki", "force_initial"),
        ("kf", "force_final"),
        ("temperature", "temperature"),
        ("energies", "energies"),
        ("width", "width"),
    ]:
        changed[key] = case[key] + step * query.get(qkey, 0)
    for key in ("constant", "linear", "quadratic"):
        changed["coupling"][key] = case["coupling"][key] + step * query.get(key, 0)
    mass = (
        np.array([float(s.specie.atomic_mass) for s in case["initial"]])
        if case["masses"] is None
        else np.asarray(case["masses"])
    )
    changed["masses"] = mass + step * query.get("masses", 0)
    return changed


def coupled_queries(case):
    a = np.array([[0.12, -0.03, 0.01], [-0.03, -0.07, 0.02], [0.01, 0.02, 0.05]])
    b = np.array([[0.08, 0.02, -0.01], [0.02, -0.04, 0.03], [-0.01, 0.03, 0.06]])
    vector = np.array([0.01, -0.006, 0.008])
    tensor = np.array([[0.01, 0.003, 0], [0.003, -0.008, 0.002], [0, 0.002, 0.012]])
    query = {
        "force_initial": pair_laplacian(a),
        "force_final": pair_laplacian(b),
        "masses": np.array([0.3, -0.2]),
        "initial_cartesian": np.array([[0.003, -0.001, 0.002], [-0.001, 0.002, 0.001]]),
        "final_cartesian": np.array([[-0.002, 0.001, 0.003], [0.001, 0.002, -0.001]]),
        "constant": 0.0005,
        "linear": np.r_[vector, -vector],
        "quadratic": pair_laplacian(tensor),
        "temperature": 0.8,
        "width": 0.0008,
        "energies": np.linspace(-0.004, 0.003, len(case["energies"])),
    }
    return [query, {"masses": np.array([0.4, 0.15])}]


def finite_quantum_response(case, query, step=2e-5, cut=10):
    low = explicit_quantum(perturb(case, query, -step), cut)["kernel"]
    high = explicit_quantum(perturb(case, query, step), cut)["kernel"]
    return (high - low) / (2 * step)


def thermal_extreme_case():
    case = isotope_case(atoms=6, temperature=200.0, omega=0.18)
    case["kf"] = case["ki"] * (0.025 / 0.18) ** 2
    case["coupling"] = {
        "constant": 0.01,
        "linear": np.zeros(18),
        "quadratic": np.zeros((18, 18)),
    }
    case["energies"] = np.array([-0.2, 0.0, 0.2, 0.35])
    case["width"] = 0.025
    return case


def positive_extreme_state_sum(case):
    wi = 0.18
    wf = 0.025
    modes = 15
    cut_i = 4
    cut_f = 151
    x, weights = np.polynomial.hermite.hermgauss(128)
    q = x * np.sqrt(2 / (wi + wf))

    def hermites(arguments, cut):
        h = np.empty((cut, len(arguments)))
        h[0] = 1.0
        h[1] = np.sqrt(2) * arguments
        for n in range(1, cut - 1):
            h[n + 1] = (
                np.sqrt(2 / (n + 1)) * arguments * h[n]
                - np.sqrt(n / (n + 1)) * h[n - 1]
            )
        return h

    hi = hermites(np.sqrt(wi) * q, cut_i)
    hf = hermites(np.sqrt(wf) * q, cut_f)
    overlap = (
        np.sqrt(2)
        * (wi * wf) ** 0.25
        / np.sqrt(np.pi * (wi + wf))
        * (hf * weights[None, :])
        @ hi.T
    )
    occupation = np.exp(-wi / (KB * case["temperature"]))
    initial = (1 - occupation) * occupation ** np.arange(cut_i)
    lo = -36 * (cut_i - 1)
    hi_index = 5 * (cut_f - 1)
    one = np.zeros(hi_index - lo + 1)
    for ni in range(cut_i):
        for nf in range(cut_f):
            one[5 * nf - 36 * ni - lo] += initial[ni] * overlap[nf, ni] ** 2
    probability = np.array([1.0])
    for _ in range(modes):
        probability = np.convolve(probability, one)
    transition = modes * 0.5 * (wf - wi) + 0.005 * (
        np.arange(len(probability)) + modes * lo
    )
    width = case["width"]
    w0 = case["coupling"]["constant"]
    kernel = np.array(
        [
            w0
            * w0
            * np.sum(probability * np.exp(-0.5 * ((e - transition) / width) ** 2))
            / (np.sqrt(2 * np.pi) * width)
            for e in case["energies"]
        ]
    )
    return {
        "kernel": kernel,
        "probability_mass": float(probability.sum()),
        "omega_initial": np.full(modes, wi),
        "omega_final": np.full(modes, wf),
    }
