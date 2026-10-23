"""Independent scientific and native-interface verification of the public kernel."""

import copy
import numpy as np
import pytest
from scipy import linalg
from pymatgen.core import Structure, Lattice
from candidate import call
from independent import (
    coupled_case,
    explicit_quantum,
    isotope_case,
    closed_isotropic,
    arguments,
    normal_coordinates,
    pair_laplacian,
    coupled_queries,
    finite_quantum_response,
    perturb,
    thermal_extreme_case,
    positive_extreme_state_sum,
)
from dense_independent import (
    make_case as dense_case,
    convolved_kernel,
    queries as dense_queries,
    shifted_case,
)
from full_state_sum import make_case as fully_mixed_case, positive_state_sum


class DeferredResult:
    def __init__(self, case):
        self.case = case
        self.value = None
        self.error = None

    def __getitem__(self, key):
        if self.value is None and self.error is None:
            args, kwargs = arguments(self.case)
            try:
                self.value = call("capture_kernel", *args, **kwargs)
            except Exception as error:
                self.error = error
        if self.error is not None:
            raise self.error
        return self.value[key]


def independent_expectation(case):
    if case.get("full_state_sum"):
        expected = positive_state_sum(case)
        assert expected["uniform_truncation_bound"] < 5e-9, (
            "Positive omitted spectral mass consumes too much tolerance"
        )
        assert expected["force_commutator_norm"] > 1e-5, (
            "The fifteen-mode surfaces must not commute"
        )
        return expected
    if "intrinsic_blocks" in case:
        expected = convolved_kernel(case, 12)
        assert abs(expected["probability_mass"] - 1) < 1e-8, (
            "Complete block state-sum normalization"
        )
        return expected
    return explicit_quantum(case, 10)


@pytest.fixture(
    scope="module", params=["three_mode", "dense_fifteen", "fully_mixed_fifteen"]
)
def coupled(request):
    case = {
        "three_mode": coupled_case,
        "dense_fifteen": dense_case,
        "fully_mixed_fifteen": fully_mixed_case,
    }[request.param]()
    expected = independent_expectation(case)
    return case, expected, DeferredResult(case)


@pytest.fixture(params=["three_mode", "dense_fifteen"])
def quadratic(request):
    case = {"three_mode": coupled_case, "dense_fifteen": dense_case}[request.param]()
    case["coupling"] = {
        "constant": -0.003,
        "linear": 0.5 * case["coupling"]["linear"],
        "quadratic": 3 * case["coupling"]["quadratic"],
    }
    if "intrinsic_blocks" in case:
        case["scales"].update(linear=0.5, quadratic=3.0)
    return case, independent_expectation(case)


@pytest.fixture(params=["three_mode", "dense_fifteen"])
def noncommuting(request):
    case = {"three_mode": coupled_case, "dense_fifteen": dense_case}[request.param]()
    if request.param == "three_mode":
        rotation = linalg.expm(
            np.array([[0.0, -0.8, 0.3], [0.8, 0.0, -0.6], [-0.3, 0.6, 0.0]])
        )
        case["kf"] = pair_laplacian(rotation @ np.diag([7.5, 13.0, 20.0]) @ rotation.T)
    return case, independent_expectation(case)


@pytest.fixture(params=["three_mode", "dense_fifteen"])
def physical_response(request):
    case = {"three_mode": coupled_case, "dense_fifteen": dense_case}[request.param]()
    if request.param == "three_mode":
        case["queries"] = coupled_queries(case)
        expected = np.array(
            [finite_quantum_response(case, q, cut=10) for q in case["queries"]]
        )
    else:
        case["queries"] = dense_queries(case)
        step = 2e-5
        expected = np.array(
            [
                (
                    convolved_kernel(shifted_case(case, q, step), 12)["kernel"]
                    - convolved_kernel(shifted_case(case, q, -step), 12)["kernel"]
                )
                / (2 * step)
                for q in case["queries"]
            ]
        )
    for direction in case["queries"]:
        for key, value in direction.items():
            assert np.max(np.abs(value)) <= 1, (
                "Direction outside public component bound: " + key
            )
    return case, expected


def test_coupled_quantum_state_spectrum(coupled):
    """ "这是对全部内部量子态的求和；sigma 是给定的 Gaussian 能量展宽的标准差，单位 eV。" Independent converged oscillator-wavefunction state sum."""
    case, expected, actual = coupled
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="complete coupled quantum kernel",
    )
    assert np.min(np.asarray(actual["kernel"])) >= -5e-8, (
        "A physical transition kernel cannot be substantially negative"
    )


def test_mode_quanta_partition_and_zero_point(coupled):
    """ "振动能量包含零点能。" Partition functions and native frequency units include the full internal quantum spectrum."""
    case, expected, actual = coupled
    np.testing.assert_allclose(
        actual["omega_initial"],
        expected["omega_initial"],
        atol=1e-7,
        rtol=3e-6,
        err_msg="initial internal phonon quanta in eV",
    )
    np.testing.assert_allclose(
        actual["omega_final"],
        expected["omega_final"],
        atol=1e-7,
        rtol=3e-6,
        err_msg="final internal phonon quanta in eV",
    )
    np.testing.assert_allclose(
        [actual["log_Z_initial"], actual["log_Z_final"]],
        [expected["log_Z_initial"], expected["log_Z_final"]],
        atol=1e-7,
        rtol=3e-6,
        err_msg="complete partition functions with zero-point energies",
    )


def test_coordinate_dependent_quadratic_coupling(quadratic):
    """ "电子跃迁耦合为 Hermitian 的实坐标乘法算符 W(x)=w0+g^T x+(1/2)x^T B x。" Linear/quadratic amplitudes and their interference are not separate rates."""
    case, expected = quadratic
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="quadratic non-Condon operator and coherent cross terms",
    )


def test_noncommuting_force_surfaces(noncommuting):
    """ "初、末态一般不能用同一套独立正常模式同时对角化。" A rotated final force surface changes multimode wavefunction overlaps."""
    case, expected = noncommuting
    p = normal_coordinates(case)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.max(np.abs(p["rotation"] - np.diag(np.diag(p["rotation"])))) > 0.1, (
        "Fixture must mix internal normal modes"
    )
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="noncommuting initial/final phonon surfaces",
    )


def test_finite_temperature_up_and_down_transitions():
    """ "E 可以为负，零点能差包含在振动能量差中。" Exact warm thermal lines include both phonon absorption and emission."""
    case = isotope_case(atoms=4, temperature=1000.0, omega=0.06)
    expected = closed_isotropic(case)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.min(expected["line_strengths"]) > 1e-7, (
        "Warm fixture must exercise every positive/negative phonon line"
    )
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="warm multi-phonon absorption/emission strengths and thermal normalization",
    )


def test_nonuniform_isotope_mass_metric():
    """ "质量 m 的单位为 amu，缺省使用初态 site.specie.atomic_mass，也可提供按初态顺序的同位素质量数组。" Unequal isotope masses alter the kinetic metric and coupling coordinates."""
    case = coupled_case()
    case["masses"] = np.array([29.1, 18.3])
    expected = explicit_quantum(case, 10)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="unequal isotope kinetic/coupling metric",
    )
    np.testing.assert_allclose(
        actual["omega_initial"],
        expected["omega_initial"],
        atol=1e-7,
        rtol=3e-6,
        err_msg="isotope-dependent native phonon energy scale",
    )


def test_native_atom_identity_and_final_force_order():
    """ "两态原子按 ID 对应。" Final force constants use the final structure order, not the initial array order."""
    case = coupled_case()
    expected = explicit_quantum(case, 10)
    permutation = np.array([1, 0])
    axes = (3 * permutation[:, None] + np.arange(3)).ravel()
    original = case["final"]
    case["final"] = Structure(
        original.lattice,
        [original[i].specie for i in permutation],
        original.cart_coords[permutation],
        coords_are_cartesian=True,
        site_properties={
            "atom_id": [original.site_properties["atom_id"][i] for i in permutation]
        },
    )
    case["kf"] = case["kf"][np.ix_(axes, axes)]
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="stable native atom ID and final Hessian correspondence",
    )


def test_periodic_images_and_cartesian_covariance():
    """ "不同内部基、简并模中的基选择、整体 Cartesian 旋转或相同原子结构的周期像表示，不改变物理结果。" Nonorthogonal cells and covariant tensors describe one physical case."""
    case = coupled_case()
    expected = explicit_quantum(case, 10)
    rot = linalg.expm(np.array([[0.0, -0.3, 0.5], [0.3, 0.0, -0.2], [-0.5, 0.2, 0.0]]))
    full = linalg.block_diag(rot, rot)
    lattice = Lattice(
        np.array([[8.0, 0.0, 0.0], [2.3, 7.8, 0.0], [-0.9, 1.2, 7.4]]) @ rot.T
    )
    for name, images in [
        ("initial", np.zeros((2, 3))),
        ("final", np.array([[2.0, -1.0, 0.0], [-1.0, 0.0, 1.0]])),
    ]:
        source = case[name]
        cart = source.cart_coords @ rot.T + images @ lattice.matrix
        case[name] = Structure(
            lattice,
            [s.specie for s in source],
            cart,
            coords_are_cartesian=True,
            site_properties=source.site_properties,
        )
    case["ki"] = full @ case["ki"] @ full.T
    case["kf"] = full @ case["kf"] @ full.T
    case["coupling"] = {
        "constant": case["coupling"]["constant"],
        "linear": full @ case["coupling"]["linear"],
        "quadratic": full @ case["coupling"]["quadratic"] @ full.T,
    }
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="physical periodic-image and rotated-tensor covariance",
    )


def test_complete_native_input_response(physical_response):
    """ "导数是完整 Phi 对该同时扰动的一阶响应，包含核动能度量、几何、两套势能面、耦合、热归一化、E 和展宽的共同变化。" Independent quantum-state perturbations exercise the whole physical chain."""
    case, expected = physical_response
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.max(np.abs(expected[0])) > 1e-5 and np.max(np.abs(expected[1])) > 1e-7, (
        "Both physical query directions must have nontrivial responses"
    )
    np.testing.assert_allclose(
        actual["kernel_response"],
        expected,
        atol=1e-6,
        rtol=2e-5,
        err_msg="joint Cartesian/force/isotope/coupling/thermal/energy/width kernel response",
    )


def test_exact_degeneracy_with_mode_splitting_response():
    """ "允许精确内部频率简并及会拆分简并的扰动；只要求物理谱核的导数，不要求个别排序简并频率的导数。" Physical response remains unique through a mode-splitting direction."""
    case = isotope_case(atoms=2, temperature=250.0, omega=0.09)
    a = np.array([[0.8, 0.4, -0.2], [0.4, -0.6, 0.3], [-0.2, 0.3, 0.3]])
    b = np.array([[-0.6, 0.2, 0.4], [0.2, 0.7, -0.3], [0.4, -0.3, 1.0]])
    case["queries"] = [
        {"force_initial": pair_laplacian(a), "force_final": pair_laplacian(b)}
    ]
    expected = finite_quantum_response(case, case["queries"][0], cut=10)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.max(np.abs(expected)) > 1e-6, (
        "Degenerate-splitting fixture must distinguish the physical derivative"
    )
    np.testing.assert_allclose(
        actual["kernel_response"][0],
        expected,
        atol=1e-6,
        rtol=2e-5,
        err_msg="basis-independent response at exact phonon degeneracy",
    )


def test_legal_thermal_frequency_extreme():
    """ "全部内部振动能量量子位于 [0.025,0.18] eV。" Independent positive 15-mode state convolution checks the legitimate thermal boundary."""
    case = thermal_extreme_case()
    expected = positive_extreme_state_sum(case)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert abs(expected["probability_mass"] - 1) < 2e-11, (
        "Independent complete-state probability normalization failed"
    )
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="low-temperature large-frequency-ratio quantum spectrum",
    )
    np.testing.assert_allclose(
        actual["omega_final"],
        expected["omega_final"],
        atol=1e-7,
        rtol=3e-6,
        err_msg="lower-frequency supported boundary",
    )


def test_full_fifteen_mode_quantum_kernel():
    """ "还需返回完整内部模式的初、末态振动能量量子 omega（eV，非负升序，各有 3N-3 个）和 log(Zi)、log(Zf)。" Exact thermal operator lines certify all fifteen internal modes."""
    case = isotope_case(atoms=6)
    expected = closed_isotropic(case)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.asarray(actual["omega_initial"]).shape == (15,) and np.asarray(
        actual["omega_final"]
    ).shape == (15,)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="complete 15-mode quadratic quantum kernel",
    )
    np.testing.assert_allclose(
        [actual["log_Z_initial"], actual["log_Z_final"]],
        [expected["log_Z_initial"], expected["log_Z_final"]],
        atol=1e-7,
        rtol=3e-6,
        err_msg="15-mode acoustic-free quantum partition functions",
    )


def test_energy_order_duplicates_and_zero_query_shape():
    """ "kernel 形状 (L,)；kernel_response 形状 (q,L)，q=0 保留零长度首维；两个 omega 形状 (3N-3,)，两个 log_Z 为实标量。" Ordered energy slots and empty response axes are preserved."""
    case = isotope_case(atoms=3)
    case["energies"] = np.array([0.07, -0.02, 0.14, 0.07])
    expected = closed_isotropic(case)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert np.asarray(actual["kernel"]).shape == (4,) and np.asarray(
        actual["kernel_response"]
    ).shape == (0, 4)
    np.testing.assert_allclose(
        actual["kernel"],
        expected["kernel"],
        atol=5e-8,
        rtol=5e-6,
        err_msg="unsorted/repeated energy slot semantics",
    )


def test_output_schema_numeric_types_and_finiteness():
    """ "数值输出必须有限，可用 ndarray 或可转换成相同形状的数值序列，不能返回编码数组字符串。" Data-only numeric outputs have the complete schema."""
    case = isotope_case(atoms=3)
    args, kwargs = arguments(case)
    actual = call("capture_kernel", *args, **kwargs)
    assert set(actual) == {
        "kernel",
        "kernel_response",
        "omega_initial",
        "omega_final",
        "log_Z_initial",
        "log_Z_final",
    }
    assert all(np.asarray(actual[key]).dtype.kind in "fiu" for key in actual), (
        "Output fields must contain real numeric values"
    )
    assert all(np.all(np.isfinite(actual[key])) for key in actual), (
        "Nonfinite physical output is not valid"
    )
    assert (
        np.asarray(actual["log_Z_initial"]).shape == ()
        and np.asarray(actual["log_Z_final"]).shape == ()
    )


def test_rejects_bad_shapes_nonfinite_and_unknown_fields():
    """ "非法字段、形状、非有限数值、非正质量、ID/元素/晶格不匹配、非对称矩阵或违反声学/耦合平移约束时抛出 ValueError；其他支持域和查询物理条件由调用方保证。" Invalid physical inputs reach and are rejected by the actual candidate API."""
    case = isotope_case(atoms=3)
    bad = {**case, "coupling": {**case["coupling"], "constant": float("nan")}}
    args, kwargs = arguments(bad)
    with pytest.raises(ValueError):
        call("capture_kernel", *args, **kwargs)
    bad = {**case, "ki": case["ki"][:-1]}
    args, kwargs = arguments(bad)
    with pytest.raises(ValueError):
        call("capture_kernel", *args, **kwargs)
    bad = {**case, "queries": [{"unknown_query": 1}]}
    args, kwargs = arguments(bad)
    with pytest.raises(ValueError):
        call("capture_kernel", *args, **kwargs)


def test_rejects_native_mismatch_without_mutating_inputs():
    """ "输入结构、数组、coupling、queries 均不得被修改；拒收时也保持原值。" Native chemistry/acoustic errors are atomic input rejections."""
    case = isotope_case(atoms=3)
    case["final"] = case["final"].copy()
    case["final"].replace(0, "Ge")
    args, kwargs = arguments(case)
    with pytest.raises(ValueError):
        call("capture_kernel", *args, **kwargs)
    case = isotope_case(atoms=3)
    case["ki"] = case["ki"].copy()
    case["ki"][0, 0] += 0.1
    args, kwargs = arguments(case)
    with pytest.raises(ValueError):
        call("capture_kernel", *args, **kwargs)
