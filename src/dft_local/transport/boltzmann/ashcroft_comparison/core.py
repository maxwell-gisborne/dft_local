from __future__ import annotations


# Canonical SI aliases used by conductivity diagnostics.
KB_J_K = 1.380649e-23
ELECTRON_CHARGE_C = 1.602176634e-19

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True, slots=True)
class VincentReference:
    temperature_K: float
    relaxation_time_s: float
    electric_field_V_per_m: np.ndarray
    expected_conductivity_S_per_m: np.ndarray
    max_fermi_weight: float
    min_fermi_weight: float
    mean_fermi_weight: float


@dataclass(frozen=True, slots=True)
class VincentInputData:
    primitive_lattice_vectors_bohr: np.ndarray
    epsilon_of_k: np.ndarray


def vincent_reference() -> VincentReference:
    return VincentReference(
        temperature_K=300.0,
        relaxation_time_s=1.0e-14,
        electric_field_V_per_m=np.array([1.0e5, 0.0], dtype=float),
        expected_conductivity_S_per_m=np.array(
            [
                [6.45179383e-02, -8.80479820e-05],
                [-8.73823365e-05, 6.44024548e-02],
            ],
            dtype=float,
        ),
        max_fermi_weight=2.499e-01,
        min_fermi_weight=0.0,
        mean_fermi_weight=3.907e-03,
    )


def domain_root() -> Path:
    return Path(__file__).resolve().parent


def load_primitive_lattice_vectors(path: Path | None = None) -> np.ndarray:
    source = path if path is not None else domain_root() / "ai.txt"
    data = np.loadtxt(source, dtype=float)

    if data.shape != (2, 2):
        raise ValueError(f"Expected two 2D primitive lattice vectors in {source}, got shape {data.shape}")

    return data


def load_epsilon_of_k(path: Path | None = None) -> np.ndarray:
    source = path if path is not None else domain_root() / "epsilon_of_k.txt"
    data = np.loadtxt(source, dtype=float)

    if data.ndim == 1:
        data = data.reshape((-1, 1))

    if data.shape[0] == 0:
        raise ValueError(f"No epsilon(k) rows found in {source}")

    return data


def load_vincent_input_data(root: Path | None = None) -> VincentInputData:
    base = root if root is not None else domain_root()

    return VincentInputData(
        primitive_lattice_vectors_bohr=load_primitive_lattice_vectors(base / "ai.txt"),
        epsilon_of_k=load_epsilon_of_k(base / "epsilon_of_k.txt"),
    )


def reciprocal_lattice_vectors_from_primitives(ai_bohr: np.ndarray) -> np.ndarray:
    """Return reciprocal basis rows b_i satisfying a_i . b_j = 2 pi delta_ij."""

    ai_bohr = np.asarray(ai_bohr, dtype=float)

    if ai_bohr.shape != (2, 2):
        raise ValueError(f"Expected primitive lattice shape (2, 2), got {ai_bohr.shape}")

    return 2.0 * np.pi * np.linalg.inv(ai_bohr).T


def relative_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    scale = np.maximum(np.abs(expected), 1.0e-300)
    return np.abs(actual - expected) / scale


# SI constants used by Vincent's reference calculation.
E_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
KB_J_PER_K = 1.380649e-23
HARTREE_TO_J = 4.3597447222071e-18
BOHR_TO_M = 0.52917721092e-10


def reciprocal_grid_fractional(shape: tuple[int, int], endpoint: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return fractional coordinates on the 2D reciprocal grid.

    endpoint=False gives n / N.
    endpoint=True gives n / (N - 1).
    """

    n1, n2 = shape
    d1 = n1 - 1 if endpoint else n1
    d2 = n2 - 1 if endpoint else n2

    u = np.arange(n1, dtype=float) / float(d1)
    v = np.arange(n2, dtype=float) / float(d2)

    return np.meshgrid(u, v, indexing="ij")


def reciprocal_grid_cartesian_per_m(ai_bohr: np.ndarray, shape: tuple[int, int], endpoint: bool = False) -> tuple[np.ndarray, np.ndarray]:
    bi_per_bohr = reciprocal_lattice_vectors_from_primitives(ai_bohr)
    uu, vv = reciprocal_grid_fractional(shape, endpoint=endpoint)

    k_bohr_inv = uu[..., None] * bi_per_bohr[0] + vv[..., None] * bi_per_bohr[1]
    k_per_m = k_bohr_inv / BOHR_TO_M

    return k_per_m[..., 0], k_per_m[..., 1]


def velocity_from_epsilon_grid(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity in m/s from epsilon(k) in Hartree on a reciprocal fractional grid."""

    epsilon_hartree = np.asarray(epsilon_hartree, dtype=float)
    n1, n2 = epsilon_hartree.shape

    du = 1.0 / float(n1 - 1 if endpoint else n1)
    dv = 1.0 / float(n2 - 1 if endpoint else n2)

    d_e_du, d_e_dv = np.gradient(epsilon_hartree, du, dv, edge_order=2)

    bi_per_bohr = reciprocal_lattice_vectors_from_primitives(ai_bohr)

    # k_cart = q B, with q = (u, v) and reciprocal vectors stored as rows.
    # Therefore grad_q E = grad_k E B^T, so grad_k E = grad_q E (B^T)^-1.
    grad_q = np.stack([d_e_du, d_e_dv], axis=-1)
    grad_k_hartree_bohr = np.einsum("...i,ij->...j", grad_q, np.linalg.inv(bi_per_bohr.T))

    grad_k_j_m = grad_k_hartree_bohr * HARTREE_TO_J * BOHR_TO_M
    velocity_m_per_s = grad_k_j_m / HBAR_J_S

    return velocity_m_per_s[..., 0], velocity_m_per_s[..., 1]


def vincent_sample_velocity_targets() -> tuple[np.ndarray, np.ndarray]:
    k_per_m = np.array(
        [
            [0.0, -0.0],
            [2.55464699e08, -1.47492613e08],
            [5.10929398e08, -2.94985226e08],
            [7.66394098e08, -4.42477839e08],
            [1.02185880e09, -5.89970451e08],
        ],
        dtype=float,
    )

    v_m_per_s = np.array(
        [
            [-7090.46879031, -12102.11158297],
            [14075.04503413, -24202.4466007],
            [28253.4799914, -24202.44660072],
            [49414.51560173, -36301.36263201],
            [77551.52350077, -60490.82133335],
        ],
        dtype=float,
    )

    return k_per_m, v_m_per_s


def sample_target_path_values(array: np.ndarray, count: int = 5) -> np.ndarray:
    return np.asarray(array[0, :count], dtype=float)


def velocity_candidates_for_vincent_samples(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return candidate velocities at Vincent's first five sample k-points.

    The target k-points follow the second reciprocal direction, i.e. grid row 0,
    columns 0..4 when endpoint=False.
    """

    eps = np.asarray(epsilon_hartree, dtype=float)
    candidates: dict[str, np.ndarray] = {}

    for name, arr in {
        "plain": eps,
        "transpose": eps.T,
        "neg_plain": -eps,
        "neg_transpose": -eps.T,
    }.items():
        vx, vy = velocity_from_epsilon_grid(arr, ai_bohr, endpoint=False)

        candidates[name] = np.stack(
            [
                sample_target_path_values(vx),
                sample_target_path_values(vy),
            ],
            axis=-1,
        )

    return candidates


def candidate_velocity_errors(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, float]:
    _target_k, target_v = vincent_sample_velocity_targets()
    candidates = velocity_candidates_for_vincent_samples(epsilon_hartree, ai_bohr)

    return {
        name: float(np.sqrt(np.mean((candidate - target_v) ** 2)))
        for name, candidate in candidates.items()
    }


def electric_field_k_shift_per_m(
    electric_field_V_per_m: np.ndarray,
    relaxation_time_s: float,
) -> np.ndarray:
    """Return semiclassical k-shift Δk = -e E tau / hbar in m^-1."""

    electric_field_V_per_m = np.asarray(electric_field_V_per_m, dtype=float)
    return -(E_CHARGE_C * electric_field_V_per_m * relaxation_time_s) / HBAR_J_S


def cartesian_k_to_fractional(k_per_m: np.ndarray, ai_bohr: np.ndarray) -> np.ndarray:
    """Convert Cartesian k in m^-1 to fractional reciprocal coordinates q.

    Reciprocal vectors are stored as rows in bohr^-1, and k_bohr^-1 = q B.
    """

    k_bohr_inv = np.asarray(k_per_m, dtype=float) * BOHR_TO_M
    bi = reciprocal_lattice_vectors_from_primitives(ai_bohr)

    return np.einsum("...i,ij->...j", k_bohr_inv, np.linalg.inv(bi))


def bilinear_periodic_sample(grid: np.ndarray, q_fractional: np.ndarray) -> np.ndarray:
    """Sample a 2D periodic grid at fractional coordinates q using bilinear interpolation."""

    grid = np.asarray(grid, dtype=float)
    q = np.asarray(q_fractional, dtype=float)

    n1, n2 = grid.shape
    x = np.mod(q[..., 0], 1.0) * n1
    y = np.mod(q[..., 1], 1.0) * n2

    i0 = np.floor(x).astype(int) % n1
    j0 = np.floor(y).astype(int) % n2
    i1 = (i0 + 1) % n1
    j1 = (j0 + 1) % n2

    tx = x - np.floor(x)
    ty = y - np.floor(y)

    return (
        (1.0 - tx) * (1.0 - ty) * grid[i0, j0]
        + tx * (1.0 - ty) * grid[i1, j0]
        + (1.0 - tx) * ty * grid[i0, j1]
        + tx * ty * grid[i1, j1]
    )


def velocity_at_cartesian_k_points(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    k_per_m: np.ndarray,
    *,
    sign: float = 1.0,
) -> np.ndarray:
    vx_grid, vy_grid = velocity_from_epsilon_grid(epsilon_hartree, ai_bohr, endpoint=False)
    q = cartesian_k_to_fractional(k_per_m, ai_bohr)

    return sign * np.stack(
        [
            bilinear_periodic_sample(vx_grid, q),
            bilinear_periodic_sample(vy_grid, q),
        ],
        axis=-1,
    )


def shifted_velocity_candidate_errors(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, float]:
    reference = vincent_reference()
    target_k, target_v = vincent_sample_velocity_targets()
    dk = electric_field_k_shift_per_m(
        reference.electric_field_V_per_m,
        reference.relaxation_time_s,
    )

    candidates = {
        "v(k)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k),
        "-v(k)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k, sign=-1.0),
        "v(k+dk)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k + dk),
        "-v(k+dk)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k + dk, sign=-1.0),
        "v(k-dk)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k - dk),
        "-v(k-dk)": velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k - dk, sign=-1.0),
    }

    return {
        name: float(np.sqrt(np.mean((candidate - target_v) ** 2)))
        for name, candidate in candidates.items()
    }


def finite_difference_velocity_basis_samples(
    epsilon_hartree: np.ndarray,
    *,
    endpoint: bool = False,
    count: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw dE/du and dE/dv based velocity-scale samples along Vincent path.

    These are not Cartesian velocities. They expose the fractional-grid derivative
    basis so we can infer whether Vincent used a different basis transform.
    """

    eps = np.asarray(epsilon_hartree, dtype=float)
    n1, n2 = eps.shape
    du = 1.0 / float(n1 - 1 if endpoint else n1)
    dv = 1.0 / float(n2 - 1 if endpoint else n2)

    d_e_du, d_e_dv = np.gradient(eps, du, dv, edge_order=2)

    # Convert Hartree to J, then multiply by Bohr/hbar just to put magnitudes in
    # velocity-like units before basis conversion.
    scale = HARTREE_TO_J * BOHR_TO_M / HBAR_J_S

    return (
        sample_target_path_values(d_e_du * scale, count=count),
        sample_target_path_values(d_e_dv * scale, count=count),
    )


def fit_fractional_derivative_basis_to_vincent(
    epsilon_hartree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit target velocity = raw fractional derivative basis @ matrix."""

    du_v, dv_v = finite_difference_velocity_basis_samples(epsilon_hartree)
    design = np.stack([du_v, dv_v], axis=-1)

    _target_k, target_v = vincent_sample_velocity_targets()

    matrix, *_ = np.linalg.lstsq(design, target_v, rcond=None)
    predicted = design @ matrix

    return matrix, predicted


def derivative_stencil_velocity_candidates(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    count: int = 5,
) -> dict[str, np.ndarray]:
    eps0 = np.asarray(epsilon_hartree, dtype=float)
    out: dict[str, np.ndarray] = {}

    bi = reciprocal_lattice_vectors_from_primitives(ai_bohr)
    scale = HARTREE_TO_J * BOHR_TO_M / HBAR_J_S
    n1, n2 = eps0.shape
    du = 1.0 / float(n1)
    dv = 1.0 / float(n2)

    def convert(d_e_du: np.ndarray, d_e_dv: np.ndarray) -> np.ndarray:
        grad_q = np.stack([d_e_du, d_e_dv], axis=-1)
        grad_k_hartree_bohr = np.einsum("...i,ij->...j", grad_q, np.linalg.inv(bi.T))
        return grad_k_hartree_bohr * scale

    for arr_name, eps in {
        "plain": eps0,
        "transpose": eps0.T,
    }.items():
        stencils: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        stencils["gradient1"] = np.gradient(eps, du, dv, edge_order=1)
        stencils["gradient2"] = np.gradient(eps, du, dv, edge_order=2)

        stencils["forward"] = (
            (np.roll(eps, -1, axis=0) - eps) / du,
            (np.roll(eps, -1, axis=1) - eps) / dv,
        )
        stencils["backward"] = (
            (eps - np.roll(eps, 1, axis=0)) / du,
            (eps - np.roll(eps, 1, axis=1)) / dv,
        )
        stencils["central_roll"] = (
            (np.roll(eps, -1, axis=0) - np.roll(eps, 1, axis=0)) / (2.0 * du),
            (np.roll(eps, -1, axis=1) - np.roll(eps, 1, axis=1)) / (2.0 * dv),
        )

        for stencil_name, (d_e_du, d_e_dv) in stencils.items():
            velocity = convert(d_e_du, d_e_dv)
            sample = velocity[0, :count, :]

            out[f"{arr_name}:{stencil_name}"] = sample
            out[f"-{arr_name}:{stencil_name}"] = -sample

    return out


def derivative_stencil_velocity_errors(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, float]:
    _target_k, target_v = vincent_sample_velocity_targets()
    candidates = derivative_stencil_velocity_candidates(epsilon_hartree, ai_bohr)

    return {
        name: float(np.sqrt(np.mean((candidate - target_v) ** 2)))
        for name, candidate in candidates.items()
    }


VINCENT_REPORTED_SHIFTED_K_PER_M = np.array([0.0, -29498522.56891833], dtype=float)


def energy_at_cartesian_k_points(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    k_per_m: np.ndarray,
) -> np.ndarray:
    q = cartesian_k_to_fractional(k_per_m, ai_bohr)
    return bilinear_periodic_sample(epsilon_hartree, q)


def directional_velocity_from_reported_shift(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    k_per_m: np.ndarray,
    *,
    shift_per_m: np.ndarray = VINCENT_REPORTED_SHIFTED_K_PER_M,
) -> np.ndarray:
    """Estimate directional velocity from Vincent's reported shifted k.

    This returns a vector parallel to shift:
        v_parallel = (1 / hbar) (E(k+dk)-E(k)) / |dk| * dk_hat

    with epsilon in Hartree and k in m^-1.
    """

    k = np.asarray(k_per_m, dtype=float)
    shift = np.asarray(shift_per_m, dtype=float)
    norm = float(np.linalg.norm(shift))

    if norm == 0.0:
        raise ValueError("shift must be non-zero")

    e0 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k)
    e1 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k + shift)

    directional_derivative_j_m = ((e1 - e0) * HARTREE_TO_J) / norm
    speed = directional_derivative_j_m / HBAR_J_S

    return speed[..., None] * (shift / norm)


def reported_shift_velocity_errors(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, float]:
    target_k, target_v = vincent_sample_velocity_targets()

    candidates = {
        "reported_shift_forward": directional_velocity_from_reported_shift(epsilon_hartree, ai_bohr, target_k),
        "reported_shift_backward": directional_velocity_from_reported_shift(
            epsilon_hartree,
            ai_bohr,
            target_k,
            shift_per_m=-VINCENT_REPORTED_SHIFTED_K_PER_M,
        ),
    }
    candidates["-reported_shift_forward"] = -candidates["reported_shift_forward"]
    candidates["-reported_shift_backward"] = -candidates["reported_shift_backward"]

    return {
        name: float(np.sqrt(np.mean((candidate - target_v) ** 2)))
        for name, candidate in candidates.items()
    }


def cartesian_component_velocity_from_steps(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    k_per_m: np.ndarray,
    *,
    dx_per_m: float,
    dy_per_m: float,
    forward: bool = True,
) -> np.ndarray:
    """Estimate Cartesian velocity components using separate finite differences.

    v_x = (1 / hbar) d epsilon / d k_x
    v_y = (1 / hbar) d epsilon / d k_y

    dx_per_m and dy_per_m are finite-difference steps in m^-1.
    """

    k = np.asarray(k_per_m, dtype=float)

    sx = np.array([dx_per_m, 0.0], dtype=float)
    sy = np.array([0.0, dy_per_m], dtype=float)

    if forward:
        e0 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k)
        ex = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k + sx)
        ey = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k + sy)

        vx = ((ex - e0) * HARTREE_TO_J) / (dx_per_m * HBAR_J_S)
        vy = ((ey - e0) * HARTREE_TO_J) / (dy_per_m * HBAR_J_S)
    else:
        ex0 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k - sx)
        ex1 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k + sx)
        ey0 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k - sy)
        ey1 = energy_at_cartesian_k_points(epsilon_hartree, ai_bohr, k + sy)

        vx = ((ex1 - ex0) * HARTREE_TO_J) / (2.0 * dx_per_m * HBAR_J_S)
        vy = ((ey1 - ey0) * HARTREE_TO_J) / (2.0 * dy_per_m * HBAR_J_S)

    return np.stack([vx, vy], axis=-1)


def search_cartesian_component_steps(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> list[tuple[float, str, float, float, np.ndarray]]:
    target_k, target_v = vincent_sample_velocity_targets()

    b2_step = target_k[1] - target_k[0]
    base = abs(VINCENT_REPORTED_SHIFTED_K_PER_M[1])

    # Candidate hidden steps: Vincent reported shift, fractions of grid step, and grid components.
    step_values = sorted(
        {
            base,
            abs(b2_step[0]),
            abs(b2_step[1]),
            abs(b2_step[0]) / 2.0,
            abs(b2_step[1]) / 2.0,
            abs(b2_step[0]) / 5.0,
            abs(b2_step[1]) / 5.0,
            abs(b2_step[0]) / 10.0,
            abs(b2_step[1]) / 10.0,
            abs(b2_step[0]) / 20.0,
            abs(b2_step[1]) / 20.0,
        }
    )

    results: list[tuple[float, str, float, float, np.ndarray]] = []

    for dx in step_values:
        for dy in step_values:
            for mode, forward in [("forward", True), ("central", False)]:
                got = cartesian_component_velocity_from_steps(
                    epsilon_hartree,
                    ai_bohr,
                    target_k,
                    dx_per_m=dx,
                    dy_per_m=dy,
                    forward=forward,
                )

                for sign_name, signed in [("+", got), ("-", -got)]:
                    err = float(np.sqrt(np.mean((signed - target_v) ** 2)))
                    results.append((err, f"{sign_name}{mode}", dx, dy, signed))

    results.sort(key=lambda item: item[0])
    return results


def central_cartesian_velocity_grid(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_k, _target_v = vincent_sample_velocity_targets()
    grid_step = float(np.linalg.norm(target_k[1] - target_k[0]))

    shape = np.asarray(epsilon_hartree).shape
    kx, ky = reciprocal_grid_cartesian_per_m(ai_bohr, shape, endpoint=False)
    k = np.stack([kx, ky], axis=-1)

    vxvy = cartesian_component_velocity_from_steps(
        epsilon_hartree,
        ai_bohr,
        k,
        dx_per_m=grid_step,
        dy_per_m=grid_step,
        forward=False,
    )

    return vxvy[..., 0], vxvy[..., 1]


def search_grid_offset_velocity_samples(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    max_offset: int = 8,
) -> list[tuple[float, str, int, int, np.ndarray]]:
    _target_k, target_v = vincent_sample_velocity_targets()
    vx, vy = central_cartesian_velocity_grid(epsilon_hartree, ai_bohr)

    n1, n2 = vx.shape
    results: list[tuple[float, str, int, int, np.ndarray]] = []

    for row_offset in range(-max_offset, max_offset + 1):
        for col_offset in range(-max_offset, max_offset + 1):
            samples = []

            for j in range(len(target_v)):
                row = row_offset % n1
                col = (j + col_offset) % n2
                samples.append([vx[row, col], vy[row, col]])

            got = np.asarray(samples, dtype=float)

            for sign_name, signed in [("+", got), ("-", -got)]:
                err = float(np.sqrt(np.mean((signed - target_v) ** 2)))
                results.append((err, sign_name, row_offset, col_offset, signed))

    results.sort(key=lambda item: item[0])
    return results


def search_baseline_subtracted_velocity_samples(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    max_baseline_offset: int = 8,
) -> list[tuple[float, str, int, int, np.ndarray]]:
    _target_k, target_v = vincent_sample_velocity_targets()
    vx, vy = central_cartesian_velocity_grid(epsilon_hartree, ai_bohr)

    n1, n2 = vx.shape
    samples = np.stack([vx[0, : len(target_v)], vy[0, : len(target_v)]], axis=-1)

    results: list[tuple[float, str, int, int, np.ndarray]] = []

    for row in range(-max_baseline_offset, max_baseline_offset + 1):
        for col in range(-max_baseline_offset, max_baseline_offset + 1):
            baseline = np.array([vx[row % n1, col % n2], vy[row % n1, col % n2]], dtype=float)

            for name, got in {
                "v-base": samples - baseline,
                "base-v": baseline - samples,
                "v+base": samples + baseline,
                "-v-base": -samples - baseline,
            }.items():
                err = float(np.sqrt(np.mean((got - target_v) ** 2)))
                results.append((err, name, row, col, got))

    results.sort(key=lambda item: item[0])
    return results


@dataclass(frozen=True, slots=True)
class VelocitySystematicErrorProbe:
    target_k_per_m: np.ndarray
    target_v_m_per_s: np.ndarray
    local_v_m_per_s: np.ndarray
    delta_v_m_per_s: np.ndarray
    percent_error: np.ndarray
    delta_step_m_per_s: np.ndarray
    rms_error_m_per_s: float
    mean_delta_m_per_s: np.ndarray


def velocity_systematic_error_probe(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> VelocitySystematicErrorProbe:
    """Compare Vincent's sample velocities against the local derivative.

    The sample path is known to be k[j] = j b2 / 100, corresponding to
    epsilon[0, j]. This probe therefore compares Vincent's reported sample
    velocities with the local derivative evaluated on the same path.
    """

    target_k, target_v = vincent_sample_velocity_targets()
    vx, vy = central_cartesian_velocity_grid(epsilon_hartree, ai_bohr)
    local_v = np.stack([vx[0, : len(target_v)], vy[0, : len(target_v)]], axis=-1)

    delta = local_v - target_v
    percent_error = 100.0 * np.abs(delta) / np.maximum(np.abs(target_v), 1.0e-300)

    delta_step = np.zeros_like(delta)
    delta_step[1:] = delta[1:] - delta[:-1]

    return VelocitySystematicErrorProbe(
        target_k_per_m=target_k,
        target_v_m_per_s=target_v,
        local_v_m_per_s=local_v,
        delta_v_m_per_s=delta,
        percent_error=percent_error,
        delta_step_m_per_s=delta_step,
        rms_error_m_per_s=float(np.sqrt(np.mean(delta**2))),
        mean_delta_m_per_s=np.mean(delta, axis=0),
    )


@dataclass(frozen=True, slots=True)
class VelocityGridMatch:
    target_index: int
    row: int
    col: int
    local_velocity_m_per_s: np.ndarray
    target_velocity_m_per_s: np.ndarray
    delta_m_per_s: np.ndarray
    error_m_per_s: float


@dataclass(frozen=True, slots=True)
class VelocityOffsetPathMatch:
    start_row: int
    start_col: int
    step_row: int
    step_col: int
    local_velocity_m_per_s: np.ndarray
    delta_m_per_s: np.ndarray
    rms_error_m_per_s: float


def nearest_velocity_grid_matches(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    count: int = 8,
) -> tuple[tuple[VelocityGridMatch, ...], ...]:
    """For each Vincent velocity sample, find nearest local velocities anywhere on the grid."""

    _target_k, target_v = vincent_sample_velocity_targets()
    vx, vy = central_cartesian_velocity_grid(epsilon_hartree, ai_bohr)
    local = np.stack([vx, vy], axis=-1)

    matches: list[tuple[VelocityGridMatch, ...]] = []

    for target_index, target in enumerate(target_v):
        errors = np.sqrt(np.sum((local - target) ** 2, axis=-1))
        flat_order = np.argsort(errors, axis=None)[:count]

        target_matches: list[VelocityGridMatch] = []
        for flat in flat_order:
            row, col = np.unravel_index(flat, errors.shape)
            local_velocity = local[row, col]
            delta = local_velocity - target

            target_matches.append(
                VelocityGridMatch(
                    target_index=target_index,
                    row=int(row),
                    col=int(col),
                    local_velocity_m_per_s=local_velocity,
                    target_velocity_m_per_s=target,
                    delta_m_per_s=delta,
                    error_m_per_s=float(errors[row, col]),
                )
            )

        matches.append(tuple(target_matches))

    return tuple(matches)


def search_velocity_offset_paths(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
    *,
    step_radius: int = 3,
    limit: int = 20,
) -> tuple[VelocityOffsetPathMatch, ...]:
    """Search straight grid paths whose local velocities best match Vincent samples.

    A candidate path is:

        row(j) = start_row + j * step_row
        col(j) = start_col + j * step_col

    with periodic wrap.
    """

    _target_k, target_v = vincent_sample_velocity_targets()
    vx, vy = central_cartesian_velocity_grid(epsilon_hartree, ai_bohr)
    local = np.stack([vx, vy], axis=-1)

    n1, n2 = vx.shape
    n_samples = len(target_v)

    results: list[VelocityOffsetPathMatch] = []

    for step_row in range(-step_radius, step_radius + 1):
        for step_col in range(-step_radius, step_radius + 1):
            if step_row == 0 and step_col == 0:
                continue

            for start_row in range(n1):
                for start_col in range(n2):
                    rows = (start_row + np.arange(n_samples) * step_row) % n1
                    cols = (start_col + np.arange(n_samples) * step_col) % n2

                    sampled = local[rows, cols]
                    delta = sampled - target_v
                    rms = float(np.sqrt(np.mean(delta**2)))

                    results.append(
                        VelocityOffsetPathMatch(
                            start_row=int(start_row),
                            start_col=int(start_col),
                            step_row=int(step_row),
                            step_col=int(step_col),
                            local_velocity_m_per_s=sampled,
                            delta_m_per_s=delta,
                            rms_error_m_per_s=rms,
                        )
                    )

    results.sort(key=lambda item: item.rms_error_m_per_s)
    return tuple(results[:limit])


@dataclass(frozen=True, slots=True)
class ShiftDiscrepancyProbe:
    expected_shift_per_m: np.ndarray
    reported_shift_per_m: np.ndarray
    delta_per_m: np.ndarray
    ratio: np.ndarray
    expected_norm_per_m: float
    reported_norm_per_m: float
    norm_ratio: float


def shift_discrepancy_probe() -> ShiftDiscrepancyProbe:
    """Compare Vincent's printed shifted k-point with the semiclassical shift.

    The semiclassical drift estimate from the stated constants is:

        delta k = -e E tau / hbar

    Vincent's printed shifted point is stored separately because it does not match
    that value in either direction or magnitude.
    """

    reference = vincent_reference()
    expected = electric_field_k_shift_per_m(
        reference.electric_field_V_per_m,
        reference.relaxation_time_s,
    )
    reported = VINCENT_REPORTED_SHIFTED_K_PER_M.copy()
    delta = reported - expected

    ratio = np.full_like(reported, np.nan, dtype=float)
    mask = np.abs(expected) > 1.0e-300
    ratio[mask] = reported[mask] / expected[mask]

    expected_norm = float(np.linalg.norm(expected))
    reported_norm = float(np.linalg.norm(reported))
    norm_ratio = reported_norm / expected_norm if expected_norm > 0.0 else np.nan

    return ShiftDiscrepancyProbe(
        expected_shift_per_m=expected,
        reported_shift_per_m=reported,
        delta_per_m=delta,
        ratio=ratio,
        expected_norm_per_m=expected_norm,
        reported_norm_per_m=reported_norm,
        norm_ratio=float(norm_ratio),
    )


@dataclass(frozen=True, slots=True)
class ShiftDiscrepancyProbe:
    expected_shift_per_m: np.ndarray
    reported_shift_per_m: np.ndarray
    delta_per_m: np.ndarray
    ratio: np.ndarray
    expected_norm_per_m: float
    reported_norm_per_m: float
    norm_ratio: float
    effective_tau_for_reported_norm_s: float
    effective_electric_field_for_reported_norm_V_per_m: float


@dataclass(frozen=True, slots=True)
class ShiftUnitHypothesis:
    name: str
    shift_per_m: np.ndarray
    norm_per_m: float
    norm_ratio_to_reported: float
    direction_comment: str


def shift_discrepancy_probe() -> ShiftDiscrepancyProbe:
    """Compare Vincent's printed shifted k-point with delta k = -e E tau / hbar."""

    reference = vincent_reference()
    expected = electric_field_k_shift_per_m(
        reference.electric_field_V_per_m,
        reference.relaxation_time_s,
    )
    reported = VINCENT_REPORTED_SHIFTED_K_PER_M.copy()
    delta = reported - expected

    ratio = np.full_like(reported, np.nan, dtype=float)
    mask = np.abs(expected) > 1.0e-300
    ratio[mask] = reported[mask] / expected[mask]

    expected_norm = float(np.linalg.norm(expected))
    reported_norm = float(np.linalg.norm(reported))
    norm_ratio = reported_norm / expected_norm if expected_norm > 0.0 else np.nan

    e_field_norm = float(np.linalg.norm(reference.electric_field_V_per_m))
    effective_tau = (
        reported_norm * HBAR_J_S / (E_CHARGE_C * e_field_norm)
        if e_field_norm > 0.0
        else np.nan
    )
    effective_e_field = (
        reported_norm * HBAR_J_S / (E_CHARGE_C * reference.relaxation_time_s)
        if reference.relaxation_time_s > 0.0
        else np.nan
    )

    return ShiftDiscrepancyProbe(
        expected_shift_per_m=expected,
        reported_shift_per_m=reported,
        delta_per_m=delta,
        ratio=ratio,
        expected_norm_per_m=expected_norm,
        reported_norm_per_m=reported_norm,
        norm_ratio=float(norm_ratio),
        effective_tau_for_reported_norm_s=float(effective_tau),
        effective_electric_field_for_reported_norm_V_per_m=float(effective_e_field),
    )


def shift_unit_hypotheses() -> tuple[ShiftUnitHypothesis, ...]:
    """Try common e / hbar / h / unit-conversion mistakes for the shifted k-point."""

    reference = vincent_reference()
    reported_norm = float(np.linalg.norm(VINCENT_REPORTED_SHIFTED_K_PER_M))
    E = reference.electric_field_V_per_m
    tau = reference.relaxation_time_s
    h_J_s = 2.0 * np.pi * HBAR_J_S

    candidates: list[tuple[str, np.ndarray]] = [
        ("correct: -e E tau / hbar", -(E_CHARGE_C * E * tau) / HBAR_J_S),
        ("use h instead of hbar: -e E tau / h", -(E_CHARGE_C * E * tau) / h_J_s),
        ("missing e: -E tau / hbar", -(E * tau) / HBAR_J_S),
        ("extra e: -e^2 E tau / hbar", -((E_CHARGE_C**2) * E * tau) / HBAR_J_S),
        ("wrong sign: +e E tau / hbar", (E_CHARGE_C * E * tau) / HBAR_J_S),
        ("treat E as V/bohr instead of V/m", -(E_CHARGE_C * (E / BOHR_TO_M) * tau) / HBAR_J_S),
        ("treat output as bohr^-1 then convert again", -(E_CHARGE_C * E * tau) / HBAR_J_S / BOHR_TO_M),
        ("forget final m^-1 conversion after bohr^-1", -(E_CHARGE_C * E * tau) / HBAR_J_S * BOHR_TO_M),
    ]

    out: list[ShiftUnitHypothesis] = []

    for name, shift in candidates:
        norm = float(np.linalg.norm(shift))
        ratio = norm / reported_norm if reported_norm > 0.0 else np.nan

        if abs(shift[1]) < 1.0e-300 and abs(VINCENT_REPORTED_SHIFTED_K_PER_M[1]) > 0.0:
            comment = "wrong direction: candidate is x-only, Vincent printout is y-only"
        elif norm == 0.0:
            comment = "zero shift"
        else:
            comment = "direction potentially comparable"

        out.append(
            ShiftUnitHypothesis(
                name=name,
                shift_per_m=shift,
                norm_per_m=norm,
                norm_ratio_to_reported=float(ratio),
                direction_comment=comment,
            )
        )

    return tuple(out)


@dataclass(frozen=True, slots=True)
class ShiftAxisSwapProbe:
    expected_x_per_m: float
    expected_y_per_m: float
    reported_x_per_m: float
    reported_y_per_m: float
    reported_y_over_expected_x: float
    reported_x_over_expected_y: float
    swapped_axis_error_per_m: float


def shift_axis_swap_probe() -> ShiftAxisSwapProbe:
    """Check whether Vincent's printed y-shift resembles the expected x-shift."""

    shift = shift_discrepancy_probe()
    expected = shift.expected_shift_per_m
    reported = shift.reported_shift_per_m

    reported_y_over_expected_x = (
        reported[1] / expected[0] if abs(expected[0]) > 1.0e-300 else np.nan
    )
    reported_x_over_expected_y = (
        reported[0] / expected[1] if abs(expected[1]) > 1.0e-300 else np.nan
    )

    # If x/y were simply swapped, reported_y should equal expected_x.
    swapped_axis_error = reported[1] - expected[0]

    return ShiftAxisSwapProbe(
        expected_x_per_m=float(expected[0]),
        expected_y_per_m=float(expected[1]),
        reported_x_per_m=float(reported[0]),
        reported_y_per_m=float(reported[1]),
        reported_y_over_expected_x=float(reported_y_over_expected_x),
        reported_x_over_expected_y=float(reported_x_over_expected_y),
        swapped_axis_error_per_m=float(swapped_axis_error),
    )


def swapped_axis_velocity_hypothesis_errors(
    epsilon_hartree: np.ndarray,
    ai_bohr: np.ndarray,
) -> dict[str, float]:
    target_k, target_v = vincent_sample_velocity_targets()

    normal = velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, target_k)

    swapped_k = target_k[:, ::-1]
    at_swapped_k = velocity_at_cartesian_k_points(epsilon_hartree, ai_bohr, swapped_k)

    candidates = {
        "normal": normal,
        "swap_output": normal[:, ::-1],
        "swap_k_input": at_swapped_k,
        "swap_k_input_and_output": at_swapped_k[:, ::-1],
        "-normal": -normal,
        "-swap_output": -normal[:, ::-1],
        "-swap_k_input": -at_swapped_k,
        "-swap_k_input_and_output": -at_swapped_k[:, ::-1],
    }

    return {
        name: float(np.sqrt(np.mean((value - target_v) ** 2)))
        for name, value in candidates.items()
    }


def swapped_field_shift() -> np.ndarray:
    reference = vincent_reference()
    swapped_e = reference.electric_field_V_per_m[::-1]
    return electric_field_k_shift_per_m(swapped_e, reference.relaxation_time_s)


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConductivityResult:
    chemical_potential_J: float
    temperature_K: float
    relaxation_time_s: float
    k_cell_area_per_m2: float
    prefactor_S_m2_per_J: float
    fermi_weight: np.ndarray
    velocity_m_per_s: np.ndarray
    raw_velocity_weight_tensor: np.ndarray
    weighted_velocity_tensor: np.ndarray
    conductivity_tensor_S: np.ndarray


def fermi_factor(epsilon_J: np.ndarray, chemical_potential_J: float, temperature_K: float) -> np.ndarray:
    beta_arg = (epsilon_J - chemical_potential_J) / (KB_J_K * temperature_K)

    # Stable logistic. Large positive => f ~ 0, large negative => f ~ 1.
    out = np.empty_like(beta_arg, dtype=np.float64)
    positive = beta_arg >= 0.0

    exp_neg = np.exp(-beta_arg[positive])
    out[positive] = exp_neg / (1.0 + exp_neg)

    exp_pos = np.exp(beta_arg[~positive])
    out[~positive] = 1.0 / (1.0 + exp_pos)

    return out


def fermi_window(epsilon_J: np.ndarray, chemical_potential_J: float, temperature_K: float) -> np.ndarray:
    f = fermi_factor(epsilon_J, chemical_potential_J, temperature_K)
    return f * (1.0 - f)


def reciprocal_cell_area_per_m2(primitive_lattice_vectors_bohr: np.ndarray, shape: tuple[int, int]) -> float:
    reciprocal_bohr = reciprocal_lattice_vectors_from_primitives(primitive_lattice_vectors_bohr)
    reciprocal_per_m = reciprocal_bohr / BOHR_TO_M
    bz_area_per_m2 = abs(float(np.linalg.det(reciprocal_per_m)))
    return bz_area_per_m2 / float(shape[0] * shape[1])


def conductivity_from_velocity_grid(
    epsilon_Ha: np.ndarray,
    velocity_m_per_s: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    spin_degeneracy: float = 1.0,
) -> ConductivityResult:
    epsilon_J = np.asarray(epsilon_Ha, dtype=np.float64) * HARTREE_TO_J
    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)

    if velocity.shape != epsilon_J.shape + (2,):
        raise ValueError(
            f"velocity shape {velocity.shape} does not match epsilon shape {epsilon_J.shape} + (2,)"
        )

    weight = fermi_window(epsilon_J, chemical_potential_J, temperature_K)

    raw = np.einsum("ija,ijb,ij->ab", velocity, velocity, weight)
    k_cell_area = reciprocal_cell_area_per_m2(primitive_lattice_vectors_bohr, epsilon_J.shape)
    weighted = raw * k_cell_area

    prefactor = spin_degeneracy * ELECTRON_CHARGE_C ** 2 * relaxation_time_s / (
        (2.0 * np.pi) ** 2 * KB_J_K * temperature_K
    )
    sigma = prefactor * weighted

    return ConductivityResult(
        chemical_potential_J=float(chemical_potential_J),
        temperature_K=float(temperature_K),
        relaxation_time_s=float(relaxation_time_s),
        k_cell_area_per_m2=float(k_cell_area),
        prefactor_S_m2_per_J=float(prefactor),
        fermi_weight=weight,
        velocity_m_per_s=velocity,
        raw_velocity_weight_tensor=raw,
        weighted_velocity_tensor=weighted,
        conductivity_tensor_S=sigma,
    )


def conductivity_from_epsilon_grid(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    spin_degeneracy: float = 1.0,
) -> ConductivityResult:
    vx, vy = central_cartesian_velocity_grid(epsilon_Ha, primitive_lattice_vectors_bohr)
    velocity = np.stack((vx, vy), axis=-1)

    return conductivity_from_velocity_grid(
        epsilon_Ha,
        velocity,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
        spin_degeneracy=spin_degeneracy,
    )



def velocity_two_pi_hypothesis_errors(epsilon_Ha: np.ndarray, primitive_lattice_vectors_bohr: np.ndarray) -> dict[str, float]:
    """Compare Vincent sample velocities against simple 2π/axis velocity conventions."""

    probe = velocity_systematic_error_probe(epsilon_Ha, primitive_lattice_vectors_bohr)
    target = probe.target_v_m_per_s
    base = probe.local_v_m_per_s

    two_pi = 2.0 * np.pi

    candidates: dict[str, np.ndarray] = {}

    base_variants = {
        "normal": base,
        "swap_output": base[:, ::-1],
        "-normal": -base,
        "-swap_output": -base[:, ::-1],
    }

    scales = {
        "x1": 1.0,
        "x2pi": two_pi,
        "/2pi": 1.0 / two_pi,
        "x(2pi)^2": two_pi ** 2,
        "/(2pi)^2": 1.0 / (two_pi ** 2),
    }

    for base_name, values in base_variants.items():
        for scale_name, scale in scales.items():
            candidates[f"{base_name}:{scale_name}"] = values * scale

    return {
        name: float(np.sqrt(np.mean((values - target) ** 2)))
        for name, values in candidates.items()
    }



def tensor_summary_metrics(tensor: np.ndarray) -> dict[str, float]:
    """Return compact symmetry/shape metrics for a 2x2 conductivity-like tensor."""

    trace = float(np.trace(tensor))
    scale = abs(trace) if trace != 0.0 else 1.0

    return {
        "trace": trace,
        "xx_over_yy": float(tensor[0, 0] / tensor[1, 1]) if tensor[1, 1] != 0.0 else np.nan,
        "anisotropy_abs_over_trace": float(abs(tensor[0, 0] - tensor[1, 1]) / scale),
        "offdiag_abs_over_trace": float(max(abs(tensor[0, 1]), abs(tensor[1, 0])) / scale),
        "antisym_abs_over_trace": float(abs(tensor[0, 1] - tensor[1, 0]) / scale),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(0.5 * (tensor + tensor.T)))),
    }


def conductivity_invariant_checks(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
) -> dict[str, float]:
    """Return numerical invariant checks for the conductivity assembly."""

    base = conductivity_from_epsilon_grid(
        epsilon_Ha,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
    )

    tau2 = conductivity_from_epsilon_grid(
        epsilon_Ha,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=2.0 * relaxation_time_s,
    )

    shifted_epsilon = epsilon_Ha + 0.12345
    shifted_mu = chemical_potential_J + 0.12345 * HARTREE_TO_J
    shifted = conductivity_from_epsilon_grid(
        shifted_epsilon,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=shifted_mu,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
    )

    velocity_scaled = conductivity_from_velocity_grid(
        epsilon_Ha,
        3.0 * base.velocity_m_per_s,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
    )

    norm = float(np.linalg.norm(base.conductivity_tensor_S))
    safe_norm = norm if norm != 0.0 else 1.0

    tau_error = float(np.linalg.norm(tau2.conductivity_tensor_S - 2.0 * base.conductivity_tensor_S) / safe_norm)
    energy_shift_error = float(np.linalg.norm(shifted.conductivity_tensor_S - base.conductivity_tensor_S) / safe_norm)
    velocity_square_error = float(np.linalg.norm(velocity_scaled.conductivity_tensor_S - 9.0 * base.conductivity_tensor_S) / safe_norm)

    summary = tensor_summary_metrics(base.conductivity_tensor_S)

    return {
        "tau_linearity_relative_error": tau_error,
        "energy_shift_relative_error": energy_shift_error,
        "velocity_square_relative_error": velocity_square_error,
        "min_eigenvalue": summary["min_eigenvalue"],
        "anisotropy_abs_over_trace": summary["anisotropy_abs_over_trace"],
        "offdiag_abs_over_trace": summary["offdiag_abs_over_trace"],
        "antisym_abs_over_trace": summary["antisym_abs_over_trace"],
    }


def conductivity_grid_subsample_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    steps: tuple[int, ...] = (1, 2, 4),
) -> list[dict[str, float]]:
    """Compute conductivity on simple strided grids to check measure stability."""

    rows: list[dict[str, float]] = []
    base_trace: float | None = None

    for step in steps:
        sub = epsilon_Ha[::step, ::step]
        result = conductivity_from_epsilon_grid(
            sub,
            primitive_lattice_vectors_bohr,
            chemical_potential_J=chemical_potential_J,
            temperature_K=temperature_K,
            relaxation_time_s=relaxation_time_s,
        )
        trace = float(np.trace(result.conductivity_tensor_S))

        if base_trace is None:
            base_trace = trace

        summary = tensor_summary_metrics(result.conductivity_tensor_S)

        rows.append({
            "step": float(step),
            "n0": float(sub.shape[0]),
            "n1": float(sub.shape[1]),
            "trace": trace,
            "trace_ratio_to_full": trace / base_trace if base_trace else np.nan,
            "xx": float(result.conductivity_tensor_S[0, 0]),
            "yy": float(result.conductivity_tensor_S[1, 1]),
            "anisotropy_abs_over_trace": summary["anisotropy_abs_over_trace"],
        })

    return rows


def conductivity_temperature_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    relaxation_time_s: float,
    temperatures_K: tuple[float, ...] = (100.0, 300.0, 600.0, 1000.0),
) -> list[dict[str, float]]:
    """Probe smooth temperature response of Fermi window and conductivity trace."""

    rows: list[dict[str, float]] = []

    for temperature_K in temperatures_K:
        result = conductivity_from_epsilon_grid(
            epsilon_Ha,
            primitive_lattice_vectors_bohr,
            chemical_potential_J=chemical_potential_J,
            temperature_K=temperature_K,
            relaxation_time_s=relaxation_time_s,
        )

        rows.append({
            "temperature_K": float(temperature_K),
            "max_fermi_weight": float(np.max(result.fermi_weight)),
            "mean_fermi_weight": float(np.mean(result.fermi_weight)),
            "trace": float(np.trace(result.conductivity_tensor_S)),
            "xx": float(result.conductivity_tensor_S[0, 0]),
            "yy": float(result.conductivity_tensor_S[1, 1]),
        })

    return rows


def conductivity_contribution_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25),
) -> list[dict[str, float]]:
    """Measure how concentrated the conductivity trace contribution is near the Fermi shell."""

    result = conductivity_from_epsilon_grid(
        epsilon_Ha,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
    )

    velocity = result.velocity_m_per_s
    contribution = (
        (velocity[..., 0] ** 2 + velocity[..., 1] ** 2)
        * result.fermi_weight
    )
    flat = np.sort(np.ravel(contribution))[::-1]
    total = float(np.sum(flat))

    rows: list[dict[str, float]] = []
    n = len(flat)

    for fraction in fractions:
        count = max(1, int(np.ceil(fraction * n)))
        partial = float(np.sum(flat[:count]))
        rows.append({
            "top_fraction": float(fraction),
            "count": float(count),
            "trace_contribution_fraction": partial / total if total != 0.0 else np.nan,
        })

    for threshold in (1.0e-3, 1.0e-2, 1.0e-1):
        mask = result.fermi_weight >= threshold * float(np.max(result.fermi_weight))
        masked = float(np.sum(contribution[mask]))
        rows.append({
            "top_fraction": -float(threshold),
            "count": float(np.count_nonzero(mask)),
            "trace_contribution_fraction": masked / total if total != 0.0 else np.nan,
        })

    return rows


def conductivity_derivative_sensitivity_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    *,
    chemical_potential_J: float,
    temperature_K: float,
    relaxation_time_s: float,
) -> list[dict[str, float]]:
    """Compare conductivity sensitivity to simple velocity scaling/stencil-like perturbations.

    This is not a replacement for a full alternative stencil implementation. It is a compact
    diagnostic for whether the final tensor is dominated by a global velocity scale.
    """

    base = conductivity_from_epsilon_grid(
        epsilon_Ha,
        primitive_lattice_vectors_bohr,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=relaxation_time_s,
    )

    rows: list[dict[str, float]] = []
    base_trace = float(np.trace(base.conductivity_tensor_S))

    velocity_variants = (
        ("central local derivative", base.velocity_m_per_s),
        ("velocity * 0.99", 0.99 * base.velocity_m_per_s),
        ("velocity * 1.01", 1.01 * base.velocity_m_per_s),
        ("velocity * 2π", (2.0 * np.pi) * base.velocity_m_per_s),
        ("velocity / 2π", base.velocity_m_per_s / (2.0 * np.pi)),
    )

    for name, velocity in velocity_variants:
        result = conductivity_from_velocity_grid(
            epsilon_Ha,
            velocity,
            primitive_lattice_vectors_bohr,
            chemical_potential_J=chemical_potential_J,
            temperature_K=temperature_K,
            relaxation_time_s=relaxation_time_s,
        )
        trace = float(np.trace(result.conductivity_tensor_S))
        rows.append({
            "name": name,
            "trace": trace,
            "trace_ratio_to_base": trace / base_trace if base_trace != 0.0 else np.nan,
            "xx": float(result.conductivity_tensor_S[0, 0]),
            "yy": float(result.conductivity_tensor_S[1, 1]),
        })

    return rows



def analytic_sinusoidal_conductivity_probe() -> dict[str, object]:
    """End-to-end analytic conductivity probe for the rendered diagnostic.

    Uses a periodic separable band on a square lattice:

        epsilon = mu + Ax cos(theta_x) + Ay cos(theta_y)

    The expected derivative is the exact central-periodic finite-difference
    derivative of the sampled cosine, not the continuum derivative. This tests
    the algorithm actually used by the calculation.
    """

    ai = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n0 = 32
    n1 = 40
    i = np.arange(n0, dtype=np.float64)[:, None]
    j = np.arange(n1, dtype=np.float64)[None, :]

    theta_x = 2.0 * np.pi * i / n0
    theta_y = 2.0 * np.pi * j / n1

    mu_Ha = -0.20
    amp_x_Ha = 2.0e-4
    amp_y_Ha = 1.4e-4

    epsilon_Ha = (
        mu_Ha
        + amp_x_Ha * np.cos(theta_x)
        + amp_y_Ha * np.cos(theta_y)
    )

    chemical_potential_J = mu_Ha * HARTREE_TO_J
    temperature_K = 300.0
    tau_s = 2.5e-14

    dkx_per_m = (2.0 * np.pi / BOHR_TO_M) / n0
    dky_per_m = (2.0 * np.pi / BOHR_TO_M) / n1

    vx_expected = (
        amp_x_Ha
        * HARTREE_TO_J
        * (-np.sin(theta_x) * np.sin(2.0 * np.pi / n0) / dkx_per_m)
        / HBAR_J_S
    )
    vy_expected = (
        amp_y_Ha
        * HARTREE_TO_J
        * (-np.sin(theta_y) * np.sin(2.0 * np.pi / n1) / dky_per_m)
        / HBAR_J_S
    )

    vx_expected = np.broadcast_to(vx_expected, epsilon_Ha.shape)
    vy_expected = np.broadcast_to(vy_expected, epsilon_Ha.shape)
    velocity_expected = np.stack((vx_expected, vy_expected), axis=-1)

    vx, vy = central_cartesian_velocity_grid(epsilon_Ha, ai)
    velocity_actual = np.stack((vx, vy), axis=-1)

    result = conductivity_from_epsilon_grid(
        epsilon_Ha,
        ai,
        chemical_potential_J=chemical_potential_J,
        temperature_K=temperature_K,
        relaxation_time_s=tau_s,
    )

    weight = fermi_window(
        epsilon_Ha * HARTREE_TO_J,
        chemical_potential_J,
        temperature_K,
    )
    raw_expected = np.einsum("ija,ijb,ij->ab", velocity_expected, velocity_expected, weight)
    k_cell_area = reciprocal_cell_area_per_m2(ai, epsilon_Ha.shape)
    prefactor = ELECTRON_CHARGE_C ** 2 * tau_s / ((2.0 * np.pi) ** 2 * KB_J_K * temperature_K)
    sigma_expected = prefactor * k_cell_area * raw_expected

    sigma_actual = result.conductivity_tensor_S
    sigma_delta = sigma_actual - sigma_expected

    sigma_norm = float(np.linalg.norm(sigma_expected))
    velocity_norm = float(np.linalg.norm(velocity_expected))

    return {
        "band": "epsilon = mu + Ax cos(theta_x) + Ay cos(theta_y)",
        "shape": epsilon_Ha.shape,
        "mu_Ha": float(mu_Ha),
        "amp_x_Ha": float(amp_x_Ha),
        "amp_y_Ha": float(amp_y_Ha),
        "temperature_K": float(temperature_K),
        "tau_s": float(tau_s),
        "max_abs_vx_error": float(np.max(np.abs(vx - vx_expected))),
        "max_abs_vy_error": float(np.max(np.abs(vy - vy_expected))),
        "relative_velocity_error": float(np.linalg.norm(velocity_actual - velocity_expected) / velocity_norm),
        "sigma_expected": sigma_expected,
        "sigma_actual": sigma_actual,
        "sigma_delta": sigma_delta,
        "relative_sigma_error": float(np.linalg.norm(sigma_delta) / sigma_norm),
        "expected_offdiag_abs": float(max(abs(sigma_expected[0, 1]), abs(sigma_expected[1, 0]))),
        "actual_offdiag_abs": float(max(abs(sigma_actual[0, 1]), abs(sigma_actual[1, 0]))),
    }



def reciprocal_basis_si_from_primitive_bohr(primitive_lattice_vectors_bohr: np.ndarray) -> np.ndarray:
    """Return reciprocal row vectors in SI units, m^-1.

    Input primitive vectors are row vectors in bohr. The returned reciprocal
    vectors satisfy a_i . b_j = 2π δ_ij after converting bohr to metres.
    """

    a_bohr = np.asarray(primitive_lattice_vectors_bohr, dtype=np.float64)

    if a_bohr.shape != (2, 2):
        raise ValueError(f"expected primitive lattice shape (2, 2), got {a_bohr.shape}")

    b_bohr_inverse = 2.0 * np.pi * np.linalg.inv(a_bohr).T

    return b_bohr_inverse / BOHR_TO_M


def vincent_k_grid_si(
    primitive_lattice_vectors_bohr: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return Vincent's k-grid in SI units.

    Vincent's construction is:

        k[i,j] = i/N * b1 + j/N * b2

    for a square N x N grid. This function also supports rectangular shapes by
    using each axis length separately.
    """

    n0, n1 = shape
    b = reciprocal_basis_si_from_primitive_bohr(primitive_lattice_vectors_bohr)

    i = np.arange(n0, dtype=np.float64)
    j = np.arange(n1, dtype=np.float64)
    I, J = np.meshgrid(i, j, indexing="ij")

    return I[:, :, None] / n0 * b[0] + J[:, :, None] / n1 * b[1]


def vincent_delaunay_velocity_grid(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
) -> np.ndarray:
    """Reproduce Vincent's Delaunay plane-fit velocity interpolation.

    The energy grid is converted from Hartree to Joules. The k-grid is built in
    SI units. For each grid point, the containing Delaunay simplex is found and a
    plane epsilon = grad epsilon . k + const is fitted through the three simplex
    vertices. The velocity is grad epsilon / hbar.

    Points outside the triangulation receive zero velocity, matching Vincent's
    supplied implementation.
    """

    epsilon_Ha = np.asarray(epsilon_Ha, dtype=np.float64)

    if epsilon_Ha.ndim != 2:
        raise ValueError(f"expected epsilon grid with ndim=2, got shape {epsilon_Ha.shape}")

    kpoints = vincent_k_grid_si(primitive_lattice_vectors_bohr, epsilon_Ha.shape)
    kpoints_flat = kpoints.reshape(-1, 2)
    epsilon_flat_J = epsilon_Ha.reshape(-1) * HARTREE_TO_J

    tri = Delaunay(kpoints_flat)

    velocities = np.zeros((kpoints_flat.shape[0], 2), dtype=np.float64)

    for idx, kq in enumerate(kpoints_flat):
        simplex = int(tri.find_simplex(kq))

        if simplex == -1:
            continue

        vertices = tri.simplices[simplex]
        k_vertices = kpoints_flat[vertices]
        eps_vertices = epsilon_flat_J[vertices]

        A = np.c_[k_vertices, np.ones(3)]
        coeff = np.linalg.lstsq(A, eps_vertices, rcond=None)[0]
        grad_eps = coeff[:2]

        velocities[idx] = grad_eps / HBAR_J_S

    return velocities.reshape(epsilon_Ha.shape + (2,))


def vincent_delaunay_velocity_at_points(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
    query_k_si: np.ndarray,
) -> np.ndarray:
    """Evaluate Vincent-style Delaunay plane-fit velocity at query k-points."""

    epsilon_Ha = np.asarray(epsilon_Ha, dtype=np.float64)
    query_k_si = np.asarray(query_k_si, dtype=np.float64)

    if epsilon_Ha.ndim != 2:
        raise ValueError(f"expected epsilon grid with ndim=2, got shape {epsilon_Ha.shape}")

    if query_k_si.ndim != 2 or query_k_si.shape[1] != 2:
        raise ValueError(f"expected query_k_si shape (M, 2), got {query_k_si.shape}")

    kpoints = vincent_k_grid_si(primitive_lattice_vectors_bohr, epsilon_Ha.shape)
    kpoints_flat = kpoints.reshape(-1, 2)
    epsilon_flat_J = epsilon_Ha.reshape(-1) * HARTREE_TO_J

    tri = Delaunay(kpoints_flat)
    velocities = np.zeros((query_k_si.shape[0], 2), dtype=np.float64)

    for idx, kq in enumerate(query_k_si):
        simplex = int(tri.find_simplex(kq))

        if simplex == -1:
            continue

        vertices = tri.simplices[simplex]
        k_vertices = kpoints_flat[vertices]
        eps_vertices = epsilon_flat_J[vertices]

        A = np.c_[k_vertices, np.ones(3)]
        coeff = np.linalg.lstsq(A, eps_vertices, rcond=None)[0]
        grad_eps = coeff[:2]

        velocities[idx] = grad_eps / HBAR_J_S

    return velocities


def vincent_delaunay_velocity_sample_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compare Vincent printed velocity samples with Vincent-style interpolation."""

    target_k, target = vincent_sample_velocity_targets()

    local = vincent_delaunay_velocity_at_points(
        epsilon_Ha,
        primitive_lattice_vectors_bohr,
        target_k,
    )

    delta = local - target
    percent = np.where(target != 0.0, 100.0 * delta / target, np.nan)

    return {
        "target": target,
        "local": local,
        "delta": delta,
        "percent": percent,
        "rms_error": float(np.sqrt(np.mean(delta**2))),
    }





def _simplex_plane_velocity(
    kpoints_flat: np.ndarray,
    epsilon_flat_J: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Return grad epsilon / hbar for one Delaunay simplex."""

    k_vertices = kpoints_flat[vertices]
    eps_vertices = epsilon_flat_J[vertices]

    A = np.c_[k_vertices, np.ones(3)]
    coeff = np.linalg.lstsq(A, eps_vertices, rcond=None)[0]

    return coeff[:2] / HBAR_J_S


def vincent_delaunay_adjacent_simplex_velocity_probe(
    epsilon_Ha: np.ndarray,
    primitive_lattice_vectors_bohr: np.ndarray,
) -> list[dict[str, object]]:
    """Inspect all Delaunay simplex gradients adjacent to Vincent sample k-points.

    Vincent evaluates velocity at points that are exactly grid vertices. A
    piecewise-linear Delaunay interpolant has non-unique gradient at such
    vertices, because several triangles meet there. This probe checks whether
    Vincent's printed velocity is reproduced by any triangle adjacent to the
    sample point.
    """

    epsilon_Ha = np.asarray(epsilon_Ha, dtype=np.float64)

    kpoints = vincent_k_grid_si(primitive_lattice_vectors_bohr, epsilon_Ha.shape)
    kpoints_flat = kpoints.reshape(-1, 2)
    epsilon_flat_J = epsilon_Ha.reshape(-1) * HARTREE_TO_J

    tri = Delaunay(kpoints_flat)

    target_probe = velocity_systematic_error_probe(epsilon_Ha, primitive_lattice_vectors_bohr)
    targets = target_probe.target_v_m_per_s

    rows: list[dict[str, object]] = []

    for sample_index, target in enumerate(targets):
        flat_index = sample_index
        kq = kpoints_flat[flat_index]

        simplex = int(tri.find_simplex(kq))
        find_simplex_velocity = (
            np.zeros(2, dtype=np.float64)
            if simplex == -1
            else _simplex_plane_velocity(kpoints_flat, epsilon_flat_J, tri.simplices[simplex])
        )

        adjacent_simplex_ids = np.flatnonzero(np.any(tri.simplices == flat_index, axis=1))

        best_id = -1
        best_velocity = np.zeros(2, dtype=np.float64)
        best_error = np.inf

        for simplex_id in adjacent_simplex_ids:
            velocity = _simplex_plane_velocity(
                kpoints_flat,
                epsilon_flat_J,
                tri.simplices[int(simplex_id)],
            )
            error = float(np.linalg.norm(velocity - target))

            if error < best_error:
                best_error = error
                best_id = int(simplex_id)
                best_velocity = velocity

        rows.append(
            {
                "sample": sample_index,
                "find_simplex": simplex,
                "adjacent_count": int(len(adjacent_simplex_ids)),
                "target": target,
                "find_simplex_velocity": find_simplex_velocity,
                "find_simplex_error": float(np.linalg.norm(find_simplex_velocity - target)),
                "best_adjacent_simplex": best_id,
                "best_adjacent_velocity": best_velocity,
                "best_adjacent_error": best_error,
            }
        )

    return rows
