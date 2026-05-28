from __future__ import annotations

import numpy as np

from dft_local.core.kernels import GdKernelArrays
from dft_local.core.numerics import AU, Units, eVag
from dft_local.transport.boltzmann.core import (
    K_B_HARTREE_PER_K,
    BoltzmannConductivity,
    fermi_window,
)


def test_dft_local_core_exports_fermi_window() -> None:
    temperature = 1.0 / K_B_HARTREE_PER_K
    window = fermi_window(
        np.array([0.0]),
        mu=0.0,
        temperature=temperature,
        units=AU,
    )

    assert np.allclose(window, [0.25])


def test_dft_local_core_exports_boltzmann_conductivity_class() -> None:
    assert BoltzmannConductivity.__name__ == "BoltzmannConductivity"
