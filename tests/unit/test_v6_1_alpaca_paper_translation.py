from __future__ import annotations

import numpy as np

from scripts.run_v6_1_alpaca_paper_translation import _apply_uniform_gross_cap


def test_uniform_gross_cap_preserves_directions_and_relative_weights() -> None:
    raw = np.array([0.8, -0.6, 0.4, -0.2])
    capped, scalar = _apply_uniform_gross_cap(raw, 1.8)
    assert np.isclose(np.abs(capped).sum(), 1.8)
    assert np.all(np.sign(capped) == np.sign(raw))
    assert np.allclose(capped, scalar * raw)
    assert np.isclose(capped[0] / capped[2], raw[0] / raw[2])


def test_uniform_gross_cap_is_inactive_below_target() -> None:
    raw = np.array([0.5, -0.3, 0.2])
    capped, scalar = _apply_uniform_gross_cap(raw, 1.8)
    assert scalar == 1.0
    assert np.array_equal(capped, raw)
