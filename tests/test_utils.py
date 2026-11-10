import numpy as np
import pytest

from regional_mom6.utils import earth_to_grid, rotate

# ---------------------------------------------------------------------------
# earth_to_grid: the shared IC/OBC velocity-rotation helper
#
# Regression coverage for the sign bug where regional_mom6.py and segment.py
# each called `rotate(u, v, +angle_dx)` (grid-relative -> earth-relative, the
# wrong direction) instead of `rotate(u, v, -angle_dx)` (earth-relative ->
# grid-relative, what MOM6 actually expects for IC/OBC velocity data, since
# MOM6 applies no rotation of its own to that data). See PR_regional_mom6_rotation.md.
# ---------------------------------------------------------------------------


def test_earth_to_grid_is_rotate_by_minus_angle():
    """earth_to_grid(u, v, angle_deg) must be exactly rotate(u, v, -radians(angle_deg))
    -- not rotate(u, v, +radians(angle_deg)), which is the bug this helper fixes."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(5, 7))
    v = rng.normal(size=(5, 7))
    angle_deg = rng.uniform(-180, 180, size=(5, 7))

    got_u, got_v = earth_to_grid(u, v, angle_deg)
    want_u, want_v = rotate(u, v, radian_angle=-np.radians(angle_deg))
    np.testing.assert_allclose(got_u, want_u)
    np.testing.assert_allclose(got_v, want_v)

    wrong_u, wrong_v = rotate(u, v, radian_angle=np.radians(angle_deg))
    assert not np.allclose(got_v, wrong_v)


def test_earth_to_grid_zero_angle_is_identity():
    """On an unrotated (plain lat-lon) grid, angle_dx is 0 everywhere, so
    earth_to_grid must be a no-op -- the bug is invisible on these grids."""
    u = np.array([1.0, -2.0, 0.5])
    v = np.array([0.0, 3.0, -1.5])
    got_u, got_v = earth_to_grid(u, v, np.zeros_like(u))
    np.testing.assert_allclose(got_u, u)
    np.testing.assert_allclose(got_v, v)


def test_earth_to_grid_eastward_flow_on_rotated_toy_grid():
    """Unit-level check with an analytically known answer: a grid whose local
    x-axis is rotated `angle_dx` degrees counter-clockwise from east should see
    a uniform eastward earth-relative flow (u=1, v=0) as
    (cos(angle_dx), -sin(angle_dx)) in its own (x, y) frame -- i.e. rotated
    *clockwise* by angle_dx relative to grid-x, not counter-clockwise."""
    angle_dx = 30.0
    u_grid, v_grid = earth_to_grid(1.0, 0.0, angle_dx)

    assert u_grid == pytest.approx(np.cos(np.radians(angle_dx)))
    assert v_grid == pytest.approx(-np.sin(np.radians(angle_dx)))

    # The old call site used +angle_dx and got the v-component's sign backwards
    # (and, away from angle_dx == 0, the wrong direction altogether).
    buggy_u, buggy_v = rotate(1.0, 0.0, radian_angle=np.radians(angle_dx))
    assert buggy_v == pytest.approx(-v_grid)


def test_earth_to_grid_matches_ic_call_site_on_real_rotated_hgrid(get_rotated_hgrid):
    """Reproduces the exact expression regional_mom6.py's IC path uses
    (`earth_to_grid(regridded_u, regridded_v, angle_deg=hgrid.angle_dx.values)`)
    against a real mom6_forge hgrid, and checks the result against the
    analytic grid-relative components for a uniform eastward earth flow."""
    angle_dx = get_rotated_hgrid.angle_dx.values

    u_earth = np.ones_like(angle_dx)
    v_earth = np.zeros_like(angle_dx)
    u_grid, v_grid = earth_to_grid(u_earth, v_earth, angle_deg=angle_dx)

    expected_u = np.cos(np.radians(angle_dx))
    expected_v = -np.sin(np.radians(angle_dx))
    np.testing.assert_allclose(u_grid, expected_u)
    np.testing.assert_allclose(v_grid, expected_v)

    # angle_dx is around +/-30 deg everywhere on this grid (not 0), so the old
    # +angle_dx call would have been wrong nearly everywhere, not just off by
    # a sign at one point.
    assert np.abs(angle_dx).min() > 20.0


V1 = np.zeros((2, 4, 3))
V2 = np.zeros((2, 4, 3))
true_V1dotV2 = np.zeros((2, 4))

for i in np.arange(2):
    for j in np.arange(4):
        sum = 0
        for k in np.arange(3):
            V1[i, j, k] = -1.0 * np.array(2 * i - 3 * j + 2 * k)
            V2[i, j, k] = 1.0 * np.array(i + j + k + 1)
            sum += -(2 * i - 3 * j + 2 * k) * (i + j + k + 1)
            true_V1dotV2[i, j] = float(sum)
