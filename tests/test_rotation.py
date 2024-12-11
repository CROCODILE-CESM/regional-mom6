import regional_mom6 as rm6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np


@pytest.fixture
def generate_curvilinear_grid(request):
    """
    Params are a tuple longitude, tuple latitude, angle
    """

    def _generate_curvilinear_grid(
        lon_point,
        lat_point,
        angle_base,
        shift_angle_by=0,
        nxp=10,
        nyp=10,
        resolution=0.1,
    ):
        """
        Generate a curvilinear grid dataset with longitude, latitude, and angle arrays.

        Parameters:
        lon_point : float
            Point for the lower left corner of the grid
        lat_point : float
            Point for the lower left corner of the grid
        angle_base : float
            Base angle (in radians) to initialize the grid angles.
        shift_angle_by : float, optional
            Maximum random variation to add or subtract from the base angle (default is 0).
        nxp : int, optional
            Number of points in the longitude direction (default is 10).
        nyp : int, optional
            Number of points in the latitude direction (default is 10).
        resolution : float, optional
            Loose resolution of the grid (default is 0.1).

        Returns:
        xarray.Dataset
            Dataset containing 'x' (longitude), 'y' (latitude), and 'angle' arrays with metadata.
        """
        # Generate logical grid
        lon = np.zeroes((nyp, nxp))
        lat = np.zeroes((nyp, nxp))

        lon[0][0] = lon_point
        lat[0][0] = lat_point

        angle = angle_base + np.random.uniform(
            -shift_angle_by, shift_angle_by, (nyp, nxp)
        )

        # based on the angle, construct the grid from these points.

        return xr.Dataset(
            {
                "x": (("nyp", "nxp"), lon),
                "y": (("nyp", "nxp"), lat),
                "angle": (("nyp", "nxp"), angle),
            }
        )

    return _generate_curvilinear_grid


def test_pseudo_hgrid_generation(generate_curvilinear_grid):
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], 10)
    pseudo_hgrid = rot.create_pseudo_hgrid(hgrid)

    # Check Size
    assert len(pseudo_hgrid.nxp) == (len(hgrid.nxp) + 2)
    assert len(pseudo_hgrid.nyp) == (len(hgrid.nyp) + 2)

    # Check pseudo_hgrid keeps the same values
    assert (pseudo_hgrid.x.values[1:-1, 1:-1] == hgrid.x.values).all()
    assert (pseudo_hgrid.y.values[1:-1, 1:-1] == hgrid.y.values).all()

    # Check extra boundary has realistic values
    diff_check = 1
    assert (
        (
            pseudo_hgrid.x.values[0, 1:-1]
            - (hgrid.x.values[0, :] - (hgrid.x.values[1, :] - hgrid.x.values[0, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.x.values[1:-1, 0]
            - (hgrid.x.values[:, 0] - (hgrid.x.values[:, 1] - hgrid.x.values[:, 0]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.x.values[-1, 1:-1]
            - (hgrid.x.values[-1, :] - (hgrid.x.values[-2, :] - hgrid.x.values[-1, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.x.values[1:-1, -1]
            - (hgrid.x.values[:, -1] - (hgrid.x.values[:, -2] - hgrid.x.values[:, -1]))
        )
        < diff_check
    ).all()

    # Check corners for the same...
    assert (
        pseudo_hgrid.x.values[0, 0]
        - (hgrid.x.values[0, 0] - (hgrid.x.values[1, 1] - hgrid.x.values[0, 0]))
    ) < diff_check
    assert (
        pseudo_hgrid.x.values[-1, 0]
        - (hgrid.x.values[-1, 0] - (hgrid.x.values[-2, 1] - hgrid.x.values[-1, 0]))
    ) < diff_check
    assert (
        pseudo_hgrid.x.values[0, -1]
        - (hgrid.x.values[0, -1] - (hgrid.x.values[1, -2] - hgrid.x.values[0, -1]))
    ) < diff_check
    assert (
        pseudo_hgrid.x.values[-1, -1]
        - (hgrid.x.values[-1, -1] - (hgrid.x.values[-2, -2] - hgrid.x.values[-1, -1]))
    ) < diff_check

    # Same for y
    assert (
        (
            pseudo_hgrid.y.values[0, 1:-1]
            - (hgrid.y.values[0, :] - (hgrid.y.values[1, :] - hgrid.y.values[0, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.y.values[1:-1, 0]
            - (hgrid.y.values[:, 0] - (hgrid.y.values[:, 1] - hgrid.y.values[:, 0]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.y.values[-1, 1:-1]
            - (hgrid.y.values[-1, :] - (hgrid.y.values[-2, :] - hgrid.y.values[-1, :]))
        )
        < diff_check
    ).all()
    assert (
        (
            pseudo_hgrid.y.values[1:-1, -1]
            - (hgrid.y.values[:, -1] - (hgrid.y.values[:, -2] - hgrid.y.values[:, -1]))
        )
        < diff_check
    ).all()

    assert (
        pseudo_hgrid.y.values[0, 0]
        - (hgrid.y.values[0, 0] - (hgrid.y.values[1, 1] - hgrid.y.values[0, 0]))
    ) < diff_check
    assert (
        pseudo_hgrid.y.values[-1, 0]
        - (hgrid.y.values[-1, 0] - (hgrid.y.values[-2, 1] - hgrid.y.values[-1, 0]))
    ) < diff_check
    assert (
        pseudo_hgrid.y.values[0, -1]
        - (hgrid.y.values[0, -1] - (hgrid.y.values[1, -2] - hgrid.y.values[0, -1]))
    ) < diff_check
    assert (
        pseudo_hgrid.y.values[-1, -1]
        - (hgrid.y.values[-1, -1] - (hgrid.y.values[-2, -2] - hgrid.y.values[-1, -1]))
    ) < diff_check

    return


def test_mom6_angle_calculation_method(generate_curvilinear_grid):
    """
    Check no rotation, up tilt, down tilt.
    """

    # Check no rotation
    top_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0]]),
            "y": (("nyp", "nxp"), [[1]]),
        }
    )
    top_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[1]]),
            "y": (("nyp", "nxp"), [[1]]),
        }
    )
    bottom_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0]]),
            "y": (("nyp", "nxp"), [[0]]),
        }
    )
    bottom_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[1]]),
            "y": (("nyp", "nxp"), [[0]]),
        }
    )
    point = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[0.5]]),
            "y": (("nyp", "nxp"), [[0.5]]),
        }
    )

    assert (
        rot.mom6_angle_calculation_method(
            2, top_left, top_right, bottom_left, bottom_right, point
        )
        == 0
    )

    # Up tilt
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], 10, 10)
    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    ds_q = rgd.get_hgrid_arakawa_c_points(hgrid, "q")

    # Reformat into x, y comps
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_t.tlon.data),
            "y": (("nyp", "nxp"), ds_t.tlat.data),
        }
    )
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_q.qlon.data),
            "y": (("nyp", "nxp"), ds_q.qlat.data),
        }
    )
    assert (
        rot.mom6_angle_calculation_method(
            hgrid.x.max() - hgrid.x.min(),
            q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
            q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
            q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
            q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
            t_points,
        )
        < 20
    ).all()

    ## Down tilt
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], -10, 10)

    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    ds_q = rgd.get_hgrid_arakawa_c_points(hgrid, "q")

    # Reformat into x, y comps
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_t.tlon.data),
            "y": (("nyp", "nxp"), ds_t.tlat.data),
        }
    )
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_q.qlon.data),
            "y": (("nyp", "nxp"), ds_q.qlat.data),
        }
    )
    assert (
        rot.mom6_angle_calculation_method(
            hgrid.x.max() - hgrid.x.min(),
            q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
            q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
            q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
            q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
            t_points,
        )
        > -20
    ).all()

    return


def test_initialize_grid_rotation_angle(generate_curvilinear_grid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], 10)
    angle = rot.initialize_grid_rotation_angle(hgrid)

    assert (angle.values == 10).all()
    return


def test_initialize_grid_rotation_angle_using_pseudo_hgrid(generate_curvilinear_grid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], 10)
    angle = rot.initialize_grid_rotation_angle_using_pseudo_hgrid(hgrid)

    assert (angle.values - hgrid.angle < 1).all()
    return
