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
        lon = np.zeros((nyp, nxp))
        lat = np.zeros((nyp, nxp))

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


# Not testing get_arakawa_c_points, coords, & create_regridder
def test_smoke_untested_funcs(generate_curvilinear_grid):
    hgrid = generate_curvilinear_grid([100, 110], [10, 20], 10)
    assert rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert rgd.coords(hgrid, "north", "segment_002")


def test_fill_missing_data():
    return


def test_add_or_update_time_dim():
    return


def test_generate_dz():
    return


def test_add_secondary_dimension():
    return


def test_add_vertical_coordinate_encoding():
    return


def test_generate_layer_thickness():
    return


def test_generate_encoding():
    return
