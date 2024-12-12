import regional_mom6 as rm6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np
import os

# Define the path where the curvilinear hgrid file is expected in the Docker container
DOCKER_FILE_PATH = "/data/small_curvilinear_hgrid.nc"


# Define the local directory where the user might have added the curvilinear hgrid file
LOCAL_FILE_PATH = (
    "/glade/u/home/manishrv/documents/nwa12_0.1/tides_dev/small_curvilinear_hgrid.nc"
)


@pytest.fixture
def get_curvilinear_hgrid():
    # Check if the file exists in the Docker-specific location
    if os.path.exists(DOCKER_FILE_PATH):
        return xr.open_dataset(DOCKER_FILE_PATH)

    # Check if the user has provided the file in a specific local directory
    elif os.path.exists(LOCAL_FILE_PATH):
        return xr.open_dataset(LOCAL_FILE_PATH)

    # If neither location contains the file, raise an error
    else:
        pytest.skip(
            f"Required file 'hgrid.nc' not found in {DOCKER_FILE_PATH} or {LOCAL_FILE_PATH}"
        )


def test_get_curvilinear_hgrid_fixture(get_curvilinear_hgrid):
    # If the fixture fails to find the file, the test will be skipped.
    assert get_curvilinear_hgrid is not None


def test_pseudo_hgrid_generation(get_curvilinear_hgrid):
    hgrid = get_curvilinear_hgrid
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


def test_mom6_angle_calculation_method(get_curvilinear_hgrid):
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

    # Angled
    hgrid = get_curvilinear_hgrid
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
        (
            rot.mom6_angle_calculation_method(
                hgrid.x.max() - hgrid.x.min(),
                q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
                t_points,
            )
            - hgrid["angle_dx"].isel(nyp=ds_t.t_points_y, nxp=ds_t.t_points_x).values
        )
        < 1
    ).all()

    return


def test_initialize_grid_rotation_angle(get_curvilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = get_curvilinear_hgrid
    angle = rot.initialize_grid_rotation_angle(hgrid)
    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert (
        (
            angle.values
            - hgrid["angle_dx"].isel(nyp=ds_t.t_points_y, nxp=ds_t.t_points_x).values
        )
        < 1
    ).all()  # Angle is correct
    assert angle.values.shape == ds_t.tlon.shape  # Shape is correct
    return


def test_initialize_grid_rotation_angle_using_pseudo_hgrid(get_curvilinear_hgrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    hgrid = get_curvilinear_hgrid
    angle = rot.initialize_grid_rotation_angles_using_pseudo_hgrid(hgrid)

    assert (angle.values - hgrid.angle_dx < 1).all()
    assert angle.values.shape == hgrid.x.shape
    return
