from .utils import setup_logger

rotation_logger = setup_logger(__name__)
# An Enum is like a dropdown selection for a menu, it essentially limits the type of input parameters. It comes with additional complexity, which of course is always a challenge.
from enum import Enum
import xarray as xr
import numpy as np
from .regridding import get_hgrid_arakawa_c_points


class RotationMethod(Enum):
    """
    This Enum defines the rotational method to be used in boundary conditions. The main regional mom6 class passes in this enum to regrid_tides and regrid_velocity_tracers to determine the method used.

    KEITH_DOUBLE_REGRIDDING: This method is used to regrid the boundary conditions to the t-points, b/c we can calculate t-point angle the same way as MOM6, rotate the conditions, and regrid again to the q-u-v, or actual, boundary
    FRED_AVERAGE: This method is used with the basis that we can find the angles at the q-u-v points by pretending we have another row/column of the hgrid with the same distances as the t-point to u/v points in the actual grid then use the four poitns to calculate the angle the exact same way MOM6 does.
    GIVEN_ANGLE: This is the original default RM6 method which expects a pre-given angle called angle_dx
    NO_ROTATION: Grids parallel to lat/lon axes, no rotation needed
    """

    KEITH_DOUBLE_REGRIDDING = 1
    FRED_AVERAGE = 2
    GIVEN_ANGLE = 3
    NO_ROTATION = 4


def initialize_grid_rotation_angles_using_pseudo_hgrid(
    hgrid: xr.Dataset,
) -> xr.Dataset:
    """
    Calculate the angle_dx in degrees from the true x (east?) direction counterclockwise) and return as dataarray

    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset
    pseudo_hgrid: xr.Dataset
        The pseudo hgrid dataset
    Returns
    -------
    xr.DataArray
        The t-point angles
    """
    # Get Fred Pseudo grid
    pseudo_hgrid = create_pseudo_hgrid(hgrid)

    return mom6_angle_calculation_method(
        pseudo_hgrid.x.max() - pseudo_hgrid.x.min(),
        pseudo_hgrid.isel(nyp=slice(2, None), nxp=slice(0, -2)),
        pseudo_hgrid.isel(nyp=slice(2, None), nxp=slice(2, None)),
        pseudo_hgrid.isel(nyp=slice(0, -2), nxp=slice(0, -2)),
        pseudo_hgrid.isel(nyp=slice(0, -2), nxp=slice(2, None)),
        hgrid,
    )


def initialize_grid_rotation_angle(hgrid: xr.Dataset) -> xr.DataArray:
    """
    Calculate the angle_dx in degrees from the true x (east?) direction counterclockwise) and return as DataArray
    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset
    Returns
    -------
    xr.DataArray
        The t-point angles
    """
    ds_t = get_hgrid_arakawa_c_points(hgrid, "t")
    ds_q = get_hgrid_arakawa_c_points(hgrid, "q")

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

    return mom6_angle_calculation_method(
        hgrid.x.max() - hgrid.x.min(),
        q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
        t_points,
    )


def modulo_around_point(x, xc, Lx):
    """
    This function calculates the modulo around a point. Return the modulo value of x in an interval [xc-(Lx/2) xc+(Lx/2)]. If Lx<=0, then it returns x without applying modulo arithmetic.
    Parameters
    ----------
    x: float
        Value to which to apply modulo arithmetic
    xc: float
        Center of modulo range
    Lx: float
        Modulo range width
    Returns
    -------
    float
        x shifted by an integer multiple of Lx to be close to xc,
    """
    if Lx <= 0:
        return x
    else:
        return ((x - (xc - 0.5 * Lx)) % Lx) - Lx / 2 + xc


def mom6_angle_calculation_method(
    len_lon,
    top_left: xr.DataArray,
    top_right: xr.DataArray,
    bottom_left: xr.DataArray,
    bottom_right: xr.DataArray,
    point: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the angle of the point using the MOM6 method in initialize_grid_rotation_angle. Built for vectorized calculations
    Parameters
    ----------
    len_lon: float
        The length of the longitude of the regional domain
    top_left, top_right, bottom_left, bottom_right: xr.DataArray
        The four points around the point to calculate the angle from the hgrid requires an x and y component
    point: xr.DataArray
        The point to calculate the angle from the hgrid
    Returns
    -------
    xr.DataArray
        The angle of the point
    """
    rotation_logger.info("Calculating grid rotation angle")
    # Direct Translation
    pi_720deg = (
        np.arctan(1) / 180
    )  # One quarter the conversion factor from degrees to radians

    # Compute lonB for all points
    lonB = np.zeros((2, 2, len(point.nyp), len(point.nxp)))

    # Vectorized computation of lonB
    # Vectorized computation of lonB
    lonB[0][0] = modulo_around_point(bottom_left.x, point.x, len_lon)  # Bottom Left
    lonB[1][0] = modulo_around_point(top_left.x, point.x, len_lon)  # Top Left
    lonB[1][1] = modulo_around_point(top_right.x, point.x, len_lon)  # Top Right
    lonB[0][1] = modulo_around_point(bottom_right.x, point.x, len_lon)  # Bottom Right

    # Compute lon_scale
    lon_scale = np.cos(
        pi_720deg * ((bottom_left.y + bottom_right.y) + (top_right.y + top_left.y))
    )

    # Compute angle
    angle = np.arctan2(
        lon_scale * ((lonB[0, 1] - lonB[1, 0]) + (lonB[1, 1] - lonB[0, 0])),
        (bottom_left.y - top_right.y) + (top_left.y - bottom_right.y),
    )
    # Assign angle to angles_arr
    angles_arr = np.rad2deg(angle) - 90

    # Assign angles_arr to hgrid
    t_angles = xr.DataArray(
        angles_arr,
        dims=["nyp", "nxp"],
        coords={
            "nyp": point.nyp.values,
            "nxp": point.nxp.values,
        },
    )
    return t_angles


def create_pseudo_hgrid(hgrid: xr.Dataset) -> xr.Dataset:
    """
    Adds an additional boundary to the hgrid to allow for the calculation of the angle_dx for the boundary points using the method in MOM6
    """
    pseudo_hgrid_x = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp) + 2), np.nan)
    pseudo_hgrid_y = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp) + 2), np.nan)

    ## Fill Boundaries
    pseudo_hgrid_x[1:-1, 1:-1] = hgrid.x.values
    pseudo_hgrid_x[0, 1:-1] = hgrid.x.values[0, :] - (
        hgrid.x.values[1, :] - hgrid.x.values[0, :]
    )  # Bottom Fill
    pseudo_hgrid_x[-1, 1:-1] = hgrid.x.values[-1, :] + (
        hgrid.x.values[-1, :] - hgrid.x.values[-2, :]
    )  # Top Fill
    pseudo_hgrid_x[1:-1, 0] = hgrid.x.values[:, 0] - (
        hgrid.x.values[:, 1] - hgrid.x.values[:, 0]
    )  # Left Fill
    pseudo_hgrid_x[1:-1, -1] = hgrid.x.values[:, -1] + (
        hgrid.x.values[:, -1] - hgrid.x.values[:, -2]
    )  # Right Fill

    pseudo_hgrid_y[1:-1, 1:-1] = hgrid.y.values
    pseudo_hgrid_y[0, 1:-1] = hgrid.y.values[0, :] - (
        hgrid.y.values[1, :] - hgrid.y.values[0, :]
    )  # Bottom Fill
    pseudo_hgrid_y[-1, 1:-1] = hgrid.y.values[-1, :] + (
        hgrid.y.values[-1, :] - hgrid.y.values[-2, :]
    )  # Top Fill
    pseudo_hgrid_y[1:-1, 0] = hgrid.y.values[:, 0] - (
        hgrid.y.values[:, 1] - hgrid.y.values[:, 0]
    )  # Left Fill
    pseudo_hgrid_y[1:-1, -1] = hgrid.y.values[:, -1] + (
        hgrid.y.values[:, -1] - hgrid.y.values[:, -2]
    )  # Right Fill

    ## Fill Corners
    pseudo_hgrid_x[0, 0] = hgrid.x.values[0, 0] - (
        hgrid.x.values[1, 1] - hgrid.x.values[0, 0]
    )  # Bottom Left
    pseudo_hgrid_x[-1, 0] = hgrid.x.values[-1, 0] - (
        hgrid.x.values[-2, 1] - hgrid.x.values[-1, 0]
    )  # Top Left
    pseudo_hgrid_x[0, -1] = hgrid.x.values[0, -1] - (
        hgrid.x.values[1, -2] - hgrid.x.values[0, -1]
    )  # Bottom Right
    pseudo_hgrid_x[-1, -1] = hgrid.x.values[-1, -1] - (
        hgrid.x.values[-2, -2] - hgrid.x.values[-1, -1]
    )  # Top Right

    pseudo_hgrid_y[0, 0] = hgrid.y.values[0, 0] - (
        hgrid.y.values[1, 1] - hgrid.y.values[0, 0]
    )  # Bottom Left
    pseudo_hgrid_y[-1, 0] = hgrid.y.values[-1, 0] - (
        hgrid.y.values[-2, 1] - hgrid.y.values[-1, 0]
    )  # Top Left
    pseudo_hgrid_y[0, -1] = hgrid.y.values[0, -1] - (
        hgrid.y.values[1, -2] - hgrid.y.values[0, -1]
    )  # Bottom Right
    pseudo_hgrid_y[-1, -1] = hgrid.y.values[-1, -1] - (
        hgrid.y.values[-2, -2] - hgrid.y.values[-1, -1]
    )  # Top Right

    pseudo_hgrid = xr.Dataset(
        {
            "x": (["nyp", "nxp"], pseudo_hgrid_x),
            "y": (["nyp", "nxp"], pseudo_hgrid_y),
        }
    )
    return pseudo_hgrid
