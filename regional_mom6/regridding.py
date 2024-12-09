"""
Helper Functions to take the user though the regridding of boundary conditions and encoding for MOM6. Built for RM6

Steps:
1. Initial Regridding -> Find the boundary of the hgrid, and regrid the forcing variables to that boundary. Call (initial_regridding) and then use the xesmf Regridder with whatever datasets you need.
2. Work on some data issues
    1. For temperature - Make sure it's in Celsius
    2. FILL IN NANS -> this is important for MOM6 (fill_missing_data) -> This diverges between 
3. For tides, we split the tides into an amplitude and a phase...
4. In some cases, here is a great place to rotate the velocities to match a curved grid.... (tidal_velocity), velocity is also a good place to do this.
5. We then add the time coordinate
6. For vars that are not just surface variables, we need to add several depth related variables
    1. Add a dz variable in layer thickness
    2. Some metadata issues later on
7. Now we do up the metadata
8. Rename variables to var_segment_num
9. (IF VERTICAL EXISTS) Rename the vertical coordinate of the variable to nz_segment_num_var
10. (IF VERTICAL EXISTS)  Declare this new vertical coordiante as a increasing series of integers
11. Re-add the "perpendicular" dimension
12. ....Add  layer thickness of dz to the vertical forcings
13. Add to encoding_dict a fill value(_FillValue) and zlib, dtype, for time, lat long, ....and each variable (no type needed though)



"""

import xesmf as xe
import xarray as xr
from pathlib import Path
import dask.array as da
import numpy as np
import netCDF4
from .utils import setup_logger

# An Enum is like a dropdown selection for a menu, it essentially limits the type of input parameters. It comes with additional complexity, which of course is always a challenge.
from enum import Enum


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


regridding_logger = setup_logger(__name__)


def coords(
    hgrid: xr.Dataset, orientation: str, segment_name: str, coords_at_t_points=False
) -> xr.Dataset:
    """
    This function:
    Allows us to call the coords for use in the xesmf.Regridder in the regrid_tides function. self.coords gives us the subset of the hgrid based on the orientation.

    Args:
        hgrid (xr.Dataset): The hgrid dataset
        orientation (str): The orientation of the boundary
        segment_name (str): The name of the segment
        coords_at_t_points (bool, optional): Whether to return the boundary t-points instead of the q/u/v of a general boundary for rotation. Defaults to False.
    Returns:
        xr.Dataset: The correct coordinate space for the orientation

    Code adapted from:
    Author(s): GFDL, James Simkins, Rob Cermak, etc..
    Year: 2022
    Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
    Version: N/A
    Type: Python Functions, Source Code
    Web Address: https://github.com/jsimkins2/nwa25

    """

    dataset_to_get_coords = None

    if coords_at_t_points:
        regridding_logger.info("Creating coordinates of the boundary t-points")

        # Calc T Point Info
        ds = get_hgrid_arakawa_c_points(hgrid, "t")

        tangle_dx = hgrid["angle_dx"][(ds.t_points_y, ds.t_points_x)]
        # Assign to dataset
        dataset_to_get_coords = xr.Dataset(
            {
                "x": ds.tlon,
                "y": ds.tlat,
                "angle_dx": (("nyp", "nxp"), tangle_dx.values),
            },
            coords={"nyp": ds.nyp, "nxp": ds.nxp},
        )
    else:
        regridding_logger.info("Creating coordinates of the boundary q/u/v points")
        # Don't have to do anything because this is the actual boundary. t-points are one-index deep and require managing.
        dataset_to_get_coords = hgrid

    # Rename nxp and nyp to locations
    if orientation == "south":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nyp=0),
                "lat": dataset_to_get_coords["y"].isel(nyp=0),
                "angle": dataset_to_get_coords["angle_dx"].isel(nyp=0),
            }
        )
        rcoord = rcoord.rename_dims({"nxp": f"nx_{segment_name}"})
        rcoord.attrs["perpendicular"] = "ny"
        rcoord.attrs["parallel"] = "nx"
        rcoord.attrs["axis_to_expand"] = (
            2  ## Need to keep track of which axis the 'main' coordinate corresponds to when re-adding the 'secondary' axis
        )
    elif orientation == "north":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nyp=-1),
                "lat": dataset_to_get_coords["y"].isel(nyp=-1),
                "angle": dataset_to_get_coords["angle_dx"].isel(nyp=-1),
            }
        )
        rcoord = rcoord.rename_dims({"nxp": f"nx_{segment_name}"})
        rcoord.attrs["perpendicular"] = "ny"
        rcoord.attrs["parallel"] = "nx"
        rcoord.attrs["axis_to_expand"] = 2
    elif orientation == "west":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nxp=0),
                "lat": dataset_to_get_coords["y"].isel(nxp=0),
                "angle": dataset_to_get_coords["angle_dx"].isel(nxp=0),
            }
        )
        rcoord = rcoord.rename_dims({"nyp": f"ny_{segment_name}"})
        rcoord.attrs["perpendicular"] = "nx"
        rcoord.attrs["parallel"] = "ny"
        rcoord.attrs["axis_to_expand"] = 3
    elif orientation == "east":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nxp=-1),
                "lat": dataset_to_get_coords["y"].isel(nxp=-1),
                "angle": dataset_to_get_coords["angle_dx"].isel(nxp=-1),
            }
        )
        rcoord = rcoord.rename_dims({"nyp": f"ny_{segment_name}"})
        rcoord.attrs["perpendicular"] = "nx"
        rcoord.attrs["parallel"] = "ny"
        rcoord.attrs["axis_to_expand"] = 3

    # Make lat and lon coordinates
    rcoord = rcoord.assign_coords(lat=rcoord["lat"], lon=rcoord["lon"])

    return rcoord


def create_regridder(
    forcing_variables: xr.Dataset,
    output_grid: xr.Dataset,
    outfile: Path = Path(".temp"),
    method: str = "bilinear",
) -> xe.Regridder:
    """
    Basic Regridder for any forcing variables, this just wraps the xesmf regridder for a few defaults
    Parameters
    ----------
    forcing_variables : xr.Dataset
        The dataset of the forcing variables
    output_grid : xr.Dataset
        The dataset of the output grid -> this is the boundary of the hgrid
    outfile : Path, optional
        The path to the output file for weights I believe, by default Path(".temp")
    method : str, optional
        The regridding method, by default "bilinear"
    Returns
    -------
    xe.Regridder
        The regridding object
    """
    regridding_logger.info("Creating Regridder")
    regridder = xe.Regridder(
        forcing_variables,
        output_grid,
        method=method,
        locstream_out=True,
        periodic=False,
        filename=outfile,
        reuse_weights=False,
    )
    return regridder


def fill_missing_data(ds: xr.Dataset, z_dim_name: str) -> xr.Dataset:
    """
    Fill in missing values with forward fill along the z dimension (We can make this more elaborate with time.... The original RM6 fill was different)
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to fill in
    z_dim_name : str
        The name of the z dimension
    Returns
    -------
    xr.Dataset
        The filled in dataset
    """
    regridding_logger.info("Forward filling in missing data along z-dim")
    ds = ds.ffill(
        dim=z_dim_name, limit=None
    )  # This fills in the nans with the forward fill along the z dimension with an unlimited num of nans
    return ds


def add_or_update_time_dim(ds: xr.Dataset, times) -> xr.Dataset:
    """
    Add the time dimension to the dataset, in tides case can be one time step.
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to add the time dimension to
    times : list, np.Array, xr.DataArray
        The list of times
    Returns
    -------
    xr.Dataset
        The dataset with the time dimension added
    """
    regridding_logger.info("Adding time dimension")

    regridding_logger.debug(f"Times: {times}")
    regridding_logger.debug(f"Make sure times is a DataArray")
    # Make sure times is an xr.DataArray
    times = xr.DataArray(times)

    if "time" in ds.dims:
        regridding_logger.debug("Time already in dataset, overwriting with new values")
        ds["time"] = times
    else:
        regridding_logger.debug("Time not in dataset, xr.Broadcasting time dimension")
        ds, _ = xr.broadcast(ds, times)

    # Make sure time is first....
    regridding_logger.debug("Transposing time to first dimension")
    new_dims = ["time"] + [dim for dim in ds.dims if dim != "time"]
    ds = ds.transpose(*new_dims)

    return ds


def generate_dz(ds: xr.Dataset, z_dim_name: str) -> xr.Dataset:
    """
    For vertical coordinates, you need to have the layer thickness or something. Generate the dz variable for the dataset
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to get the z variable from
    z_dim_name : str
        The name of the z dimension
    Returns
    -------
    xr.Dataset
        the dz variable
    """
    dz = ds[z_dim_name].diff(z_dim_name)
    dz.name = "dz"
    dz = xr.concat([dz, dz[-1]], dim=z_dim_name)
    return dz


def add_secondary_dimension(
    ds: xr.Dataset, var: str, coords, segment_name: str
) -> xr.Dataset:
    """Add the perpendiciular dimension to the dataset, even if it's like one val. It's required.
    Parameters
    -----------
    ds : xr.Dataset
        The dataset to add the perpendicular dimension to
    var : str
        The variable to add the perpendicular dimension to
    coords : xr.Dataset
        The coordinates from the function coords...
    segment_name : str
        The segment name
    Returns
    -------
    xr.Dataset
        The dataset with the perpendicular dimension added


    """

    # Check if we need to insert the dim earlier or later
    regridding_logger.info("Adding perpendicular dimension to {}".format(var))

    regridding_logger.debug(
        "Checking if nz or constituent is in dimensions, then we have to bump the perpendicular dimension up by one"
    )
    insert_behind_by = 0
    if any(coord.startswith("nz") or coord == "constituent" for coord in ds[var].dims):
        regridding_logger.debug("Bump it by one")
        insert_behind_by = 0
    else:
        # Missing vertical dim or tidal coord means we don't need to offset the perpendicular
        insert_behind_by = 1

    regridding_logger.debug(f"Expand dimensions")
    ds[var] = ds[var].expand_dims(
        f"{coords.attrs['perpendicular']}_{segment_name}",
        axis=coords.attrs["axis_to_expand"] - insert_behind_by,
    )
    return ds


def vertical_coordinate_encoding(
    ds: xr.Dataset, var: str, segment_name: str, old_vert_coord_name: str
) -> xr.Dataset:
    """
    Rename vertical coordinate to nz_..., then change it to regular increments

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to rename the vertical coordinate in
    var : str
        The variable to rename the vertical coordinate in
    segment_name : str
        The segment name
    old_vert_coord_name : str
        The old vertical coordinate name
    """

    regridding_logger.info("Renaming vertical coordinate to nz_... in {}".format(var))
    section = "_seg"
    base_var = var[: var.find(section)] if section in var else var
    ds[var] = ds[var].rename({old_vert_coord_name: f"nz_{segment_name}_{base_var}"})

    ## Replace the old depth coordinates with incremental integers
    regridding_logger.info("Replacing old depth coordinates with incremental integers")
    ds[f"nz_{segment_name}_{base_var}"] = np.arange(
        ds[f"nz_{segment_name}_{base_var}"].size
    )

    return ds


def generate_layer_thickness(
    ds: xr.Dataset, var: str, segment_name: str, old_vert_coord_name: str
) -> xr.Dataset:
    """
    Generate Layer Thickness Variable, needed for vars with vertical dimensions
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to generate the layer thickness for
    var : str
        The variable to generate the layer thickness for
    segment_name : str
        The segment name
    old_vert_coord_name : str
        The old vertical coordinate name
    Returns
    -------
    xr.Dataset
        The dataset with the layer thickness variable added
    """
    regridding_logger.debug("Generating layer thickness variable for {}".format(var))
    dz = generate_dz(ds, old_vert_coord_name)
    ds[f"dz_{var}"] = (
        [
            "time",
            f"nz_{var}",
            f"ny_{segment_name}",
            f"nx_{segment_name}",
        ],
        da.broadcast_to(
            dz.data[None, :, None, None],
            ds[var].shape,
            chunks=(
                1,
                None,
                None,
                None,
            ),  ## Chunk in each time, and every 5 vertical layers
        ),
    )

    return ds


def generate_encoding(
    ds: xr.Dataset, encoding_dict, default_fill_value=netCDF4.default_fillvals["f8"]
) -> xr.Dataset:
    """
    Generate the encoding dictionary for the dataset
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to generate the encoding for
    encoding_dict : dict
        The starting encoding dict with some specifications needed for time and other vars, this will be updated with encodings in this function
    default_fill_value : float, optional
        The default fill value, by default 1.0e20
    Returns
    -------
    dict
        The encoding dictionary
    """
    regridding_logger.info("Generating encoding dictionary")
    for var in ds:
        if "_segment_" in var and not "nz" in var:
            encoding_dict[var] = {
                "_FillValue": default_fill_value,
            }
    for var in ds.coords:
        if "nz_" in var:
            encoding_dict[var] = {
                "dtype": "int32",
            }

    return encoding_dict


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


def initialize_grid_rotation_angle(hgrid: xr.Dataset) -> xr.Dataset:
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
    regridding_logger.info("Initializing grid rotation angle")
    # Direct Translation
    pi_720deg = (
        np.arctan(1) / 180
    )  # One quarter the conversion factor from degrees to radians

    ## Check length of longitude
    G_len_lon = (
        hgrid.x.max() - hgrid.x.min()
    )  # We're always going to be working with the regional case.... in the global case len_lon is different, and is a check in the actual MOM code.
    len_lon = G_len_lon

    # Get the tlon and tlat
    ds_t = get_hgrid_arakawa_c_points(hgrid, "t")
    tlon = ds_t.tlon
    ds_q = get_hgrid_arakawa_c_points(hgrid, "q")
    qlon = ds_q.qlon
    qlat = ds_q.qlat
    angles_arr = np.zeros((len(tlon.nyp), len(tlon.nxp)))

    # Compute lonB for all points
    lonB = np.zeros((2, 2, len(tlon.nyp), len(tlon.nxp)))

    # Vectorized computation of lonB
    for n in np.arange(0, 2):
        for m in np.arange(0, 2):
            lonB[m, n] = modulo_around_point(
                qlon[
                    np.arange(m, (m - 1 + len(qlon.nyp))),
                    np.arange(n, (n - 1 + len(qlon.nxp))),
                ],
                tlon,
                len_lon,
            )

    # Compute lon_scale
    lon_scale = np.cos(
        pi_720deg
        * ((qlat[0:-1, 0:-1] + qlat[1:, 1:]) + (qlat[1:, 0:-1] + qlat[0:-1, 1:]))
    )

    # Compute angle
    angle = np.arctan2(
        lon_scale * ((lonB[0, 1] - lonB[1, 0]) + (lonB[1, 1] - lonB[0, 0])),
        (qlat[:-1, :-1] - qlat[1:, 1:]) + (qlat[1:, 0:-1] - qlat[0:-1, 1:]),
    )
    # Assign angle to angles_arr
    angles_arr = np.rad2deg(angle) - 90

    # Assign angles_arr to hgrid
    t_angles = xr.DataArray(
        angles_arr,
        dims=["nyp", "nxp"],
        coords={
            "nyp": tlon.nyp.values,
            "nxp": tlon.nxp.values,
        },
    )
    return t_angles


def get_hgrid_arakawa_c_points(hgrid: xr.Dataset, point_type="t") -> xr.Dataset:
    """
    Get the Arakawa C points from the Hgrid, originally written by Fred (Castruccio) and moved to RM6
    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset
    Returns
    -------
    xr.Dataset
        The specific points x, y, & point indexes
    """
    if point_type not in "uvqth":
        raise ValueError("point_type must be one of 'uvqht'")

    regridding_logger.info("Getting {} points..".format(point_type))

    # Figure out the maths for the offset
    k = 2
    kp2 = k // 2
    offset_one_by_two_y = np.arange(kp2, len(hgrid.x.nyp), k)
    offset_one_by_two_x = np.arange(kp2, len(hgrid.x.nxp), k)
    by_two_x = np.arange(0, len(hgrid.x.nxp), k)
    by_two_y = np.arange(0, len(hgrid.x.nyp), k)

    # T point locations
    if point_type == "t" or point_type == "h":
        points = (offset_one_by_two_y, offset_one_by_two_x)
    # U point locations
    elif point_type == "u":
        points = (offset_one_by_two_y, by_two_x)
    # V point locations
    elif point_type == "v":
        points = (by_two_y, offset_one_by_two_x)
    # Corner point locations
    elif point_type == "q":
        points = (by_two_y, by_two_x)
    else:
        raise ValueError("Invalid Point Type (u, v, q, or t/h only)")

    point_dataset = xr.Dataset(
        {
            "{}lon".format(point_type): hgrid.x[points],
            "{}lat".format(point_type): hgrid.y[points],
            "{}_points_y".format(point_type): points[0],
            "{}_points_x".format(point_type): points[1],
        }
    )
    point_dataset.attrs["description"] = (
        "Arakawa C {}-points of supplied h-grid".format(point_type)
    )
    return point_dataset


def create_pseudo_hgrid(hgrid: xr.Dataset) -> xr.Dataset:
    """
    Adds an additional boundary to the hgrid to allow for the calculation of the angle_dx for the boundary points using the method in MOM6
    """
    pseudo_hgrid_x = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp)+2), np.nan)
    pseudo_hgrid_y = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp)+2), np.nan)

    ## Fill Boundaries
    pseudo_hgrid_x[1:-1, 1:-1] = hgrid.x.values
    pseudo_hgrid_x[0, 1:-1] = hgrid.x.values[0,:] - (hgrid.x.values[1,:] - hgrid.x.values[0,:]) # Bottom Fill
    pseudo_hgrid_x[-1, 1:-1] = hgrid.x.values[-1,:] + (hgrid.x.values[-1,:] - hgrid.x.values[-2,:]) # Top Fill
    pseudo_hgrid_x[1:-1, 0] = hgrid.x.values[:,0] - (hgrid.x.values[:,1] - hgrid.x.values[:,0]) # Left Fill
    pseudo_hgrid_x[1:-1, -1] = hgrid.x.values[:,-1] + (hgrid.x.values[:,-1] - hgrid.x.values[:,-2]) # Right Fill

    pseudo_hgrid_y[1:-1, 1:-1] = hgrid.y.values
    pseudo_hgrid_y[0, 1:-1] = hgrid.y.values[0,:] - (hgrid.y.values[1,:] - hgrid.y.values[0,:]) # Bottom Fill
    pseudo_hgrid_y[-1, 1:-1] = hgrid.y.values[-1,:] + (hgrid.y.values[-1,:] - hgrid.y.values[-2,:]) # Top Fill
    pseudo_hgrid_y[1:-1, 0] = hgrid.y.values[:,0] - (hgrid.y.values[:,1] - hgrid.y.values[:,0]) # Left Fill
    pseudo_hgrid_y[1:-1, -1] = hgrid.y.values[:,-1] + (hgrid.y.values[:,-1] - hgrid.y.values[:,-2]) # Right Fill


    ## Fill Corners
    pseudo_hgrid_x[0, 0] = hgrid.x.values[0,0] - (hgrid.x.values[1,1] - hgrid.x.values[0,0]) # Bottom Left
    pseudo_hgrid_x[-1, 0] = hgrid.x.values[-1,0] - (hgrid.x.values[-2,1] - hgrid.x.values[-1,0]) # Top Left
    pseudo_hgrid_x[0, -1] = hgrid.x.values[0,-1] - (hgrid.x.values[1,-2] - hgrid.x.values[0,-1]) # Bottom Right
    pseudo_hgrid_x[-1, -1] = hgrid.x.values[-1,-1] - (hgrid.x.values[-2,-2] - hgrid.x.values[-1,-1]) # Top Right

    pseudo_hgrid_y[0, 0] = hgrid.y.values[0,0] - (hgrid.y.values[1,1] - hgrid.y.values[0,0]) # Bottom Left
    pseudo_hgrid_y[-1, 0] = hgrid.y.values[-1,0] - (hgrid.y.values[-2,1] - hgrid.y.values[-1,0]) # Top Left
    pseudo_hgrid_y[0, -1] = hgrid.y.values[0,-1] - (hgrid.y.values[1,-2] - hgrid.y.values[0,-1]) # Bottom Right
    pseudo_hgrid_y[-1, -1] = hgrid.y.values[-1,-1] - (hgrid.y.values[-2,-2] - hgrid.y.values[-1,-1]) # Top Right

    pseudo_hgrid = xr.Dataset( 
    {
        "x": (["nyp", "nxp"], pseudo_hgrid_x),
        "y": (["nyp", "nxp"], pseudo_hgrid_y),
    }
    )
    return pseudo_hgrid

def initialize_hgrid_rotation_angles_using_pseudo_hgrid(hgrid: xr.Dataset, pseudo_hgrid:xr.Dataset)->xr.Dataset:
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
    # Direct Translation
    pi_720deg = (
        np.arctan(1) / 180
    )  # One quarter the conversion factor from degrees to radians

    ## Check length of longitude
    G_len_lon = pseudo_hgrid.x.max() - pseudo_hgrid.x.min() # We're always going to be working with the regional case.... in the global case len_lon is different, and is a check in the actual MOM code.
    len_lon = G_len_lon

    angles_arr = np.zeros((len(hgrid.nyp), len(hgrid.nxp)))

    # Compute lonB for all points
    lonB = np.zeros((2, 2, len(hgrid.nyp), len(hgrid.nxp)))

    # Vectorized computation of lonB
    lonB[0][0] = modulo_around_point(
        pseudo_hgrid.x[:-2, :-2], hgrid.x, len_lon
    ) # Bottom Left
    lonB[1][0] = modulo_around_point(
        pseudo_hgrid.x[2:, :-2], hgrid.x, len_lon
    ) # Top Left
    lonB[1][1] = modulo_around_point(
        pseudo_hgrid.x[2:, 2:], hgrid.x, len_lon
    ) # Top Right
    lonB[0][1] = modulo_around_point(
        pseudo_hgrid.x[:-2, 2:], hgrid.x, len_lon
    ) # Bottom Right


    # Compute lon_scale
    lon_scale = np.cos(
        pi_720deg
        * ((pseudo_hgrid.y[:-2, :-2] + pseudo_hgrid.y[:-2, 2:]) + (pseudo_hgrid.y[2:, 2:] + pseudo_hgrid.y[2:, :-2]))
    )

    # Compute angle
    angle = np.arctan2(
        lon_scale * ((lonB[0, 1] - lonB[1, 0]) + (lonB[1, 1] - lonB[0, 0])),
        (pseudo_hgrid.y[:-2, :-2] -pseudo_hgrid.y[2:, 2:]) + (pseudo_hgrid.y[2:, :-2] - pseudo_hgrid.y[:-2, 2:]),
    )
    # Assign angle to angles_arr
    angles_arr = np.rad2deg(angle) - 90

    # Assign angles_arr to hgrid
    t_angles = xr.DataArray(
        angles_arr,
        dims=["nyp", "nxp"],
        coords={
            "nyp": hgrid.nyp.values,
            "nxp": hgrid.nxp.values,
        },
    )
    return t_angles



