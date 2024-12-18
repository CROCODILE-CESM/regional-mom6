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


regridding_logger = setup_logger(__name__)


def coords(
    hgrid: xr.Dataset,
    orientation: str,
    segment_name: str,
    coords_at_t_points=False,
    angle_variable_name="angle_dx",
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

        tangle_dx = hgrid[angle_variable_name][(ds.t_points_y, ds.t_points_x)]
        # Assign to dataset
        dataset_to_get_coords = xr.Dataset(
            {
                "x": ds.tlon,
                "y": ds.tlat,
                angle_variable_name: (("nyp", "nxp"), tangle_dx.values),
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
                "angle": dataset_to_get_coords[angle_variable_name].isel(nyp=0),
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
                "angle": dataset_to_get_coords[angle_variable_name].isel(nyp=-1),
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
                "angle": dataset_to_get_coords[angle_variable_name].isel(nxp=0),
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
                "angle": dataset_to_get_coords[angle_variable_name].isel(nxp=-1),
            }
        )
        rcoord = rcoord.rename_dims({"nyp": f"ny_{segment_name}"})
        rcoord.attrs["perpendicular"] = "nx"
        rcoord.attrs["parallel"] = "ny"
        rcoord.attrs["axis_to_expand"] = 3

    # Make lat and lon coordinates
    rcoord = rcoord.assign_coords(lat=rcoord["lat"], lon=rcoord["lon"])

    return rcoord


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


def create_regridder(
    forcing_variables: xr.Dataset,
    output_grid: xr.Dataset,
    outfile: Path = None,
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
    ds: xr.Dataset, var: str, coords, segment_name: str, to_beginning=False
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
    to_beginning : bool, optional
        Whether to add the perpendicular dimension to the beginning or to the selected position, by default False
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
    if not to_beginning:

        if any(
            coord.startswith("nz") or coord == "constituent" for coord in ds[var].dims
        ):
            regridding_logger.debug("Bump it by one")
            insert_behind_by = 0
        else:
            # Missing vertical dim or tidal coord means we don't need to offset the perpendicular
            insert_behind_by = 1
    else:
        insert_behind_by = coords.attrs[
            "axis_to_expand"
        ]  # Just magic to add dim to the beginning

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


def get_boundary_mask(
    hgrid: xr.Dataset, bathy: xr.Dataset, side: str, segment_name: str, minimum_depth=0
) -> np.ndarray:
    """
    Mask out the boundary conditions based on the bathymetry. We don't want to have boundary conditions on land.
    Parameters
    ----------
    hgrid : xr.Dataset
        The hgrid dataset
    bathy : xr.Dataset
        The bathymetry dataset
    side : str
        The side of the boundary, "north", "south", "east", or "west"
    segment_name : str
        The segment name
    minimum_depth : float, optional
        The minimum depth to consider land, by default 0
    Returns
    -------
    np.Array
        The boundary mask
    """

    # Hide the bathy as an angle field so we can take advantage of the coords function to get the boundary points.
    bathy = bathy.rename({"lath": "nyp", "lonh": "nxp"})

    # Copy Hgrid
    bathy_2 = hgrid.copy(deep=True)

    # Create new depth field
    bathy_2["depth"] = bathy_2["angle_dx"]
    bathy_2["depth"][:, :] = np.nan

    # Fill at t_points (what bathy is determined at)
    ds_t = get_hgrid_arakawa_c_points(hgrid, "t")
    bathy_2["depth"][ds_t.t_points_y.values, ds_t.t_points_x.values] = bathy.depth

    bathy_2_coords = coords(
        bathy_2,
        side,
        segment_name,
        angle_variable_name="depth",
        coords_at_t_points=True,
    )

    # Get the Boundary Depth
    bathy_2_coords["boundary_depth"] = bathy_2_coords["angle"]
    land = 0
    ocean = 1.0
    boundary_mask = np.full(np.shape(coords(hgrid, side, segment_name).angle), ocean)

    ## Mask2DCu is the mask for the u/v points on the hgrid and is set to OBCmaskCy as well...
    for i in range(len(bathy_2_coords["boundary_depth"]) - 1):
        if bathy_2_coords["boundary_depth"][i] <= minimum_depth:
            # The points to the left and right of this t-point are land points
            boundary_mask[(i * 2) + 2] = land
            boundary_mask[(i * 2) + 1] = (
                land  # u/v point on the second level just like mask2DCu
            )
            boundary_mask[(i * 2)] = land

    # Corner Q-points defined as wet
    boundary_mask[0] = ocean
    boundary_mask[-1] = ocean

    # Looks like in the boundary between land and ocean - in NWA for example - we basically need to remove 3 points closest to ocean as a buffer.
    # Search for intersections
    beaches_before = []
    beaches_after = []
    for index in range(1, len(boundary_mask) - 1):
        if boundary_mask[index - 1] == land and boundary_mask[index] == ocean:
            beaches_before.append(index)
        elif boundary_mask[index + 1] == land and boundary_mask[index] == ocean:
            beaches_after.append(index)
    for beach in beaches_before:
        for i in range(3):
            if beach - 1 - i >= 0:
                boundary_mask[beach - 1 - i] = ocean
    for beach in beaches_before:
        for i in range(3):
            if beach + 1 + i < len(beaches_before):
                boundary_mask[beach + 1 + i] = ocean
    boundary_mask[np.where(boundary_mask == land)] = np.nan

    return boundary_mask


def mask_dataset(
    ds: xr.Dataset,
    hgrid: xr.Dataset,
    bathymetry: xr.Dataset,
    orientation,
    segment_name: str,
) -> xr.Dataset:
    """
    This function masks the dataset to the provided bathymetry. If bathymetry is not provided, it fills all NaNs with 0.
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to mask
    hgrid : xr.Dataset
        The hgrid dataset
    bathymetry : xr.Dataset
        The bathymetry dataset
    orientation : str
        The orientation of the boundary
    segment_name : str
        The segment name
    """
    ## Add Boundary Mask ##
    if bathymetry is not None:
        regridding_logger.info(
            "Masking to bathymetry. If you don't want this, set bathymetry_path to None in the segment class."
        )
        mask = get_boundary_mask(
            hgrid,
            bathymetry,
            orientation,
            segment_name,
            minimum_depth=0,
        )
        if orientation in ["east", "west"]:
            mask = mask[:, np.newaxis]
        else:
            mask = mask[np.newaxis, :]

        for var in ds.data_vars.keys():

            ## Compare the dataset to the mask by reducing dims##
            dataset_reduce_dim = ds[var]
            for index in range(ds[var].ndim - 2):
                dataset_reduce_dim = dataset_reduce_dim[0]
            if orientation in ["east", "west"]:
                dataset_reduce_dim = dataset_reduce_dim[:, 0]
                mask_reduce = mask[:, 0]
            else:
                dataset_reduce_dim = dataset_reduce_dim[0, :]
                mask_reduce = mask[0, :]
            loc_nans_data = np.where(np.isnan(dataset_reduce_dim))
            loc_nans_mask = np.where(np.isnan(mask_reduce))

            ## Check if all nans in the data are in the mask without corners ##
            if not np.isin(loc_nans_data[1:-1], loc_nans_mask[1:-1]).all():
                regridding_logger.warning(
                    f"NaNs in {var} not in mask. This values are filled with zeroes b/c they could cause issues with boundary conditions."
                )

                ## Remove Nans if needed ##
                ds[var] = ds[var].fillna(0)

            ## Apply the mask ##
            ds[var] = ds[var] * mask
    else:
        regridding_logger.warning(
            "All NaNs filled b/c bathymetry wasn't provided to the function. Add bathymetry_path to the segment class to avoid this"
        )
        ds = ds.fillna(
            0
        )  # Without bathymetry, we can't assume the nans will be allowed in Boundary Conditions
    return ds


def generate_encoding(
    ds: xr.Dataset, encoding_dict, default_fill_value=netCDF4.default_fillvals["f8"]
) -> dict:
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
