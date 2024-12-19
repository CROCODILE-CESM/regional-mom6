import regional_mom6 as rm6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np


# Not testing get_arakawa_c_points, coords, & create_regridder
def test_smoke_untested_funcs(get_curvilinear_hgrid, generate_silly_vt_dataset):
    hgrid = get_curvilinear_hgrid
    ds = generate_silly_vt_dataset
    ds["lat"] = ds.silly_lat
    ds["lon"] = ds.silly_lat
    assert rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert rgd.coords(hgrid, "north", "segment_002")
    assert rgd.create_regridder(ds, ds)


def test_fill_missing_data(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds["temp"][0, 0, 6:10, 0] = np.nan

    ds = rgd.fill_missing_data(ds, "silly_depth")

    assert (
        ds["temp"][0, 0, 6:10, 0] == (ds["temp"][0, 0, 5, 0])
    ).all()  # Assert if we are forward filling in time

    ds_2 = generate_silly_vt_dataset
    ds_2["temp"][0, 0, 6:10, 0] = ds["temp"][0, 0, 5, 0]
    assert (ds["temp"] == (ds_2["temp"])).all()  # Assert everything else is the same


def test_add_or_update_time_dim(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds = rgd.add_or_update_time_dim(ds, xr.DataArray([0]))

    assert ds["time"].values == [0]  # Assert time is added
    assert ds["temp"].dims[0] == "time"  # Check time is first dim


def test_generate_dz(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    dz = rgd.generate_dz(ds, "silly_depth")
    z = np.linspace(0, 1000, 10)
    dz_check = np.full(z.shape, z[1] - z[0])
    assert (
        (dz.values - dz_check) < 0.00001
    ).all()  # Assert dz is generated correctly (some roundingleniency)


def test_add_secondary_dimension(get_curvilinear_hgrid, generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    hgrid = get_curvilinear_hgrid

    # N/S Boundary
    coords = rgd.coords(hgrid, "north", "segment_002")
    ds = rgd.add_secondary_dimension(ds, "temp", coords, "segment_002")
    assert ds["temp"].dims == (
        "silly_lat",
        "ny_segment_002",
        "silly_lon",
        "silly_depth",
        "time",
    )

    # E/W Boundary
    coords = rgd.coords(hgrid, "east", "segment_003")
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(ds, "v", coords, "segment_003")
    assert ds["v"].dims == (
        "silly_lat",
        "silly_lon",
        "nx_segment_003",
        "silly_depth",
        "time",
    )

    # Beginning
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(
        ds, "temp", coords, "segment_003", to_beginning=True
    )
    assert ds["temp"].dims[0] == "nx_segment_003"

    # NZ dim E/W Boundary
    ds = generate_silly_vt_dataset
    ds = ds.rename({"silly_depth": "nz"})
    ds = rgd.add_secondary_dimension(ds, "u", coords, "segment_003")
    assert ds["u"].dims == (
        "silly_lat",
        "silly_lon",
        "nz",
        "nx_segment_003",
        "time",
    )


def test_vertical_coordinate_encoding(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds = rgd.vertical_coordinate_encoding(ds, "temp", "segment_002", "silly_depth")
    assert "nz_segment_002_temp" in ds["temp"].dims
    assert "nz_segment_002_temp" in ds
    assert (
        ds["nz_segment_002_temp"] == np.arange(ds[f"nz_segment_002_temp"].size)
    ).all()


def test_generate_layer_thickness(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds["temp"] = ds["temp"].transpose("time", "silly_depth", "silly_lat", "silly_lon")
    ds = rgd.generate_layer_thickness(ds, "temp", "segment_002", "silly_depth")
    assert "dz_temp" in ds
    assert ds["dz_temp"].dims == ("time", "nz_temp", "ny_segment_002", "nx_segment_002")
    assert (
        ds["temp"]["silly_depth"].shape == ds["dz_temp"]["nz_temp"].shape
    )  # Make sure the depth dimension was broadcasted correctly


def test_generate_encoding(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    encoding_dict = {}
    ds["temp_segment_002"] = ds["temp"]
    ds.coords["temp_segment_003_nz_"] = ds.silly_depth
    encoding_dict = rgd.generate_encoding(ds, encoding_dict, default_fill_value="-3")
    assert (
        encoding_dict["temp_segment_002"]["_FillValue"] == "-3"
        and "dtype" not in encoding_dict["temp_segment_002"]
    )
    assert encoding_dict["temp_segment_003_nz_"]["dtype"] == "int32"


## TBD - Boundary Mask Functions


def test_get_boundary_mask(get_curvilinear_hgrid, dummy_bathymetry_data):
    hgrid = get_curvilinear_hgrid
    bathy_og = get_curvilinear_hgrid
    bathy = bathy_og.isel(
        nxp=slice(0, bathy_og.dims["nxp"] // 2), nyp=slice(0, bathy_og.dims["nyp"] // 2)
    )
    mask = rgd.get_boundary_mask(hgrid, bathy, "north", "segment_002")
    assert True
