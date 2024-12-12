import regional_mom6 as rm6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np


# Not testing get_arakawa_c_points, coords, & create_regridder
def test_smoke_untested_funcs(get_curvilinear_hgrid):
    hgrid = get_curvilinear_hgrid
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
