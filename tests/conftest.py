import pytest
import os
import xarray as xr
import numpy as np

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


@pytest.fixture()
def generate_silly_vt_dataset(tmp_path):
    latitude_extent = [30, 40]
    longitude_extent = [-80, -70]
    eastern_boundary = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 5, 10)),
                dims=["silly_lat", "silly_lon", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
        }
    )
    return eastern_boundary


@pytest.fixture()
def dummy_bathymetry_data():
    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    bathymetry = np.random.random((100, 100)) * (-100)
    bathymetry = xr.DataArray(
        bathymetry,
        dims=["silly_lat", "silly_lon"],
        coords={
            "silly_lat": np.linspace(
                latitude_extent[0] - 5, latitude_extent[1] + 5, 100
            ),
            "silly_lon": np.linspace(
                longitude_extent[0] - 5, longitude_extent[1] + 5, 100
            ),
        },
    )
    bathymetry.name = "silly_depth"
    return bathymetry
