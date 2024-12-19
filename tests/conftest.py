import pytest
import os
import xarray as xr

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
