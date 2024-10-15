import pytest
import regional_mom6 as rmom6
from pathlib import Path
import os


def test_create_config():
    expt_name = "testing"

    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

    ## Place where all your input files go
    input_dir = Path(
        os.path.join(
            expt_name,
            "inputs",
        )
    )

    ## Directory where you'll run the experiment from
    run_dir = Path(
        os.path.join(
            expt_name,
            "run_files",
        )
    )
    data_path = Path("data")
    for path in (run_dir, input_dir, data_path):
        os.makedirs(str(path), exist_ok=True)

    ## User-1st, test if we can even read the angled nc files.
    expt = rmom6.experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=0.05,
        number_vertical_layers=75,
        layer_thickness_ratio=10,
        depth=4500,
        minimum_depth=5,
        mom_run_dir=run_dir,
        mom_input_dir=input_dir,
        toolpath_dir="",
    )
    config_dict = expt.write_config_file()
    print(config_dict)


def test_load_config():
    expt = rmom6.load_experiment(
        "/home/manishrv/regional-mom6/testing/run_files/rmom6_config.json"
    )
    print(expt)
