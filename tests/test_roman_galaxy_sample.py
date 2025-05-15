import numpy as np
import os
import shutil
import pytest
from scripts.roman_galaxy_sample import RomanGalaxySample

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
YAML_SOURCE = os.path.join(PROJECT_ROOT, "parameters", "pz_parameters.yaml")

@pytest.fixture
def roman_sample(tmp_path):
    param_dir = tmp_path / "parameters"
    param_dir.mkdir()
    shutil.copy(YAML_SOURCE, param_dir / "pz_parameters.yaml")
    os.chdir(tmp_path)
    return RomanGalaxySample(forecast_year="1", decimal_places=4)

def test_file_saving_safe(tmp_path):
    param_dir = tmp_path / "parameters"
    param_dir.mkdir()
    shutil.copy(YAML_SOURCE, param_dir / "pz_parameters.yaml")

    os.chdir(tmp_path)
    rs = RomanGalaxySample(forecast_year="1", decimal_places=2)
    rs.lens_sample(save_file=True)

    expected_file = tmp_path / "roman_lens_sample_pz_year_1.npy"
    assert expected_file.exists()

def test_redshift_range_shape(roman_sample):
    assert isinstance(roman_sample.redshift_range, np.ndarray)
    assert len(roman_sample.redshift_range) > 10

def test_lens_sample_output_shape(roman_sample):
    pz = roman_sample.lens_sample(save_file=False)
    assert isinstance(pz, np.ndarray)
    assert pz.shape == roman_sample.redshift_range.shape

def test_source_sample_output_shape(roman_sample):
    pz = roman_sample.source_sample(save_file=False)
    assert isinstance(pz, np.ndarray)
    assert pz.shape == roman_sample.redshift_range.shape

def test_lens_bins_key_count(roman_sample):
    bins = roman_sample.lens_bins(save_file=False)
    assert isinstance(bins, dict)
    assert len(bins) == roman_sample.get_n_tomo_bins("lens")

def test_source_bins_key_count(roman_sample):
    bins = roman_sample.source_bins(save_file=False)
    assert isinstance(bins, dict)
    assert len(bins) == roman_sample.get_n_tomo_bins("source")

def test_bin_centers_format(roman_sample):
    centers = roman_sample.lens_bin_centers(save_file=False)
    assert isinstance(centers, dict)
    for k, v in centers.items():
        assert isinstance(k, int)
        assert isinstance(v, float)
