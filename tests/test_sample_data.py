import os.path

import pyorc


def test_sample_pyorc_data():
    path = pyorc.sample_data.get_hommerich_pyorc_files()
    # check if recipe exists
    assert os.path.exists(os.path.join(path, "hommerich.yml"))
