"""Retrieval of sample dataset."""

import os
import zipfile


def get_hommerich_dataset():
    """Retrieve and cache sample dataset of Sheaf river."""
    try:
        import pooch
    except ImportError:
        raise ImportError("This function needs pooch. Install pyorc with pip install pyopenrivercam[extra]")

    # Define the DOI link
    filename = "20241010_081717.mp4"
    base_url = "doi:10.5281/zenodo.15002591"
    url = base_url + "/" + filename
    print(f"Retrieving or providing cached version of dataset from {url}")
    # Create a Pooch registry to manage downloads
    registry = pooch.create(
        # Define the cache directory
        path=pooch.os_cache("pyorc"),
        # Define the base URL for fetching data
        base_url=base_url,
        # Define the registry with the file we're expecting to download
        registry={filename: None},
    )
    # Fetch the dataset
    file_path = registry.fetch(filename, progressbar=True)
    print(f"Hommerich video is available in {file_path}")
    return file_path


def get_hommerich_pyorc_zip():
    """Retrieve and cache sample dataset of Sheaf river."""
    try:
        import pooch
    except ImportError:
        raise ImportError("This function needs pooch. Install pyorc with pip install pyopenrivercam[extra]")

    # Define the DOI link
    filename = "hommerich_20241010_081717_pyorc_data.zip.zip"
    base_url = "doi:10.5281/zenodo.15002591"
    url = base_url + "/" + filename
    print(f"Retrieving or providing cached version of dataset from {url}")
    # Create a Pooch registry to manage downloads
    registry = pooch.create(
        # Define the cache directory
        path=pooch.os_cache("pyorc"),
        # Define the base URL for fetching data
        base_url=base_url,
        # Define the registry with the file we're expecting to download
        registry={filename: None},
    )
    # Fetch the dataset
    file_path = registry.fetch(filename, progressbar=True)
    print(f"Hommerich video is available in {file_path}")
    return file_path


def get_hommerich_pyorc_files():
    """Unzip hommerich pyorc files and return file list."""
    zip_file = get_hommerich_pyorc_zip()
    trg_dir = os.path.split(zip_file)[0]
    if not os.path.exists(os.path.join(trg_dir, "hommerich.yml")):
        # apparently the pyorc files are not yet preset. Unzip the record.
        print("Unzipping sample data...")
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(trg_dir)
    return trg_dir
