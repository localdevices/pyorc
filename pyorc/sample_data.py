"""Retrieval of sample dataset."""

import os
import time
import zipfile

import platformdirs
import requests

cache_dir = platformdirs.user_cache_dir("pyorc")


def zenodo_pooch(record_id, cache_name):
    """Retrieve files from Zenodo record."""
    try:
        import pooch
    except ImportError:
        raise ImportError("This function needs pooch. Install pyorc with pip install pyopenrivercam[extra]")
    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}",
        timeout=30,
        # headers=headers
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch metadata for record {record_id}. {r.status_code} {r.text}")
    meta = r.json()
    urls = {f["key"]: f["links"]["self"] for f in meta["files"]}

    return pooch.create(
        path=pooch.os_cache(cache_name),
        base_url="",
        urls=urls,
        registry={name: None for name in urls},
    )


def get_hommerich_dataset():
    """Retrieve and cache sample dataset of Sheaf river."""
    # Define the DOI link
    filename = "20241010_081717.mp4"
    file_path = os.path.join(cache_dir, filename)
    # Fetch the dataset
    if not os.path.exists(file_path):
        for attempt in range(5):
            registry = zenodo_pooch(
                record_id=15002591,
                cache_name="pyorc",
            )
            try:
                file_path = registry.fetch(filename, progressbar=True)
                break
            except Exception as e:
                if attempt == 4:
                    raise f"Download failed with error: {e}."
                else:
                    print(f"Download failed with error: {e}. Retrying...")
                    time.sleep(1)
        print(f"Hommerich video is available in {file_path}")
    return file_path


def get_hommerich_pyorc_zip():
    """Retrieve and cache sample dataset of Sheaf river."""
    #
    # # Define the DOI link
    filename = "hommerich_20241010_081717_pyorc_data.zip.zip"
    file_path = os.path.join(cache_dir, filename)
    # Fetch the dataset
    if not os.path.exists(file_path):
        for attempt in range(5):
            registry = zenodo_pooch(
                record_id=15002591,
                cache_name="pyorc",
            )
            try:
                file_path = registry.fetch(filename, progressbar=True)
                break
            except Exception as e:
                if attempt == 4:
                    raise f"Download failed with error: {e}."
                else:
                    print(f"Download failed with error: {e}. Retrying...")
                    time.sleep(1)
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
