# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
import os
from PyInstaller.utils.hooks import collect_data_files # this is very helpful
env_path = os.environ['CONDA_PREFIX']
dlls = os.path.join(env_path, 'DLLs')
bins = os.path.join(env_path, 'Library', 'bin')

paths = [
    os.getcwd(),
    env_path,
    dlls,
    bins,
]

# these binary paths might be different on your installation. 
# modify as needed. 
# caveat emptor
binaries = [
    (os.path.join(bins,'geos.dll'), '.'),
    (os.path.join(bins,'geos_c.dll'), '.'),
    (os.path.join(bins,'spatialindex_c-64.dll'), '.'),
    (os.path.join(bins,'spatialindex-64.dll'),'.'),
]

hidden_imports = [
    'ctypes',
    'ctypes.util',
    'fiona',
    'osgeo.gdal',
    'geos',
    'numpy',
    'shapely',
    'shapely.geometry',
    'shapely._geos',
    'rasterio',
    'rasterio._shim',
    'rasterio.control',
    'rasterio.crs',
    'rasterio.sample',
    'rasterio.vrt',
    'rasterio._features',
    'pyproj',
    'rtree',
    'geopandas.datasets',
    'pytest',
    'pandas._libs.tslibs.timedeltas',
]

_geos_pyds = collect_data_files('shapely', include_py_files=True)
geos_pyds = []
for p, lib in geos_pyds:
    if '.pyd' in p or '.pxd' in p:
        geos_pyds.append((p, '.'))

print(geos_pyds)

_osgeo_pyds = collect_data_files('osgeo', include_py_files=True)

osgeo_pyds = []
for p, lib in _osgeo_pyds:
    if '.pyd' in p:
        osgeo_pyds.append((p, '.'))

binaries += osgeo_pyds
binaries += geos_pyds

a = Analysis(
    ['main.py'],
    pathex=paths,
    binaries=binaries,
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pyorc',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
