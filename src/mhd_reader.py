"""
LUNA16 Raw MHD/RAW Reader using pure standard library and numpy.
No heavy third-party ITK dependencies required.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image

MET_TYPES = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


def parse_mhd_header(mhd_path: str) -> dict:
    """Parse .mhd metadata header file."""
    header = {}
    with open(mhd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [s.strip() for s in line.split("=", 1)]
            header[key] = val
    return header


def read_mhd(mhd_path: str):
    """
    Read .mhd and corresponding .raw file into 3D numpy array.
    Returns: volume (Z, Y, X), origin (X, Y, Z), spacing (X, Y, Z)
    """
    mhd_path = Path(mhd_path)
    header = parse_mhd_header(str(mhd_path))

    dim_size = [int(x) for x in header.get("DimSize", "512 512 1").split()]
    element_type = header.get("ElementType", "MET_SHORT")
    dtype = MET_TYPES.get(element_type, np.int16)

    origin = np.array([float(x) for x in header.get("Offset", "0 0 0").split()])
    spacing = np.array([float(x) for x in header.get("ElementSpacing", "1 1 1").split()])

    raw_filename = header.get("ElementDataFile", mhd_path.stem + ".raw")
    raw_path = mhd_path.parent / raw_filename

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # Read binary voxel buffer
    raw_data = np.fromfile(str(raw_path), dtype=dtype)
    # Shape is (Z, Y, X)
    shape = (dim_size[2], dim_size[1], dim_size[0])
    volume = raw_data.reshape(shape)

    return volume, origin, spacing


def world_to_voxel(world_coord: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert Cartesian world coordinates (mm) to 3D voxel index (x, y, z)."""
    voxel_coord = np.round(np.abs(world_coord - origin) / spacing).astype(int)
    return voxel_coord


def apply_lung_window(slice_2d: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    """
    Apply standard medical CT Lung Windowing (HU).
    center: -600 HU, width: 1500 HU -> range [-1350, 150] HU.
    Returns normalized uint8 array [0, 255].
    """
    min_hu = center - (width / 2.0)
    max_hu = center + (width / 2.0)
    windowed = np.clip(slice_2d, min_hu, max_hu)
    normalized = ((windowed - min_hu) / (max_hu - min_hu) * 255.0).astype(np.uint8)
    return normalized


def extract_axial_slice(volume: np.ndarray, z_idx: int) -> Image.Image:
    """Extract single 2D axial slice and convert to PIL Image with lung windowing."""
    z_idx = max(0, min(z_idx, volume.shape[0] - 1))
    slice_data = volume[z_idx, :, :]
    windowed = apply_lung_window(slice_data)
    return Image.fromarray(windowed).convert("RGB")
