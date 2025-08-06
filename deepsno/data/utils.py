import zlib
from pathlib import Path

import h5py


def checksum_file(path: str | Path, chunk_size: int = 65536) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} is not a file")

    running_checksum = 1
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            running_checksum = zlib.adler32(chunk, running_checksum)
    return running_checksum & 0xFFFFFFFF


def checksum_h5_file(h5_file: h5py.File):
    if isinstance(h5_file, h5py.File):
        path = h5_file.filename
    else:
        path = h5_file

    with h5py.File(path, "r+") as f:
        f.attrs["checksum"] = checksum_file(path)


def str_from_many_paths(paths: tuple[str], n=3) -> str:
    paths = sorted(paths)
    if len(paths) > n:
        paths = paths[: n - 1] + paths[n - 1 :]

    output_str = paths[0]
    for path in paths[1 : n - 1]:
        output_str = f"{output_str}, {path}"

    if len(paths) > n:
        output_str = f"{output_str}, ..."
        output_str = f"{output_str}, {paths[-1]}"

    return output_str
