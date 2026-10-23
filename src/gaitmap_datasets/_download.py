"""Helpers to download and unpack dataset archives."""

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class _RemoteFileInfo:
    filename: str
    version_identifier: str


def _relevant_file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))


def _infer_dataset_root(path: Path) -> Path:
    children = [p for p in path.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    child_dirs = [p for p in children if p.is_dir()]
    child_files = [p for p in children if p.is_file()]

    if len(child_dirs) == 1 and not child_files:
        return child_dirs[0]
    return path


def _version_identifier_from_metadata(metadata: dict) -> str:
    extra = metadata.get("attributes", {}).get("extra", {})
    version = extra.get("version")
    sha256 = extra.get("hashes", {}).get("sha256")

    version_parts = []
    if version is not None:
        version_parts.append(f"v{version}")
    if sha256:
        version_parts.append(sha256[:8])

    if version_parts:
        return "_".join(version_parts)
    return "latest"


def _get_osf_file_info(download_url: str) -> _RemoteFileInfo:
    request = Request(download_url, headers={"User-Agent": "gaitmap-datasets"}, method="HEAD")
    with urlopen(request) as response:
        metadata = json.loads(response.headers["x-waterbutler-metadata"])
    return _RemoteFileInfo(
        filename=metadata["attributes"]["name"],
        version_identifier=_version_identifier_from_metadata(metadata),
    )


def _download_url_to_file(download_url: str, destination: Path) -> None:
    request = Request(download_url, headers={"User-Agent": "gaitmap-datasets"})
    with urlopen(request) as response, destination.open("wb") as file_obj:
        shutil.copyfileobj(response, file_obj)


def _download_and_extract_zip(
    base_path: str | Path,
    *,
    dataset_name: str,
    download_url: str,
    version_identifier: str,
    min_existing_files: int = 5,
) -> Path:
    """Download and unpack a zipped dataset archive.

    The dataset is stored inside ``base_path / dataset_name / version_identifier``.
    The returned path is the folder that should be passed to the dataset class.
    """
    version_path = Path(base_path) / dataset_name / version_identifier

    if version_path.exists() and _relevant_file_count(version_path) >= min_existing_files:
        return _infer_dataset_root(version_path)

    if version_path.exists():
        shutil.rmtree(version_path)
    version_path.mkdir(parents=True, exist_ok=True)

    temp_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=version_path.parent) as tmp:
            temp_file = Path(tmp.name)
        _download_url_to_file(download_url, temp_file)
        with zipfile.ZipFile(temp_file) as zip_ref:
            zip_ref.extractall(version_path)
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()

    return _infer_dataset_root(version_path)


def download_and_extract_osf_zip(
    base_path: str | Path,
    *,
    dataset_name: str,
    download_url: str,
    min_existing_files: int = 5,
) -> Path:
    """Download and unpack a zipped dataset from OSF."""
    file_info = _get_osf_file_info(download_url)
    return _download_and_extract_zip(
        base_path,
        dataset_name=dataset_name,
        download_url=download_url,
        version_identifier=file_info.version_identifier,
        min_existing_files=min_existing_files,
    )


def download_and_extract_zip(
    base_path: str | Path,
    *,
    dataset_name: str,
    download_url: str,
    version_identifier: str,
    min_existing_files: int = 5,
) -> Path:
    """Download and unpack a zipped dataset with a caller-provided version identifier."""
    return _download_and_extract_zip(
        base_path,
        dataset_name=dataset_name,
        download_url=download_url,
        version_identifier=version_identifier,
        min_existing_files=min_existing_files,
    )


def raise_manual_download_error(dataset_name: str, instructions: str) -> None:
    """Raise a consistent error for datasets without automatic download support."""
    raise RuntimeError(f"Automatic download is not supported for {dataset_name}.\n{instructions}")
