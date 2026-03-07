"""Tests for the download helper."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from gaitmap_datasets import EgaitParameterValidation2013, EgaitSegmentationValidation2014, Kluge2017
from gaitmap_datasets._download import _RemoteFileInfo, download_and_extract_osf_zip, download_and_extract_zip
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Hallway, PyShoe2019Stairs, PyShoe2019Vicon


def _create_zip(zip_path: Path, file_paths: list[str]) -> None:
    with ZipFile(zip_path, "w") as zip_file:
        for relative_path in file_paths:
            zip_file.writestr(relative_path, relative_path)


def test_download_and_extract_osf_zip_returns_single_extracted_root(tmp_path, monkeypatch):
    """A single extracted top-level folder is returned as dataset root."""
    source_zip = tmp_path / "source.zip"
    _create_zip(
        source_zip,
        [
            "dataset_root/data/file_1.txt",
            "dataset_root/data/file_2.txt",
            "dataset_root/meta/file_3.txt",
            "dataset_root/meta/file_4.txt",
            "dataset_root/meta/file_5.txt",
        ],
    )

    monkeypatch.setattr(
        "gaitmap_datasets._download._get_osf_file_info",
        lambda _: _RemoteFileInfo(filename="dataset.zip", version_identifier="v1_deadbeef"),
    )
    monkeypatch.setattr(
        "gaitmap_datasets._download._download_url_to_file",
        lambda _url, destination: destination.write_bytes(source_zip.read_bytes()),
    )

    result = download_and_extract_osf_zip(tmp_path / "downloads", dataset_name="example_dataset", download_url="test")

    assert result == tmp_path / "downloads" / "example_dataset" / "v1_deadbeef" / "dataset_root"
    assert result.is_dir()
    assert (result / "data" / "file_1.txt").exists()


def test_download_and_extract_osf_zip_skips_existing_version(tmp_path, monkeypatch):
    """Existing populated version folders are reused without downloading again."""
    version_path = tmp_path / "downloads" / "example_dataset" / "v1_deadbeef"
    dataset_root = version_path / "dataset_root"
    dataset_root.mkdir(parents=True)
    for i in range(5):
        (dataset_root / f"file_{i}.txt").write_text("existing", encoding="utf8")

    monkeypatch.setattr(
        "gaitmap_datasets._download._get_osf_file_info",
        lambda _: _RemoteFileInfo(filename="dataset.zip", version_identifier="v1_deadbeef"),
    )

    def _unexpected_download(_url, _destination):
        raise AssertionError("Download should have been skipped for an existing dataset version.")

    monkeypatch.setattr("gaitmap_datasets._download._download_url_to_file", _unexpected_download)

    result = download_and_extract_osf_zip(tmp_path / "downloads", dataset_name="example_dataset", download_url="test")

    assert result == dataset_root


def test_download_and_extract_zip_uses_explicit_version_identifier(tmp_path, monkeypatch):
    """Generic zip downloads use the provided version identifier in the target path."""
    source_zip = tmp_path / "source.zip"
    _create_zip(
        source_zip,
        [
            "dataset_root/data/file_1.txt",
            "dataset_root/data/file_2.txt",
            "dataset_root/meta/file_3.txt",
            "dataset_root/meta/file_4.txt",
            "dataset_root/meta/file_5.txt",
        ],
    )

    monkeypatch.setattr(
        "gaitmap_datasets._download._download_url_to_file",
        lambda _url, destination: destination.write_bytes(source_zip.read_bytes()),
    )

    result = download_and_extract_zip(
        tmp_path / "downloads",
        dataset_name="example_dataset",
        download_url="test",
        version_identifier="record_123",
    )

    assert result == tmp_path / "downloads" / "example_dataset" / "record_123" / "dataset_root"


@pytest.mark.parametrize(
    ("dataset_class", "message_fragment"),
    [
        (EgaitParameterValidation2013, "contact the data owner"),
        (EgaitSegmentationValidation2014, "contact the data owner"),
        (Kluge2017, "https://osf.io/cfb7e/"),
        (PyShoe2019Vicon, "https://github.com/utiasSTARS/pyshoe"),
        (PyShoe2019Hallway, "https://github.com/utiasSTARS/pyshoe"),
        (PyShoe2019Stairs, "https://github.com/utiasSTARS/pyshoe"),
    ],
)
def test_manual_download_methods_raise_dataset_specific_error(dataset_class, message_fragment):
    """Datasets without automatic download support still expose the download contract."""
    with pytest.raises(RuntimeError, match=message_fragment):
        dataset_class.download("/tmp")
