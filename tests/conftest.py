import pytest

from tests._regression_utils import PyTestSnapshotTest


@pytest.fixture()
def snapshot(request):
    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser):
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )
