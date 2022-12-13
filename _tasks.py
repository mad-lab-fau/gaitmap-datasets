import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def task_prepare_release():
    update_version(sys.argv[1])
    task_docs()
    task_upload_docs()
    new_version = _get_poetry_version()
    subprocess.run(["git", "add", "."], shell=False, check=True)
    subprocess.run(["git", "commit", "-m", "Release {}".format(new_version)], shell=False, check=True)


def task_docs():
    """Build the html docs using Sphinx."""
    # Delete Autogenerated files from previous run
    shutil.rmtree(str(HERE / "docs/modules/generated"), ignore_errors=True)

    # Clear the build directory
    shutil.rmtree(
        [HERE / "docs/_build/doctrees", *(v for v in (HERE / "docs/_build/html").iterdir() if v.name != ".git")],
        ignore_errors=True,
    )

    if platform.system() == "Windows":
        subprocess.run([HERE / "docs/make.bat", "html"], shell=False, check=True)
    else:
        subprocess.run(["make", "-C", HERE / "docs", "html"], shell=False, check=True)


def task_upload_docs():
    """Upload docs to the gh-pages branch."""
    html_dir = HERE / "docs/_build/html"
    subprocess.run(["git", "add", "."], shell=False, check=True, cwd=html_dir)
    subprocess.run(["git", "commit", "--amend", "--no-edit"], shell=False, check=True, cwd=html_dir)
    subprocess.run(["git", "push", "origin", "gh-pages", "--force"], shell=False, check=True, cwd=html_dir)


def update_version_strings(file_path, new_version):
    # taken from:
    # https://stackoverflow.com/questions/57108712/replace-updated-version-strings-in-files-via-python
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with open(file_path, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: '{}{}"'.format(match.group(1), new_version),
                content,
            )
        )
        f.truncate()


def _get_poetry_version():
    return (
        subprocess.run(["poetry", "version"], shell=False, check=True, capture_output=True)
        .stdout.decode()
        .strip()
        .split(" ", 1)[1]
    )


def update_version(version):
    subprocess.run(["poetry", "version", version], shell=False, check=True)
    new_version = _get_poetry_version()
    update_version_strings(HERE.joinpath("gaitmap_datasets/__init__.py"), new_version)


def task_update_version():
    update_version(sys.argv[1])
