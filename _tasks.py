import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def task_prepare_release() -> None:
    update_version(sys.argv[1])
    html_dir = HERE / "docs/_build/html"
    subprocess.run(["git", "switch", "gh-pages"], shell=False, check=True, cwd=html_dir)
    task_docs()
    task_upload_docs()
    new_version = _get_project_version()
    subprocess.run(["git", "add", "."], shell=False, check=True)
    subprocess.run(["git", "commit", "-m", f"Release {new_version}"], shell=False, check=True)


def task_docs() -> None:
    """Build the html docs using Sphinx."""
    shutil.rmtree(HERE / "docs/modules/generated", ignore_errors=True)
    shutil.rmtree(HERE / "docs/_build/doctrees", ignore_errors=True)

    html_dir = HERE / "docs/_build/html"
    if html_dir.exists():
        for path in html_dir.iterdir():
            if path.name == ".git":
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()

    subprocess.run(
        ["sphinx-build", "-b", "html", "-j", "auto", "-d", "docs/_build/doctrees", "docs", "docs/_build/html"],
        shell=False,
        check=True,
        cwd=HERE,
    )


def task_upload_docs() -> None:
    """Upload docs to the gh-pages branch."""
    html_dir = HERE / "docs/_build/html"
    subprocess.run(["git", "add", "."], shell=False, check=True, cwd=html_dir)
    subprocess.run(["git", "commit", "--amend", "--no-edit"], shell=False, check=True, cwd=html_dir)
    subprocess.run(["git", "push", "origin", "gh-pages", "--force"], shell=False, check=True, cwd=html_dir)


def update_version_strings(file_path: Path, new_version: str) -> None:
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with file_path.open("r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: f'{match.group(1)}{new_version}"',
                content,
            )
        )
        f.truncate()


def _get_project_version() -> str:
    return (
        subprocess.run(["uv", "version", "--short"], shell=False, check=True, capture_output=True, cwd=HERE)
        .stdout.decode()
        .strip()
    )


def update_version(version: str) -> None:
    subprocess.run(["uv", "version", version, "--frozen"], shell=False, check=True, cwd=HERE)
    new_version = _get_project_version()
    update_version_strings(HERE / "gaitmap_datasets/__init__.py", new_version)


def task_update_version() -> None:
    update_version(sys.argv[1])
