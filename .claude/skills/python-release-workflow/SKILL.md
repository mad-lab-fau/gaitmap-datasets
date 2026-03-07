---
name: python-release-workflow
description: Use when releasing gaitmap-datasets from this repo with uv+poe+GitHub Actions+manual gh-pages docs publishing; supports patch/minor/major and enforces CI before creating a GitHub release.
---

# Python Release Workflow

## Scope
- Repository: `mad-lab-fau/gaitmap-datasets`
- Tooling: `uv`, `poe`, `gh`
- Release types: `patch`, `minor`, `major`
- Package layout: single package `gaitmap-datasets`
- Docs hosting: manual `gh-pages` publish from `docs/_build/html`

## Inputs
- `bump`: `patch` | `minor` | `major`
- `base_ref`: `main`
- Release notes source: latest release section in `CHANGELOG.md`

## Semver Quick Rule
- `patch`: fixes/docs/tooling/compat, no breaking API
- `minor`: backward-compatible features
- `major`: breaking behavior/API

## Flow
1. **Preflight**
   - `git status -sb` (must be clean or intentionally scoped)
   - `git log -5 --oneline`

2. **Local verification**
   - `uv sync --group dev`
   - `uv run poe ci_check`
   - `uv run pytest tests/test_config.py`
   - Optional full local tests, if dataset paths are configured:
     - `uv run poe test`
   - `uv build`

3. **Release prep**
   - Finalize release notes in `CHANGELOG.md`
   - Bump version: `uv run poe version --bump=<patch|minor|major>`
   - Verify touched files (typically):
     - `pyproject.toml`
     - `gaitmap_datasets/__init__.py`
     - `uv.lock`
     - `CHANGELOG.md`

4. **Docs publish**
   - Ensure the `docs/_build/html` gh-pages worktree/submodule is initialized
   - Run: `uv run poe prepare_release --bump=<patch|minor|major>`
   - Verify docs changes were pushed to `gh-pages`

5. **Commit strategy**
   - Commit code/docs fixes first (if needed)
   - Commit release prep separately (version + changelog + lockfile)

6. **Push + CI gate**
   - `git push`
   - `REL_SHA=$(git rev-parse HEAD)`
   - `gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5`
   - `gh run watch <run-id>`
   - Required: workflow conclusion is `success` for the release commit SHA

7. **Create GitHub release**
   - Tag format: `vX.Y.Z`
   - `gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "...from changelog..."`

8. **Post-release checks**
   - Watch publish workflow triggered by release (`Upload Python Package`)
   - Confirm the PyPI publish job succeeded
   - Confirm the package is visible on PyPI

## Hard Gates
- Never create a GitHub release before CI is green for the release SHA.
- Never claim docs are published without command evidence from `uv run poe prepare_release` or `uv run poe upload_docs`.
- Never claim done without command evidence.

## Fast Command Set
```bash
uv sync --group dev
uv run poe ci_check
uv run pytest tests/test_config.py
uv build
uv run poe version --bump=minor
uv run poe prepare_release --bump=minor
git push
REL_SHA=$(git rev-parse HEAD)
gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5
gh run watch <run-id>
gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "<changelog section>"
```
