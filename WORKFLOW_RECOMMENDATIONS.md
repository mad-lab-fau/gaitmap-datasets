# Workflow Recommendations

This file captures follow-up ideas after the initial `uv` migration. These are not part of the required port, but
they are worth deciding on explicitly.

## Recommended Follow-Ups

### 1. Make CI lockfile-strict

Change the GitHub Actions install step from:

```bash
uv sync --group dev
```

to:

```bash
uv sync --group dev --frozen
```

Why:
- This guarantees CI always uses the committed `uv.lock`.
- It fails fast if `pyproject.toml` and `uv.lock` drift apart.

Tradeoff:
- Contributors need to remember to regenerate `uv.lock` whenever dependency metadata changes.

### 2. Add a broader CI test target

Current CI only runs `tests/test_config.py`, which is correct for now because the rest of the suite depends on local
dataset installations.

Recommended next step:
- Either check in a very small fixture subset for one dataset, or
- Split more pure logic into dataset-free tests.

Why:
- This would give us real runtime coverage for the loaders and helpers on every PR.

### 3. Revisit docs automation

Docs publishing still relies on the local `gh-pages` checkout in `docs/_build/html` and manual pushes via
`uv run poe upload_docs`.

Recommended next step:
- Decide whether to keep that workflow, or
- Move toward an automated docs build/publish path once dataset-heavy examples can be isolated.

Why:
- The current flow works, but it is unusual and depends on local git state.

## Ruff Backlog

The repo currently ignores a small set of Ruff rules that were relaxed during the migration to avoid mixing a large
annotation refactor into the tooling port.

### Current ignored cleanup scope

Counts from `ruff check gaitmap_datasets --select UP006,UP035,PD101,RUF005,RUF046 --statistics`:

- `UP006`: 91 findings
- `UP035`: 35 findings
- `RUF046`: 3 findings
- `RUF005`: 1 finding
- `PD101`: 1 finding

### What can be autofixed

`ruff check gaitmap_datasets --select UP006,UP035,PD101,RUF005,RUF046 --fix --unsafe-fixes --diff`
showed that 95 findings can be autofixed immediately:

- `UP006`: replace `List`/`Dict`/`Tuple` annotations with builtin generics
- `UP035`: replace deprecated `typing` imports
- `RUF046`: remove unnecessary `int(...)` casts
- `RUF005`: replace list concatenation with iterable unpacking

### What still needs manual work

- `PD101`: 1 manual fix in `gaitmap_datasets/kluge_2017/helper.py`

### Recommended cleanup plan

If we decide to tighten these rules later:

1. Run Ruff unsafe autofixes for `UP006`, `UP035`, `RUF046`, and `RUF005`.
2. Apply the single manual `PD101` fix.
3. Re-run the lightweight CI checks.
4. If dataset paths are available locally, run the dataset-backed tests as a regression check.

Estimated effort:
- Low for the autofix pass itself.
- Moderate overall because the typing rewrite touches many files and should be regression-checked.
