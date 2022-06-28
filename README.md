# mad-datasets

Helper to access to open-source gait datasets of the MaD-Lab


## Testing

The `/tests` directory contains a set of tests to check the functionality of the library.
However, most tests rely on the existence of the respective datasets in certain folders outside the library.
Therefore, the tests can only be run locally and not on the CI server.

To run them locally, make sure datasets are downloaded into the correct folders and then run `poe test`.

## Documentation (build instructions)

Like the tests, the documentation requires the datasets to be downloaded into the correct folders to execute the 
examples.
Therefore, we can not build the docs automatically on RTD.
Instead we host the docs via github pages.
The HTML source can be found in the `gh-pages` branch of this repo.

To make the deplowment as easy as possible, we "mounted" the `gh-pages` branch as a submodule in the `docs/_build/html`
folder.
Hence, before you attempt to build the docs, you need to initialize the submodule.

```
git submodule update --init --recursive
```

After that you can run `poe docs` to build the docs and then `poe upload_docs` to push the changes to the gh-pages
branch.
We will always just update a single commit on the gh-pages branch to keep the effective file size small.

**WARNING: ** Don't delete the `docs/_build` folder manually or by running the sphinx make file!
This will delete the submodule and might cause issues.
The `poe` task is configured to clean all relevant files in the `docs/_build` folder before each run.
