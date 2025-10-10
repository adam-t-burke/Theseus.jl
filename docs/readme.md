# Theseus docs — developer notes

This file documents how to (re)build the documentation locally and what the GitHub Actions workflow does automatically.

## Build docs locally

From the repository root (recommended)
- Builds docs using the docs environment and the docs/make.jl script:
  julia --project=docs docs/make.jl

Or from the docs directory
- Activate the docs environment and run make.jl:
  cd docs
  julia --project=. make.jl

If you need to recreate the docs environment (clean install)
- Remove the old manifest (optional, helpful when things are inconsistent):
  rm docs/Manifest.toml
- Then run:
  julia --project=docs -e 'using Pkg; Pkg.develop(path=".."); Pkg.add("Documenter"); Pkg.instantiate()'
- Finally build:
  julia --project=docs docs/make.jl

Serve the built site locally
- After a successful build, serve the site with live reload:
  julia --project=docs -e 'using Documenter; Documenter.serve("docs/build"; host="127.0.0.1", port=8000, live=true)'
- Open http://127.0.0.1:8000 in your browser.

Notes
- Running make.jl will: update/create docs/Project.toml (copy_deps), activate the docs env, ensure Documenter and the local package are available (Pkg.develop(path="..")), instantiate the environment, then run makedocs.
- If you change dependencies in the main Project.toml, re-run the recreate steps above to refresh the docs environment.

## What happens automatically on GitHub

- The workflow at `.github/workflows/docs.yml` is triggered on pushes to `main`/`master` and on manual dispatch.
- The CI job does roughly the same steps as `make.jl`: it builds the docs (using `julia --project=docs docs/make.jl`) and then publishes `docs/build` to the `gh-pages` branch using the actions-gh-pages action.
- After a successful run the site will be available via GitHub Pages, typically at:
  `https://<github-username>.github.io/Theseus.jl/` (set `baseurl` in `make.jl` if needed)
- On failure check the Actions tab for logs. Common failures:
  - Missing dependencies in the docs env — inspect the workflow log and run the recreate steps locally if needed.
  - Documenter errors about missing docs — add `@autodocs` / `@docs` entries in `docs/src` pages or relax checks.

## Troubleshooting tips

- If Documenter complains about docstrings not included:
  - Add an `@autodocs` block in `docs/src/api.md` or list the symbols explicitly with `@docs`.
- If Pkg errors about missing sources or manifests:
  - Delete `docs/Manifest.toml` and re-run the recreate commands above.
- Use the Actions > build-and-deploy run logs for the deployed workflow to see exact errors.

## Helpful commands summary

- Build: julia --project=docs docs/make.jl
- Clean & recreate docs env:
  rm docs/Manifest.toml
  julia --project=docs -e 'using Pkg; Pkg.develop(path=".."); Pkg.add("Documenter"); Pkg.instantiate()'
- Serve: julia --project=docs -e 'using Documenter; Documenter.serve("docs/build"; host="127.0.0.1", port=8000, live=true)'
