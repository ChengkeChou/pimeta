# Releasing

The procedure is short and every step has a failure mode that has already
happened once. It is written down so the next release does not re-discover them.

## 0. Before anything: the tree is boring

```bash
git status --short          # empty
git log --oneline -1 origin/main   # == HEAD
```

A release built from a dirty tree is a release nobody can reproduce, and a tag
pushed from a branch that is behind the remote names the wrong commit.

## 1. The suite, the linter, the types, the docs

```bash
python -m pytest tests -q            # ~14 min; expect 0 failed, 6 skipped
ruff check src tests tools
python -m mypy src/pimeta            # the CI venv, not the anaconda one
python tools/gen_api_docs.py         # then re-run `git status`: docs/ must be clean
```

`gen_api_docs.py` regenerating nothing is the proof that the public surface did
not move. If it writes pages, the change is a documentation change and belongs
in the changelog.

## 2. The version and the changelog

* `pyproject.toml`: `version = "X.Y.Z"`.
* `CHANGELOG.md`: the `[Unreleased]` section becomes `## [X.Y.Z] — YYYY-MM-DD`,
  and the link block at the bottom gains the new tag URL.

Do not bump the version in a commit that also changes behaviour — the diff of
the version commit should be the version, the changelog and nothing else. That
is what makes "which code is in the wheel" answerable from `git show`.

## 3. Build, and prove the artefact

```bash
cd ..                                  # NOT inside the repo: a local ./build/ shadows the build package
python -m build pimeta --outdir pimeta/dist
```

Then install the **wheel** into a fresh venv and reproduce a pinned number:

```bash
python -m venv /tmp/check
/tmp/check/bin/python -m pip install pimeta/dist/pimeta-X.Y.Z-py3-none-any.whl
/tmp/check/bin/python -c "
import numpy as np, pimeta as ma
d = np.genfromtxt('pimeta/tests/data/dynhet_spline.csv', delimiter=',', names=True)
fit = ma.tau_trend(d['yi'], d['vi'], d['year'], tau='spline')
print(ma.__version__, round(fit.sigma_s, 6), round(fit.edf, 6))
"
```

The pinned values are `4.121267` and `1.532920` (`tests/test_dynhet.py`,
`SPLINE_SIGMA_S` / `SPLINE_EDF`). A wheel that installs but does not reproduce a
pinned number is not a release candidate; it is a wheel.

Two things that cost an hour the first time: `python -m build` from inside the
repository imports the local `./build/` directory instead of the build package,
and `--no-build-isolation` fails because the test venv has no `setuptools`.
Both are avoided by building from the parent directory.

## 4. Tag and push

```bash
git tag -a vX.Y.Z -m "pimeta X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

The push to this remote fails intermittently; retry in a loop and check
`git ls-remote --tags origin vX.Y.Z` rather than trusting the exit code of one
attempt.

### When the tag already exists and points somewhere else

This happened with `v1.1.0`: the remote already had a tag of that name from an
earlier build, so every push was rejected and `ls-remote` kept showing the old
object. Confirm before assuming:

```bash
git rev-parse vX.Y.Z            # local tag object
git ls-remote --tags origin vX.Y.Z   # remote tag object
```

If the two differ, a remote tag exists that does not name the release. There are
two honest responses and no third:

* **Do not move it.** Release the content under the next patch version
  (`v1.1.0` → `v1.1.1`) and say so in the changelog. This is what was done, and
  it is the safe answer when anything may already have been built from the old
  tag.
* **Replace it**, only with the maintainer's explicit agreement, because
  deleting a remote tag is destructive and anything that pinned it breaks:

  ```bash
  git push origin :refs/tags/vX.Y.Z   # destructive
  git push origin vX.Y.Z
  ```

Never force-push a tag silently, and never re-point one to a commit whose
content was not tested at that version number.

## 5. Upload

```bash
twine upload dist/pimeta-X.Y.Z*
```

Needs a PyPI API token; the maintainer does this step or supplies the token. The
artefacts in `dist/` are the ones built in step 3 and verified in step 3 — do not
rebuild before uploading.

## 6. Afterwards

* `CHANGELOG.md` gains a fresh empty `## [Unreleased]` section.
* Check that the GitHub release page links the changelog entry and, if the
  measurement numbers moved, that `tests/reference/coverage_*.json` and the
  docstrings were updated together (the guard in
  `tests/test_coverage_baselines.py` fails if they were not).
