# GitHub Pages

This repository can publish MkDocs output through GitHub Actions.

## Required repository settings

1. Commit `mkdocs.yml`, `requirements-docs.txt`, the `docs/` folder, and `.github/workflows/docs.yml`.
2. Push the changes to GitHub.
3. Open **Settings** → **Pages**.
4. Set **Source** to **GitHub Actions**.
5. Run the workflow or push to the default branch.

The published site will use the repository Pages URL configured in `mkdocs.yml`.

## Important edits before publishing

Replace these placeholders in `mkdocs.yml`:

```yaml
site_url: https://<your-github-username>.github.io/Foo/
repo_url: https://github.com/<your-github-username>/Foo
repo_name: <your-github-username>/Foo
```
