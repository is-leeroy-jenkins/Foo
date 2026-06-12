# GitHub Pages

Foo documentation can be published as a static documentation site with MkDocs and GitHub Pages. The
recommended deployment path is to build the MkDocs site with GitHub Actions and publish the
generated static site as a GitHub Pages deployment.

GitHub supports publishing Pages from a branch or through a custom GitHub Actions workflow. For a
MkDocs site, GitHub Actions is the better fit because MkDocs must build the Markdown source into a
static `site/` directory before publishing. GitHub’s documentation states that a Pages site can be
published from a branch or by using a GitHub Actions workflow, and recommends Actions when the site
uses a build process other than Jekyll.

## 🧭 Purpose

The purpose of this page is to document how Foo’s MkDocs documentation should be published to GitHub
Pages.

The publishing workflow should:

* Keep Markdown source files in `docs/`.
* Keep `mkdocs.yml` in the repository root.
* Keep generated documentation images in `docs/images/`.
* Build the site with MkDocs.
* Publish the generated static site with GitHub Actions.
* Avoid manually committing the generated `site/` directory.
* Keep the published documentation synchronized with the repository.

## 🧱 Required Repository Structure

A clean Foo documentation structure should look like this:

```text
Foo/
├─ .github/
│  └─ workflows/
│     └─ docs.yml
├─ docs/
│  ├─ images/
│  │  ├─ foo-architecture.png
│  │  └─ foo-classes.png
│  ├─ api/
│  │  ├─ index.md
│  │  ├─ core.md
│  │  ├─ data.md
│  │  ├─ fetchers.md
│  │  ├─ generators.md
│  │  ├─ loaders.md
│  │  ├─ models.md
│  │  ├─ scrapers.md
│  │  └─ writers.md
│  ├─ index.md
│  ├─ app.md
│  ├─ architecture.md
│  ├─ user-guide.md
│  ├─ loading.md
│  ├─ fetching-scraping.md
│  ├─ generation.md
│  ├─ data-management.md
│  ├─ output.md
│  ├─ development.md
│  └─ github-pages.md
├─ mkdocs.yml
├─ requirements.txt
├─ requirements-docs.txt
├─ app.py
├─ core.py
├─ data.py
├─ fetchers.py
├─ generators.py
├─ loaders.py
├─ models.py
├─ scrapers.py
└─ writers.py
```

The generated `site/` directory should not normally be committed. GitHub Actions should build it
during deployment.

## ⚙️ MkDocs Configuration

The root `mkdocs.yml` file controls the documentation site.

At minimum, confirm these values are correct:

```yaml
site_name: Foo Documentation
site_description: Documentation for Foo, a Streamlit and Python application for information acquisition and analysis workflows.
site_author: Terry D. Eppler
site_url: https://<github-username>.github.io/Foo/
repo_url: https://github.com/<github-username>/Foo
repo_name: <github-username>/Foo
```

Replace:

```text
<github-username>
```

with the actual GitHub account or organization name.

For example:

```yaml
site_url: https://is-leeroy-jenkins.github.io/Foo/
repo_url: https://github.com/is-leeroy-jenkins/Foo
repo_name: is-leeroy-jenkins/Foo
```

The `site_url` value matters because it affects canonical links, social links, sitemap output, and
some theme behavior.

## 🧭 Recommended Navigation

The navigation should include the manual application pages and the generated API reference pages.

Recommended `nav` section:

```yaml
nav:
  - Home: index.md
  - Application: app.md
  - Architecture: architecture.md
  - User Guide: user-guide.md
  - Loading Data: loading.md
  - Fetching and Scraping: fetching-scraping.md
  - Generation: generation.md
  - Data Management: data-management.md
  - Output: output.md
  - Development: development.md
  - GitHub Pages: github-pages.md
  - API Reference:
      - Overview: api/index.md
      - Core: api/core.md
      - Data: api/data.md
      - Fetchers: api/fetchers.md
      - Generators: api/generators.md
      - Loaders: api/loaders.md
      - Models: api/models.md
      - Scrapers: api/scrapers.md
      - Writers: api/writers.md
```

Every page listed in `nav` must exist under `docs/`. If a Markdown file exists under `docs/` but is
not included in navigation, MkDocs may warn that the file exists but is not included in the
navigation.

## 📦 Documentation Dependencies

Documentation-specific dependencies should be installed from:

```text
requirements-docs.txt
```

That file should include MkDocs and the plugins or themes required to build the documentation site.

Typical documentation dependencies include:

```text
mkdocs
mkdocs-material
mkdocstrings[python]
```

If the build reports that signature formatting requires Black or Ruff, install one of them or add
one to `requirements-docs.txt`:

```text
black
```

or:

```text
ruff
```

Keep documentation dependencies separate from application runtime dependencies in
`requirements.txt`.

## 🧪 Build Locally First

Always build the documentation locally before pushing changes.

From the repository root:

```powershell
python -m pip install -r requirements-docs.txt
mkdocs build
```

For stricter validation:

```powershell
mkdocs build --strict
```

Use `--strict` before publishing because it turns warnings into build failures. That helps catch
missing images, missing navigation entries, broken links, malformed API documentation, and Griffe
parsing problems before GitHub Actions runs.

## 🖥️ Preview Locally

Run the local development server:

```powershell
mkdocs serve
```

Open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8000/
```

Review:

* Home page.
* Application page.
* Architecture page.
* User guide.
* Loading page.
* Fetching and scraping page.
* Generation page.
* Data management page.
* Output page.
* Development page.
* API reference pages.
* Image rendering.
* Navigation behavior.
* Search behavior.
* Code block formatting.

Do not publish until the local site renders cleanly.

## 🖼️ Images

Documentation images should live under:

```text
docs/images/
```

For Foo, the expected documentation images are:

```text
docs/images/foo-architecture.png
docs/images/foo-classes.png
```

Reference them from Markdown with relative paths.

From `docs/index.md`:

```markdown
![Foo Architecture](images/foo-architecture.png)
```

From `docs/architecture.md`:

```markdown
![Foo Architecture](images/foo-architecture.png)

![Foo Class Map](images/foo-classes.png)
```

From a file under `docs/api/`, use one directory up:

```markdown
![Foo Architecture](../images/foo-architecture.png)
```

If MkDocs reports that an image target is not found, confirm:

* The file exists under `docs/images/`.
* The filename matches exactly.
* The extension matches exactly.
* The link uses the correct relative path.
* The image was committed to Git.

## 🧾 Recommended GitHub Actions Workflow

Create this file:

```text
.github/workflows/docs.yml
```

Recommended workflow:

```yaml
name: Deploy documentation

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    name: Build MkDocs site
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install documentation dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements-docs.txt

      - name: Build documentation
        run: |
          mkdocs build --strict

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    name: Deploy to GitHub Pages
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
      - name: Deploy Pages artifact
        id: deployment
        uses: actions/deploy-pages@v4
```

This workflow builds the MkDocs site from source and deploys the generated `site/` directory as the
GitHub Pages artifact.

The workflow uses the built-in `GITHUB_TOKEN` permissions model rather than requiring a manually
created personal access token. GitHub recommends using the built-in `GITHUB_TOKEN` for API
authentication in GitHub Actions workflows and granting permissions with the workflow `permissions`
key.

## ⚙️ GitHub Repository Settings

After the workflow file is committed, configure GitHub Pages in the repository settings.

In the GitHub repository:

1. Open **Settings**.
2. Open **Pages** under **Code and automation**.
3. Under **Build and deployment**, select **GitHub Actions** as the source.
4. Save changes if prompted.
5. Push a commit to `main` or run the workflow manually.

GitHub’s Pages documentation describes the Pages settings area and the choice of publishing source.

## 🚀 Publishing Workflow

The normal publishing process is:

```text
Edit docs or source docstrings
        |
        v
Run mkdocs build --strict locally
        |
        v
Commit changes
        |
        v
Push to main
        |
        v
GitHub Actions builds MkDocs site
        |
        v
GitHub Actions deploys Pages artifact
        |
        v
GitHub Pages serves the documentation
```

After deployment, the site should be available at:

```text
https://<github-username>.github.io/Foo/
```

For example:

```text
https://is-leeroy-jenkins.github.io/Foo/
```

## 🧪 Manual Workflow Run

The workflow includes:

```yaml
workflow_dispatch:
```

That means the workflow can be run manually from GitHub.

To run it manually:

1. Open the repository on GitHub.
2. Select **Actions**.
3. Select **Deploy documentation**.
4. Select **Run workflow**.
5. Choose the branch.
6. Run the workflow.

Use this when you want to redeploy without making another commit.

## 🔎 Checking Deployment Status

After pushing or manually running the workflow:

1. Open the repository on GitHub.
2. Select **Actions**.
3. Select the latest **Deploy documentation** workflow run.
4. Confirm the `build` job passed.
5. Confirm the `deploy` job passed.
6. Open the deployment URL shown in the workflow summary or repository Pages settings.

If the site does not update immediately, wait briefly and refresh. GitHub Pages deployments are
usually fast but not always instant.

## ⚠️ Common Build Problems

| Problem                                                          | Likely Cause                                                                                          | Fix                                                                                  |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `Config value 'site_name': Required configuration not provided.` | `mkdocs.yml` is missing `site_name` or the workflow is not running from the repository root.          | Add `site_name` and confirm `mkdocs.yml` is in the root.                             |
| Page exists but is not in nav                                    | Markdown file exists under `docs/` but is not listed in `nav`.                                        | Add the page to `mkdocs.yml` navigation or remove the file.                          |
| Image target not found                                           | Markdown references an image that does not exist under `docs/` or uses the wrong relative path.       | Move image to `docs/images/` and fix the relative link.                              |
| API page fails during import                                     | mkdocstrings imports a module that has missing dependencies or import-time side effects.              | Install dependencies or avoid rendering import-unsafe modules directly.              |
| Griffe warnings                                                  | Malformed docstring sections or incorrect return documentation.                                       | Fix Google-style docstrings in the source file.                                      |
| Site shows README instead of MkDocs site                         | Pages source is still set to branch/root or the workflow is not deploying the MkDocs artifact.        | Set Pages source to GitHub Actions and confirm workflow deployment.                  |
| 404 after deployment                                             | Pages artifact may be missing a top-level `index.html`, deployment failed, or site URL/path is wrong. | Confirm `mkdocs build` produced `site/index.html` and the workflow deployed `site/`. |

GitHub notes that when a Pages site is published through GitHub Actions, the deployed artifact must
include the entry file at the top level of the artifact. For MkDocs, that means the workflow should
upload the generated `site/` directory containing `index.html`.

## 🧰 Common API Documentation Problems

Foo uses mkdocstrings to render API documentation from Python source files. That makes source
docstring quality part of the documentation build.

Common issues include:

| Warning Pattern                          | Likely Cause                                         | Fix                                                            |
| ---------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| Failed to get `name: description` pair   | Malformed Google-style `Args:` entry.                | Ensure each argument line starts with the real parameter name. |
| No type or annotation for returned value | Bad `Returns:` section or missing return annotation. | Add a return annotation and clean `Returns:` prose.            |
| Constructor return warning               | `__init__` has a `Returns:` section.                 | Remove constructor `Returns:` section.                         |
| Section parsing warning                  | Old NumPy-style or malformed headings.               | Use clean Google-style docstrings only.                        |
| Module import failure                    | mkdocstrings cannot import the module.               | Install dependency or avoid rendering import-unsafe modules.   |

Avoid these docstring patterns:

```text
Parameters:
-----------
Returns:
--------
Returns:
    None: This method does not return a value.
```

Prefer:

```text
Purpose:
Args:
Returns:
```

Only include `Returns:` when a function or method returns a meaningful value.

## 🧱 Should `app.py` Be in the API Reference?

`app.py` should be documented manually through `docs/app.md`.

Do not automatically render `app.py` with mkdocstrings unless it is refactored to be import-safe.

The reason is that `app.py` contains Streamlit runtime logic and session-state behavior. If
mkdocstrings imports it during the build, the import may execute UI code, assume Streamlit context,
require optional dependencies, or trigger runtime warnings.

Recommended approach:

```text
docs/app.md       Manual application documentation.
docs/api/*.md     Generated API documentation for import-safe modules.
```

If `app.py` is later refactored into import-safe functions or a `ui/` package, generated API
documentation can be reconsidered.

## 🛡️ Security and Secrets

Do not commit secrets to the repository.

Do not put secrets in:

* `mkdocs.yml`
* Markdown pages.
* GitHub Actions workflow files.
* API documentation examples.
* Output artifacts.
* Generated logs.
* Screenshots.
* Source comments.
* Error messages.

Use repository secrets or environment variables for values that must be available in GitHub Actions.

For documentation builds, avoid requiring live provider credentials unless the documentation build
genuinely needs them. API documentation should normally render from source docstrings without
calling providers.

## 📦 GitHub Pages Limits

GitHub Pages has documented operational limits, including a published site size limit and deployment
timeout limits. GitHub’s current documentation states that published Pages sites may be no larger
than 1 GB and Pages deployments time out if they take longer than 10 minutes.

For Foo, this means:

* Do not publish large generated datasets as part of the documentation site.
* Do not publish model files.
* Do not publish large raw outputs.
* Keep images optimized.
* Keep generated artifacts out of `docs/` unless they are intended documentation assets.
* Keep the documentation build fast.

## 🧪 Pre-Publish Checklist

Before pushing documentation changes:

* `mkdocs.yml` exists in the repository root.
* `site_name` is set.
* `site_url` matches the repository Pages URL.
* `repo_url` points to the correct GitHub repository.
* Every page in `nav` exists.
* Every intended public page is included in `nav`.
* Images are under `docs/images/`.
* Image links are relative and valid.
* `requirements-docs.txt` includes required MkDocs dependencies.
* API modules compile.
* Docstrings are Google-style and Griffe-safe.
* `mkdocs build --strict` succeeds locally.
* The generated `site/` directory contains `index.html`.
* The workflow file exists at `.github/workflows/docs.yml`.
* GitHub Pages source is set to GitHub Actions.

## ✅ Deployment Checklist

After pushing:

* GitHub Actions workflow starts.
* `build` job succeeds.
* `deploy` job succeeds.
* The deployment URL is shown in the workflow.
* The site opens at the expected Pages URL.
* Navigation renders correctly.
* API pages render correctly.
* Images render correctly.
* Search works.
* The site is not showing the repository README.
* The site is not returning 404.

## 🧭 Summary

Foo documentation should be published with MkDocs and GitHub Actions. Keep Markdown source in
`docs/`, keep `mkdocs.yml` in the repository root, build locally with `mkdocs build --strict`, and
deploy the generated `site/` artifact through GitHub Pages. This keeps the documentation source
reviewable, the API reference tied to source docstrings, and the published site synchronized with
the repository.
