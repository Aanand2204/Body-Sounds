---
description: How to build and upload the package to PyPI
---

# PyPI Deployment Guide

Follow these steps to build and publish your `heart-murmur-analysis` package to PyPI.

### 1. Preparation
Ensure your `pyproject.toml` has the correct version. Every time you upload to PyPI, you **must** increment the version (e.g., from `0.4.1` to `0.4.2`).

### 2. Install Build Tools
You'll need `build` and `twine`. Use `python -m pip` to ensure they are installed in your active virtual environment.
```powershell
python -m pip install --upgrade build twine
```

### 3. Build the Package
Run this command from the root directory (`Body-Sound-Detection1-main`). This will create a `dist/` folder with `.whl` and `.tar.gz` files.
```powershell
python -m build
```

### 4. Upload to TestPyPI (Recommended)
It's best to verify everything is correct by uploading to TestPyPI first.
```powershell
python -m twine upload --repository testpypi dist/*
```
*Note: You will need a separate account on [test.pypi.org](https://test.pypi.org/).*

### 5. Upload to PyPI (Final)
Once verified, upload to the real PyPI.
```powershell
python -m twine upload dist/*
```

---

## Troubleshooting hatch-specific build
Since you are using `hatchling`, you can also use [hatch](https://hatch.pypa.io/) directly:
1. `pip install hatch`
2. `hatch build`
3. `hatch publish`
