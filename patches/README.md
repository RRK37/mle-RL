# Patches Directory

This directory contains modified versions of files from the `mle-dojo` submodule.

## Purpose

Since we don't have write access to the original MLE-Dojo repository, we maintain our customizations here. The `main.py` script adds this directory to the Python path **before** the submodule, so our patched versions take precedence.

## Patched Files

### `mledojo/utils.py`
- Enhanced `load_config()` to preserve agent settings from YAML config
- Updated `load_agent_config()` to search multiple paths (works from both root and submodule directories)

### `mledojo/agent/aide/buildup.py`
- Added support for applying agent configuration from the main config file
- Properly merges settings for steps, k_fold_validation, code generation, and search parameters

## How It Works

In `main.py`, we set the path order:
```python
sys.path.insert(0, 'patches')           # Our modifications (highest priority)
sys.path.insert(0, 'submodules/mle-dojo')  # Original submodule
```

This way, when Python imports `mledojo.utils`, it finds our patched version first.

## Updating

If the submodule is updated and conflicts arise:
1. Review changes in the original files
2. Merge necessary updates into our patched versions
3. Test thoroughly

## Alternative Approaches

If you prefer to contribute these changes upstream:
1. Fork the MLE-Dojo repository
2. Apply these patches to your fork
3. Update `.gitmodules` to point to your fork
4. Submit a pull request to the original repository
