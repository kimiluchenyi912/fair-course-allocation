# Fair Course Allocation

## Configuration validation

Run the Version 1 configuration and template validator:

```bash
python -m src.validation
```

The validator checks `data/config/` and `data/templates/` before synthetic
request generation or allocation algorithms are run.

To treat current TPHS baseline deviations as errors instead of warnings:

```bash
python -m src.validation --strict-policy
```
