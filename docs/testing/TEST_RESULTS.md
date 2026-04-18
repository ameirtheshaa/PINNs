# Automated test results

This file records outcomes of **`pytest`** runs intended for CI and agent verification. Update it whenever the suite meaningfully changes or after fixing failures.

## How to run

```bash
# Optional: use the Cloud Agent venv from .cursor/install.sh
source ~/.pinns/venv/bin/activate

cd /path/to/PINNs
pip install -r requirements.txt
pytest tests/ -q
```

## Latest run

| Field | Value |
|--------|--------|
| Date (UTC) | 2026-04-18 15:15 |
| Git revision | *(fill with `git rev-parse --short HEAD` when updating)* |
| Command | `pytest tests/ -q` |
| Result | PASS |
| Notes | 13 tests; active `src/` stack and root shims only. `deprecated/` not executed. |

### Console summary

```
.............                                                            [100%]
13 passed in 5.24s
```

---

## Scope and limits

- **Included:** Repository root shims, `main('smoke_env_check_cli', ...)` with local-safe config overrides, import of core `src/` modules (via package names under `src.*`), minimal `PINN` forward pass on CPU.
- **Not included:** Full CFD data pipelines, multi-hour training, Paraview Python bindings, or every script under `deprecated/` (unless explicitly added to tests later).
