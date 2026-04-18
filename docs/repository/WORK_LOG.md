# Working log (append-only)

Agents and maintainers append new entries at the **top** (below this line). Keep each entry short: date (UTC), author/agent summary, changes, verification.

---

## 2026-04-18 — Documentation, agent rules, pytest suite

- Added `docs/repository/FILE_LAYOUT.md`, `AGENTS.md`, this log, `docs/testing/TEST_RESULTS.md`.
- Added `tests/` (pytest): smoke env check, shim/config checks, PINN forward, `src.*` import matrix; `pytest.ini`, `requirements.txt` dev dep `pytest`.
- Updated `README.md` with pointers to layout, agents, tests, work log.
- Verification: `pytest tests/ -q` → 13 passed (see `docs/testing/TEST_RESULTS.md`).
