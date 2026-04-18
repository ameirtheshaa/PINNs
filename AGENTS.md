# Agent instructions (Cursor and automation)

Follow these rules for every task in this repository.

## Scope

- Prefer changing `src/` for real logic; root files like `definitions.py` are **shims** unless you are fixing import wiring.
- Do not delete or rewrite `deprecated/` unless the user explicitly asks.

## After you add or change code

1. **README.md** — If you introduced a new script, module, command, config pattern, or workflow, add a short subsection under the relevant heading (Installation, Running, Repository structure, Testing). Keep it accurate; remove stale commands.
2. **FILE_LAYOUT.md** — If you added directories or moved entry points, update `docs/repository/FILE_LAYOUT.md`.
3. **WORK_LOG.md** — Append one dated entry to `docs/repository/WORK_LOG.md`: what changed, why, and what you verified (tests/commands).

## Testing and documentation of results

- Run **`pytest`** from the repository root (with the project venv activated if you use one).
- Update **`docs/testing/TEST_RESULTS.md`** with:
  - date,
  - git revision (`git rev-parse --short HEAD`),
  - command run,
  - pass/fail summary,
  - any skipped or known gaps (e.g. deprecated tree not executed).

If tests fail, fix or narrow scope and document the actual outcome—do not claim green without command output.

## Cursor Cloud agents

- Environment bootstrap lives in `.cursor/environment.json` and `.cursor/install.sh`.
- Cloud sessions run the install/update script before work starts; keep it **idempotent**.

## Communication style

- Prefer small, focused commits with messages that state intent.
- Do not commit secrets or large binary artifacts unless the project already tracks them.
