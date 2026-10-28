# Progress

## YOU must do

## Step A0 — Project rules and base repository setup
- Date: 2026-10-28 (Asia/Singapore).
- Feature/task IDs touched: Step A0 only; no feature work, brainstorming or planning.
- Done: added project rules, exact cluster rules and job template, decision template, Python 3.11 src-layout packaging, pytest/ruff configuration, package placeholders and base directories.
- Validation: Python 3.11.18; `.venv/bin/python -m pytest` -> 1 passed in 0.01s; `.venv/bin/ruff check .` -> All checks passed!; `.venv/bin/ruff format --check .` -> 29 files already formatted. Cluster rules and tool routing compared verbatim with the supplied request (Markdown code fences excluded).
- Commands the user must run: none for Step A0.
- Open questions: confirm whether `ssh soc` works inside WSL or ssh/scp must run from PowerShell using `\\wsl$` paths. The existing constitution remains an unfilled Spec Kit template.
- Exact next message to send: "Step A0 is complete. My SSH setup is: [choose: ssh soc works inside WSL / I run ssh/scp from PowerShell; give me PowerShell commands with \\wsl$ paths]. Wait for my next project instruction."
- Repository: created `main` from the existing `master` commit for this setup; pre-existing changes under `.agents/` and `.specify/` are preserved and excluded from the setup commit.
