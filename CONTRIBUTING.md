# Contributing

1. Open an issue before starting a non-trivial change.
2. Set up and run locally:
   ```bash
   uv run uvicorn app:app --reload
   ```
3. Before opening a PR, run these checks:
   ```bash
   uvx ruff check .
   uv run python core.py   # self-check for the audio/text helpers
   ```
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` and so on), and keep each PR to one change.

Never commit audio of real people. `data/` is ignored by git for this reason.
