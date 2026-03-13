# Contributing to Agent Platform

Thank you for your interest in contributing! We welcome help in making this the most powerful local-first agent orchestrator.

## How Can I Contribute?

### Reporting Bugs
- Use the GitHub Issue Tracker.
- Provide a clear description and steps to reproduce.

### Suggesting Enhancements
- Open an issue titled "[Enhancement] ..."
- Describe the feature and why it would be useful.

### Pull Requests
1. Fork the repo.
2. Create a new branch (`feature/your-feature`).
3. Ensure all tests pass (`pytest`).
4. Submit a PR.

## Development Setup
1. Standard installation (see README).
2. Run tests: `pytest`.
3. Keep logic modular: add new LLM providers to `llm/` and new tools to `plugins/`.

## Coding Standards
- Use type hints wherever possible.
- Functional changes must include a corresponding test in `tests/`.
- Document new tools clearly in their docstrings.
