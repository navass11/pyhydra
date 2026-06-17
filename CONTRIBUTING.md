# Contributing

Contributions are welcome. Please open an issue before submitting a pull request
to discuss what you would like to change.

## Development setup

```bash
git clone https://github.com/navass11/pyhydra.git
cd pyhydra
pip install -e ".[geo]"
pip install pytest
```

## Running tests

```bash
pytest tests/
```

## Code style

- Follow PEP 8
- Type hints where practical
- No docstrings required for internal helpers; public API functions should have
  a one-line summary and parameter descriptions
