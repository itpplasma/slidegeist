# Contributing to Slidegeist

Thank you for your interest in contributing to Slidegeist!

## Development Setup

### Prerequisites

- Python ≥ 3.10
- FFmpeg installed and available in PATH
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/krystophny/slidegeist.git
cd slidegeist

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=slidegeist --cov-report=html

# Run specific test file
pytest tests/test_export.py

# Run with verbose output
pytest -v
```

## Code Quality

We use several tools to maintain code quality:

### Linting

```bash
# Run Ruff linter
ruff check slidegeist/

# Auto-fix issues
ruff check --fix slidegeist/
```

### Type Checking

```bash
# Run mypy type checker
mypy slidegeist/
```

### Formatting

```bash
# Check formatting
ruff format --check slidegeist/

# Auto-format code
ruff format slidegeist/
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure quality checks pass**
   ```bash
   pytest
   ruff check slidegeist/
   mypy slidegeist/
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a pull request on GitHub

5. **PR Requirements**
   - All tests pass
   - Code follows style guidelines
   - Documentation is updated
   - Meaningful commit messages

## Coding Guidelines

- Follow PEP 8 style guide
- Use type hints for function arguments and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Add logging for important operations
- Handle errors gracefully with helpful messages

## Project Structure

```
slidegeist/
├── slidegeist/          # Main package
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── ffmpeg.py       # FFmpeg wrapper
│   ├── slides.py       # Slide extraction
│   ├── transcribe.py   # Audio transcription
│   ├── export.py       # SRT export
│   └── pipeline.py     # Main orchestration
├── tests/              # Test suite
├── pyproject.toml      # Project configuration
└── README.md           # User documentation
```

## Reporting Issues

When reporting bugs, please include:

- Slidegeist version (`slidegeist --version`)
- Python version
- Operating system
- FFmpeg version (`ffmpeg -version`)
- Minimal reproduction steps
- Error messages and stack traces

## Feature Requests

We welcome feature suggestions! Please:

- Check existing issues first
- Describe the use case
- Explain expected behavior
- Consider implementation complexity

## Questions?

Feel free to open an issue for questions or discussion.

## Release Process

Slidegeist uses CalVer versioning with the format YYYY.MM.DD.

### Creating a Release

1. **Ensure all tests pass locally**
   ```bash
   pytest tests/test_download.py tests/test_export.py tests/test_ffmpeg.py -v
   mypy slidegeist/download.py slidegeist/cli.py --strict
   ```

2. **Update pyproject.toml version**
   ```bash
   # Update version to CalVer format, e.g., 2025.10.22
   vim pyproject.toml
   ```

3. **Commit version bump**
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 2025.10.22"
   git push origin main
   ```

4. **Create and push tag**
   ```bash
   git tag v2025.10.22
   git push origin v2025.10.22
   ```

5. **Automated workflow**
   - GitHub Actions will automatically build the package
   - Create a GitHub release with release notes
   - Publish to PyPI using trusted publishing

### PyPI Setup

Before the first release, configure PyPI trusted publishing:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - PyPI Project Name: `slidegeist`
   - Owner: `krystophny`
   - Repository name: `slidegeist`
   - Workflow name: `release.yml`
   - Environment name: leave blank

### Version Format

- **Releases**: YYYY.MM.DD (e.g., 2025.10.22)
- **Development**: Use dev suffix in pyproject.toml between releases

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
