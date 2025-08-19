# Contributing to COCO Utils

Thank you for your interest in contributing to COCO Utils! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package versions)
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Use cases and examples
- Any relevant mockups or diagrams

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Set up your development environment:**
   ```bash
   git clone https://github.com/felipe-parodi/coco_utils.git
   cd coco_utils
   pip install -e .[dev]
   pre-commit install
   ```

3. **Make your changes:**
   - Write clear, concise commit messages
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run quality checks:**
   ```bash
   # Format code
   black coco_utils tests
   isort coco_utils tests
   
   # Run linting
   flake8 coco_utils tests
   mypy coco_utils
   
   # Run tests
   pytest tests -v --cov=coco_utils
   ```

5. **Submit your pull request:**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all CI checks pass

## Development Guidelines

### Code Style

- We use [Black](https://github.com/psf/black) for code formatting (line length: 100)
- We use [isort](https://github.com/PyCQA/isort) for import sorting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function signatures
- Write Google-style docstrings for all public functions and classes

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage (aim for >80%)
- Use pytest for testing
- Place tests in the `tests/` directory with `test_` prefix

### Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Add examples for new functionality
- Keep documentation clear and concise

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Example: `feat: add support for filtering annotations by category`

## Project Structure

```
coco_utils/
├── coco_utils/          # Main package code
│   ├── __init__.py
│   ├── coco_file_utils.py
│   ├── coco_labels_utils.py
│   ├── coco_viz_utils.py
│   ├── exceptions.py
│   └── logger.py
├── tests/               # Test files
│   ├── conftest.py
│   ├── test_coco_file_utils.py
│   ├── test_coco_labels_utils.py
│   └── test_coco_viz_utils.py
├── examples/            # Example scripts and notebooks
├── docs/                # Documentation
└── .github/             # GitHub Actions workflows
```

## Getting Help

If you need help, you can:

- Open an issue on GitHub
- Check existing documentation
- Review closed issues for similar problems

## Recognition

Contributors will be recognized in the project's README. We appreciate all contributions, no matter how small!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.