# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
make venv

# Install package in editable mode
pip install -e .
```

### Code Quality
```bash
# Run linting and formatting
ruff check .
ruff format .

# Run type checking
mypy cods/

# Run tests
pytest tests/
pytest tests/od/test_cp.py  # Run specific test file
pytest tests/classif/       # Run classification tests
```

### Documentation
```bash
# Generate API documentation
./gen_docs.sh

# Build documentation site (if MkDocs is configured)
mkdocs serve
```

### Development Environment
- Python 3.8+ required
- Use pip for dependency management
- Dependencies in `requirements.txt` (runtime) and `requirements-dev.txt` (development)

## Code Architecture

### Overall Structure
CODS follows a hierarchical, task-oriented architecture with three main layers:

1. **Base Layer** (`cods/base/`): Abstract classes and common functionality
2. **Task Layers**: Specialized implementations for different computer vision tasks
   - `cods/classif/`: Conformal classification 
   - `cods/od/`: Conformal object detection
   - `cods/seg/`: Conformal segmentation (work in progress)

### Key Architectural Components

#### Base Classes (`cods/base/`)
- **`Model`**: Abstract base for all ML models with prediction save/load capabilities
- **`Conformalizer`**: Abstract base for conformal prediction with `calibrate()`, `conformalize()`, `evaluate()` methods
- **Data Structures**: `Predictions`, `Parameters`, `ConformalizedPredictions`, `Results`

#### Task-Specific Extensions
- **Classification**: Single prediction per image with class probabilities
- **Object Detection**: Complex multi-prediction per image with spatial reasoning and multiple conformalizers:
  - `LocalizationConformalizer`: Bounding box expansion
  - `ConfidenceConformalizer`: Objectness thresholding  
  - `ODClassificationConformalizer`: Class prediction sets
  - `ODConformalizer`: Orchestrates all three with multiple testing correction

#### Data Pipeline
```
Raw Model → Predictions → Parameters (calibration) → ConformalizedPredictions → Results
```

### Model Implementation
- **Model Agnostic**: Works with any PyTorch model through wrapper classes
- **Caching**: Built-in prediction saving/loading with hash-based caching
- **Device Management**: Consistent GPU/CPU handling
- **Supported Models**: DETR, YOLO for object detection; extensible for new architectures

### Conformal Prediction Implementation
- **Risk-Controlling Conformal Prediction (RCCP)** with sophisticated multi-aspect guarantees
- **Sequential Calibration**: Confidence → Matching → Localization → Classification  
- **Optimization**: Binary search and Gaussian Process optimizers for parameter tuning
- **Multiple Testing Correction**: Bonferroni correction across prediction aspects

## Development Guidelines

### Adding New Tasks
1. Inherit from base classes in `cods/base/`
2. Create task-specific `Model`, `Conformalizer`, and data structure classes
3. Implement required abstract methods: `build_predictions()`, `calibrate()`, `conformalize()`, `evaluate()`

### Adding New Models
1. Inherit from task-specific model classes (e.g., `ODModel` for object detection)
2. Implement model-specific prediction building logic
3. Ensure proper device handling and caching integration

### Testing
- Tests located in `tests/` with same structure as main package
- Use pytest for all testing
- Test files follow `test_*.py` naming convention

### Code Style
- Configured with Ruff for linting and formatting (line length: 79)
- MyPy for type checking with strict mode
- Docstring format: Configured in pyproject.toml