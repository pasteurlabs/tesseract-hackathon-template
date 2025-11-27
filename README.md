# Tesseract Hackathon Template

A ready-to-use template for building projects with [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation.

## Overview

This template shows how to:
- Build and serve tesseracts locally
- Call tesseracts separately
- Compose tesseracts into a pipeline

### Included Tesseracts

**1. scaler**
- Scales input vectors by a given factor
- Computes L2 norm (magnitude) of scaled vectors

**2. dotproduct**
- Computes dot product between two vectors
- Calculates cosine similarity

## Quick Start

### Prerequisites

- Python 3.9 or higher
- tesseract-core and tesseract-jax installed
- JAX installed (CPU or GPU version)

### Installation

1. **Clone or use this template**
   ```bash
   git clone <your-repo-url>
   cd tesseract-hackathon-template
   ```

2. **Set up environment**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

3. **Build tesseracts**
   ```bash
   ./buildall.sh
   ```

4. **Run the demo**
   ```bash
   python main.py
   ```

## Project Structure

```
tesseract-hackathon-template/
├── tesseracts/
│   ├── scaler/
│   │   ├── tesseract_api.py
│   │   ├── tesseract_config.yaml
│   │   └── tesseract_requirements.txt
│   └── dotproduct/
│       ├── tesseract_api.py
│       ├── tesseract_config.yaml
│       └── tesseract_requirements.txt
│
├── main.py              # Demo script
├── setup.sh             # Initial setup
├── buildall.sh          # Build all tesseracts
└── requirements.txt     # Python dependencies
```

## What the Demo Shows

Running `python main.py` demonstrates two paths:

**Path 1: Calling tesseracts separately**
- Load each tesseract
- Call scaler twice (for two vectors)
- Call dotproduct once (on scaled vectors)

**Path 2: Composing tesseracts together**
- Wrap tesseract calls in a single function
- Create a reusable pipeline
- Get the same result with cleaner code

## Usage Examples

### Basic Usage

```python
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import jax.numpy as jnp

# Load and serve
scaler = Tesseract.from_image("scaler")
scaler.serve()

# Apply tesseract
inputs = {
    "data": {
        "vector": jnp.array([1.0, 2.0, 3.0]),
        "scale_factor": 2.0
    }
}
outputs = apply_tesseract(scaler, inputs)

# Cleanup
scaler.teardown()
```

### Composing Tesseracts

```python
def scale_and_similarity(v_a, v_b, scale):
    """Composed pipeline"""
    out_a = apply_tesseract(scaler, {
        "data": {"vector": v_a, "scale_factor": scale}
    })
    out_b = apply_tesseract(scaler, {
        "data": {"vector": v_b, "scale_factor": scale}
    })

    sim = apply_tesseract(dotproduct, {
        "data": {
            "vector_a": out_a['result']['scaled'],
            "vector_b": out_b['result']['scaled']
        }
    })

    return sim['result']['cosine_similarity']
```

## Creating Your Own Tesseracts

### 1. Create Directory Structure

```bash
mkdir -p tesseracts/my_tesseract
cd tesseracts/my_tesseract
```

### 2. Create Required Files

**tesseract_config.yaml**
```yaml
name: my_tesseract
version: 1.0.0
description: "Your tesseract description"

build:
  target_platform: native
```

**tesseract_requirements.txt**
```
jax[cpu]
equinox
```

**tesseract_api.py**
```python
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any

class InputSchema(eqx.Module):
    # Define your input schema
    pass

class OutputSchema(eqx.Module):
    # Define your output schema
    pass

def apply(inputs: dict[str, Any]) -> dict[str, Any]:
    # Your implementation
    pass
```

### 3. Build and Test

```bash
cd ../..
tesseract build tesseracts/my_tesseract

# Test it
python -c "
from tesseract_core import Tesseract
tess = Tesseract.from_image('my_tesseract')
tess.serve()
tess.teardown()
"
```

## Development Tips

### Rebuilding After Changes

```bash
./buildall.sh
python main.py
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'tesseract_core'`
- **Solution**: Activate the virtual environment: `source venv/bin/activate`

**Issue**: `tesseract: command not found`
- **Solution**: Install tesseract-core from [official repo](https://github.com/pasteurlabs/tesseract-core)

**Issue**: Tesseract build fails
- **Solution**: Check that all three files (api, config, requirements) exist and are valid

## Resources

- [Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax)
- [Tesseract-Core Documentation](https://github.com/pasteurlabs/tesseract-core)
- [JAX Documentation](https://jax.readthedocs.io/)

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [Tesseract](https://github.com/pasteurlabs) by Pasteur Labs.

---

**Happy Hacking!** 🚀
