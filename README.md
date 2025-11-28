# Tesseract Hackathon Template

A ready-to-use template for building projects with [tesseract-core](https://github.com/pasteurlabs/tesseract-core) and [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation.

## Overview

This template shows how to:
- Build and serve tesseracts locally
- Call tesseracts separately
- Compose tesseracts into a pipeline

### Included Tesseracts

**1. scaler**
- Scales input vectors by a given factor

**2. dotproduct**
- Computes dot product between two vectors
- Calculates cosine similarity

## Quick Start

### Prerequisites

- Python 3.10 or higher
- tesseract-core and tesseract-jax installed
- JAX installed

### Installation

1. **Clone or use this template**
   ```bash
   git clone <your-repo-url>
   cd tesseract-hackathon-template
   ```

2. **Set up environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Build tesseracts**
   ```bash
   ./buildall.sh
   ```

4. **Run the demo**
   ```bash
   python main.py
   ```

## What the Demo Shows

Running `python main.py` demonstrates two paths:

**Path 1: Calling tesseracts separately**
- Load each tesseract
- Call scaler twice (for two vectors)
- Call dotproduct once (on scaled vectors)

**Path 2: Composing tesseracts together**
- Wrap tesseract calls in a single jax function
- Create a reusable pipeline
- Get the same result with cleaner code

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
