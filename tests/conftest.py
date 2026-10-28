"""The tests are tiny; run them on CPU so they do not depend on the GPU/cuDNN setup."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
