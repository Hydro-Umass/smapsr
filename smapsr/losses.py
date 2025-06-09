import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from functools import partial

def mse_loss(img1, img2, mask):
    batch_size = img1.shape[0]
    mask = _prepare_mask(mask, batch_size)
    return jnp.sum(mask * (img1 - img2)**2) / jnp.sum(mask)

def mae_loss(img1, img2, mask):
    batch_size = img1.shape[0]
    mask = _prepare_mask(mask, batch_size)
    return jnp.sum(jnp.abs(img1 - img2) * mask) / jnp.sum(mask)

def gradient_loss(img1, img2):
    dy1, dx1 = jnp.gradient(img1, axis=(1, 2))
    dy2, dx2 = jnp.gradient(img2, axis=(1, 2))
    return jnp.mean((dy2 - dy1)**2) + jnp.mean((dx2 - dx1)**2)

def gaussian_kernel(size=11, sigma=1.5):
    """Generates a 2D Gaussian kernel."""
    x = jnp.arange(-size // 2 + 1, size // 2 + 1)
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return jnp.outer(kernel, kernel)

def _prepare_mask(mask, batch_size):
    """Ensure mask is 3D (B,H,W). If input is 2D, broadcast to batch."""
    if mask.ndim == 2:
        return jnp.broadcast_to(mask, (batch_size, *mask.shape))
    elif mask.ndim == 3:
        return mask
    else:
        raise ValueError(f"Mask must be 2D (H,W) or 3D (B,H,W), got {mask.ndim}D")

def masked_ssim(img1, img2, mask, max_val=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    Computes SSIM for batched single-channel images with missing data.

    Args:
        img1: First image batch (B,H,W)
        img2: Second image batch (B,H,W)
        mask: Binary mask (H,W) or (B,H,W) where 1=valid, 0=missing
        max_val: Maximum pixel value
        kernel_size, sigma: Gaussian kernel parameters
        k1, k2: SSIM stability constants

    Returns:
        SSIM values for each image in batch (shape [B])
    """
    # Validate inputs and prepare mask
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match, got {img1.shape} vs {img2.shape}")
    batch_size = img1.shape[0]
    mask = _prepare_mask(mask, batch_size)
    # Convert inputs and handle NaNs
    img1 = jnp.where(jnp.isnan(img1), 0.0, img1.astype(jnp.float32))
    img2 = jnp.where(jnp.isnan(img2), 0.0, img2.astype(jnp.float32))
    mask = mask.astype(jnp.float32)
    # Generate kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Constants
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    # Vectorized convolution function
    def conv_fn(x, m):
        x_masked = (x * m)[..., None]  # Add channel dim
        kernel_3d = kernel[..., None]
        valid_mask = convolve(m[..., None], kernel_3d, mode='same')[..., 0]
        conv_result = convolve(x_masked, kernel_3d, mode='same')[..., 0]
        return conv_result / jnp.maximum(valid_mask, 1e-12)
    # Compute statistics
    mu1 = jax.vmap(conv_fn)(img1, mask)
    mu2 = jax.vmap(conv_fn)(img2, mask)
    # Compute SSIM components
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = jax.vmap(conv_fn)(img1**2, mask) - mu1_sq
    sigma2_sq = jax.vmap(conv_fn)(img2**2, mask) - mu2_sq
    sigma12 = jax.vmap(conv_fn)(img1 * img2, mask) - mu1_mu2
    # Compute SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    # Average over valid regions
    valid_pixels = jax.vmap(conv_fn)(mask, jnp.ones_like(mask))  # Convolve mask with kernel
    return jnp.sum(ssim_map * mask, axis=(-2, -1)) / jnp.maximum(jnp.sum(mask, axis=(-2, -1)), 1e-12)

def masked_ssim_loss(img1, img2, mask, **kwargs):
    """SSIM loss (1 - SSIM) for batched inputs."""
    return 1 - jnp.mean(masked_ssim(img1, img2, mask, **kwargs))
