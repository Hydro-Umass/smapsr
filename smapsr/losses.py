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

def batch_correlation_masked(arr1, arr2, mask):
    """
    Calculate correlation between two JAX arrays along the batch dimension,
    only considering valid (masked) pixels.

    Args:
        arr1: JAX array of shape (B, H, W)
        arr2: JAX array of shape (B, H, W)
        mask: JAX array of shape (B, H, W) with 1 for valid pixels, 0 for invalid

    Returns:
        JAX array of shape (H, W) containing correlation coefficients for valid pixels
    """
    batch_size = arr1.shape[0]
    mask = _prepare_mask(mask, batch_size)
    B, H, W = arr1.shape
    # Reshape arrays to (B, H*W)
    arr1_flat = arr1.reshape(B, H * W)
    arr2_flat = arr2.reshape(B, H * W)
    mask_flat = mask.reshape(B, H * W)
    # Count valid samples per pixel location
    valid_count = jnp.sum(mask_flat, axis=0)  # Shape: (H*W,)
    # Avoid division by zero - set minimum count to 1
    valid_count = jnp.maximum(valid_count, 1)
    # Calculate masked means
    masked_sum1 = jnp.sum(arr1_flat * mask_flat, axis=0)
    masked_sum2 = jnp.sum(arr2_flat * mask_flat, axis=0)
    mean1 = masked_sum1 / valid_count
    mean2 = masked_sum2 / valid_count
    # Center the data (only for valid pixels)
    centered1 = (arr1_flat - mean1[None, :]) * mask_flat
    centered2 = (arr2_flat - mean2[None, :]) * mask_flat
    # Calculate masked covariance
    covariance = jnp.sum(centered1 * centered2, axis=0) / valid_count
    # Calculate masked standard deviations
    var1 = jnp.sum(centered1**2, axis=0) / valid_count
    var2 = jnp.sum(centered2**2, axis=0) / valid_count
    std1 = jnp.sqrt(var1)
    std2 = jnp.sqrt(var2)
    # Calculate correlation coefficient
    eps = 1e-8
    correlation = covariance / (std1 * std2 + eps)
    # Set correlation to 0 where there are no valid pixels
    correlation = jnp.where(jnp.sum(mask_flat, axis=0) > 1, correlation, 0.0)
    return correlation.reshape(H, W)

def corr_loss_masked(arr1, arr2, mask):
    """
    Masked correlation loss - only compute loss for valid regions.

    Args:
        arr1: JAX array of shape (B, H, W)
        arr2: JAX array of shape (B, H, W)
        mask: JAX array of shape (B, H, W) with 1 for valid pixels, 0 for invalid

    Returns:
        Scalar loss value
    """
    if mask is None:
        return correlation_loss_negative(arr1, arr2)
    # Get masked correlation matrix
    corr_matrix = batch_correlation_masked(arr1, arr2, mask)
    # Create spatial mask (1 where we have valid correlations)
    spatial_mask = (jnp.sum(mask, axis=0) > 1).astype(jnp.float32)
    # Apply loss function only to valid spatial locations
    loss_per_pixel = (1 - corr_matrix) * spatial_mask
    # Average over valid spatial locations
    total_loss = jnp.sum(loss_per_pixel)
    valid_pixels = jnp.sum(spatial_mask)
    return total_loss / jnp.maximum(valid_pixels, 1)
