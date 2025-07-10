import xarray as xr
import json
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from smapsr.train import prepare_data
from smapsr.models import NeuralODE
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

def predict(sl, sh, region, model, m, s):
    """Generate predictions using a trained NeuralODE model. This function processes spatio-temporal data through a NeuralODE model to generate predictions, handling data normalization and coordinate transformations.

    Parameters:
    - sl : xarray.DataArray Low-resolution time series data.
    - sh : xarray.DataArray High-resolution reference data for spatial coordinates.
    - region : list Region specification for data preparation ([xmin, ymax, xmax, ymin]).
    - model : NeuralODE Trained NeuralODE model.
    - m : float Mean value for data normalization.
    - s : float Standard deviation for data normalization.

    Returns:
    - xarray.DataArray Predicted values with proper coordinates and time dimension.
    """
    sm = []
    dt = []
    ts = jnp.linspace(0., 1., 100)
    for t in tqdm(sl.time.to_pandas()):
        slr, shr, slri, y0, ys = prepare_data(sl.sel(time=t), sh.isel(time=0), region)
        if slr.notnull().any():
            y0 = (y0 - m) / s
            out = jax.vmap(model, in_axes=(None, 0))(ts, y0)
            smout = xr.DataArray(out.reshape(shr.shape), dims=shr.dims, coords=shr.coords)
            smout = smout * s + m
            sm.append(smout)
            dt.append(t)
    dt = pd.DatetimeIndex(dt, name='time')
    return xr.concat(sm, dim=dt)

def ubrmse(obs, pred):
    """Calculate unbiased RMSE."""
    bias = np.mean(obs - pred)
    rmse = np.sqrt(np.mean((obs - pred)**2))
    return np.sqrt(rmse**2 - bias**2)

def split_into_tiles(data_array, tile_size):
    """Partition data array in tiles for training."""
    x_coords = data_array.x.data
    y_coords = data_array.y.data
    x_size, y_size = tile_size
    tile_id = 0
    tiles_dict = {}
    for i in range(0, len(x_coords), x_size):
        for j in range(0, len(y_coords), y_size):
            x_start, x_end = i, min(i + x_size, len(x_coords))
            y_start, y_end = j, min(j + y_size, len(y_coords))
            tile_x_min, tile_x_max = float(x_coords[x_start]), float(x_coords[x_end - 1])
            tile_y_min, tile_y_max = float(y_coords[y_end - 1]), float(y_coords[y_start])
            tiles_dict[tile_id] = [tile_x_min, tile_y_max, tile_x_max, tile_y_min]
            tile_id += 1
    return tiles_dict

def make(mask, *, key, size, width, depth, data_mean, data_std):
    return NeuralODE(size, width, depth, mask, key=key), data_mean, data_std

def save(filename, hyperparams, model):
    """Save a model and its hyperparameters to a file."""
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename, mask):
    """Load a model and its hyperparameters from a file."""
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model, dmean, dstd = make(mask=mask, key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model), dmean, dstd

def ssim_with_mask(img1, img2, mask, window_size=11, sigma=1.5, k1=0.01, k2=0.03, data_range=1.0):
    """Calculate the Structural Similarity Index (SSIM) between two images with missing data. This function computes SSIM while properly handling missing or invalid pixels indicated by a mask. The algorithm applies Gaussian filtering to both the images and the mask, then normalizes statistics by the filtered mask to account for missing data."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)
    # Apply mask to images
    img1_masked = img1 * mask
    img2_masked = img2 * mask
    # Compute local means
    mu1 = gaussian_filter(img1_masked, sigma)
    mu2 = gaussian_filter(img2_masked, sigma)
    mask_filtered = gaussian_filter(mask, sigma)
    # Normalize by filtered mask to get proper means
    valid_pixels = mask_filtered > 1e-8
    mu1 = np.divide(mu1, mask_filtered, out=np.zeros_like(mu1), where=valid_pixels)
    mu2 = np.divide(mu2, mask_filtered, out=np.zeros_like(mu2), where=valid_pixels)
    # Compute local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1_masked ** 2, sigma)
    sigma1_sq = np.divide(sigma1_sq, mask_filtered, out=np.zeros_like(sigma1_sq), where=valid_pixels) - mu1_sq
    sigma2_sq = gaussian_filter(img2_masked ** 2, sigma)
    sigma2_sq = np.divide(sigma2_sq, mask_filtered, out=np.zeros_like(sigma2_sq), where=valid_pixels) - mu2_sq
    sigma12 = gaussian_filter(img1_masked * img2_masked, sigma)
    sigma12 = np.divide(sigma12, mask_filtered, out=np.zeros_like(sigma12), where=valid_pixels) - mu1_mu2
    # SSIM constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    # Compute SSIM
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>0)
    # Only consider valid regions
    valid_regions = valid_pixels & (denominator > 0)
    if np.sum(valid_regions) == 0:
        return 0.0
    return np.mean(ssim_map[valid_regions]), ssim_map

def psnr_map(sr_image, ref_image, window_size=11, max_pixel=255.0):
    """
    Calculate a PSNR map between a super-resolved image and a reference image, handling NaN values.

    Parameters:
    - sr_image: numpy array of the super-resolved image (H x W)
    - ref_image: numpy array of the reference image (same shape as sr_image)
    - window_size: size of the sliding window (odd integer, default 11)
    - max_pixel: maximum possible pixel value (default 255 for 8-bit images)

    Returns:
    - psnr_map: numpy array of the PSNR map (H x W) with local PSNR values
    """
    # Convert to float64 for precision
    sr_image = sr_image.astype(np.float64)
    ref_image = ref_image.astype(np.float64)
    pad = window_size // 2
    sr_padded = np.pad(sr_image, pad, mode='reflect')
    ref_padded = np.pad(ref_image, pad, mode='reflect')
    H, W = sr_image.shape
    psnr_map = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            window_sr = sr_padded[i:i+window_size, j:j+window_size]
            window_ref = ref_padded[i:i+window_size, j:j+window_size]
            # Mask to ignore NaN values in both windows
            valid_mask = ~np.isnan(window_sr) & ~np.isnan(window_ref)
            if np.any(valid_mask):
                mse = np.mean((window_sr[valid_mask] - window_ref[valid_mask]) ** 2)
                if mse == 0:
                    psnr_map[i, j] = 100  # Arbitrarily high value for perfect match
                else:
                    psnr_map[i, j] = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
            else:
                # If no valid pixels, assign NaN to indicate undefined PSNR
                psnr_map[i, j] = np.nan
    return psnr_map
