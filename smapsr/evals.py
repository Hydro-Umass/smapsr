import xarray as xr
import json
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd
from smapsr.train import prepare_data
from smapsr.models import NeuralODE
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

def predict(sl, sh, region, model, m, s):
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
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename, mask):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model, dmean, dstd = make(mask=mask, key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model), dmean, dstd
