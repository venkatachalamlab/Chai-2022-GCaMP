import json
import os
import glob
from math import floor
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sklearn.neighbors as sk

import h5py

from array_writer import TimestampedArrayWriter
from ..annotation.annotations_io import load_annotations

def _coord_from_idx(idx: int, shape: int) -> float:
    return (idx + 0.5)/shape

def _idx_from_coord(coord: float, shape: int) -> int:
    return max(floor(coord*shape - 1e-6), 0)

def get_metadata(dataset_path: Path):
    json_filename = dataset_path / "metadata.json"
    with open(json_filename) as json_file:
        metadata = json.load(json_file)
    return metadata
  
def get_times(dataset_path: Path):
    h5_filename = dataset_path / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return f["times"][:]
  
def get_slice(dataset_path: Path, t):
    h5_filename = dataset_path / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return f["data"][t]

def apply_lut(x: np.ndarray, lo: float, hi: float, newtype=None) -> np.ndarray:
    if newtype is None:
        newtype = x.dtype
    y_float = (x-lo)/(hi-lo)
    y_clipped = np.clip(y_float, 0, 1)
    if np.issubdtype(newtype, np.integer):
        maxval = np.iinfo(newtype).max
    else:
        maxval = 1.0
    return (maxval*y_clipped).astype(newtype)
  
def mip_threeview(vol: np.ndarray, scale=(4,1,1)) -> np.ndarray:
    S = vol.shape[:3] * np.array(scale)
    output_shape = (S[1] + S[0],
                    S[2] + S[0])
    if vol.ndim == 4:
        output_shape = (*output_shape, 3)
    vol = np.repeat(vol, scale[0], axis=0)
    vol = np.repeat(vol, scale[1], axis=1)
    vol = np.repeat(vol, scale[2], axis=2)
    x = mip_x(vol)
    y = mip_y(vol)
    z = mip_z(vol)
    output = np.zeros(output_shape, dtype=vol.dtype)
    output[:S[1], :S[2]] = z
    output[:S[1], S[2]:] = x
    output[S[1]:, :S[2]] = y

    return output

def compress(data_path, l_quantile=0.75, h_quantile=1.0, noise_decider_t=0):
    '''
    Combines and compresses raw data collected with LAMBDA.

            Parameters:
                    data_path (str): path of raw datasets
                    l_quantile (float): quantile used as the floor of the lookup-
                                        table for each channel
                    h_quantile (float): quantile used as the ceiling of the lookup-
                                        table for both channels
                    noise_decider_t (int): l_qunatile uses the data at this time point
                                           to calculate the baseline noise for each
                                           channel

    '''
    data_path = Path(data_path)

    if not os.path.exists(data_path / "compressed"):
        os.mkdir(data_path / "compressed")
    if os.path.isfile(data_path / "compressed" / "data.h5"):
        os.remove(data_path / "compressed" / "data.h5")


    raw_files = glob.glob(os.path.join(data_path, '*.h5'))
    hdf1 = None
    hdf2 = None
    for p in raw_files:
        if Path(p).name[:7] == 'camera1':
            hdf1 = Path(p)
        if Path(p).name[:7] == 'camera2':
            hdf2 = Path(p)

    if hdf1 is not None and hdf2 is not None:
        f1 = h5py.File(hdf1, 'r')
        f2 = h5py.File(hdf2, 'r')
        times = np.stack([f1["times"], f2["times"]]).max(axis=0)

    elif hdf1 is not None and hdf2 is None:
        f1 = h5py.File(hdf1, 'r')
        times = f1["times"]

    elif hdf1 is None and hdf2 is not None:
        f2 = h5py.File(hdf2, 'r')
        times = f2["times"]
    else:
        print("dataset not found")
        return


    def get_4D_volume(t):

        if hdf1 is not None and hdf2 is not None:
            vol = np.stack([f1["data"][t], f2["data"][t]])

        elif hdf1 is not None and hdf2 is None:
            vol = f1["data"][t]

        elif hdf1 is None and hdf2 is not None:
            vol = f2["data"][t]

        if vol.ndim == 2:
            return np.expand_dims(vol, axis=(0, 1))
        elif vol.ndim == 3:
            return np.expand_dims(vol, axis=0)
        return vol

    temp_vol = get_4D_volume(noise_decider_t)
    if temp_vol.shape[1] > 2:
        temp_vol = temp_vol[:, 1:-1, ...]

    shape = temp_vol.shape

    if np.issubdtype(temp_vol.dtype, np.integer):
        maxval = np.iinfo(temp_vol.dtype).max
    else:
        maxval = 1.0

    lut_high = h_quantile * maxval
    if l_quantile == 0.0:
        noise_values = [0.0 for c in range(shape[0])]
    else:
        noise_values = [np.quantile(temp_vol[c], l_quantile) for c in range(shape[0])]

    writer = TimestampedArrayWriter(None, data_path / "compressed" / "data.h5",
                                    shape, dtype=np.float32, groupname=None,
                                    compression="gzip", compression_opts=5)

    for i, time in tqdm(enumerate(times)):
        vol = get_4D_volume(i)
        if vol.shape[1] > 2:
            vol = vol[:, 1:-1, ...]
        vol_float32 = np.empty_like(vol, dtype=np.float32)
        for channel in range(shape[0]):
            vol_float32[channel] = apply_lut(
                vol[channel],
                noise_values[channel],
                lut_high, np.float32)
        writer.append_data((time, vol_float32))
    writer.close()

    metadata = {
        "shape_t": len(times),
        "shape_c": shape[0],
        "shape_z": shape[1],
        "shape_y": shape[2],
        "shape_x": shape[3],
        "dtype": "float32"
    }
    with open(data_path / "compressed" / "metadata.json", 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


def get_spherical_mask(r):
    s = [2 * r + 1] * 3
    c = [r] * 3
    z, y, x = np.ogrid[:s[0], :s[1], :s[2]]
    d = np.sqrt(
        (z - c[0])**2 + \
        (y - c[1])**2 + \
        (x - c[2])**2
    )
    mask = d <= r
    return mask


def extract_traces(
    data_path, scale=(4, 1, 1), radius=5,
    pixels_to_keep=10, overlap_ratio=0.8
    ):
    
    path = Path(data_path)
    a, w = load_annotations(path)
    metadata = get_metadata(path)
    
    shape = (
        metadata["shape_t"],
        metadata["shape_c"],
        metadata["shape_z"],
        metadata["shape_y"],
        metadata["shape_x"]
    )
    sphere = np.repeat(
        get_spherical_mask(radius)[np.newaxis, ...],
        shape[1],
        axis=0
    )
    mask = np.repeat(
        np.zeros(
            [8 * radius + 1] * 3,
            dtype=bool
        )[np.newaxis, ...],
        shape[1],
        axis=0
    )
    padded_v = np.zeros(
        (shape[1],
         shape[2] * scale[0] + 8 * radius,
         shape[3] * scale[1] + 8 * radius,
         shape[4] * scale[2] + 8 * radius)
    )
    n_neurons = w.df.shape[0]
    annotated_timepoints = np.unique(a.df["t_idx"])

    traces = np.zeros((shape[1], n_neurons, shape[0]))
    traces[:] = np.NaN
    
    
    for t in tqdm(annotated_timepoints):
        v_t = np.repeat(get_slice(path, t), scale[0], axis=1)
        padded_v[
            :,
            4 * radius: 4 * radius + shape[2] * scale[0],
            4 * radius: 4 * radius + shape[3] * scale[1],
            4 * radius: 4 * radius + shape[4] * scale[2]
        ] = v_t
        
        
        a_t = a.df[a.df['t_idx'] == t]
        neurons_t = np.unique(a_t["worldline_id"])
        zyx_t = np.zeros((neurons_t.shape[0], 3))
        
        for n, track in enumerate(neurons_t):
            a_t_n = a_t[a_t['worldline_id'] == track]
            zyx_t[n] = [_idx_from_coord(a_t_n['z'], shape[2] * scale[0]) + 4 * radius,
                        _idx_from_coord(a_t_n['y'], shape[3] * scale[1]) + 4 * radius,
                        _idx_from_coord(a_t_n['x'], shape[4] * scale[2]) + 4 * radius]
            
        tree = sk.KDTree(zyx_t)
        zyx_t = zyx_t.astype(int)
        
        for n, track in enumerate(neurons_t):
            v_t_n = padded_v[
                :,
                zyx_t[n,0] - 4 * radius: zyx_t[n,0] + 4 * radius + 1,
                zyx_t[n,1] - 4 * radius: zyx_t[n,1] + 4 * radius + 1,
                zyx_t[n,2] - 4 * radius: zyx_t[n,2] + 4 * radius + 1
            ]
            
            idx, dis = tree.query_radius(
                X=np.array([zyx_t[n]]),
                r=2 * radius,
                return_distance=True,
                sort_results=True
            )
            neighbors_idx = idx[0][1:]
            neighbors_dis = dis[0][1:]
            
            neuron_mask = mask.copy()
            neuron_mask[
                :,
                3 * radius: 5 * radius + 1,
                3 * radius: 5 * radius + 1,
                3 * radius: 5 * radius + 1,
            ] = sphere
                
            neighbors_mask = 1 - mask.copy()
            for idx, dist in zip(neighbors_idx, neighbors_dis):
                neighbor_r = int(overlap_ratio * dist)
                neighbor_sphere = np.repeat(
                    get_spherical_mask(neighbor_r)[np.newaxis, ...],
                    shape[1],
                    axis=0
                )
                temp = 1 - mask.copy()
                lower_idx = [
                    zyx_t[idx][i] - zyx_t[n][i] + 4 * radius - neighbor_r for i in range(3)
                ]
                upper_idx = [
                    zyx_t[idx][i] - zyx_t[n][i] + 4 * radius + neighbor_r + 1 for i in range(3)
                ]
                temp[
                    :,
                    lower_idx[0]: upper_idx[0],
                    lower_idx[1]: upper_idx[1],
                    lower_idx[2]: upper_idx[2]
                ] = 1 - neighbor_sphere
                neighbors_mask = np.logical_and(temp, neighbors_mask)
                
            
                
            masked_v = neuron_mask * neighbors_mask * v_t_n
            
            
            for c in range(shape[1]):
                non_zero_v = masked_v[c][np.nonzero(masked_v[c])]
                if non_zero_v.shape[0] != 0:
                    traces[c, track, t] = np.mean(
                        np.sort(non_zero_v)[-pixels_to_keep * np.prod(scale):]
                    )

    file_name = path / "traces.npy"
    np.save(file_name, traces)