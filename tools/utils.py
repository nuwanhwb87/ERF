import numpy as np
import torch
from torch.utils.data import Sampler
import random
# -----------------------------------------------------------------------------
# NOTE ON `spinup`
# -----------------------------------------------------------------------------
# `spinup` controls how many initial timestamps are *skipped* when building the
# index array. It ensures that when we fetch conditioning inputs at negative
# offsets (e.g., [-24, -6, 0]) we never index before the start of the dataset.
#
# IMPORTANT:
# - Units: spinup is measured in *index steps*, not hours/days. If your dataset
#   is hourly, "24" means 24 hours. If it is weekly (freq="7D"), "2" means 2
#   weekly steps.
# - Minimum safe value: spinup >= max(0, -min(conditioning_times))
#   (because we access: start_index + conditioning_times)
# - Do NOT hardcode hour-based buffers (e.g., +24) if your frequency changes.
#
# Rules of thumb:
# - Hourly data, conditioning_times = [-24, -6, 0] → spinup >= 24
# - Weekly data, conditioning_times = [-3, 0]      → spinup >= 3
# -----------------------------------------------------------------------------

# NOTE:
# The duplicate imports and duplicate spinup note block that previously appeared
# here were intentionally commented out/removed from execution as non-used
# duplicates for this workflow. If needed in future, recover from git history.


class MarineDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path,  # str: Path to the dataset files.
                 dataset_mode,  # str: Dataset dataset_mode ('train', 'val', 'test').
                 sample_counts,
                 # tuple: Total, training, and validation sample counts (total_samples, train_samples, val_samples).
                 dimensions,  # tuple: Dimensions of the dataset (variables, latitude, longitude).
                 lead_time,  # int: Current lead time for forecasting.
                 max_horizon,  # int: Maximum lead time we want to forecast. Used for not going outside dataset
                 norm_factors,  # tuple: Mean and standard deviation for normalization (mean, std_dev).
                 device,  # torch.device: Device on which tensors will be loaded.
                 lead_time_range,  # Range of lead time
                 spinup=0,  # int: Number of samples to discard at the start for stability.
                 spacing=1,  # int: Sample selection interval for data reduction.
                 dtype='float32',  # str: Data type of the dataset (default 'float32').
                 conditioning_times=[0, ],  # list: Times to condition on for forecasting.
                 static_data_path=None,  # str: Path to the static data file.
                 random_lead_time=0,  # bool: Whether to use random lead time

                 ):
        """
        Initialize a custom Dataset for lazily loading WB samples from a memory-mapped file,
        which allows for efficient data handling without loading the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup + 30  # Change this if we ever look back more than 7 days
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None

        self.static_vars = 0
        if static_data_path != None:
            self.static_fields = torch.tensor(self.load_static_data(), dtype=torch.float32)
            self.static_vars = self.static_fields.shape[0]

        self.conditioning_times = conditioning_times
        self.input_times = self.num_variables * len(self.conditioning_times)
        self.output_times = self.num_variables * (
            len(self.lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

        self.index_array = self._generate_indices()

        self.mmap = self.create_mmap()

        self.random_lead_time = random_lead_time

    def create_mmap(self):
        """Creates a memory-mapped array for the dataset to facilitate large data handling."""
        return np.memmap(self.dataset_path, dtype=np.float32, mode='r',
                         shape=(self.n_samples, self.num_variables, self.n_lat, self.n_lon))

    def load_static_data(self):
        """Load and normalize static fields."""
        static_fields = np.load(self.static_data_path)

        min_vals = np.min(static_fields, axis=(1, 2))
        max_vals = np.max(static_fields, axis=(1, 2))

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Replace zero range with one to avoid division by zero

        # Apply min-max scaling: (x - min) / (max - min)
        scaled_static_fields = (static_fields - min_vals[:, None, None]) / range_vals[:, None, None]
        return scaled_static_fields

    def _generate_indices(self):
        """Generates indices for dataset partitioning according to the specified dataset_mode."""
        if self.dataset_mode == 'train':
            start, stop = self.spinup, self.n_train
        elif self.dataset_mode == 'val':
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        elif self.dataset_mode == 'test':
            start, stop = self.spinup + self.n_train + self.n_val, self.n_samples

        return np.arange(start, stop - self.max_horizon)[::self.spacing]

    def set_lead_time(self, lead_time):
        """ Updates the lead time lead_time for generating future or past indices."""
        self.lead_time = lead_time

    def set_lead_time_range(self, lead_time_range):
        self.t_min, self.t_max, self.delta_t = lead_time_range

    def get_lead_time(self):
        if self.random_lead_time:
            num_steps = 1 + (self.t_max - self.t_min) // self.delta_t
            return self.t_min + self.delta_t * np.random.randint(0, num_steps)
        return self.lead_time

    def __len__(self):
        """Returns the number of samples available in the dataset based on the computed indices."""
        return self.index_array.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding future or past state from the dataset."""
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()

        x_index = start_index + self.conditioning_times
        y_index = start_index + lead_times

        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        X_sample = (X_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        Y_sample = (Y_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]

        X_sample = torch.tensor(X_sample, dtype=torch.float32).view(self.input_times, self.n_lat, self.n_lon)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).view(self.output_times, self.n_lat, self.n_lon)

        if self.static_vars != 0:
            X_sample = torch.cat([X_sample, self.static_fields], dim=0)

        return X_sample, Y_sample, lead_times

class ERA5Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path,     # str: Path to the dataset files.
                 dataset_mode,     # str: Dataset dataset_mode ('train', 'val', 'test').
                 sample_counts,    # tuple: Total, training, and validation sample counts (total_samples, train_samples, val_samples).
                 dimensions,        # tuple: Dimensions of the dataset (variables, latitude, longitude).
                 lead_time,        # int: Current lead time for forecasting.
                 max_horizon,    # int: Maximum lead time we want to forecast. Used for not going outside dataset
                 norm_factors,     # tuple: Mean and standard deviation for normalization (mean, std_dev).
                 device,           # torch.device: Device on which tensors will be loaded.
                 lead_time_range,  # Range of lead time
                 spinup = 0,       # int: Number of samples to discard at the start for stability.
                 spacing = 1,      # int: Sample selection interval for data reduction.
                 dtype='float32',   # str: Data type of the dataset (default 'float32').
                 conditioning_times=[0,], # list: Times to condition on for forecasting.
                 static_data_path = None, # str: Path to the static data file.
                 random_lead_time = 0, # bool: Whether to use random lead time

                ):
        """
        Initialize a custom Dataset for lazily loading WB samples from a memory-mapped file,
        which allows for efficient data handling without loading the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup + 24 # Change this if we ever look back more than 24h
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None

        self.static_vars = 0
        if static_data_path != None:
            self.static_fields = torch.tensor(self.load_static_data(), dtype=torch.float32)
            self.static_vars = self.static_fields.shape[0]
        
        self.conditioning_times = conditioning_times
        self.input_times = self.num_variables * len(self.conditioning_times)
        self.output_times = self.num_variables * (len(self.lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

        self.index_array = self._generate_indices()

        self.mmap = self.create_mmap()

        self.random_lead_time = random_lead_time


    def create_mmap(self):
        """Creates a memory-mapped array for the dataset to facilitate large data handling."""
        return np.memmap(self.dataset_path, dtype=np.float32, mode='r', shape=(self.n_samples, self.num_variables, self.n_lat, self.n_lon))

    def load_static_data(self):
        """Load and normalize static fields."""
        static_fields = np.load(self.static_data_path)

        min_vals = np.min(static_fields, axis=(1,2))
        max_vals = np.max(static_fields, axis=(1,2))

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Replace zero range with one to avoid division by zero

        # Apply min-max scaling: (x - min) / (max - min)
        scaled_static_fields = (static_fields - min_vals[:, None, None]) / range_vals[:, None, None]
        return scaled_static_fields

    def _generate_indices(self):
        """Generates indices for dataset partitioning according to the specified dataset_mode."""
        if self.dataset_mode == 'train':
            start, stop = self.spinup, self.n_train
        elif self.dataset_mode == 'val':
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        elif self.dataset_mode == 'test':
            start, stop = self.spinup + self.n_train + self.n_val, self.n_samples

        return np.arange(start, stop - self.max_horizon)[::self.spacing]

    def set_lead_time(self, lead_time):
        """ Updates the lead time lead_time for generating future or past indices."""
        self.lead_time = lead_time

    def set_lead_time_range(self, lead_time_range):
        self.t_min, self.t_max, self.delta_t = lead_time_range

    def get_lead_time(self):
        if self.random_lead_time:
            num_steps = 1 + (self.t_max - self.t_min) // self.delta_t
            return self.t_min + self.delta_t * np.random.randint(0, num_steps)
        return self.lead_time

    def __len__(self):
        """Returns the number of samples available in the dataset based on the computed indices."""
        return self.index_array.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding future or past state from the dataset."""
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()

        x_index = start_index + self.conditioning_times
        y_index = start_index + lead_times

        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        X_sample = (X_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        Y_sample = (Y_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]

        X_sample = torch.tensor(X_sample, dtype=torch.float32).view(self.input_times, self.n_lat, self.n_lon)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).view(self.output_times, self.n_lat, self.n_lon)

        if self.static_vars != 0:
            X_sample = torch.cat([X_sample, self.static_fields], dim=0)

        return X_sample, Y_sample, lead_times


def get_uniform_t_dist_fn(t_min, t_max, delta_t):
    """ Create the update function """

    def uniform_t_dist(dataset):
        new_lead_time = t_min + delta_t * np.random.randint(0, 1 + (t_max - t_min) // delta_t)
        dataset.set_lead_time(new_lead_time)
    
    return uniform_t_dist

class DynamicKBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, t_update_callback, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.t_update_callback = t_update_callback
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle indices at the beginning of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            if len(batch) == self.batch_size:
                self.t_update_callback(self.dataset)  # Update `lead_time` before yielding the batch
                yield batch
                batch = []
            batch.append(idx)
        if batch and not self.drop_last:
            self.t_update_callback(self.dataset)  # Update `lead_time` for the last batch if not dropping it
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def _summarize(v):
    # Tensors
    if isinstance(v, torch.Tensor):
        shape = tuple(v.shape)
        return f"Tensor(shape={shape}, dtype={v.dtype}, device={v.device})"
    # NumPy arrays
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    # PyTorch device
    if isinstance(v, torch.device):
        return str(v)
    # Paths
    from pathlib import Path
    if isinstance(v, Path):
        return str(v)
    # Lists/Tuples: summarize recursively
    if isinstance(v, (list, tuple)):
        return type(v)([_summarize(x) for x in v])
    # Dicts: summarize recursively
    if isinstance(v, dict):
        return {k: _summarize(val) for k, val in v.items()}
    return v  # numbers, strings, etc.

def pretty_print_kwargs(kwargs: dict, width: int = 100):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=2, width=width, compact=False, sort_dicts=False)
    pp.pprint(_summarize(kwargs))


import pandas as pd

def make_time_index(start_datetime: str, end_datetime: str, time_freq: str) -> pd.DatetimeIndex:
    """Build time index from config. Works for '1H', '6H', '7D', 'W', 'W-TUE', etc."""
    start = pd.to_datetime(start_datetime)
    end   = pd.to_datetime(end_datetime)
    try:
        # pandas >= 1.4
        return pd.date_range(start, end, freq=time_freq, inclusive="both")
    except TypeError:
        # older pandas fallback
        ti = pd.date_range(start, end, freq=time_freq)
        if len(ti) and ti[-1] == end:
            return ti
        # If end lies exactly on the grid but wasn't included due to version quirks
        if len(ti) and (end - ti[-1]) % pd.tseries.frequencies.to_offset(time_freq) == pd.Timedelta(0):
            return ti.append(pd.DatetimeIndex([end]))
        return ti


def split_time_index(ti: pd.DatetimeIndex, split_mode: str,
                     train_year_end: int | None = None, val_year_end: int | None = None,
                     train_until: str | None = None,   val_until: str | None = None):
    """
    Return (n_samples, n_train, n_val) using:
      - split_mode='years':   <= train_year_end, then (train_year_end, val_year_end]
      - split_mode='dates':   <= train_until,    then (train_until,   val_until]
    All bounds are inclusive at the right end of each split.
    """
    mode = split_mode.lower()
    if mode == "years":
        assert train_year_end is not None and val_year_end is not None, "years split requires train_year_end & val_year_end"
        train_mask = (ti.year <= train_year_end)
        val_mask   = (ti.year >  train_year_end) & (ti.year <= val_year_end)

    elif mode == "dates":
        assert train_until is not None and val_until is not None, "dates split requires train_until & val_until"
        tr_end = pd.to_datetime(train_until)
        va_end = pd.to_datetime(val_until)
        train_mask = (ti <= tr_end)
        val_mask   = (ti > tr_end) & (ti <= va_end)

    else:
        raise ValueError(f"Unknown split_mode: {split_mode} (use 'years' or 'dates')")

    n_samples = len(ti)
    n_train   = int(train_mask.sum())
    n_val     = int(val_mask.sum())
    return n_samples, n_train, n_val


def save_checkpoint(path, model, optimizer, scheduler, warmup_scheduler,
                    epoch, best_val_loss):
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model": (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "warmup_scheduler": warmup_scheduler.state_dict(),
        # RNG (optional but good)
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler, warmup_scheduler, device):
    """
    Load checkpoint and restore model, optimizer, scheduler states, and RNG states.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        warmup_scheduler: Warmup scheduler to load state into
        device: Device to load checkpoint on
    
    Returns:
        dict with keys: epoch, best_val_loss
    """
    # weights_only=False is needed because checkpoint contains numpy RNG states
    # Load to CPU first to preserve RNG state types, then move model to device
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Load model state (move to device after loading)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Load scheduler states
    scheduler.load_state_dict(checkpoint["scheduler"])
    warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
    
    # Restore RNG states (must be on CPU and correct type)
    # Ensure rng_torch is uint8 tensor on CPU
    rng_torch = checkpoint["rng_torch"]
    if rng_torch.dtype != torch.uint8:
        rng_torch = rng_torch.to(torch.uint8)
    if rng_torch.device.type != 'cpu':
        rng_torch = rng_torch.cpu()
    torch.set_rng_state(rng_torch)
    
    # Restore CUDA RNG states if available
    if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    
    # Restore numpy and python RNG states
    np.random.set_state(checkpoint["rng_numpy"])
    random.setstate(checkpoint["rng_python"])
    
    print(f"Loaded checkpoint from {path}")
    print(f"Resuming from epoch {checkpoint['epoch'] + 1}, best_val_loss: {checkpoint['best_val_loss']:.4f}")
    
    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"]
    }