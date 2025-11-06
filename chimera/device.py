import torch
import psutil

def get_device(force_device=None):
    """
    Auto-detect optimal device: CUDA > MPS > CPU

    Args:
        force_device: Optional string to force specific device ('cuda', 'mps', 'cpu')

    Returns:
        torch.device object
    """
    if force_device:
        if force_device == 'cuda':
            if torch.cuda.is_available():
                print("> Using CUDA (forced)")
                return torch.device("cuda")
            else:
                print("> CUDA not available, falling back to CPU")
                return torch.device("cpu")
        elif force_device == 'mps':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("> Using MPS (forced)")
                return torch.device("mps")
            else:
                print("> MPS not available, falling back to CPU")
                return torch.device("cpu")
        else:
            print("> Using CPU (forced)")
            return torch.device("cpu")

    # Auto-detect: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"> CUDA detected ({gpu_name})")
        return device
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("> MPS detected")
        device = torch.device("mps")
        return device
    else:
        print("> Using CPU")
        return torch.device("cpu")


def get_available_memory():
    """
    Get available system memory in bytes

    Returns:
        int: Available memory in bytes
    """
    memory = psutil.virtual_memory()
    return memory.available


def get_memory_limit(memory_fraction=0.5, verbose=True):
    """
    Calculate memory limit based on available memory and fraction to use

    Args:
        memory_fraction: Fraction of available memory to use (0.0-1.0)
        verbose: Print information

    Returns:
        int: Memory limit in bytes
    """
    available = get_available_memory()
    limit = int(available * memory_fraction)

    if verbose:
        print(f"> Available memory: {available / (1024**3):.2f} GB")
        print(f"> Using {memory_fraction*100:.0f}% = {limit / (1024**3):.2f} GB for processing")

    return limit


def calculate_optimal_batch_size(memory_limit, fragment_size, n_fragments, device, verbose=True):
    """
    Calculate optimal batch size based on available memory

    Estimates memory per position and determines how many can fit in memory

    Args:
        memory_limit: Memory limit in bytes
        fragment_size: Tuple of (width, height) for fragments
        n_fragments: Number of fragments in library
        device: torch.device object
        verbose: Print information

    Returns:
        int: Optimal batch size
    """
    # Estimate memory per position (fragment comparison operation)
    bytes_per_element = 4  # float32
    pixels_per_region = fragment_size[0] * fragment_size[1]
    channels = 3  # LAB color space

    # Memory for one region comparison with all fragments
    memory_per_position = (
        pixels_per_region * channels * bytes_per_element +  # region tensor
        pixels_per_region * channels * n_fragments * bytes_per_element  # comparison arrays
    )

    # Conservative estimate: use 70% of limit for batch processing (rest for overhead)
    usable_memory = memory_limit * 0.7
    batch_size = int(usable_memory / memory_per_position)

    # Ensure reasonable bounds
    batch_size = max(100, min(batch_size, 50000))

    if verbose:
        if device.type in ["mps", "cuda"]:
            print(f"> GPU batch size: {batch_size:,} positions")
        else:
            print(f"> CPU batch size: {batch_size:,} positions")

    return batch_size


def clear_cache(device):
    """
    Clear device cache to free memory

    Args:
        device: torch.device object
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
