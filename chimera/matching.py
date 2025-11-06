import torch
import numpy as np
from skimage import color


def rgb_to_lab_torch(rgb_tensor):
    """
    Convert RGB tensor to LAB color space using PyTorch

    Args:
        rgb_tensor: (B, H, W, 3) or (H, W, 3) tensor with values in [0, 1]

    Returns:
        LAB tensor of same shape
    """
    # Use skimage for conversion (reliable, but requires CPU transfer)
    # For full GPU pipeline, could implement native PyTorch LAB conversion
    if rgb_tensor.device.type in ["mps", "cuda"]:
        rgb_np = rgb_tensor.cpu().numpy()
    else:
        rgb_np = rgb_tensor.numpy()

    if len(rgb_np.shape) == 4:
        # Batch processing
        lab_list = [color.rgb2lab(img) for img in rgb_np]
        lab_np = np.stack(lab_list, axis=0)
    else:
        lab_np = color.rgb2lab(rgb_np)

    return torch.from_numpy(lab_np).to(rgb_tensor.device)


def compute_masked_mae_batch(fragment_lab_tensors, fragment_masks_tensors,
                               region_lab_batch, device):
    """
    Vectorized MAE calculation for batch of regions

    Computes mean absolute error between each region and all fragments,
    considering transparency masks

    Args:
        fragment_lab_tensors: List of (N_pixels, 3) tensors, one per fragment
        fragment_masks_tensors: List of (H, W) boolean tensors
        region_lab_batch: (B, H, W, 3) tensor of canvas regions in LAB space
        device: torch device

    Returns:
        (B, N_fragments) tensor of MAE errors
    """
    batch_size = region_lab_batch.shape[0]
    n_fragments = len(fragment_lab_tensors)

    errors = torch.zeros((batch_size, n_fragments), device=device)

    for frag_idx in range(n_fragments):
        mask = fragment_masks_tensors[frag_idx]  # (H, W)
        frag_lab = fragment_lab_tensors[frag_idx]  # (N_pixels, 3)
        n_pixels = frag_lab.shape[0]

        if n_pixels == 0:
            # Fully transparent fragment, assign high error
            errors[:, frag_idx] = float('inf')
            continue

        # Extract masked pixels from each region in batch
        # Use vectorized indexing: region_lab_batch[:, mask, :]
        # This extracts masked positions for all batch items at once
        # Result: (B, N_pixels, 3)
        region_masked = region_lab_batch[:, mask, :]  # (B, N_pixels, 3)

        # Compute absolute error: (B, N_pixels, 3)
        abs_error = torch.abs(region_masked - frag_lab.unsqueeze(0))

        # Sum and normalize: (B,)
        errors[:, frag_idx] = torch.sum(abs_error, dim=(1, 2)) / n_pixels

    return errors


def find_best_fragments_batch(region_batch, fragment_lab_tensors, fragment_masks_tensors, device):
    """
    Find best matching fragment for each region in batch

    Args:
        region_batch: (B, H, W, 4) tensor of RGBA regions
        fragment_lab_tensors: List of LAB tensors for fragments
        fragment_masks_tensors: List of mask tensors
        device: torch device

    Returns:
        best_indices: (B,) tensor of best fragment indices
    """
    # Convert regions to LAB (only RGB channels)
    region_rgb = region_batch[..., :3]  # (B, H, W, 3)
    region_lab = rgb_to_lab_torch(region_rgb)

    # Compute errors for all fragments
    errors = compute_masked_mae_batch(
        fragment_lab_tensors,
        fragment_masks_tensors,
        region_lab,
        device
    )

    # Find best match for each region
    best_indices = torch.argmin(errors, dim=1)

    return best_indices


def preprocess_fragments(fragment_images, fragment_size, device):
    """
    Preprocess fragment images for GPU-accelerated matching

    Args:
        fragment_images: List of PIL Images (RGBA)
        fragment_size: Tuple of (width, height)
        device: torch device

    Returns:
        tuple: (fragment_images, fragment_lab_tensors, fragment_masks_tensors)
    """
    fragment_arrays = [np.array(img, dtype='float32') / 255. for img in fragment_images]

    # Convert to tensors and move to device
    fragment_tensors = [torch.from_numpy(arr).to(device) for arr in fragment_arrays]

    # Extract LAB and masks
    fragment_lab_tensors = []
    fragment_masks_tensors = []

    for frag_tensor in fragment_tensors:
        # Convert to LAB
        frag_rgb = frag_tensor[..., :3]
        frag_lab = rgb_to_lab_torch(frag_rgb)

        # Extract alpha mask
        alpha = frag_tensor[..., 3]
        mask = alpha > 0

        # Store masked LAB values
        frag_lab_masked = frag_lab[mask]  # (N_pixels, 3)

        fragment_lab_tensors.append(frag_lab_masked)
        fragment_masks_tensors.append(mask)

    return fragment_images, fragment_lab_tensors, fragment_masks_tensors
