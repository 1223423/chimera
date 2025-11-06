import numpy as np
from PIL import Image


def load_fragments(fragment_paths, fragment_size):
    """
    Load and resize fragment images

    Args:
        fragment_paths: List of file paths to fragment images
        fragment_size: Tuple of (width, height) to resize to

    Returns:
        List of PIL Images (RGBA format)
    """
    fragment_images = []
    for fpath in fragment_paths:
        img = Image.open(fpath).convert('RGBA').resize(fragment_size)
        fragment_images.append(img)
    return fragment_images


def extract_canvas_regions(canvas, positions, fragment_size):
    """
    Extract canvas regions at specified positions

    Args:
        canvas: PIL Image
        positions: List of (x, y) tuples
        fragment_size: Tuple of (width, height)

    Returns:
        List of numpy arrays (H, W, 4) normalized to [0, 1]
    """
    regions = []
    for x, y in positions:
        region = canvas.crop((x, y, x + fragment_size[0], y + fragment_size[1]))
        region_array = np.array(region, dtype='float32') / 255.0
        regions.append(region_array)
    return regions


def filter_valid_positions(positions, canvas_width, canvas_height, fragment_size):
    """
    Filter positions to ensure fragments fit within canvas bounds

    Args:
        positions: List of (x, y) tuples
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        fragment_size: Tuple of (width, height)

    Returns:
        List of valid (x, y) tuples
    """
    valid = [
        (x, y) for x, y in positions
        if x + fragment_size[0] <= canvas_width and y + fragment_size[1] <= canvas_height
    ]
    return valid


def create_output_canvas(width, height, background='black'):
    """
    Create blank output canvas

    Args:
        width: Canvas width
        height: Canvas height
        background: Background color (default: 'black')

    Returns:
        PIL Image (RGBA)
    """
    return Image.new('RGBA', (width, height), background)


def save_reduced_library(fragment_paths, kept_indices, output_dir):
    """
    Copy reduced fragment library to new directory

    Args:
        fragment_paths: Original fragment file paths
        kept_indices: Indices of fragments to keep
        output_dir: Directory to save reduced library

    Returns:
        List of new file paths
    """
    import os
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    new_paths = []
    for i, idx in enumerate(kept_indices):
        original_path = fragment_paths[idx]
        filename = os.path.basename(original_path)
        new_path = os.path.join(output_dir, filename)

        shutil.copy2(original_path, new_path)
        new_paths.append(new_path)

    return new_paths
