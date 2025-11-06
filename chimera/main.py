import argparse
from glob import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .device import (
    get_device,
    get_memory_limit,
    calculate_optimal_batch_size,
    clear_cache
)
from .sampling import PATTERN_GENERATORS
from .fragment_reduction import reduce_fragments
from .matching import preprocess_fragments, find_best_fragments_batch
from .utils import (
    load_fragments,
    extract_canvas_regions,
    filter_valid_positions,
    create_output_canvas
)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Chimera: Reconstruct a mosaic from fragment images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling patterns:
  gaussian       Centralized gaussian distribution (default)
  uniform        Random uniform distribution
  poisson        Poisson disk sampling (evenly spaced)
  spiral         Spiral pattern from center
  concentric     Concentric circles
  halton         Low-discrepancy quasi-random
  jittered_grid  Regular grid with jitter

Examples:
  # Basic usage with auto-reduction (default)
  python -m chimera.main --canvas image.jpg --samples 90000

  # Disable fragment reduction
  python -m chimera.main --no-reduce-fragments

  # Custom reduction target
  python -m chimera.main --target-library-size 500

  # Different sampling pattern
  python -m chimera.main --pattern halton --samples 150000

  # Use more memory for faster processing
  python -m chimera.main --memory-fraction 0.7
        """
    )

    parser.add_argument('--canvas', type=str, default='./canvas/puppy.jpeg',
                        help='Path to canvas image')
    parser.add_argument('--fragments', type=str, default='./fragments/*.png',
                        help='Glob pattern for fragment images')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output file path')
    parser.add_argument('--samples', type=int, default=90000,
                        help='Number of positions to sample')
    parser.add_argument('--scaling', type=float, default=3.0,
                        help='Canvas scaling factor')
    parser.add_argument('--fragment-size', type=int, nargs=2, default=[32, 32],
                        metavar=('W', 'H'), help='Fragment size (width height)')
    parser.add_argument('--memory-fraction', type=float, default=0.5,
                        help='Fraction of available memory to use (0.0-1.0, default: 0.5)')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                        help='Device to use: auto (default), cuda, mps, cpu')
    parser.add_argument('--pattern', type=str,
                        choices=list(PATTERN_GENERATORS.keys()),
                        default='gaussian',
                        help='Sampling pattern (default: gaussian)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display output image when done')

    # Fragment reduction options
    parser.add_argument('--no-reduce-fragments', action='store_true',
                        help='Disable automatic fragment library reduction')
    parser.add_argument('--target-library-size', type=int, default=800,
                        help='Target fragment library size after reduction (default: 800)')
    parser.add_argument('--similarity-threshold', type=float, default=0.88,
                        help='Similarity threshold for fragment deduplication (default: 0.88)')

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    print("=" * 60)
    print("CHIMERA - Mosaic Reconstruction")
    print("=" * 60)

    # Device detection
    if args.device == 'auto':
        device = get_device()
    else:
        device = get_device(force_device=args.device)

    # Memory management
    memory_limit = get_memory_limit(args.memory_fraction)

    # Load fragment paths
    FRAGMENT_PATHS = glob(args.fragments)
    if not FRAGMENT_PATHS:
        print(f"❌ Error: No fragment images found matching pattern: {args.fragments}")
        exit(1)

    print(f"> Found {len(FRAGMENT_PATHS):,} fragment images")

    # Fragment reduction (enabled by default)
    if not args.no_reduce_fragments and len(FRAGMENT_PATHS) > args.target_library_size:
        print("\n" + "=" * 60)
        print("Fragment Library Reduction")
        print("=" * 60)

        kept_indices = reduce_fragments(
            FRAGMENT_PATHS,
            target_size=args.target_library_size,
            similarity_threshold=args.similarity_threshold,
            verbose=True
        )

        # Use reduced library
        FRAGMENT_PATHS = [FRAGMENT_PATHS[i] for i in kept_indices]
        print(f"> Using reduced library: {len(FRAGMENT_PATHS):,} fragments")
    else:
        if args.no_reduce_fragments:
            print("> Fragment reduction disabled")

    # Configuration
    SCALING_FACTOR = args.scaling
    FRAGMENT_SIZE = tuple(args.fragment_size)
    CANVAS_PATH = args.canvas
    N_SAMPLES = args.samples

    print(f"> Sampling pattern: {args.pattern}")
    print(f"> Number of samples: {N_SAMPLES:,}")

    # Calculate optimal batch size
    batch_size = calculate_optimal_batch_size(
        memory_limit,
        FRAGMENT_SIZE,
        len(FRAGMENT_PATHS),
        device
    )

    # Load and prepare canvas
    print("\n" + "=" * 60)
    print("Loading canvas...")
    img = Image.open(CANVAS_PATH).convert('RGBA')
    w, h = img.size
    img = img.resize((int(w * SCALING_FACTOR), int(h * SCALING_FACTOR)))
    canvas = img
    canvas_width, canvas_height = int(w * SCALING_FACTOR), int(h * SCALING_FACTOR)
    canvas_output = create_output_canvas(canvas_width, canvas_height)
    print(f"> Canvas size: {canvas_width}x{canvas_height}")

    # Load and preprocess fragments
    print("\n" + "=" * 60)
    print(f"Loading and preprocessing {len(FRAGMENT_PATHS):,} fragments...")
    fragment_images = load_fragments(FRAGMENT_PATHS, FRAGMENT_SIZE)

    fragment_images, fragment_lab_tensors, fragment_masks_tensors = preprocess_fragments(
        fragment_images,
        FRAGMENT_SIZE,
        device
    )

    print(f"> Fragments preprocessed and loaded to {device}")

    # Generate positions
    print("\n" + "=" * 60)
    print(f"Generating {N_SAMPLES:,} positions using '{args.pattern}' pattern...")

    positions = PATTERN_GENERATORS[args.pattern](N_SAMPLES, canvas_width, canvas_height)
    print(f"> Generated {len(positions):,} positions")

    # Filter valid positions
    valid_positions = filter_valid_positions(
        positions,
        canvas_width,
        canvas_height,
        FRAGMENT_SIZE
    )
    print(f"> Valid positions: {len(valid_positions):,}")

    # Process in batches
    print("\n" + "=" * 60)
    print("Processing mosaic...")
    n_batches = (len(valid_positions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(valid_positions))
        batch_positions = valid_positions[batch_start:batch_end]

        # Extract regions for this batch
        regions = extract_canvas_regions(canvas, batch_positions, FRAGMENT_SIZE)

        # Stack into batch tensor
        region_batch = torch.from_numpy(np.stack(regions, axis=0)).to(device)

        # Find best fragments (GPU accelerated)
        with torch.no_grad():
            best_indices = find_best_fragments_batch(
                region_batch,
                fragment_lab_tensors,
                fragment_masks_tensors,
                device
            )

        # Paste fragments
        best_indices_cpu = best_indices.cpu().numpy()
        for i, (x, y) in enumerate(batch_positions):
            best_fragment = fragment_images[best_indices_cpu[i]]
            canvas_output.paste(best_fragment, (x, y), best_fragment)

        # Clear cache periodically to prevent memory buildup
        if batch_idx % 10 == 0:
            clear_cache(device)

    # Save output
    print("\n" + "=" * 60)
    print(f"Saving output to {args.output}...")
    canvas_output.save(args.output)
    print(f"> Saved: {args.output}")

    if not args.no_show:
        canvas_output.show()

    print("\n" + "=" * 60)
    print("> Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
