from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import color
from multiprocessing import Pool

def generate_gaussian(n, width, height):
    '''
        Gaussian possition generator for an interesting centralized effect
    '''
    positions = set()
    center_x, center_y = width / 2, height / 2
    std_dev_x, std_dev_y = width / 6, height / 6

    while len(positions) < n:
        x = int(np.random.normal(center_x, std_dev_x))
        y = int(np.random.normal(center_y, std_dev_y))

        if 0 <= x < width and 0 <= y < height:
            positions.add((x, y))
    return list(positions)

def generate_grid(width, height, increment_width, increment_height):
    '''
        Grid generator; mostly used to oversample before applying uniform random positions
    '''
    return list(itertools.product(range(0, width, int(increment_width)), range(0, height, int(increment_height))))

def generate_uniform(width, height):
    '''
        Uniform generator; mostly used for adding random over-sampled jitter aesthetic
    '''
    return list(zip(np.random.randint(0, width, N_SAMPLES), np.random.randint(0, height, N_SAMPLES)))

def masked_mae(fragment_lab_masked, region_lab_masked, non_transparent_pixels):
    '''
        For each image fragment calculates mean absolute error per pixel, the sum of which is weighted by transparency
    '''
    mae_errors = np.zeros(len(fragment_lab_masked))
    for i in range(len(fragment_lab_masked)):
        abs_error = np.abs(fragment_lab_masked[i] - region_lab_masked[i])
        total_abs_error = np.sum(abs_error)
        mae_error = total_abs_error / non_transparent_pixels[i] 
        mae_errors[i] = mae_error
    return mae_errors

def place_fragment(args):
    '''
        Given a region of the canvas returns the closest resulting fragment
    '''
    (x, y), region = args
    region = np.array(region, dtype='float32') / 255.
    region_lab = color.rgb2lab(region[..., :3])
    region_lab_masked = [region_lab[mask] for mask in fragment_masks]
    errors = masked_mae(fragment_lab_masked, region_lab_masked, non_transparent_pixels)
    best_idx = np.argmin(errors)
    best_fragment = fragment_images[best_idx]
    return best_fragment, x, y


# Configuration
SCALING_FACTOR = 3
FRAGMENT_SIZE = (32, 32)
CHUNK_SIZE = 100000
CANVAS_PATH = './canvas/puppy.jpeg'
FRAGMENT_PATHS = glob('./fragments/*.png')
N_SAMPLES = 90000

# Configure reference canvas and blank output canvas
img = Image.open(CANVAS_PATH).convert('RGBA')
w, h = img.size
img = img.resize((w*SCALING_FACTOR, h*SCALING_FACTOR))
canvas = img
canvas_output = Image.new('RGBA', (w*SCALING_FACTOR, h*SCALING_FACTOR), 'black')

# Precompute fragments into LAB colorspace, transparency masks, transparency weights, and masked matrices
fragment_images = [Image.open(filename).convert('RGBA').resize(FRAGMENT_SIZE) for filename in FRAGMENT_PATHS]
fragment_arrays = [np.array(img, dtype='float32') / 255. for img in fragment_images]
lab_fragments = [color.rgb2lab(img[..., :3]) for img in fragment_arrays]
alpha_fragments = [img[..., 3] for img in fragment_arrays]
fragment_masks = [alpha > 0 for alpha in alpha_fragments]
non_transparent_pixels = [np.sum(mask) for mask in fragment_masks]
fragment_lab_masked = [lab[mask] for lab, mask in zip(lab_fragments, fragment_masks)]

# Should I trust python GC?
del fragment_arrays, lab_fragments, alpha_fragments

# Generate positions and handle chunking
positions = generate_gaussian(N_SAMPLES, w*SCALING_FACTOR, h*SCALING_FACTOR)
n_chunks = np.round(len(positions)/CHUNK_SIZE)

if __name__ == "__main__":

    for chunk, chunk_start in enumerate(range(0, len(positions), CHUNK_SIZE)):
        print(f'Rendering chunk {chunk+1} of {int(n_chunks)}')
        chunk_end = min(chunk_start + CHUNK_SIZE, len(positions))
        chunk_positions = positions[chunk_start:chunk_end]

        args = [((x, y), canvas.crop((x, y, x+FRAGMENT_SIZE[0], y+FRAGMENT_SIZE[1]))) for x, y in chunk_positions 
                if x + FRAGMENT_SIZE[0] <= w*SCALING_FACTOR and y + FRAGMENT_SIZE[1] <= h*SCALING_FACTOR]

        with Pool() as pool:
            for best_fragment, x, y in tqdm(pool.imap_unordered(place_fragment, args), total=len(args)):
                canvas_output.paste(best_fragment, (x, y), best_fragment)

    canvas_output.save('output.png')
    canvas_output.show()