import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from PIL import Image
from skimage import color as skcolor


def extract_color_features(fragment_array, mask):
    """
    Extract color features from a fragment for clustering

    Args:
        fragment_array: RGBA array (H, W, 4) normalized to [0, 1]
        mask: Boolean mask of non-transparent pixels

    Returns:
        numpy array: LAB color features (L, a, b)
    """
    # Convert to LAB
    rgb = fragment_array[..., :3]
    lab = skcolor.rgb2lab(rgb)

    # Get average LAB color (weighted by alpha)
    if np.any(mask):
        masked_lab = lab[mask]
        avg_lab = np.mean(masked_lab, axis=0)
    else:
        avg_lab = np.array([50.0, 0.0, 0.0])  # Neutral gray

    return avg_lab


def compute_lab_histogram(fragment_array, mask, bins=(4, 4, 4)):
    """
    Compute LAB color histogram for detailed similarity

    Args:
        fragment_array: RGBA array (H, W, 4) normalized to [0, 1]
        mask: Boolean mask of non-transparent pixels
        bins: Number of bins per LAB channel

    Returns:
        numpy array: Flattened histogram
    """
    rgb = fragment_array[..., :3]
    lab = skcolor.rgb2lab(rgb)

    # Only consider non-transparent pixels
    if not np.any(mask):
        return np.zeros(bins[0] * bins[1] * bins[2])

    masked_lab = lab[mask]

    # LAB ranges: L [0, 100], a [-127, 127], b [-127, 127]
    hist, _ = np.histogramdd(
        masked_lab,
        bins=bins,
        range=[[0, 100], [-127, 127], [-127, 127]]
    )

    # Normalize
    hist = hist.flatten()
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def histogram_similarity(hist1, hist2):
    """
    Compute Bhattacharyya coefficient between histograms

    Args:
        hist1, hist2: Normalized histograms

    Returns:
        float: Similarity in [0, 1], 1 = identical
    """
    return np.sum(np.sqrt(hist1 * hist2))


def reduce_fragments(fragment_paths, target_size=800, n_color_clusters=40,
                     similarity_threshold=0.88, min_per_cluster=3, verbose=True):
    """
    Reduce fragment library while preserving diversity

    Strategy:
    1. Cluster fragments by dominant color (preserves color diversity)
    2. Within each cluster, remove similar fragments (removes redundancy)
    3. Use greedy diverse sampling (maximizes coverage)

    Args:
        fragment_paths: List of file paths to fragment images
        target_size: Target library size
        n_color_clusters: Number of color clusters to preserve diversity
        similarity_threshold: Similarity above which fragments are considered duplicates
        min_per_cluster: Minimum fragments to keep per cluster
        verbose: Print progress information

    Returns:
        list: Indices of fragments to keep
    """
    n_fragments = len(fragment_paths)

    if n_fragments <= target_size:
        if verbose:
            print(f"> Fragment count ({n_fragments}) already below target ({target_size}), skipping reduction")
        return list(range(n_fragments))

    if verbose:
        print(f"> Reducing {n_fragments:,} fragments to ~{target_size:,}")
        print("> Stage 1/3: Extracting color features...")

    # Stage 1: Extract color features for all fragments
    color_features = []
    histograms = []

    for i, fpath in enumerate(fragment_paths):
        if verbose and (i + 1) % 500 == 0:
            print(f">   Processed {i+1:,}/{n_fragments:,} fragments...")

        img = Image.open(fpath).convert('RGBA')
        arr = np.array(img, dtype='float32') / 255.0
        mask = arr[..., 3] > 0

        # Extract features
        color_feat = extract_color_features(arr, mask)
        hist = compute_lab_histogram(arr, mask)

        color_features.append(color_feat)
        histograms.append(hist)

    color_features = np.array(color_features)
    histograms = np.array(histograms)

    # Stage 2: Cluster by color
    if verbose:
        print(f"> Stage 2/3: Clustering into {n_color_clusters} color groups...")

    # Use MiniBatchKMeans for speed with large datasets
    kmeans = MiniBatchKMeans(n_clusters=n_color_clusters, random_state=42, batch_size=256)
    cluster_labels = kmeans.fit_predict(color_features)

    # Stage 3: Reduce within each cluster
    if verbose:
        print("> Stage 3/3: Reducing within clusters...")

    kept_indices = []

    for cluster_id in range(n_color_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        n_in_cluster = len(cluster_indices)

        if n_in_cluster == 0:
            continue

        # Calculate target for this cluster (proportional to cluster size)
        cluster_target = max(
            min_per_cluster,
            int(target_size * n_in_cluster / n_fragments)
        )

        if n_in_cluster <= cluster_target:
            # Keep all if cluster is small
            kept_indices.extend(cluster_indices.tolist())
        else:
            # Reduce using histogram similarity
            cluster_histograms = histograms[cluster_indices]

            # Greedy diverse sampling
            selected = greedy_diverse_sampling(
                cluster_indices,
                cluster_histograms,
                max_keep=cluster_target,
                similarity_threshold=similarity_threshold
            )
            kept_indices.extend(selected)

    # Final adjustment to hit target size
    if len(kept_indices) > target_size:
        if verbose:
            print(f"> Final trim: {len(kept_indices):,} → {target_size:,}")
        # Keep most diverse
        kept_indices = final_diverse_sampling(
            kept_indices,
            histograms[kept_indices],
            target_size
        )

    if verbose:
        print(f"> Reduction complete: {n_fragments:,} → {len(kept_indices):,} fragments")
        print(f">   Reduction ratio: {len(kept_indices)/n_fragments:.1%}")

    return kept_indices


def greedy_diverse_sampling(indices, histograms, max_keep, similarity_threshold):
    """
    Greedy diverse sampling within a cluster

    Iteratively selects the most different fragment from already-selected ones

    Args:
        indices: Array of fragment indices
        histograms: Histogram features for these fragments
        max_keep: Maximum number to keep
        similarity_threshold: Stop if all remaining are too similar

    Returns:
        list: Selected indices
    """
    n = len(indices)
    if n <= max_keep:
        return indices.tolist()

    kept = []
    available = set(range(n))

    # Start with a random fragment (for speed, not always most distinct)
    first_idx = np.random.randint(0, n)
    kept.append(first_idx)
    available.remove(first_idx)

    # Greedy addition
    while len(kept) < max_keep and available:
        # Find fragment most dissimilar to all kept
        min_max_similarity = float('inf')
        best_candidate = None

        for candidate in available:
            # Compute similarity to all kept fragments
            similarities = [
                histogram_similarity(histograms[candidate], histograms[k])
                for k in kept
            ]
            max_similarity = max(similarities)

            # We want the candidate with minimum max_similarity (most dissimilar to kept set)
            if max_similarity < min_max_similarity:
                min_max_similarity = max_similarity
                best_candidate = candidate

        # If even the most different is too similar, stop
        if min_max_similarity > similarity_threshold:
            break

        if best_candidate is not None:
            kept.append(best_candidate)
            available.remove(best_candidate)

    return [indices[i] for i in kept]


def final_diverse_sampling(indices, histograms, target_size):
    """
    Final sampling to reach exact target size

    Uses distance-based sampling to maximize diversity

    Args:
        indices: Fragment indices
        histograms: Histogram features
        target_size: Exact number to keep

    Returns:
        list: Selected indices
    """
    n = len(indices)
    if n <= target_size:
        return indices

    # Compute pairwise histogram similarities
    # For efficiency, sample if too large
    if n > 2000:
        # Random sample
        sample_indices = np.random.choice(n, target_size, replace=False)
        return [indices[i] for i in sample_indices]

    # Farthest point sampling
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[0] = True  # Start with first

    for _ in range(1, target_size):
        # Find point farthest from any kept point
        kept_indices_local = np.where(kept_mask)[0]
        kept_hists = histograms[kept_indices_local]

        distances = []
        for i in range(n):
            if kept_mask[i]:
                distances.append(-1)
            else:
                # Minimum similarity to any kept (maximum distance)
                sims = [histogram_similarity(histograms[i], kh) for kh in kept_hists]
                distances.append(1.0 - max(sims))  # Convert to distance

        # Select farthest
        farthest = np.argmax(distances)
        kept_mask[farthest] = True

    return [indices[i] for i in np.where(kept_mask)[0]]
