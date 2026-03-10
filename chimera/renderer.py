from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Callable, Iterator

import numpy as np
from PIL import Image
import torch

from .device import (
    MemoryBudget,
    choose_chunk_size_original,
    choose_fragment_block_size,
    choose_placement_batch_size,
    clear_device_cache,
    resolve_memory_budget,
)
from .matching import FragmentLibrary, match_best_fragments_lab, rgb_to_lab_torch


ProgressCallback = Callable[["ProgressUpdate"], None]
PreviewCallback = Callable[[np.ndarray], None]
CancelCallback = Callable[[], bool]


class RenderCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class RenderConfig:
    canvas_path: str
    fragments_glob: str
    output_path: str
    scaling: float = 10.0
    target_coverage: float = 1.0
    fragment_size: tuple[int, int] = (32, 32)
    memory_limit_gb: float = 25.0
    device_request: str = "auto"
    chunk_size_original: int | None = 200
    placement_batch_size: int | None = None
    fragment_block_size: int | None = None
    temp_dir: str | None = None
    keep_temp: bool = False
    seed: int = 42
    preview_max_size: int = 400
    preview_interval_chunks: int = 1


@dataclass(slots=True)
class ResourcePlan:
    memory_budget: MemoryBudget
    fragment_block_size: int
    placement_batch_size: int
    chunk_size_original: int


@dataclass(slots=True)
class ProgressUpdate:
    completed_chunks: int
    total_chunks: int
    message: str

    @property
    def progress_fraction(self) -> float:
        if self.total_chunks <= 0:
            return 0.0
        return self.completed_chunks / self.total_chunks


@dataclass(slots=True)
class RenderSummary:
    output_path: str
    device_label: str
    fragment_count: int
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    elapsed_seconds: float
    resource_plan: ResourcePlan


def validate_render_config(config: RenderConfig) -> None:
    if config.scaling <= 0:
        raise ValueError("Scaling must be greater than 0.")
    if not (0 < config.target_coverage <= 1.0):
        raise ValueError("Target coverage must be in the range (0, 1].")
    if config.fragment_size[0] <= 0 or config.fragment_size[1] <= 0:
        raise ValueError("Fragment size must be positive.")
    if config.memory_limit_gb <= 0:
        raise ValueError("Memory limit must be greater than 0.")
    if config.preview_max_size <= 0:
        raise ValueError("Preview size must be greater than 0.")
    if config.preview_interval_chunks <= 0:
        raise ValueError("Preview interval must be greater than 0.")


def plan_resources(
    config: RenderConfig,
    source_size: tuple[int, int],
    fragment_total_bytes: int,
    device: torch.device,
) -> ResourcePlan:
    memory_budget = resolve_memory_budget(
        device=device,
        memory_limit_gb=config.memory_limit_gb,
        fragment_bytes=fragment_total_bytes,
    )

    fragment_block_size = choose_fragment_block_size(device, config.fragment_block_size)
    placement_batch_size = choose_placement_batch_size(
        working_bytes=memory_budget.working_bytes,
        fragment_size=config.fragment_size,
        fragment_block_size=fragment_block_size,
        requested=config.placement_batch_size,
        device=device,
    )
    chunk_size_original = choose_chunk_size_original(
        source_size=source_size,
        scaling=config.scaling,
        working_bytes=memory_budget.working_bytes,
        requested_chunk_size=config.chunk_size_original,
    )

    return ResourcePlan(
        memory_budget=memory_budget,
        fragment_block_size=fragment_block_size,
        placement_batch_size=placement_batch_size,
        chunk_size_original=chunk_size_original,
    )


class DiskBackedCanvas:
    def __init__(self, width: int, height: int, temp_dir: str | None, keep_temp: bool) -> None:
        self.width = width
        self.height = height
        self.keep_temp = keep_temp
        self._owns_temp_dir = temp_dir is None

        base_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.mkdtemp(prefix="chimera_"))
        base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = base_dir
        self.data_path = self.temp_dir / "mosaic_rgba.dat"
        self._canvas = np.memmap(self.data_path, dtype=np.uint8, mode="w+", shape=(height, width, 4))
        self._canvas[:] = 0

    def read_window(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return np.array(self._canvas[y0:y1, x0:x1], copy=True)

    def write_window(self, x0: int, y0: int, window: np.ndarray) -> None:
        height, width = window.shape[:2]
        self._canvas[y0 : y0 + height, x0 : x0 + width] = window

    def save(self, output_path: str) -> None:
        image = Image.frombuffer(
            "RGBA",
            (self.width, self.height),
            memoryview(self._canvas),
            "raw",
            "RGBA",
            0,
            1,
        )
        image.save(output_path)

    def preview_rgba(self, max_size: int) -> np.ndarray:
        step = max(1, math.ceil(max(self.width / max_size, self.height / max_size)))
        preview = np.array(self._canvas[::step, ::step], copy=True)

        if preview.shape[1] > max_size or preview.shape[0] > max_size:
            preview_image = Image.fromarray(preview)
            preview_image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
            preview = np.asarray(preview_image, dtype=np.uint8)

        return preview

    def flush(self) -> None:
        self._canvas.flush()

    def close(self) -> None:
        self.flush()
        del self._canvas

        if self.keep_temp:
            return

        if self.data_path.exists():
            self.data_path.unlink()

        if self._owns_temp_dir and self.temp_dir.exists():
            self.temp_dir.rmdir()


class ChunkedMosaicRenderer:
    def __init__(
        self,
        config: RenderConfig,
        source_image: Image.Image,
        library: FragmentLibrary,
        device: torch.device,
        resource_plan: ResourcePlan,
        progress_callback: ProgressCallback | None = None,
        preview_callback: PreviewCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> None:
        self.config = config
        self.source_image = source_image
        self.library = library
        self.device = device
        self.resource_plan = resource_plan
        self.progress_callback = progress_callback
        self.preview_callback = preview_callback
        self.cancel_callback = cancel_callback

        self.source_width, self.source_height = source_image.size
        self.output_width = int(round(self.source_width * config.scaling))
        self.output_height = int(round(self.source_height * config.scaling))

        self.canvas = DiskBackedCanvas(
            width=self.output_width,
            height=self.output_height,
            temp_dir=config.temp_dir,
            keep_temp=config.keep_temp,
        )

        fragment_width, fragment_height = self.library.fragment_size
        center_x, center_y = self.library.centered_xy
        self.left_margin = center_x
        self.top_margin = center_y
        self.right_margin = fragment_width - center_x - 1
        self.bottom_margin = fragment_height - center_y - 1
        self.fragment_x_offsets = torch.arange(fragment_width, device=device, dtype=torch.long)
        self.fragment_y_offsets = torch.arange(fragment_height, device=device, dtype=torch.long)
        self.rng = np.random.default_rng(config.seed)

    def run(self) -> None:
        completed_chunks = 0
        total_chunks = self.total_chunks

        try:
            for chunk_bounds in self._iter_original_chunks():
                self._check_cancelled()
                self._process_chunk(*chunk_bounds)
                completed_chunks += 1

                if self.progress_callback is not None:
                    self.progress_callback(
                        ProgressUpdate(
                            completed_chunks=completed_chunks,
                            total_chunks=total_chunks,
                            message=f"Chunk {completed_chunks} of {total_chunks}",
                        )
                    )

                if self.preview_callback is not None and (
                    completed_chunks == total_chunks
                    or completed_chunks % self.config.preview_interval_chunks == 0
                ):
                    self.preview_callback(self.canvas.preview_rgba(self.config.preview_max_size))

            self.canvas.flush()
            self.canvas.save(self.config.output_path)
        finally:
            self.canvas.close()
            clear_device_cache(self.device)

    @property
    def total_chunks(self) -> int:
        chunk_size = self.resource_plan.chunk_size_original
        return math.ceil(self.source_height / chunk_size) * math.ceil(self.source_width / chunk_size)

    def _iter_original_chunks(self) -> Iterator[tuple[int, int, int, int]]:
        chunk_size = self.resource_plan.chunk_size_original
        for y0 in range(0, self.source_height, chunk_size):
            y1 = min(self.source_height, y0 + chunk_size)
            for x0 in range(0, self.source_width, chunk_size):
                x1 = min(self.source_width, x0 + chunk_size)
                yield y0, y1, x0, x1

    def _check_cancelled(self) -> None:
        if self.cancel_callback is not None and self.cancel_callback():
            raise RenderCancelled("Render cancelled.")

    def _scaled_coord(self, value: int) -> int:
        return int(round(value * self.config.scaling))

    def _process_chunk(self, source_y0: int, source_y1: int, source_x0: int, source_x1: int) -> None:
        scaled_x0 = self._scaled_coord(source_x0)
        scaled_x1 = self._scaled_coord(source_x1)
        scaled_y0 = self._scaled_coord(source_y0)
        scaled_y1 = self._scaled_coord(source_y1)

        if scaled_x1 <= scaled_x0 or scaled_y1 <= scaled_y0:
            return

        extended_x0 = max(0, scaled_x0 - self.left_margin)
        extended_y0 = max(0, scaled_y0 - self.top_margin)
        extended_x1 = min(self.output_width, scaled_x1 + self.right_margin)
        extended_y1 = min(self.output_height, scaled_y1 + self.bottom_margin)

        working_tile_rgba = self.canvas.read_window(extended_x0, extended_y0, extended_x1, extended_y1)

        chunk_rel_x0 = scaled_x0 - extended_x0
        chunk_rel_x1 = scaled_x1 - extended_x0
        chunk_rel_y0 = scaled_y0 - extended_y0
        chunk_rel_y1 = scaled_y1 - extended_y0

        local_mask = working_tile_rgba[chunk_rel_y0:chunk_rel_y1, chunk_rel_x0:chunk_rel_x1, 3] > 0
        total_pixels = int(local_mask.size)
        if total_pixels == 0:
            return
        covered_pixels = int(local_mask.sum())
        target_pixels = max(1, int(math.ceil(self.config.target_coverage * total_pixels)))
        if covered_pixels >= target_pixels:
            return

        working_tile = rgba_u8_to_premultiplied(working_tile_rgba)
        target_window = render_scaled_window(
            source=self.source_image,
            scaling=self.config.scaling,
            x0=extended_x0,
            y0=extended_y0,
            x1=extended_x1,
            y1=extended_y1,
        )
        with torch.inference_mode():
            target_window_lab = rgb_to_lab_torch(
                torch.from_numpy(np.ascontiguousarray(target_window[..., :3])).to(self.device, dtype=torch.float32) / 255.0
            ).to(dtype=torch.float16)
        del target_window

        center_x, center_y = self.library.centered_xy
        chunk_width = local_mask.shape[1]
        local_mask_flat = local_mask.ravel()
        free_pool = np.flatnonzero(~local_mask_flat).astype(np.int32)

        stagnant_batches = 0
        compact_after_batches = 4
        batches_since_compaction = 0
        while covered_pixels < target_pixels:
            self._check_cancelled()

            if batches_since_compaction >= compact_after_batches:
                free_pool = free_pool[~local_mask_flat[free_pool]]
                batches_since_compaction = 0

            if free_pool.size == 0:
                break

            batch_size = min(self.resource_plan.placement_batch_size, int(free_pool.size))
            sample_count = min(free_pool.size, max(batch_size, batch_size * 2))
            sampled_positions = self.rng.choice(free_pool.size, size=sample_count, replace=False)
            sampled_pixels = free_pool[sampled_positions]
            sampled_pixels = sampled_pixels[~local_mask_flat[sampled_pixels]]
            if sampled_pixels.size == 0:
                free_pool = free_pool[~local_mask_flat[free_pool]]
                batches_since_compaction = 0
                continue
            if sampled_pixels.size > batch_size:
                sampled_pixels = sampled_pixels[:batch_size]
            elif sampled_pixels.size < batch_size:
                free_pool = free_pool[~local_mask_flat[free_pool]]
                batches_since_compaction = 0

            ys, xs = np.divmod(sampled_pixels, chunk_width)
            centers_x = scaled_x0 + xs.astype(np.int32)
            centers_y = scaled_y0 + ys.astype(np.int32)

            top_left_x = centers_x - center_x
            top_left_y = centers_y - center_y

            region_lab_batch = extract_regions_lab_torch(
                target_window_lab=target_window_lab,
                region_top_left_x=torch.from_numpy(top_left_x).to(self.device, dtype=torch.long),
                region_top_left_y=torch.from_numpy(top_left_y).to(self.device, dtype=torch.long),
                target_window_x0=extended_x0,
                target_window_y0=extended_y0,
                x_offsets=self.fragment_x_offsets,
                y_offsets=self.fragment_y_offsets,
            )
            with torch.inference_mode():
                best_indices = match_best_fragments_lab(
                    region_lab_batch=region_lab_batch,
                    library=self.library,
                    fragment_block_size=self.resource_plan.fragment_block_size,
                    require_center_coverage=True,
                )

            best_indices_cpu = best_indices.detach().cpu().numpy()
            gained_pixels = 0

            for batch_index, fragment_index in enumerate(best_indices_cpu):
                fragment_x = int(top_left_x[batch_index])
                fragment_y = int(top_left_y[batch_index])

                alpha_composite_premultiplied_inplace(
                    destination=working_tile,
                    source=self.library.premultiplied_rgba_cpu[fragment_index],
                    source_global_x=fragment_x,
                    source_global_y=fragment_y,
                    destination_global_x0=extended_x0,
                    destination_global_y0=extended_y0,
                )

                gained_pixels += update_local_mask(
                    local_mask=local_mask,
                    fragment_mask=self.library.alpha_cpu[fragment_index],
                    fragment_global_x=fragment_x,
                    fragment_global_y=fragment_y,
                    chunk_global_x0=scaled_x0,
                    chunk_global_y0=scaled_y0,
                    chunk_global_x1=scaled_x1,
                    chunk_global_y1=scaled_y1,
                )

            if gained_pixels <= 0:
                stagnant_batches += 1
                if stagnant_batches >= 6:
                    break
            else:
                covered_pixels += gained_pixels
                stagnant_batches = 0
                batches_since_compaction += 1

        self.canvas.write_window(extended_x0, extended_y0, premultiplied_rgba_to_u8(working_tile))


def render_scaled_window(
    source: Image.Image,
    scaling: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 4), dtype=np.uint8)

    inv_scale = 1.0 / scaling
    affine = (inv_scale, 0.0, x0 * inv_scale, 0.0, inv_scale, y0 * inv_scale)
    window = source.transform(
        (width, height),
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(window, dtype=np.uint8)


def extract_regions_lab_torch(
    target_window_lab: torch.Tensor,
    region_top_left_x: torch.Tensor,
    region_top_left_y: torch.Tensor,
    target_window_x0: int,
    target_window_y0: int,
    x_offsets: torch.Tensor,
    y_offsets: torch.Tensor,
) -> torch.Tensor:
    if target_window_lab.ndim != 3 or target_window_lab.shape[-1] != 3:
        raise ValueError("target_window_lab must have shape (H, W, 3)")

    window_height, window_width = target_window_lab.shape[:2]
    x_positions = region_top_left_x.unsqueeze(1) - target_window_x0 + x_offsets.unsqueeze(0)
    y_positions = region_top_left_y.unsqueeze(1) - target_window_y0 + y_offsets.unsqueeze(0)

    valid_x = (x_positions >= 0) & (x_positions < window_width)
    valid_y = (y_positions >= 0) & (y_positions < window_height)

    x_clamped = x_positions.clamp(0, window_width - 1)
    y_clamped = y_positions.clamp(0, window_height - 1)

    regions = target_window_lab[y_clamped[:, :, None], x_clamped[:, None, :]]
    valid = valid_y[:, :, None] & valid_x[:, None, :]
    return regions * valid.unsqueeze(-1).to(dtype=target_window_lab.dtype)


def rgba_u8_to_premultiplied(rgba: np.ndarray) -> np.ndarray:
    premultiplied = rgba.astype(np.float32) / 255.0
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    return premultiplied


def premultiplied_rgba_to_u8(premultiplied_rgba: np.ndarray) -> np.ndarray:
    alpha = premultiplied_rgba[..., 3:4]
    rgb = np.divide(
        premultiplied_rgba[..., :3],
        np.clip(alpha, 1e-6, 1.0),
        out=np.zeros_like(premultiplied_rgba[..., :3]),
        where=alpha > 1e-6,
    )
    rgba = np.empty_like(premultiplied_rgba)
    rgba[..., :3] = rgb
    rgba[..., 3:4] = alpha
    return np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


def alpha_composite_premultiplied_inplace(
    destination: np.ndarray,
    source: np.ndarray,
    source_global_x: int,
    source_global_y: int,
    destination_global_x0: int,
    destination_global_y0: int,
) -> None:
    source_height, source_width = source.shape[:2]
    destination_height, destination_width = destination.shape[:2]

    dst_x0 = source_global_x - destination_global_x0
    dst_y0 = source_global_y - destination_global_y0
    dst_x1 = dst_x0 + source_width
    dst_y1 = dst_y0 + source_height

    intersect_x0 = max(0, dst_x0)
    intersect_y0 = max(0, dst_y0)
    intersect_x1 = min(destination_width, dst_x1)
    intersect_y1 = min(destination_height, dst_y1)
    if intersect_x0 >= intersect_x1 or intersect_y0 >= intersect_y1:
        return

    src_x0 = intersect_x0 - dst_x0
    src_y0 = intersect_y0 - dst_y0
    src_x1 = src_x0 + (intersect_x1 - intersect_x0)
    src_y1 = src_y0 + (intersect_y1 - intersect_y0)

    src_rgba = source[src_y0:src_y1, src_x0:src_x1]
    dst_rgba = destination[intersect_y0:intersect_y1, intersect_x0:intersect_x1]

    src_alpha = src_rgba[..., 3:4]
    inv_alpha = 1.0 - src_alpha

    dst_rgba[..., :3] *= inv_alpha
    dst_rgba[..., :3] += src_rgba[..., :3]
    dst_rgba[..., 3:4] *= inv_alpha
    dst_rgba[..., 3:4] += src_alpha


def update_local_mask(
    local_mask: np.ndarray,
    fragment_mask: np.ndarray,
    fragment_global_x: int,
    fragment_global_y: int,
    chunk_global_x0: int,
    chunk_global_y0: int,
    chunk_global_x1: int,
    chunk_global_y1: int,
) -> int:
    fragment_height, fragment_width = fragment_mask.shape

    frag_x0 = fragment_global_x
    frag_y0 = fragment_global_y
    frag_x1 = frag_x0 + fragment_width
    frag_y1 = frag_y0 + fragment_height

    intersect_x0 = max(frag_x0, chunk_global_x0)
    intersect_y0 = max(frag_y0, chunk_global_y0)
    intersect_x1 = min(frag_x1, chunk_global_x1)
    intersect_y1 = min(frag_y1, chunk_global_y1)
    if intersect_x0 >= intersect_x1 or intersect_y0 >= intersect_y1:
        return 0

    local_x0 = intersect_x0 - chunk_global_x0
    local_y0 = intersect_y0 - chunk_global_y0
    local_x1 = local_x0 + (intersect_x1 - intersect_x0)
    local_y1 = local_y0 + (intersect_y1 - intersect_y0)

    fragment_x0 = intersect_x0 - frag_x0
    fragment_y0 = intersect_y0 - frag_y0
    fragment_x1 = fragment_x0 + (intersect_x1 - intersect_x0)
    fragment_y1 = fragment_y0 + (intersect_y1 - intersect_y0)

    local_view = local_mask[local_y0:local_y1, local_x0:local_x1]
    fragment_view = fragment_mask[
        fragment_y0:fragment_y1,
        fragment_x0:fragment_x1,
    ]
    newly_covered = int(np.count_nonzero(fragment_view & ~local_view))
    local_view |= fragment_view
    return newly_covered
