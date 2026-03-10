from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
import torch


def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)

    srgb_mask = rgb > 0.04045
    linear = torch.where(srgb_mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    rgb_to_xyz = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=linear.dtype,
        device=linear.device,
    )

    flat = linear.reshape(-1, 3)
    xyz = torch.matmul(flat, rgb_to_xyz.T).reshape(linear.shape)

    ref_white = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    normalized = xyz / ref_white

    epsilon = 0.008856
    kappa = 903.3
    lab_basis = torch.where(
        normalized > epsilon,
        normalized.pow(1.0 / 3.0),
        (kappa * normalized + 16.0) / 116.0,
    )

    l_channel = 116.0 * lab_basis[..., 1] - 16.0
    a_channel = 500.0 * (lab_basis[..., 0] - lab_basis[..., 1])
    b_channel = 200.0 * (lab_basis[..., 1] - lab_basis[..., 2])

    return torch.stack((l_channel, a_channel, b_channel), dim=-1)


def recenter_fragment_by_alpha_centroid(fragment_rgba: np.ndarray) -> np.ndarray:
    alpha = fragment_rgba[..., 3] > 0
    if not np.any(alpha):
        return fragment_rgba

    ys, xs = np.nonzero(alpha)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    height, width = alpha.shape
    target_x = (width - 1) / 2.0
    target_y = (height - 1) / 2.0

    shift_x = int(round(target_x - centroid_x))
    shift_y = int(round(target_y - centroid_y))

    shifted = np.zeros_like(fragment_rgba)

    src_x0 = max(0, -shift_x)
    src_y0 = max(0, -shift_y)
    src_x1 = min(width, width - shift_x)
    src_y1 = min(height, height - shift_y)

    dst_x0 = max(0, shift_x)
    dst_y0 = max(0, shift_y)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x0 < src_x1 and src_y0 < src_y1:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = fragment_rgba[src_y0:src_y1, src_x0:src_x1]

    return shifted


@dataclass(slots=True)
class FragmentLibrary:
    fragment_size: tuple[int, int]
    centered_xy: tuple[int, int]
    premultiplied_rgba_cpu: list[np.ndarray]
    alpha_cpu: list[np.ndarray]
    lab_tensor: torch.Tensor
    alpha_weights_tensor: torch.Tensor
    center_opaque_tensor: torch.Tensor
    mask_pixel_counts: torch.Tensor

    @property
    def count(self) -> int:
        return len(self.premultiplied_rgba_cpu)

    @property
    def total_bytes(self) -> int:
        cpu_bytes = sum(fragment.nbytes for fragment in self.premultiplied_rgba_cpu)
        cpu_bytes += sum(mask.nbytes for mask in self.alpha_cpu)

        tensor_bytes = self.lab_tensor.numel() * self.lab_tensor.element_size()
        tensor_bytes += self.alpha_weights_tensor.numel() * self.alpha_weights_tensor.element_size()
        tensor_bytes += self.center_opaque_tensor.numel() * self.center_opaque_tensor.element_size()
        tensor_bytes += self.mask_pixel_counts.numel() * self.mask_pixel_counts.element_size()

        return int(cpu_bytes + tensor_bytes)

    @classmethod
    def from_paths(
        cls,
        fragment_paths: Sequence[str],
        fragment_size: tuple[int, int],
        device: torch.device,
    ) -> "FragmentLibrary":
        fragment_width, fragment_height = fragment_size
        center_x = fragment_width // 2
        center_y = fragment_height // 2

        premultiplied_rgba_cpu: list[np.ndarray] = []
        alpha_cpu: list[np.ndarray] = []
        lab_list: list[torch.Tensor] = []
        alpha_list: list[torch.Tensor] = []
        center_opaque_list: list[bool] = []
        mask_counts: list[float] = []

        for path in fragment_paths:
            with Image.open(path) as image_handle:
                fragment = image_handle.convert("RGBA").resize(
                    (fragment_width, fragment_height),
                    Image.Resampling.LANCZOS,
                )

            fragment_rgba = np.asarray(fragment, dtype=np.uint8)
            fragment_rgba = recenter_fragment_by_alpha_centroid(fragment_rgba)

            alpha = fragment_rgba[..., 3] > 0
            if not np.any(alpha):
                continue

            premultiplied_rgba = fragment_rgba.astype(np.float32) / 255.0
            premultiplied_rgba[..., :3] *= premultiplied_rgba[..., 3:4]

            premultiplied_rgba_cpu.append(premultiplied_rgba)
            alpha_cpu.append(alpha)

            rgb_tensor = torch.from_numpy(fragment_rgba[..., :3]).to(device=device, dtype=torch.float32) / 255.0
            lab_list.append(rgb_to_lab_torch(rgb_tensor).to(dtype=torch.float16))
            alpha_list.append(torch.from_numpy(alpha).to(device=device, dtype=torch.float16))

            center_opaque_list.append(bool(alpha[center_y, center_x]))
            mask_counts.append(float(alpha.sum()))

        if not premultiplied_rgba_cpu:
            raise RuntimeError("No usable fragments were loaded after preprocessing.")

        return cls(
            fragment_size=fragment_size,
            centered_xy=(center_x, center_y),
            premultiplied_rgba_cpu=premultiplied_rgba_cpu,
            alpha_cpu=alpha_cpu,
            lab_tensor=torch.stack(lab_list, dim=0),
            alpha_weights_tensor=torch.stack(alpha_list, dim=0),
            center_opaque_tensor=torch.tensor(center_opaque_list, device=device, dtype=torch.bool),
            mask_pixel_counts=torch.tensor(mask_counts, device=device, dtype=torch.float32),
        )


def match_best_fragments_lab(
    region_lab_batch: torch.Tensor,
    library: FragmentLibrary,
    fragment_block_size: int,
    require_center_coverage: bool = True,
) -> torch.Tensor:
    if region_lab_batch.ndim != 4 or region_lab_batch.shape[-1] != 3:
        raise ValueError("region_lab_batch must have shape (B, H, W, 3)")

    device = library.lab_tensor.device
    region_lab_batch = region_lab_batch.to(device=device, dtype=torch.float16)
    expanded_regions = region_lab_batch.unsqueeze(1)

    batch_size = region_lab_batch.shape[0]
    best_errors = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=device)
    best_indices = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for start in range(0, library.count, fragment_block_size):
        end = min(start + fragment_block_size, library.count)

        fragment_lab = library.lab_tensor[start:end]
        alpha_weights = library.alpha_weights_tensor[start:end].unsqueeze(0).unsqueeze(-1)

        differences = torch.abs(expanded_regions - fragment_lab.unsqueeze(0)) * alpha_weights
        errors = differences.sum(dim=(2, 3, 4)).to(dtype=torch.float32)
        errors = errors / library.mask_pixel_counts[start:end].unsqueeze(0)

        if require_center_coverage:
            center_ok = library.center_opaque_tensor[start:end]
            if torch.any(center_ok):
                errors[:, ~center_ok] += 1e6

        block_errors, block_indices = torch.min(errors, dim=1)
        improved = block_errors < best_errors
        best_errors[improved] = block_errors[improved]
        best_indices[improved] = block_indices[improved] + start

    return best_indices
