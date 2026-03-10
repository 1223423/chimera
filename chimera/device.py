from __future__ import annotations

from dataclasses import dataclass
import math

import psutil
import torch


BYTES_PER_GB = 1024 ** 3


@dataclass(slots=True)
class AcceleratorStatus:
    key: str
    label: str
    available: bool


@dataclass(slots=True)
class MemoryBudget:
    requested_bytes: int
    total_budget_bytes: int
    fragment_bytes: int
    working_bytes: int


def list_accelerators() -> list[AcceleratorStatus]:
    return [
        AcceleratorStatus("cuda", "CUDA", torch.cuda.is_available()),
        AcceleratorStatus(
            "mps",
            "MPS",
            torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        ),
        AcceleratorStatus("cpu", "CPU", True),
    ]


def best_available_accelerator() -> AcceleratorStatus:
    for status in list_accelerators():
        if status.available:
            return status
    return AcceleratorStatus("cpu", "CPU", True)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        return torch.device(best_available_accelerator().key)

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")

    if requested == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {requested}")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return "CUDA"
    if device.type == "mps":
        return "MPS"
    return "CPU"


def system_total_memory_bytes() -> int:
    return int(psutil.virtual_memory().total)


def system_available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


def system_total_memory_gb() -> float:
    return system_total_memory_bytes() / BYTES_PER_GB


def available_device_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        total = int(properties.total_memory)
        reserved = int(torch.cuda.memory_reserved(0))
        allocated = int(torch.cuda.memory_allocated(0))
        used = max(reserved, allocated)
        return max(total - used, BYTES_PER_GB)

    return system_available_memory_bytes()


def resolve_memory_budget(
    device: torch.device,
    memory_limit_gb: float,
    fragment_bytes: int,
) -> MemoryBudget:
    if memory_limit_gb <= 0:
        raise ValueError("memory_limit_gb must be > 0")

    requested_bytes = int(memory_limit_gb * BYTES_PER_GB)
    available_bytes = available_device_memory_bytes(device)
    total_budget = min(requested_bytes, available_bytes)

    safety_margin = int(total_budget * 0.12)
    safe_budget = max(total_budget - safety_margin, BYTES_PER_GB)
    working_bytes = max(safe_budget - fragment_bytes, 512 * 1024 * 1024)

    return MemoryBudget(
        requested_bytes=requested_bytes,
        total_budget_bytes=safe_budget,
        fragment_bytes=fragment_bytes,
        working_bytes=working_bytes,
    )


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"

    value = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"

    return f"{num_bytes} B"


def choose_fragment_block_size(device: torch.device, requested: int | None) -> int:
    if requested is not None:
        return max(4, requested)
    if device.type == "cuda":
        return 32
    if device.type == "mps":
        return 16
    return 8


def choose_placement_batch_size(
    working_bytes: int,
    fragment_size: tuple[int, int],
    fragment_block_size: int,
    requested: int | None,
    device: torch.device,
) -> int:
    if requested is not None:
        return max(64, requested)

    pixels = fragment_size[0] * fragment_size[1]
    region_bytes = pixels * 7 * 4
    pairwise_bytes = pixels * 3 * 2 * fragment_block_size
    bytes_per_placement = max(region_bytes + pairwise_bytes, 1)

    max_matching_bytes = max(int(working_bytes * 0.25), 64 * 1024 * 1024)
    batch_size = max_matching_bytes // bytes_per_placement

    if device.type == "cuda":
        return max(256, min(batch_size, 4096))
    if device.type == "mps":
        return max(128, min(batch_size, 2048))
    return max(64, min(batch_size, 1024))


def choose_chunk_size_original(
    source_size: tuple[int, int],
    scaling: float,
    working_bytes: int,
    requested_chunk_size: int | None,
) -> int:
    bytes_per_scaled_pixel = 12
    chunk_budget = max(int(working_bytes * 0.40), 128 * 1024 * 1024)
    chunk_scaled_side = int(math.sqrt(chunk_budget / bytes_per_scaled_pixel))
    computed_original = max(32, int(chunk_scaled_side / max(scaling, 1e-6)))

    if requested_chunk_size is not None:
        chunk_size = min(max(32, requested_chunk_size), computed_original)
    else:
        chunk_size = min(200, computed_original)

    max_source_side = max(source_size[0], source_size[1])
    return max(16, min(chunk_size, max_source_side))


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
