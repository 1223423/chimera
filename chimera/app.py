from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
import time

from PIL import Image
import torch

from .device import describe_device, resolve_device
from .matching import FragmentLibrary
from .renderer import (
    CancelCallback,
    ChunkedMosaicRenderer,
    PreviewCallback,
    ProgressCallback,
    RenderConfig,
    RenderSummary,
    ResourcePlan,
    plan_resources,
    validate_render_config,
)


@dataclass(slots=True)
class PreparedRender:
    config: RenderConfig
    source_image: Image.Image
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    fragment_paths: list[str]
    library: FragmentLibrary
    device: torch.device
    device_label: str
    resource_plan: ResourcePlan


def collect_fragment_paths(pattern: str) -> list[str]:
    return sorted(glob(pattern))


def prepare_render(config: RenderConfig) -> PreparedRender:
    validate_render_config(config)

    fragment_paths = collect_fragment_paths(config.fragments_glob)
    if not fragment_paths:
        raise RuntimeError(f"No fragment images matched: {config.fragments_glob}")

    device = resolve_device(config.device_request)

    with Image.open(config.canvas_path) as image_handle:
        source_image = image_handle.convert("RGBA")

    library = FragmentLibrary.from_paths(
        fragment_paths=fragment_paths,
        fragment_size=config.fragment_size,
        device=device,
    )

    source_size = source_image.size
    output_size = (
        int(round(source_size[0] * config.scaling)),
        int(round(source_size[1] * config.scaling)),
    )
    resource_plan = plan_resources(
        config=config,
        source_size=source_size,
        fragment_total_bytes=library.total_bytes,
        device=device,
    )

    return PreparedRender(
        config=config,
        source_image=source_image,
        source_size=source_size,
        output_size=output_size,
        fragment_paths=fragment_paths,
        library=library,
        device=device,
        device_label=describe_device(device),
        resource_plan=resource_plan,
    )


def run_render(
    config: RenderConfig,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> RenderSummary:
    prepared = prepare_render(config)
    renderer = ChunkedMosaicRenderer(
        config=prepared.config,
        source_image=prepared.source_image,
        library=prepared.library,
        device=prepared.device,
        resource_plan=prepared.resource_plan,
        progress_callback=progress_callback,
        preview_callback=preview_callback,
        cancel_callback=cancel_callback,
    )

    started_at = time.perf_counter()
    renderer.run()
    elapsed = time.perf_counter() - started_at

    return RenderSummary(
        output_path=prepared.config.output_path,
        device_label=prepared.device_label,
        fragment_count=prepared.library.count,
        source_size=prepared.source_size,
        output_size=prepared.output_size,
        elapsed_seconds=elapsed,
        resource_plan=prepared.resource_plan,
    )


def build_fragment_glob(folder: str, pattern: str) -> str:
    folder_path = Path(folder).expanduser()
    return str(folder_path / pattern)
