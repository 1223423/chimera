from __future__ import annotations

import argparse
import sys

from .app import run_render
from .device import format_bytes
from .renderer import ProgressUpdate, RenderConfig


class ConsoleProgress:
    def __init__(self) -> None:
        self._last_percent = -1

    def __call__(self, update: ProgressUpdate) -> None:
        percent = int(update.progress_fraction * 100)
        if percent == self._last_percent:
            return
        self._last_percent = percent
        sys.stdout.write(f"\rRendering... {percent:3d}%")
        sys.stdout.flush()

        if update.completed_chunks >= update.total_chunks:
            sys.stdout.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chimera renderer CLI")
    parser.add_argument("--canvas", type=str, default="./canvas/puppy.jpeg")
    parser.add_argument("--fragments", type=str, default="./fragments/*.png")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--scaling", type=float, default=10.0)
    parser.add_argument("--fragment-size", type=int, nargs=2, metavar=("W", "H"), default=(32, 32))
    parser.add_argument("--target-coverage", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--memory-limit-gb", type=float, default=25.0)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--placement-batch-size", type=int, default=None)
    parser.add_argument("--fragment-block-size", type=int, default=None)
    parser.add_argument("--temp-dir", type=str, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RenderConfig(
        canvas_path=args.canvas,
        fragments_glob=args.fragments,
        output_path=args.output,
        scaling=args.scaling,
        target_coverage=args.target_coverage,
        fragment_size=(int(args.fragment_size[0]), int(args.fragment_size[1])),
        memory_limit_gb=args.memory_limit_gb,
        device_request=args.device,
        chunk_size_original=args.chunk_size,
        placement_batch_size=args.placement_batch_size,
        fragment_block_size=args.fragment_block_size,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
        seed=args.seed,
    )

    summary = run_render(config=config, progress_callback=ConsoleProgress())
    plan = summary.resource_plan

    print(f"Output: {summary.output_path}")
    print(f"Device: {summary.device_label}")
    print(f"Fragments: {summary.fragment_count:,}")
    print(f"Source size: {summary.source_size[0]}x{summary.source_size[1]}")
    print(f"Output size: {summary.output_size[0]}x{summary.output_size[1]}")
    print(f"Elapsed: {summary.elapsed_seconds:.2f}s")
    print(f"Chunk size: {plan.chunk_size_original}")
    print(f"Placement batch size: {plan.placement_batch_size}")
    print(f"Fragment block size: {plan.fragment_block_size}")
    print(f"Working memory budget: {format_bytes(plan.memory_budget.working_bytes)}")


if __name__ == "__main__":
    main()
