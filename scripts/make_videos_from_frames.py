"""Create videos from extracted frame images.

This scans a folder for image frames and creates one video per sequence.

Default behavior (single folder):
- Groups frames by the prefix before "_frame" in the filename.
  Example: "myvideo_frame_000123.jpg" -> output "myvideo.mp4"

Alternative behavior:
- Use --by-subfolder to create one video per immediate subfolder,
  naming the output video after the subfolder.

Requires: opencv-python-headless (already in this repo's requirements).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class _FrameFile:
    path: Path
    sort_key: Tuple[int, str]


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTS


def _extract_last_int(text: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    n = _extract_last_int(path.stem)
    # Put numbered frames first (stable), then non-numbered.
    if n is None:
        return (10**18, path.name.lower())
    return (n, path.name.lower())


def _group_key_from_filename(filename: str, pattern: re.Pattern[str]) -> Optional[str]:
    m = pattern.search(filename)
    if not m:
        return None
    key = m.group("video")
    key = re.sub(r"[\s\-_]+$", "", key)  # trim trailing separators
    return key.strip() or None


def _collect_frames_in_dir(frames_dir: Path) -> List[_FrameFile]:
    frames: List[_FrameFile] = []
    for p in frames_dir.iterdir():
        if _is_image(p):
            frames.append(_FrameFile(path=p, sort_key=_frame_sort_key(p)))
    frames.sort(key=lambda f: f.sort_key)
    return frames


def _write_video(
    frames: List[_FrameFile],
    output_path: Path,
    fps: float,
    fourcc: str,
    resize_to_first: bool,
    overwrite: bool,
) -> None:
    if not frames:
        raise ValueError("No frames provided")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path} (use --overwrite)")

    first = cv2.imread(str(frames[0].path), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0].path}")

    height, width = first.shape[0], first.shape[1]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer for {output_path}. "
            f"Try a different --fourcc (e.g. mp4v, avc1)."
        )

    try:
        for frame in frames:
            img = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read frame: {frame.path}")
            if img.shape[0] != height or img.shape[1] != width:
                if not resize_to_first:
                    raise RuntimeError(
                        f"Frame size mismatch for {frame.path}: "
                        f"got {img.shape[1]}x{img.shape[0]}, expected {width}x{height}. "
                        f"Use --resize-to-first to auto-resize."
                    )
                img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


def _iter_immediate_subdirs(root: Path) -> Iterable[Path]:
    for p in root.iterdir():
        if p.is_dir():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create .mp4 videos from frame images (e.g. *_frame_0001.jpg)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Folder containing frames (or subfolders of frames with --by-subfolder)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder (default: input folder)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="OpenCV fourcc codec (default: mp4v)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Output extension without dot (default: mp4)",
    )
    parser.add_argument(
        "--by-subfolder",
        action="store_true",
        help="Create one video per immediate subfolder (video name = folder name)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"^(?P<video>.+?)_frame",
        help=(
            "Regex used to extract video name from a filename. "
            "Must contain a named group 'video'. Default: '^(?P<video>.+?)_frame'"
        ),
    )
    parser.add_argument(
        "--resize-to-first",
        action="store_true",
        help="Resize all frames to match the first frame size",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing videos",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    output_dir = Path(args.output).expanduser().resolve() if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(args.pattern, re.IGNORECASE)

    jobs: List[Tuple[str, Path, List[_FrameFile]]] = []  # (name, out_path, frames)

    if args.by_subfolder:
        for sub in _iter_immediate_subdirs(input_dir):
            frames = _collect_frames_in_dir(sub)
            if not frames:
                continue
            name = sub.name
            out_path = output_dir / f"{name}.{args.ext.lstrip('.')}"
            jobs.append((name, out_path, frames))
    else:
        grouped: Dict[str, List[_FrameFile]] = {}
        ungrouped: List[Path] = []
        for p in input_dir.iterdir():
            if not _is_image(p):
                continue
            key = _group_key_from_filename(p.name, pattern)
            if key is None:
                ungrouped.append(p)
                continue
            grouped.setdefault(key, []).append(_FrameFile(path=p, sort_key=_frame_sort_key(p)))

        for key, frames in grouped.items():
            frames.sort(key=lambda f: f.sort_key)
            out_path = output_dir / f"{key}.{args.ext.lstrip('.')}"
            jobs.append((key, out_path, frames))

        if ungrouped:
            print(
                f"Warning: {len(ungrouped)} image(s) did not match --pattern and were skipped. "
                f"Example: {ungrouped[0].name}"
            )

    if not jobs:
        print("No frame sequences found.")
        return 0

    print(f"Found {len(jobs)} video(s) to create.")

    for name, out_path, frames in jobs:
        print(f"- {name}: {len(frames)} frame(s) -> {out_path}")
        if args.dry_run:
            continue
        _write_video(
            frames=frames,
            output_path=out_path,
            fps=args.fps,
            fourcc=args.fourcc,
            resize_to_first=bool(args.resize_to_first),
            overwrite=bool(args.overwrite),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
