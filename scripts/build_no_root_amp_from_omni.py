"""Write no-root AMP clips by dropping root velocity columns from omni clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Convert 62-col omni AMP txt files to 56-col no-root files.")
parser.add_argument(
    "--input_root",
    type=Path,
    default=Path("booster_assets/motions/K1/motion_amp_expert/omni"),
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path("booster_assets/motions/K1/motion_amp_expert"),
)
parser.add_argument("--categories", nargs="+", default=["backward"])
args = parser.parse_args()


def convert_file(src: Path, dst: Path) -> None:
    with src.open() as f:
        data = json.load(f)

    frames = data["Frames"]
    if not frames:
        raise ValueError(f"{src}: no frames")
    width = len(frames[0])
    if width == 56:
        new_frames = frames
    elif width == 62:
        new_frames = [frame[:56] for frame in frames]
    else:
        raise ValueError(f"{src}: expected 56 or 62 columns, got {width}")

    data["Frames"] = new_frames
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"{src} ({width}) -> {dst} (56)")


def main() -> None:
    for category in args.categories:
        src_dir = args.input_root / category
        if not src_dir.is_dir():
            raise FileNotFoundError(src_dir)
        for src in sorted(src_dir.glob("*.txt")):
            convert_file(src, args.output_root / category / src.name)


if __name__ == "__main__":
    main()
