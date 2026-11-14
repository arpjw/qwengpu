"""Render the v4 release graphic from qualified CUDA reports.

Run with ``uv run --no-project --with pillow python scripts/render_v4_validation.py REPORTS OUT``.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("install Arial or DejaVu Sans to render the chart")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    inherited = read(args.reports, "inherited-merged.json")["cases"]
    groups = (read(args.reports, "interleaved-merged.json")["cases"]
              + read(args.reports, "sequential-merged.json")["cases"])
    text = read(args.reports, "text.json")
    matrix = read(args.reports, "matmul16-single-launch.json")
    for case in inherited + groups:
        if not all(value for key, value in case.items() if key.endswith("_exact")):
            raise ValueError(f"nonexact fixture: {case['fixture']}")
    if not matrix["exact"] or matrix["generation_calls"] != 1:
        raise ValueError("the single-launch matrix is not exact")
    if text["merged_chip_tokens"] != 0:
        raise ValueError("native text emitted GPU token IDs")
    cells = sum(len(row) for row in matrix["matrix"])
    if (len(inherited), len(groups), len(text["records"]), cells) != (27, 8, 24, 256):
        raise ValueError("unexpected v4 report shape")
    rows = [
        ("Inherited tile fixtures", len(inherited), 27, "#47dbc7"),
        ("Workgroup fixtures", len(groups), 8, "#47dbc7"),
        ("Native first tokens", text["next_top1_same"], 24, "#a59af7"),
        ("Native continuations", text["greedy_identical"], 24, "#a59af7"),
        ("One-launch matrix cells", cells, 256, "#47dbc7"),
    ]
    canvas = Image.new("RGB", (1120, 570), "#0d1425")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 30), "Merged Qwen3.5-9B v4", fill="#e9edff", font=font(35, True))
    draw.text((42, 82), "Exact results on frozen CUDA suites", fill="#9babcb", font=font(23))
    for index, (label, value, total, color) in enumerate(rows):
        y = 148 + index * 76
        draw.text((42, y - 3), label, fill="#e9edff", font=font(22, True))
        draw.rounded_rectangle((374, y, 944, y + 24), radius=12, fill="#344365")
        width = round(570 * value / total)
        draw.rounded_rectangle((374, y, 374 + width, y + 24), radius=12, fill=color)
        draw.text((970, y - 4), f"{value} / {total}", fill=color, font=font(23, True))
    draw.text((42, 530), "The 16 x 16 matrix uses one generation call. Apple MPS is a separate gate.",
              fill="#9babcb", font=font(18))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)


if __name__ == "__main__":
    main()
