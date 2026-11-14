"""Build standalone resident generation weights on a host with sufficient RAM."""

import argparse
from pathlib import Path

from qwengpu.compiled_tile8 import build_standalone
from qwengpu import resident_generation as gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--generated-tilegrid", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    model, circuit = build_standalone(generate=True, generated_tilegrid=args.generated_tilegrid)
    model.config.qwengpu_layout = gen.TILEGRID_LAYOUT if args.generated_tilegrid else gen.LAYOUT
    model.save_pretrained(args.out)
    print(f"Saved {circuit.n_layers()} circuit layers to {args.out}")


if __name__ == "__main__":
    main()
