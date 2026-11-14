"""Build standalone v3 weights on a host with sufficient RAM."""

import argparse
from pathlib import Path

from qwengpu.compiled_tile8 import build_standalone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    model, circuit = build_standalone(generate=True)
    model.save_pretrained(args.out)
    print(f"Saved {circuit.n_layers()} circuit layers to {args.out}")


if __name__ == "__main__":
    main()
