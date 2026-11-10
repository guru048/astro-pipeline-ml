"""
Generate a synthetic astronomical catalogue and save as parquet.

Usage
-----
python scripts/generate_catalogue.py --n-sources 1000000 --output data/catalogue.parquet
"""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.catalogue_generator import generate_catalogue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sources", type=int, default=1_000_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--field",     type=float, default=10.0)
    parser.add_argument("--output",    default="data/catalogue.parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Generating {args.n_sources:,} sources ...")
    cat = generate_catalogue(n=args.n_sources, seed=args.seed, field_size_deg=args.field)
    cat.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}  ({len(cat):,} rows, {os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
