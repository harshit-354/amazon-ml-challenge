"""
ML Challenge: Business Entity Resolution
CLI & Batch Pipeline to Normalize Datasets

Usage:
    python -m src.normalize_dataset \
        --input student_resource/dataset/train/train_source1.tsv \
        --output data/normalized/train_source1_norm.parquet \
        --nrows 50000
"""

import os
import sys
import argparse
import time
import pandas as pd
from typing import Optional

from src.normalization import normalize_dataframe


def process_tsv_file(
    input_path: str,
    output_path: Optional[str] = None,
    chunksize: int = 100000,
    nrows: Optional[int] = None,
    format: str = "tsv",
) -> None:
    """Normalize a dataset file in streaming chunks for memory efficiency."""
    print(f"[*] Processing {input_path}...")
    start_time = time.time()
    
    # Check total rows if small or process chunks
    chunks = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        nrows=nrows,
    )
    
    total_processed = 0
    first_chunk = True
    
    for i, chunk in enumerate(chunks):
        t0 = time.time()
        chunk = normalize_dataframe(chunk)
        total_processed += len(chunk)
        t1 = time.time()
        
        print(f"  Chunk {i+1}: processed {len(chunk)} rows ({len(chunk)/(t1-t0):.0f} rows/sec)")
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            if format == "tsv":
                mode = "w" if first_chunk else "a"
                header = first_chunk
                chunk.to_csv(output_path, sep="\t", index=False, mode=mode, header=header)
            elif format == "parquet":
                # For parquet, save chunk or accumulate
                chunk.to_parquet(
                    output_path.replace(".parquet", f"_part_{i}.parquet"),
                    index=False
                )
        first_chunk = False
        
    elapsed = time.time() - start_time
    print(f"[✓] Finished {total_processed:,} records in {elapsed:.2f}s ({total_processed/elapsed:.0f} rows/s)")


def main():
    parser = argparse.ArgumentParser(description="Normalize Business Entity Resolution Datasets")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input TSV file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path (optional)")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size for streaming processing")
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows to process (for testing)")
    parser.add_argument("--format", type=str, choices=["tsv", "parquet"], default="tsv", help="Output format")
    
    args = parser.parse_args()
    process_tsv_file(
        input_path=args.input,
        output_path=args.output,
        chunksize=args.chunksize,
        nrows=args.nrows,
        format=args.format,
    )


if __name__ == "__main__":
    main()