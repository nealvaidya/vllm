#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect the format/layout of hidden-state safetensors files.

Handles both plain .safetensors and zstd-compressed .safetensors.zst files.
Recursively scans a directory or inspects a single file.

Usage:
    python examples/offline_inference/inspect_hidden_states.py /tmp/hidden_states
    python examples/offline_inference/inspect_hidden_states.py /tmp/hidden_states/req_abc.safetensors
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch


def read_raw_bytes(path: str) -> bytes:
    """Read file bytes, decompressing zstd if needed."""
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".zst"):
        import zstandard as zstd
        data = zstd.ZstdDecompressor().decompress(data)
    return data


def parse_safetensors_header(raw: bytes) -> dict:
    """Parse the safetensors header without loading tensors into memory."""
    if len(raw) < 8:
        raise ValueError("File too small to be safetensors")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header_bytes = raw[8 : 8 + header_len]
    return json.loads(header_bytes)


def inspect_file(path: str) -> None:
    disk_size = os.path.getsize(path)
    compressed = path.endswith(".zst")
    raw = read_raw_bytes(path)
    header = parse_safetensors_header(raw)

    metadata = header.pop("__metadata__", {})
    payload_size = len(raw) - 8 - len(json.dumps(header).encode())

    print(f"\n{'=' * 70}")
    print(f"File: {path}")
    print(f"  Disk size:        {disk_size:,} bytes ({disk_size / 1024:.1f} KB)")
    if compressed:
        ratio = disk_size / len(raw) if len(raw) else 0
        print(f"  Uncompressed:     {len(raw):,} bytes ({len(raw) / 1024:.1f} KB)")
        print(f"  Compression:      zstd ({ratio:.2%} of original)")
    print(f"  Payload size:     ~{payload_size:,} bytes")
    print(f"  Tensor count:     {len(header)}")

    if metadata:
        print(f"  Metadata:         {json.dumps(metadata, indent=4)}")

    print(f"\n  {'Tensor':<25} {'DType':<12} {'Shape':<30} {'Offset range'}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 30} {'-' * 25}")

    for name, info in sorted(header.items()):
        dtype = info.get("dtype", "?")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])
        size = offsets[1] - offsets[0]
        shape_str = "x".join(str(s) for s in shape) if shape else "scalar"
        print(
            f"  {name:<25} {dtype:<12} {shape_str:<30} "
            f"[{offsets[0]:>10} .. {offsets[1]:>10}] ({size:,} B)"
        )

    # Quick sanity: try loading with safetensors to show actual torch dtypes
    try:
        from safetensors import safe_open
        import tempfile

        load_path = path
        if compressed:
            tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
            tmp.write(raw)
            tmp.close()
            load_path = tmp.name

        with safe_open(load_path, "pt") as f:
            print(f"\n  Loaded tensor details:")
            for key in f.keys():
                t: torch.Tensor = f.get_tensor(key)
                print(
                    f"    {key:<25} torch.{t.dtype}  shape={tuple(t.shape)}  "
                    f"numel={t.numel():,}"
                )
                if key == "token_ids":
                    preview = t.flatten()[:20].tolist()
                    print(f"      first tokens: {preview}")
                if key == "hidden_states":
                    print(f"      min={t.min().item():.4f}  max={t.max().item():.4f}  "
                          f"mean={t.mean().item():.4f}  std={t.std().item():.4f}")
                    print(t[:10,:,:1])

        if compressed:
            os.unlink(load_path)

    except ImportError:
        print("\n  (install safetensors + torch for detailed tensor inspection)")


def find_files(root: str) -> list[str]:
    """Find all .safetensors and .safetensors.zst files under root."""
    root_path = Path(root)
    if root_path.is_file():
        return [str(root_path)]
    files = sorted(root_path.rglob("*.safetensors")) + sorted(
        root_path.rglob("*.safetensors.zst")
    )
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect hidden-state safetensors files"
    )
    parser.add_argument(
        "path",
        help="File or directory to inspect",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max files to inspect when scanning a directory (default: 10)",
    )
    args = parser.parse_args()

    files = find_files(args.path)
    if not files:
        print(f"No safetensors files found in {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} safetensors file(s)")
    for f in files[: args.max_files]:
        inspect_file(f)

    if len(files) > args.max_files:
        print(f"\n... and {len(files) - args.max_files} more (use --max-files to show)")


if __name__ == "__main__":
    main()
