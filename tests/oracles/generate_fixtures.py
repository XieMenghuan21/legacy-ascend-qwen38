#!/usr/bin/env python3
"""Generate exact FP16 scatter references and check every untouched bit."""
from pathlib import Path
import json
import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    destination = parser.parse_args().out
    destination.mkdir(exist_ok=True)
    random = np.random.default_rng(1910910)
    source = random.normal(0, 3, size=1024).astype(np.float32)
    source[:16] = np.array([0, -0.0, 1, -1, 0.5 + 2**-12, 0.5 + 3 * 2**-12,
                           1 + 2**-11, 1 + 3 * 2**-11, 65504, -65504,
                           2**-14, -2**-14, 2**-24, -2**-24, 2**-25, -2**-25], dtype=np.float32)
    # Include all possible half bit patterns in untouched cache: preservation
    # must cover signed zero, subnormal, infinity, and NaN payloads as well.
    initial = np.arange(1024 * 256, dtype=np.uint32).astype(np.uint16).view(np.float16)
    source.tofile(destination / "source.f32.bin")
    initial.tofile(destination / "initial.f16.bin")
    converted = source.astype(np.float16)
    cases: list[dict[str, object]] = []
    for position in (0, 1, 7, 15, 16, 17, 31, 63, 127, 128, 239, 240, 254, 255):
        key_index = np.array([position, 0, 0, 0], dtype=np.int64)
        value_index = np.arange(1024, dtype=np.int64) * 256 + position
        for kind, indices in (("k", key_index), ("v", value_index)):
            expected = initial.copy()
            changed = np.zeros(expected.size, dtype=np.bool_)
            if kind == "k":
                offsets = position * 1024 + np.arange(1024)
            else:
                offsets = value_index
            expected[offsets] = converted
            changed[offsets] = True
            assert np.array_equal(expected.view(np.uint16)[~changed], initial.view(np.uint16)[~changed])
            assert np.array_equal(expected[offsets].view(np.uint16), converted.view(np.uint16))

            # Simulate the exact block-level RMW layout and independently prove
            # no two core ownership ranges share a 32-byte cache block.
            actual = initial.copy()
            owned_blocks: set[int] = set()
            for core in range(32):
                start = core * 32
                if kind == "k":
                    actual[position * 1024 + start: position * 1024 + start + 32] = converted[start:start + 32]
                else:
                    for channel in range(start, start + 32):
                        block_start = channel * 256 + position // 16 * 16
                        block_number = block_start // 16
                        assert block_number not in owned_blocks
                        owned_blocks.add(block_number)
                        tile = actual[block_start:block_start + 16].copy()
                        tile[position % 16] = converted[channel]
                        actual[block_start:block_start + 16] = tile
            assert np.array_equal(actual.view(np.uint16), expected.view(np.uint16))
            stem = f"{kind}-p{position}"
            indices.tofile(destination / f"{stem}.i64.bin")
            expected.tofile(destination / f"{stem}.expected.f16.bin")
            cases.append({"kind": kind, "position": position, "source": "source.f32.bin",
                          "initial": "initial.f16.bin", "indices": f"{stem}.i64.bin",
                          "expected": f"{stem}.expected.f16.bin", "comparison": "bit_exact_uint16"})
    (destination / "manifest.json").write_text(json.dumps({"cases": cases, "count": len(cases)}, indent=2) + "\n")
    print(f"Validated and generated {len(cases)} bit-exact full-cache fixtures in {destination}")


if __name__ == "__main__":
    main()
