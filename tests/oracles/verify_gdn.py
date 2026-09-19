"""Compare a device kernel's float32 raw outputs against GDN reference vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(expected_path: Path, actual_path: Path, rtol: float, atol: float) -> dict[str, float | int | bool | None]:
    expected = np.fromfile(expected_path, dtype="<f4")
    actual = np.fromfile(actual_path, dtype="<f4")
    if expected.size != actual.size:
        raise ValueError(f"{actual_path.name}: expected {expected.size} float32 elements, got {actual.size}")
    finite = bool(np.isfinite(actual).all())
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    permitted = atol + rtol * np.abs(expected)
    return {
        "passed": finite and bool(np.all(error <= permitted)),
        "finite": finite,
        "elements": int(actual.size),
        "max_abs_error": float(error.max()) if finite else None,
        "rms_error": float(np.sqrt(np.mean(error**2))) if finite else None,
        "max_relative_error_above_1e_6": float(np.max(error / np.maximum(np.abs(expected), 1e-6))) if finite else None,
        "failed_elements": int(np.count_nonzero((error > permitted) | ~np.isfinite(actual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--actual-output", type=Path, required=True)
    parser.add_argument("--actual-state", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    results = {
        "output": compare(args.vectors / "expected_output.f32.bin", args.actual_output, args.rtol, args.atol),
        "state": compare(args.vectors / "expected_state_vk.f32.bin", args.actual_state, args.rtol, args.atol),
        "rtol": args.rtol,
        "atol": args.atol,
    }
    serialized = json.dumps(results, indent=2, allow_nan=False) + "\n"
    print(serialized, end="")
    if args.report:
        args.report.write_text(serialized, encoding="utf-8")
    raise SystemExit(0 if results["output"]["passed"] and results["state"]["passed"] else 1)


if __name__ == "__main__":
    main()
