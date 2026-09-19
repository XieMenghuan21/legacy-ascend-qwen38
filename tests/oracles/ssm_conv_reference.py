"""GGML SSM_CONV reference: x[B,C,T+3], weight[C,4], output[B,T,C]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def ssm_conv(x: FloatArray, weight: FloatArray) -> FloatArray:
    """Four-term depthwise valid cross-correlation; no bias, activation or state update."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[1] != 4:
        raise ValueError("Expected x[B,C,T+3] and weight[C,4]")
    batch, channels, history_tokens = x.shape
    tokens = history_tokens - 3
    if channels != weight.shape[0] or min(batch, channels) <= 0 or tokens < 0:
        raise ValueError("Invalid dimensions")
    if not np.isfinite(x).all() or not np.isfinite(weight).all():
        raise ValueError("Inputs must be finite")
    result_bct = np.zeros((batch, channels, tokens), dtype=np.float32)
    for tap in range(4):
        result_bct += x[:, :, tap:tap + tokens] * weight[None, :, tap:tap + 1]
    return np.ascontiguousarray(result_bct.transpose(0, 2, 1))


def oracle(x: FloatArray, weight: FloatArray) -> FloatArray:
    windows = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), 4, axis=-1)
    result_bct = np.einsum("bctk,ck->bct", windows, weight.astype(np.float64))
    return np.ascontiguousarray(result_bct.transpose(0, 2, 1), dtype=np.float32)


def self_test() -> None:
    # Distinct taps distinguish cross-correlation from reversed convolution.
    x = np.array([[[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]]], dtype=np.float32)
    weight = np.array([[1, 10, 100, 1000], [1, 0, 0, 0]], dtype=np.float32)
    expected = np.array([[[4321, 10], [5432, 20], [6543, 30]]], dtype=np.float32)
    np.testing.assert_array_equal(ssm_conv(x, weight), expected)
    # Appending tokens one at a time must equal prefill when the last 3 values are preserved.
    pref = ssm_conv(x, weight)
    state = x[:, :, :3].copy()
    individual = []
    for token in range(3):
        window = np.concatenate((state, x[:, :, 3 + token:4 + token]), axis=-1)
        individual.append(ssm_conv(window, weight))
        state = window[:, :, -3:].copy()
    np.testing.assert_array_equal(np.concatenate(individual, axis=1), pref)


def generate(directory: Path, tokens: int, channels: int = 10240, batch: int = 1, seed: int = 20260919) -> None:
    if min(tokens, channels, batch) <= 0:
        raise ValueError("Fixture dimensions must be positive")
    self_test()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, channels, tokens + 3)).astype(np.float32)
    weight = (rng.normal(size=(channels, 4)) * 0.25).astype(np.float32)
    expected = ssm_conv(x, weight)
    precise = oracle(x, weight)
    np.testing.assert_allclose(expected, precise, rtol=2e-5, atol=2e-6)
    arrays = {"input_x": x, "weight": weight, "expected_output": expected}
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        array.astype("<f4", copy=False).tofile(directory / (name + ".f32.bin"))
    np.savez(directory / "ssm_conv_vectors.npz", **arrays)
    manifest = {
        "operator": "GGML_OP_SSM_CONV",
        "dtype": "float32 little-endian",
        "x_layout": "[B,C,T+3] C contiguous; time/history contiguous",
        "weight_layout": "[C,4] C contiguous; taps contiguous; no reversal",
        "output_layout": "[B,T,C] C contiguous; channels contiguous",
        "equation": "y[b,t,c] = sum(x[b,c,t+i] * weight[c,i], i=0..3)",
        "includes_silu": False,
        "updates_state": False,
        "seed": seed,
        "x_ggml_ne": [tokens + 3, channels, batch, 1],
        "x_ggml_nb": [4, 4 * (tokens + 3), 4 * (tokens + 3) * channels, 4 * (tokens + 3) * channels * batch],
        "weight_ggml_ne": [4, channels, 1, 1],
        "weight_ggml_nb": [4, 16, 16 * channels, 16 * channels],
        "output_ggml_ne": [channels, tokens, batch, 1],
        "output_ggml_nb": [4, 4 * channels, 4 * channels * tokens, 4 * channels * tokens * batch],
        "arrays": {name: {"shape": list(array.shape), "bytes": array.nbytes} for name, array in arrays.items()},
        "oracle_max_abs_error": float(np.max(np.abs(expected - precise))),
        "verification": "PASS: distinct taps, prefill-vs-decode history recurrence, float64 oracle",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--channels", type=int, default=10240)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    generate(args.out, args.tokens, args.channels, args.batch, args.seed)


if __name__ == "__main__":
    main()
