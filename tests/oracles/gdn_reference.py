"""Single-token Gated DeltaNet reference for the local llama.cpp Qwen3.6 path.

State uses [batch, value_heads, value_dim, key_dim], with key_dim contiguous.
Inputs q/k are already L2 normalized; g_log is log decay, beta is sigmoid output.
Head mapping intentionally matches this llama.cpp tree: hv % num_key_heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def normalize_qk(x: FloatArray, epsilon: float = 1e-6) -> FloatArray:
    """Matches models.h: rms_norm(x, epsilon / D) / sqrt(D)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    x32 = np.asarray(x, dtype=np.float32)
    return x32 / np.sqrt(np.sum(x32 * x32, axis=-1, keepdims=True) + np.float32(epsilon))


def _validated_inputs(
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    g_log: FloatArray,
    beta: FloatArray,
    state_vk: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    inputs = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in (q, k, v, g_log, beta, state_vk))
    q32, k32, v32, g32, b32, s32 = inputs
    if q32.ndim != 3 or k32.shape != q32.shape or v32.ndim != 3:
        raise ValueError("q/k must be equal [B,Hk,Dk] tensors; v must have shape [B,Hv,Dv]")
    batch, key_heads, key_dim = q32.shape
    value_batch, value_heads, value_dim = v32.shape
    if min(batch, key_heads, key_dim, value_heads, value_dim) <= 0:
        raise ValueError("all dimensions must be positive")
    if batch != value_batch or value_heads % key_heads:
        raise ValueError("batch dimensions must match and Hv must be divisible by Hk")
    if key_dim != value_dim:
        raise ValueError("this llama.cpp implementation requires Dk == Dv")
    if g32.shape != (batch, value_heads) or b32.shape != g32.shape:
        raise ValueError("scalar GDN g_log/beta must have shape [B,Hv]")
    if s32.shape != (batch, value_heads, value_dim, key_dim):
        raise ValueError("state must have shape [B,Hv,Dv,Dk]")
    if not all(np.isfinite(x).all() for x in inputs):
        raise ValueError("all inputs must be finite")
    return q32, k32, v32, g32, b32, s32


def gdn_decode(
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    g_log: FloatArray,
    beta: FloatArray,
    state_vk: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return output [B,Hv,Dv], updated state [B,Hv,Dv,Dk], all float32.

    Inputs are not mutated. Decay and beta transforms are outside the kernel
    except exp(g_log), which is part of llama.cpp's GDN operator.
    """
    q32, k32, v32, g32, b32, s32 = _validated_inputs(q, k, v, g_log, beta, state_vk)
    key_heads, key_dim = q32.shape[1:]
    value_heads = v32.shape[1]
    head_index = np.arange(value_heads) % key_heads
    qh, kh = q32[:, head_index, :], k32[:, head_index, :]
    decayed = s32 * np.exp(g32)[..., None, None]
    prediction = np.einsum("bhvk,bhk->bhv", decayed, kh, optimize=False)
    delta = (v32 - prediction) * b32[..., None]
    new_state = decayed + delta[..., :, None] * kh[..., None, :]
    output = np.einsum("bhvk,bhk->bhv", new_state, qh, optimize=False)
    output *= np.float32(1.0 / np.sqrt(key_dim))
    return output, new_state


def scalar_oracle(
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    g_log: FloatArray,
    beta: FloatArray,
    state_vk: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Independent float64 oracle using conceptual S[key,value] matrices."""
    q32, k32, v32, g32, b32, s32 = _validated_inputs(q, k, v, g_log, beta, state_vk)
    output = np.empty_like(v32)
    new_state = np.empty_like(s32)
    for batch in range(v32.shape[0]):
        for head in range(v32.shape[1]):
            kh = k32[batch, head % k32.shape[1]].astype(np.float64)
            qh = q32[batch, head % q32.shape[1]].astype(np.float64)
            conceptual = s32[batch, head].T.astype(np.float64)
            conceptual *= np.exp(float(g32[batch, head]))
            delta = float(b32[batch, head]) * (v32[batch, head].astype(np.float64) - kh @ conceptual)
            conceptual += np.outer(kh, delta)
            new_state[batch, head] = conceptual.T
            output[batch, head] = (qh @ conceptual) / np.sqrt(qh.size)
    return output, new_state


def make_inputs(batch: int, key_heads: int, value_heads: int, dim: int, seed: int) -> dict[str, FloatArray]:
    if min(batch, key_heads, value_heads, dim) <= 0 or value_heads % key_heads:
        raise ValueError("dimensions must be positive and Hv divisible by Hk")
    rng = np.random.default_rng(seed)
    q = normalize_qk(rng.normal(size=(batch, key_heads, dim)).astype(np.float32))
    k = normalize_qk(rng.normal(size=(batch, key_heads, dim)).astype(np.float32))
    return {
        "q": q,
        "k": k,
        "v": rng.normal(size=(batch, value_heads, dim)).astype(np.float32),
        "g_log": -rng.uniform(0.001, 2.0, size=(batch, value_heads)).astype(np.float32),
        "beta": rng.uniform(0.001, 0.999, size=(batch, value_heads)).astype(np.float32),
        "state_vk": (rng.normal(size=(batch, value_heads, dim, dim)) * 0.125).astype(np.float32),
    }


def self_test() -> dict[str, float | int | str]:
    errors: list[float] = []
    for shape in ((1, 2, 4, 8), (2, 3, 6, 32), (1, 16, 32, 128)):
        values = make_inputs(*shape, seed=20260919)
        output, state = gdn_decode(**values)
        oracle_output, oracle_state = scalar_oracle(**values)
        for actual, expected in ((output, oracle_output), (state, oracle_state)):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            errors.append(float(np.max(np.abs(actual - expected))))

        # beta=0 makes this pure exponential forgetting.
        zero_beta = values | {"beta": np.zeros_like(values["beta"])}
        _, forgotten = gdn_decode(**zero_beta)
        np.testing.assert_array_equal(forgotten, values["state_vk"] * np.exp(values["g_log"])[..., None, None])

        # No decay and no correction must preserve state exactly.
        unchanged = zero_beta | {"g_log": np.zeros_like(values["g_log"])}
        _, preserved = gdn_decode(**unchanged)
        np.testing.assert_array_equal(preserved, values["state_vk"])

        # k=0 means no state correction; q=0 means identically zero output.
        zeros = values | {"q": np.zeros_like(values["q"]), "k": np.zeros_like(values["k"])}
        zero_output, pure_decay = gdn_decode(**zeros)
        np.testing.assert_array_equal(zero_output, np.zeros_like(zero_output))
        np.testing.assert_array_equal(pure_decay, forgotten)

        # Vanishing gate removes all dependence on the previous state.
        hard_reset = values | {"g_log": np.full_like(values["g_log"], -1000.0)}
        reset_o, reset_s = gdn_decode(**hard_reset)
        alternate_o, alternate_s = gdn_decode(**(hard_reset | {"state_vk": np.zeros_like(values["state_vk"])}))
        np.testing.assert_array_equal(reset_o, alternate_o)
        np.testing.assert_array_equal(reset_s, alternate_s)

    # Multiple decode steps catch orientation errors hidden by zero-state tests.
    state = make_inputs(1, 4, 8, 32, 7)["state_vk"]
    oracle_state = state.copy()
    for index in range(128):
        values = make_inputs(1, 4, 8, 32, index)
        output, state = gdn_decode(**(values | {"state_vk": state}))
        expected_output, oracle_state = scalar_oracle(**(values | {"state_vk": oracle_state}))
        np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(state, oracle_state, rtol=2e-5, atol=2e-6)
    return {"status": "PASS", "shape_cases": 3, "recurrent_steps": 128, "max_abs_error_single_step": max(errors)}


def write_vectors(directory: Path, batch: int, key_heads: int, value_heads: int, dim: int, seed: int, steps: int = 1) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    values = make_inputs(batch, key_heads, value_heads, dim, seed)
    state = values["state_vk"].copy()
    oracle_state = state.copy()
    for _ in range(steps):
        output, state = gdn_decode(**(values | {"state_vk": state}))
        oracle_output, oracle_state = scalar_oracle(**(values | {"state_vk": oracle_state}))
        np.testing.assert_allclose(output, oracle_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(state, oracle_state, rtol=2e-5, atol=2e-6)
    head_index = np.arange(value_heads) % key_heads
    g_padded = np.zeros((batch, value_heads, 8), dtype=np.float32)
    beta_padded = np.zeros_like(g_padded)
    g_padded[..., 0] = values["g_log"]
    beta_padded[..., 0] = values["beta"]
    arrays = values | {
        "q_expanded": np.ascontiguousarray(values["q"][:, head_index, :]),
        "k_expanded": np.ascontiguousarray(values["k"][:, head_index, :]),
        "g_padded": g_padded,
        "beta_padded": beta_padded,
        "expected_output": output,
        "expected_state_vk": state,
    }
    np.savez(directory / "gdn_vectors.npz", **arrays)
    for name, data in arrays.items():
        data.astype("<f4", copy=False).tofile(directory / f"{name}.f32.bin")
    manifest = {
        "format": "little-endian float32, C contiguous",
        "state_layout": "[batch, value_head, value_dim, key_dim]",
        "head_mapping": "value_head % key_heads (local llama.cpp internal ordering)",
        "gate_contract": "g_log is log-decay; kernel computes exp(g_log); beta is already sigmoid",
        "qk_contract": "already L2-normalized; output is scaled by 1/sqrt(key_dim)",
        "seed": seed,
        "recurrent_steps": steps,
        "recurrence_contract": "same q/k/v/g/beta each step; update state after every step; expected outputs are final step",
        "arrays": {name: {"shape": list(data.shape), "bytes": data.nbytes} for name, data in arrays.items()},
        "self_test": self_test(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.out is None:
        print(json.dumps(self_test(), indent=2))
    else:
        write_vectors(args.out, args.batch, args.key_heads, args.value_heads, args.dim, args.seed, args.steps)


if __name__ == "__main__":
    main()
