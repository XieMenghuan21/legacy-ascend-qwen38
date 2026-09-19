"""Build FP32 RMS_NORM decode kernels for the original Ascend 910.

ABI: input, output, both device FP32 pointers. Epsilon and dimensions are
specialized at compile time. Input can have a padded row stride; output is
contiguous. This operator does NOT apply learned normalization weights.
"""

import argparse
import json
import math
from pathlib import Path


DEFAULT_SHAPES = ((5120, 1, 5120), (128, 48, 128), (256, 24, 512), (256, 4, 256))
EPSILON = 1.0e-6
L2_EPSILON = EPSILON / 128


def build(width: int, rows: int, stride: int, epsilon: float = EPSILON) -> dict[str, object]:
    if width < 64 or width > 8192 or width % 64:
        raise ValueError("width must be a multiple of 64 in [64, 8192]")
    if rows < 1 or rows > 256:
        raise ValueError("rows must be in [1, 256]")
    if stride < width or stride % 8:
        raise ValueError("stride must be >= width and 32-byte aligned")
    if epsilon not in (EPSILON, L2_EPSILON):
        raise ValueError("epsilon must be 1e-6 or 1e-6/128")

    from tbe import tik

    name = f"rms_910_w{width}_r{rows}_s{stride}"
    if epsilon == L2_EPSILON:
        name += "_l2"
    blocks = min(rows, 32)
    chunks = math.ceil(rows / blocks)
    repeats = width // 64
    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    x = kernel.Tensor("float32", (rows * stride,), name="x", scope=tik.scope_gm)
    y = kernel.Tensor("float32", (rows * width,), name="y", scope=tik.scope_gm)

    with kernel.for_range(0, blocks, block_num=blocks) as core:
        xu = kernel.Tensor("float32", (width,), name="xu", scope=tik.scope_ubuf)
        square = kernel.Tensor("float32", (width,), name="square", scope=tik.scope_ubuf)
        reduction = kernel.Tensor("float32", (8,), name="reduction", scope=tik.scope_ubuf)
        # vec_reduce_add uses intermediate per-repeat reductions.
        scratch = kernel.Tensor("float32", (max(128, repeats * 8),), name="scratch", scope=tik.scope_ubuf)
        inverse = kernel.Tensor("float32", (8,), name="inverse", scope=tik.scope_ubuf)
        correction = kernel.Tensor("float32", (8,), name="correction", scope=tik.scope_ubuf)
        scale = kernel.Scalar("float32", name="scale")

        with kernel.for_range(0, chunks) as chunk:
            row = core + chunk * blocks
            with kernel.if_scope(row < rows):
                kernel.data_move(xu, x[row * stride], 0, 1, width // 8, 0, 0)
                kernel.vec_mul(64, square, xu, xu, repeats, 8, 8, 8)
                kernel.vec_reduce_add(64, reduction, square, scratch, repeats, 8)
                kernel.vec_muls(1, reduction, reduction, 1.0 / width, 1, 1, 1)
                kernel.vec_adds(1, reduction, reduction, epsilon, 1, 1, 1)
                kernel.vec_rsqrt(1, inverse, reduction, 1, 1, 1)

                # Original 910's approximate rsqrt needs refinement for FP32.
                # y <- y * (1.5 - 0.5 * x * y * y), twice, stays in FP32.
                for _ in range(2):
                    kernel.vec_mul(1, correction, inverse, inverse, 1, 1, 1, 1)
                    kernel.vec_mul(1, correction, correction, reduction, 1, 1, 1, 1)
                    kernel.vec_muls(1, correction, correction, -0.5, 1, 1, 1)
                    kernel.vec_adds(1, correction, correction, 1.5, 1, 1, 1)
                    kernel.vec_mul(1, inverse, inverse, correction, 1, 1, 1, 1)

                scale.set_as(inverse[0])
                kernel.vec_muls(64, xu, xu, scale, repeats, 8, 8)
                kernel.data_move(y[row * width], xu, 0, 1, width // 8, 0, 0)

    kernel.BuildCCE(kernel_name=name, inputs=[x], outputs=[y])
    return {"name": name, "width": width, "rows": rows, "stride": stride,
            "epsilon": epsilon, "blocks": blocks, "input_dtype": "float32",
            "output_dtype": "float32", "weights_applied": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", metavar="WIDTH,ROWS,STRIDE")
    parser.add_argument("--l2-only", action="store_true", help="Only build GDN q/k 128x16 with epsilon=1e-6/128")
    args = parser.parse_args()
    if args.l2_only:
        if args.shape:
            raise ValueError("--shape and --l2-only are mutually exclusive")
        entries = [build(128, 16, 128, L2_EPSILON)]
        Path("kernel_meta/rms_910_l2_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
        return
    shapes = DEFAULT_SHAPES if not args.shape else [tuple(map(int, value.split(","))) for value in args.shape]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError("--shape expects WIDTH,ROWS,STRIDE")
    entries = [build(*shape) for shape in shapes]
    Path("kernel_meta/rms_910_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
