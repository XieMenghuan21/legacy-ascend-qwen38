"""Fused F32 partial RoPE for legacy 910: H=4/24, D=256, n_rot=64, T=1.

ABI pointers: x[H,256], sine[64], cosine[64], output[H,256], all F32.
The existing IMRoPE cache generator has already selected position axes and
computed trigonometric values. Only its first 32 sine/cosine entries are read.
Each head owns one core; launch exactly H blocks. Tail 192 channels are copied
as raw F32 storage, without arithmetic. In-place x/output is also supported.
"""


def build(heads: int) -> None:
    from tbe import tik

    if heads not in (4, 24):
        raise ValueError("Only Qwen3.8 head counts 4 and 24 are supported")
    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    x = kernel.Tensor("float32", (heads, 256), name="x", scope=tik.scope_gm)
    sine = kernel.Tensor("float32", (64,), name="sine", scope=tik.scope_gm)
    cosine = kernel.Tensor("float32", (64,), name="cosine", scope=tik.scope_gm)
    output = kernel.Tensor("float32", (heads, 256), name="output", scope=tik.scope_gm)

    with kernel.for_range(0, heads, block_num=heads) as head:
        values = kernel.Tensor("float32", (256,), name="values", scope=tik.scope_ubuf)
        sin_ub = kernel.Tensor("float32", (32,), name="sin_ub", scope=tik.scope_ubuf)
        cos_ub = kernel.Tensor("float32", (32,), name="cos_ub", scope=tik.scope_ubuf)
        p0 = kernel.Tensor("float32", (32,), name="p0", scope=tik.scope_ubuf)
        p1 = kernel.Tensor("float32", (32,), name="p1", scope=tik.scope_ubuf)
        p2 = kernel.Tensor("float32", (32,), name="p2", scope=tik.scope_ubuf)
        p3 = kernel.Tensor("float32", (32,), name="p3", scope=tik.scope_ubuf)
        kernel.data_move(values, x[head, 0], 0, 1, 32, 0, 0)
        kernel.data_move(sin_ub, sine, 0, 1, 4, 0, 0)
        kernel.data_move(cos_ub, cosine, 0, 1, 4, 0, 0)

        # Preserve composite-path arithmetic order: four separate products,
        # followed by subtract/add. No fused multiply-add or half conversion.
        kernel.vec_mul(32, p0, values, cos_ub, 1, 8, 8, 8)
        kernel.vec_mul(32, p1, values[32], sin_ub, 1, 8, 8, 8)
        kernel.vec_mul(32, p2, values, sin_ub, 1, 8, 8, 8)
        kernel.vec_mul(32, p3, values[32], cos_ub, 1, 8, 8, 8)
        kernel.vec_sub(32, values, p0, p1, 1, 8, 8, 8)
        kernel.vec_add(32, values[32], p2, p3, 1, 8, 8, 8)
        kernel.data_move(output[head, 0], values, 0, 1, 32, 0, 0)

    kernel.BuildCCE(kernel_name=f"rope_910_h{heads}_t1", inputs=[x, sine, cosine], outputs=[output])


if __name__ == "__main__":
    build(4)
    build(24)
