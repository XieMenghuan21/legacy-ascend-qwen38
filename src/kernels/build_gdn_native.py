"""Build fixed-shape FP32 Gated DeltaNet decode kernels for Ascend 910A.

Set GDN_QK_HEADS=16 (native grouped inputs) or 48 (already expanded inputs).
The eight-pointer ABI is state, q, k, v, g, beta, out_state, out.
All inputs are contiguous; g and beta contain 48 floats without padding.
Only B=1, T=1, Hv=48, Dk=Dv=128 are supported. q/k are already normalized.
"""

import os


VALUE_HEADS = 48
DIM = 128
CORES = 24
QK_HEADS = int(os.environ.get("GDN_QK_HEADS", "16"))
if QK_HEADS not in (16, 48):
    raise ValueError("GDN_QK_HEADS must be 16 or 48")
KERNEL_NAME = "gdn_native_h48_q%d" % QK_HEADS


def build_kernel() -> None:
    # Import only after shape validation, so invalid configurations fail clearly.
    from tbe import tik

    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    state = kernel.Tensor("float32", (VALUE_HEADS, DIM, DIM), name="state", scope=tik.scope_gm)
    q = kernel.Tensor("float32", (QK_HEADS, DIM), name="q", scope=tik.scope_gm)
    k = kernel.Tensor("float32", (QK_HEADS, DIM), name="k", scope=tik.scope_gm)
    v = kernel.Tensor("float32", (VALUE_HEADS, DIM), name="v", scope=tik.scope_gm)
    g = kernel.Tensor("float32", (VALUE_HEADS,), name="g", scope=tik.scope_gm)
    beta = kernel.Tensor("float32", (VALUE_HEADS,), name="beta", scope=tik.scope_gm)
    out_state = kernel.Tensor("float32", (VALUE_HEADS, DIM, DIM), name="out_state", scope=tik.scope_gm)
    out = kernel.Tensor("float32", (VALUE_HEADS, DIM), name="out", scope=tik.scope_gm)

    with kernel.for_range(0, CORES, block_num=CORES) as core:
        su = kernel.Tensor("float32", (DIM,), name="su", scope=tik.scope_ubuf)
        qu = kernel.Tensor("float32", (DIM,), name="qu", scope=tik.scope_ubuf)
        ku = kernel.Tensor("float32", (DIM,), name="ku", scope=tik.scope_ubuf)
        vu = kernel.Tensor("float32", (DIM,), name="vu", scope=tik.scope_ubuf)
        gu = kernel.Tensor("float32", (8,), name="gu", scope=tik.scope_ubuf)
        bu = kernel.Tensor("float32", (8,), name="bu", scope=tik.scope_ubuf)
        ou = kernel.Tensor("float32", (DIM,), name="ou", scope=tik.scope_ubuf)
        temp = kernel.Tensor("float32", (DIM,), name="temp", scope=tik.scope_ubuf)
        reduction = kernel.Tensor("float32", (8,), name="reduction", scope=tik.scope_ubuf)
        scratch = kernel.Tensor("float32", (64,), name="scratch", scope=tik.scope_ubuf)
        decay = kernel.Scalar("float32", name="decay")
        gate = kernel.Scalar("float32", name="gate")
        delta = kernel.Scalar("float32", name="delta")
        prediction = kernel.Scalar("float32", name="prediction")
        value = kernel.Scalar("float32", name="value")

        # Each core owns heads core and core+24; no output regions overlap.
        with kernel.for_range(0, VALUE_HEADS // CORES) as part:
            head = core + CORES * part
            qk_head = head % QK_HEADS
            gate_block = (head // 8) * 8
            gate_lane = head % 8
            kernel.data_move(qu, q[qk_head, 0], 0, 1, 16, 0, 0)
            kernel.data_move(ku, k[qk_head, 0], 0, 1, 16, 0, 0)
            kernel.data_move(vu, v[head, 0], 0, 1, 16, 0, 0)

            # DMA transfers whole aligned 32-byte blocks; select this head's lane.
            kernel.data_move(gu, g[gate_block], 0, 1, 1, 0, 0)
            kernel.data_move(bu, beta[gate_block], 0, 1, 1, 0, 0)
            decay.set_as(gu[gate_lane])
            gate.set_as(bu[gate_lane])
            gu[0].set_as(decay)
            kernel.vec_exp(1, gu, gu, 1, 1, 1)
            decay.set_as(gu[0])
            kernel.vec_muls(64, qu, qu, DIM ** -0.5, 2, 8, 8)

            with kernel.for_range(0, DIM) as row:
                kernel.data_move(su, state[head, row, 0], 0, 1, 16, 0, 0)
                kernel.vec_muls(64, su, su, decay, 2, 8, 8)
                kernel.vec_mul(64, temp, su, ku, 2, 8, 8, 8)
                kernel.vec_reduce_add(64, reduction, temp, scratch, 2, 8)
                prediction.set_as(reduction[0])
                value.set_as(vu[row])
                delta.set_as((value - prediction) * gate)
                kernel.vec_muls(64, temp, ku, delta, 2, 8, 8)
                kernel.vec_add(64, su, su, temp, 2, 8, 8, 8)
                kernel.data_move(out_state[head, row, 0], su, 0, 1, 16, 0, 0)
                kernel.vec_mul(64, temp, su, qu, 2, 8, 8, 8)
                kernel.vec_reduce_add(64, reduction, temp, scratch, 2, 8)
                ou[row].set_as(reduction[0])
            kernel.data_move(out[head, 0], ou, 0, 1, 16, 0, 0)

    kernel.BuildCCE(
        kernel_name=KERNEL_NAME,
        inputs=[state, q, k, v, g, beta],
        outputs=[out_state, out],
    )


if __name__ == "__main__":
    build_kernel()
