"""Build a fixed-shape FP32 four-tap SSM_CONV decode kernel for Ascend 910A.

ABI: x[10240,4], weight[10240,4], out[10240], all contiguous device FP32.
Only B=1 and T=1 are supported. Tap order is preserved; no SiLU/state update.
Launch ssm_conv_910_t1 with 32 blocks, each computing 320 channels.
"""


CHANNELS = 10240
TAPS = 4
CORES = 32
CHANNELS_PER_CORE = CHANNELS // CORES
VALUES_PER_CORE = CHANNELS_PER_CORE * TAPS
KERNEL_NAME = "ssm_conv_910_t1"


def build_kernel() -> None:
    from tbe import tik

    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    x = kernel.Tensor("float32", (CHANNELS, TAPS), name="x", scope=tik.scope_gm)
    weight = kernel.Tensor("float32", (CHANNELS, TAPS), name="weight", scope=tik.scope_gm)
    out = kernel.Tensor("float32", (CHANNELS,), name="out", scope=tik.scope_gm)

    with kernel.for_range(0, CORES, block_num=CORES) as core:
        xu = kernel.Tensor("float32", (VALUES_PER_CORE,), name="xu", scope=tik.scope_ubuf)
        wu = kernel.Tensor("float32", (VALUES_PER_CORE,), name="wu", scope=tik.scope_ubuf)
        products = kernel.Tensor("float32", (VALUES_PER_CORE,), name="products", scope=tik.scope_ubuf)
        ou = kernel.Tensor("float32", (CHANNELS_PER_CORE,), name="ou", scope=tik.scope_ubuf)
        accumulator = kernel.Scalar("float32", name="accumulator")
        term = kernel.Scalar("float32", name="term")
        first_channel = core * CHANNELS_PER_CORE

        # Each input tile is 1,280 floats, exactly 160 aligned DMA blocks.
        kernel.data_move(xu, x[first_channel, 0], 0, 1, VALUES_PER_CORE // 8, 0, 0)
        kernel.data_move(wu, weight[first_channel, 0], 0, 1, VALUES_PER_CORE // 8, 0, 0)
        kernel.vec_mul(64, products, xu, wu, VALUES_PER_CORE // 64, 8, 8, 8)

        # Keep the reference tap accumulation order and FP32 rounding.
        with kernel.for_range(0, CHANNELS_PER_CORE) as channel:
            offset = channel * TAPS
            accumulator.set_as(products[offset])
            term.set_as(products[offset + 1])
            accumulator.set_as(accumulator + term)
            term.set_as(products[offset + 2])
            accumulator.set_as(accumulator + term)
            term.set_as(products[offset + 3])
            accumulator.set_as(accumulator + term)
            ou[channel].set_as(accumulator)

        kernel.data_move(out[first_channel], ou, 0, 1, CHANNELS_PER_CORE // 8, 0, 0)

    kernel.BuildCCE(kernel_name=KERNEL_NAME, inputs=[x, weight], outputs=[out])


if __name__ == "__main__":
    build_kernel()
