"""Fixed one-token FP32 -> FP16 KV updates for legacy Ascend 910.

K ABI: x[1024] f32, index[4] i64 (only index[0] used), out[256,1024] f16.
V ABI: x[1024] f32, index[1024] i64, cache[1024,256] f16, out[1024,256] f16.
V cache and out MUST point to the same allocation. index[j] MUST equal
j*256+index[0]. A caller provenance marker is required before dispatch.
Both kernels launch 32 blocks, each owning 32 channels.
"""

CHANNELS = 1024
CONTEXT = 256
CORES = 32
PER_CORE = CHANNELS // CORES
HALF_PER_DMA = 16


def build_key() -> None:
    from tbe import tik

    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    x = kernel.Tensor("float32", (CHANNELS,), name="x", scope=tik.scope_gm)
    index = kernel.Tensor("int64", (4,), name="index", scope=tik.scope_gm)
    out = kernel.Tensor("float16", (CONTEXT, CHANNELS), name="out", scope=tik.scope_gm)
    with kernel.for_range(0, CORES, block_num=CORES) as core:
        xu = kernel.Tensor("float32", (PER_CORE,), name="xu", scope=tik.scope_ubuf)
        yu = kernel.Tensor("float16", (PER_CORE,), name="yu", scope=tik.scope_ubuf)
        iu = kernel.Tensor("int64", (4,), name="iu", scope=tik.scope_ubuf)
        position = kernel.Scalar("int64", name="position")
        kernel.data_move(iu, index, 0, 1, 1, 0, 0)
        position.set_as(iu[0])
        with kernel.if_scope(tik.all(position >= 0, position < CONTEXT)):
            kernel.data_move(xu, x[core * PER_CORE], 0, 1, PER_CORE // 8, 0, 0)
            kernel.vec_conv(PER_CORE, "none", yu, xu, 1, 4, 8)
            kernel.data_move(out[position, core * PER_CORE], yu, 0, 1, PER_CORE // HALF_PER_DMA, 0, 0)
    kernel.BuildCCE(kernel_name="setrows_k_910_t1", inputs=[x, index], outputs=[out])


def build_value() -> None:
    from tbe import tik

    kernel = tik.Tik(tik.Dprofile("v100", "cloud"))
    x = kernel.Tensor("float32", (CHANNELS,), name="x", scope=tik.scope_gm)
    index = kernel.Tensor("int64", (CHANNELS,), name="index", scope=tik.scope_gm)
    cache = kernel.Tensor("float16", (CHANNELS, CONTEXT), name="cache", scope=tik.scope_gm)
    out = kernel.Tensor("float16", (CHANNELS, CONTEXT), name="out", scope=tik.scope_gm)
    with kernel.for_range(0, CORES, block_num=CORES) as core:
        xu = kernel.Tensor("float32", (PER_CORE,), name="xu", scope=tik.scope_ubuf)
        yu = kernel.Tensor("float16", (PER_CORE,), name="yu", scope=tik.scope_ubuf)
        iu = kernel.Tensor("int64", (4,), name="iu", scope=tik.scope_ubuf)
        tile = kernel.Tensor("float16", (PER_CORE * HALF_PER_DMA,), name="tile", scope=tik.scope_ubuf)
        position = kernel.Scalar("int64", name="position")
        aligned = kernel.Scalar("int64", name="aligned")
        lane = kernel.Scalar("int64", name="lane")
        value = kernel.Scalar("float16", name="value")
        kernel.data_move(iu, index, 0, 1, 1, 0, 0)
        position.set_as(iu[0])
        with kernel.if_scope(tik.all(position >= 0, position < CONTEXT)):
            aligned.set_as(position // HALF_PER_DMA * HALF_PER_DMA)
            lane.set_as(position % HALF_PER_DMA)
            kernel.data_move(xu, x[core * PER_CORE], 0, 1, PER_CORE // 8, 0, 0)
            kernel.vec_conv(PER_CORE, "none", yu, xu, 1, 4, 8)
            # One strided DMA gathers the aligned 16-half cache segment from
            # each of this core's 32 channels. No other core owns these blocks.
            kernel.data_move(tile, cache[core * PER_CORE, aligned], 0, PER_CORE, 1,
                             CONTEXT // HALF_PER_DMA - 1, 0)
            with kernel.for_range(0, PER_CORE) as channel:
                value.set_as(yu[channel])
                tile[channel * HALF_PER_DMA + lane].set_as(value)
            kernel.data_move(out[core * PER_CORE, aligned], tile, 0, PER_CORE, 1,
                             0, CONTEXT // HALF_PER_DMA - 1)
    kernel.BuildCCE(kernel_name="setrows_v_910_t1", inputs=[x, index, cache], outputs=[out])


if __name__ == "__main__":
    build_key()
    build_value()
