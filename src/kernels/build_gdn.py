"""Compile a fixed-shape FP32 Gated DeltaNet decode kernel for legacy Ascend 910.

Set GDN_HEADS=32 (default) or GDN_HEADS=48 before compiling.
State layout: [value heads, 128 value channels, 128 key channels].
q/k are already L2-normalized and expanded to value heads by the caller.
g contains log-decay; beta is the sigmoid gate, both padded to 8 floats/head.
"""
import os

H = int(os.environ.get('GDN_HEADS', '32'))
if H not in (32, 48):
    raise ValueError('GDN_HEADS must be 32 or 48')
D = 128
CORES = 32 if H == 32 else 24
KERNEL_NAME = 'gdn_decode_910' if H == 32 else 'gdn_decode_910_h48'

from tbe import tik
kernel = tik.Tik(tik.Dprofile('v100', 'cloud'))
state = kernel.Tensor('float32', (H, D, D), name='state', scope=tik.scope_gm)
q = kernel.Tensor('float32', (H, D), name='q', scope=tik.scope_gm)
k = kernel.Tensor('float32', (H, D), name='k', scope=tik.scope_gm)
v = kernel.Tensor('float32', (H, D), name='v', scope=tik.scope_gm)
g = kernel.Tensor('float32', (H, 8), name='g', scope=tik.scope_gm)
beta = kernel.Tensor('float32', (H, 8), name='beta', scope=tik.scope_gm)
out_state = kernel.Tensor('float32', (H, D, D), name='out_state', scope=tik.scope_gm)
out = kernel.Tensor('float32', (H, D), name='out', scope=tik.scope_gm)
with kernel.for_range(0, CORES, block_num=CORES) as core:
    su = kernel.Tensor('float32', (D,), name='su', scope=tik.scope_ubuf)
    qu = kernel.Tensor('float32', (D,), name='qu', scope=tik.scope_ubuf)
    ku = kernel.Tensor('float32', (D,), name='ku', scope=tik.scope_ubuf)
    vu = kernel.Tensor('float32', (D,), name='vu', scope=tik.scope_ubuf)
    gu = kernel.Tensor('float32', (8,), name='gu', scope=tik.scope_ubuf)
    bu = kernel.Tensor('float32', (8,), name='bu', scope=tik.scope_ubuf)
    ou = kernel.Tensor('float32', (D,), name='ou', scope=tik.scope_ubuf)
    temp = kernel.Tensor('float32', (D,), name='temp', scope=tik.scope_ubuf)
    reduction = kernel.Tensor('float32', (8,), name='reduction', scope=tik.scope_ubuf)
    scratch = kernel.Tensor('float32', (64,), name='scratch', scope=tik.scope_ubuf)
    decay = kernel.Scalar('float32', name='decay')
    gate = kernel.Scalar('float32', name='gate')
    delta = kernel.Scalar('float32', name='delta')
    prediction = kernel.Scalar('float32', name='prediction')
    value = kernel.Scalar('float32', name='value')
    # 48 heads are evenly split across 24 cores; each core reuses its UB.
    with kernel.for_range(0, H // CORES) as part:
        head = core + CORES * part
        kernel.data_move(qu, q[head, 0], 0, 1, 16, 0, 0)
        kernel.data_move(ku, k[head, 0], 0, 1, 16, 0, 0)
        kernel.data_move(vu, v[head, 0], 0, 1, 16, 0, 0)
        kernel.data_move(gu, g[head, 0], 0, 1, 1, 0, 0)
        kernel.data_move(bu, beta[head, 0], 0, 1, 1, 0, 0)
        kernel.vec_exp(8, gu, gu, 1, 1, 1)
        decay.set_as(gu[0])
        gate.set_as(bu[0])
        kernel.vec_muls(64, qu, qu, D ** -0.5, 2, 8, 8)
        with kernel.for_range(0, D) as row:
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
kernel.BuildCCE(kernel_name=KERNEL_NAME, inputs=[state, q, k, v, g, beta], outputs=[out_state, out])
