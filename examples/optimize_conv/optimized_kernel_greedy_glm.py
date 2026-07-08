import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 16, 'BLOCK_OW': 8}, num_stages=4, num_warps=8),
    ],
    # C_IN is a tl.constexpr, so it is not in the runtime args list.
    # Including it in the key causes an IndexError in the autotuner.
    # Different C_IN values automatically trigger recompilation.
    key=['C_out', 'H', 'W'],
)
@triton.jit
def _fused_conv_relu_bias_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, C_out, H, W, H_out, W_out,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_woc, stride_wkh, stride_wkw, stride_wcin,
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    C_IN: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_n = tl.program_id(2)

    num_ow_tiles = tl.cdiv(W_out, BLOCK_OW)
    pid_ow = pid_spatial % num_ow_tiles
    pid_oh = pid_spatial // num_ow_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < C_out

    # Explicitly annotate as constexpr so tl.zeros, tl.reshape, and tl.arange
    # accept it as a compile-time shape element.
    BLOCK_HW: tl.constexpr = BLOCK_OH * BLOCK_OW

    # Fused stage 1: load conv bias and initialize accumulator with it.
    # This fuses the bias-add of the convolution into the accumulator init,
    # avoiding a separate pass over the output.
    conv_bias = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_OC, BLOCK_HW], dtype=tl.float32) + conv_bias[:, None]

    # Fused stage 2: 3x3 convolution (9 kernel positions).
    # Weight is pre-permuted to (C_out, KH, KW, C_in) and accessed via
    # block pointers for coalesced loads. Input x is in channels-last layout.
    for kh in range(3):
        for kw in range(3):
            w_block_ptr = tl.make_block_ptr(
                base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                shape=(C_out, C_IN),
                strides=(stride_woc, stride_wcin),
                offsets=(pid_oc * BLOCK_OC, 0),
                block_shape=(BLOCK_OC, C_IN),
                order=(1, 0),
            )
            w = tl.load(w_block_ptr, boundary_check=(0, 1))

            x_block_ptr = tl.make_block_ptr(
                base=x_ptr + pid_n * stride_xn,
                shape=(H, W * C_IN),
                strides=(stride_xh, 1),
                offsets=(pid_oh * BLOCK_OH + kh, (pid_ow * BLOCK_OW + kw) * C_IN),
                block_shape=(BLOCK_OH, BLOCK_OW * C_IN),
                order=(1, 0),
            )
            x_val = tl.load(x_block_ptr, boundary_check=(0, 1))
            x_flat = tl.reshape(x_val, (BLOCK_HW, C_IN))

            acc = tl.dot(w, tl.trans(x_flat), acc)

    # Fused stage 3: ReLU activation (fused with conv output, no separate pass)
    acc = tl.maximum(acc, 0.0)

    # Fused stage 4: add extra bias (fused with ReLU output, no separate pass)
    extra_bias = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc = acc + extra_bias[:, None]

    # Fused stage 5: store output (channels-last layout).
    # Transpose accumulator from (BLOCK_OC, BLOCK_HW) to (BLOCK_HW, BLOCK_OC)
    # so the spatial dimension is contiguous for coalesced stores.
    acc_t = tl.trans(acc)

    oh_flat = tl.arange(0, BLOCK_HW) // BLOCK_OW
    ow_flat = tl.arange(0, BLOCK_HW) % BLOCK_OW
    oh_val = pid_oh * BLOCK_OH + oh_flat
    ow_val = pid_ow * BLOCK_OW + ow_flat
    mask_hw = (oh_val < H_out) & (ow_val < W_out)

    out_ptrs = (
        out_ptr + pid_n * stride_on
        + oh_val[:, None] * stride_oh
        + ow_val[:, None] * stride_ow
        + offs_oc[None, :] * stride_oc
    )
    out_mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptrs, acc_t.to(out_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(x, conv_weight, conv_bias, extra_bias):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = conv_weight.shape
    assert KH == 3 and KW == 3
    H_out = H - KH + 1
    W_out = W - KW + 1

    orig_dtype = x.dtype
    x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)

    w_bf16_orig = conv_weight if conv_weight.dtype == torch.bfloat16 else conv_weight.to(torch.bfloat16)
    w_bf16 = w_bf16_orig.permute(0, 2, 3, 1).contiguous()

    conv_bias_bf16 = conv_bias if conv_bias.dtype == torch.bfloat16 else conv_bias.to(torch.bfloat16)
    extra_bias_bf16 = extra_bias if extra_bias.dtype == torch.bfloat16 else extra_bias.to(torch.bfloat16)

    x_cl = x_bf16.to(memory_format=torch.channels_last)

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=orig_dtype).to(memory_format=torch.channels_last)

    grid = lambda META: (
        triton.cdiv(C_out, META['BLOCK_OC']),
        triton.cdiv(H_out, META['BLOCK_OH']) * triton.cdiv(W_out, META['BLOCK_OW']),
        N,
    )

    _fused_conv_relu_bias_kernel[grid](
        x_cl, w_bf16, conv_bias_bf16, extra_bias_bf16, out,
        N, C_out, H, W, H_out, W_out,
        x_cl.stride(0), x_cl.stride(2), x_cl.stride(3), x_cl.stride(1),
        w_bf16.stride(0), w_bf16.stride(1), w_bf16.stride(2), w_bf16.stride(3),
        out.stride(0), out.stride(2), out.stride(3), out.stride(1),
        C_IN=C_in,
    )

    return out