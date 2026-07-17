import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_gelu_softmax_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    inv_sqrt_2 = 0.70710678118
    
    # Pass 1: Compute GEMM + GELU, write to HBM, and compute online softmax stats
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=k_mask[None, :] & mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]
        
        gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt_2))
        gelu_out = tl.where(mask_n[None, :], gelu_out, -float("inf"))
        
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, gelu_out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])
        
        m_ij = tl.maximum(m_i, tl.max(gelu_out, axis=1))
        m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp(m_i - m_ij_safe)
        p = tl.math.exp(gelu_out - m_ij_safe[:, None])
        p = tl.where(m_ij[:, None] == -float("inf"), 0.0, p)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    # Pass 2: Read GELU from HBM and apply softmax normalization
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        gelu_out = tl.load(out_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=-float("inf")).to(tl.float32)
        
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        p = tl.math.exp(gelu_out - m_i_safe[:, None])
        
        l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
        p = p / l_i_safe[:, None]
        
        tl.store(out_ptrs, p.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(x, linear):
    if hasattr(linear, 'weight'):
        w = linear.weight
        bias = linear.bias
    else:
        w = linear
        bias = getattr(linear, 'bias', None)
    
    assert w is not None, "Could not resolve weight tensor"
    assert w.dim() == 2, "Linear weight must be 2D"
    
    M, K = x.shape
    N, _ = w.shape
    
    assert K == w.shape[1], "Incompatible dimensions between x and weight"
    
    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=w.dtype)
        
    assert bias.dim() == 1, "Linear bias must be 1D"
    assert N == bias.shape[0], "Incompatible dimensions between weight and bias"
    
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    
    _fused_linear_gelu_softmax_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
    )
    
    return out