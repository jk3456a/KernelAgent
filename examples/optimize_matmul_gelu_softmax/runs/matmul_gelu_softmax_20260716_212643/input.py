import torch
import triton
import triton.language as tl

# ==============================================================================
# FUSION EXPLANATION
# ------------------
# The target operation is a sequence of Linear (MatMul + Bias), GELU, and Softmax.
# We fuse the entire pipeline into a single Triton kernel to minimize memory
# traffic and kernel launch overhead.
#
# 1. MatMul + Bias:
#    - We accumulate the matrix product in FP32 for numerical stability.
#    - The bias is loaded once per output column and added directly to the
#      accumulator in FP32.
# 2. GELU:
#    - We use the exact GELU formulation: 0.5 * x * (1 + erf(x / sqrt(2))).
#    - This is computed in FP32 directly on the matmul accumulator.
# 3. Softmax:
#    - We use an online softmax algorithm to compute the row-wise maximum and
#      sum in a single pass over the N dimension (epilogue subtiling).
#    - First pass: Compute GELU(MatMul + Bias) for all column subtiles, track 
#      running max and sum.
#    - Second pass: Normalize the GELU outputs by the final sum and store in BF16.
# ==============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
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
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    
    # --- PASS 1: Fused MatMul + Bias + GELU + Softmax Max & Sum ---
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    
    for start_n in range(0, num_n_tiles):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Matmul accumulation loop
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        # Reset a_ptrs for the next column tile
        a_ptrs -= tl.cdiv(K, BLOCK_K) * BLOCK_K * stride_ak
        
        # Add bias
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]
        
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt_2 = 0.70710678118
        erf_val = tl.math.erf(acc * inv_sqrt_2)
        gelu_out = 0.5 * acc * (1.0 + erf_val)
        
        # Mask out-of-bounds elements before reduction
        gelu_out = tl.where(mask_n[None, :], gelu_out, -float("inf"))
        
        # Online Softmax: update max and sum
        m_ij = tl.maximum(m_i, tl.max(gelu_out, axis=1))
        alpha = tl.math.exp(m_i - m_ij)
        p = tl.math.exp(gelu_out - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij
        
    # --- PASS 2: Normalize and Store ---
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    
    for start_n in range(0, num_n_tiles):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            
        a_ptrs -= tl.cdiv(K, BLOCK_K) * BLOCK_K * stride_ak
        
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]
        
        inv_sqrt_2 = 0.70710678118
        erf_val = tl.math.erf(acc * inv_sqrt_2)
        gelu_out = 0.5 * acc * (1.0 + erf_val)
        
        # Softmax output
        p = tl.math.exp(gelu_out - m_i[:, None]) / l_i[:, None]
        
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, p.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(x, linear):
    """
    Wrapper function for the fused Linear -> GELU -> Softmax kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        linear (torch.Tensor or torch.nn.Parameter): The weights of the linear layer.
                                  
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features) in BF16.
    """
    # Robustly resolve the weight and bias tensors
    if hasattr(linear, 'weight'):
        w = linear.weight
        bias = linear.bias
    else:
        w = linear
        bias = getattr(linear, 'bias', None)
    
    assert w is not None, "Could not resolve weight tensor"
    
    # PyTorch stores Linear weights as (out_features, in_features).
    # Our kernel expects B as (in_features, out_features), so we need the transpose.
    # To avoid an explicit PyTorch compute call (.t()), we pass the original
    # tensor and swap the stride arguments to the Triton kernel.
    assert w.dim() == 2, "Linear weight must be 2D"
    
    M, K = x.shape
    N, _ = w.shape
    
    assert K == w.shape[1], "Incompatible dimensions between x and weight"
    
    # Allocate bias if it was not provided
    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=w.dtype)
        
    assert bias.dim() == 1, "Linear bias must be 1D"
    assert N == bias.shape[0], "Incompatible dimensions between weight and bias"
    
    # Allocate output tensor
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    
    # Configure grid
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    
    # Launch kernel
    _fused_linear_gelu_softmax_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # Swapped strides for transpose
        out.stride(0), out.stride(1),
    )
    
    return out