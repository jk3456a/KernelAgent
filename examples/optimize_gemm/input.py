import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Tiled matrix multiplication kernel with FP32 accumulation.

    Computes C = A @ B using a blocked (tiled) algorithm. The accumulator
    is always FP32 (matching PyTorch's matmul behavior for both BF16 and
    FP32 inputs). For FP32 inputs, tl.dot uses TF32 tensor cores on
    Ampere+, which matches PyTorch's default CUDA matmul precision.

    Fusion analysis:
        The operator pipeline specified by the test is a single matrix
        multiplication (C = A @ B). There are no adjacent operators
        (bias, activation, normalization, etc.) to fuse, so the entire
        computation is performed in one Triton kernel call.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row/column offsets for the output tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers for the A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # FP32 accumulator (regardless of input dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in BLOCK_SIZE_K tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_mask = offs_k[None, :] < k_remaining
        b_mask = offs_k[:, None] < k_remaining
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # tl.dot accumulates into `accumulator` (FP32). For BF16 inputs this
        # is native FP32 accumulation; for FP32 inputs Triton uses TF32
        # tensor cores, matching torch.matmul's default CUDA behavior.
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result, casting back to the output dtype
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def kernel_function(A, B):
    """Wrapper for tiled matrix multiplication C = A @ B with FP32 accumulation.

    Args:
        A: Input tensor of shape (M, K) on CUDA (BF16 or FP32).
        B: Input tensor of shape (K, N) on CUDA (same dtype as A).

    Returns:
        C: Output tensor of shape (M, N), same dtype as A.
    """
    # --- Validation (no compute here) ---
    assert A.is_cuda and B.is_cuda, "Both inputs must be on CUDA"
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D matrices"
    assert A.shape[1] == B.shape[0], (
        f"Incompatible dimensions: A={tuple(A.shape)}, B={tuple(B.shape)}"
    )

    M, K = A.shape
    _, N = B.shape

    # --- Allocation only (no compute) ---
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # --- Fixed configuration (avoids autotune compilation overhead) ---
    # For 2048x2048 matmul, BLOCK_SIZE 128x128x64 with 8 warps and 3 stages
    # is a well-proven high-performance configuration on modern GPUs.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # --- Grid configuration: 1D grid over (M-tiles x N-tiles) ---
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    # --- Launch the Triton kernel (all math happens here) ---
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=3,
        num_warps=8,
    )
    return C