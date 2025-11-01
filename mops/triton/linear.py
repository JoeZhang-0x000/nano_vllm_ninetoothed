import torch
import triton
import triton.language as tl
from icecream import ic
from .registry import register_triton_op

# `triton.jit` 装饰器表示这是一个 Triton 内核
@triton.jit
def gemm_nt_kernel(
    # 指向矩阵的指针
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # 矩阵的维度
    M, N, K,
    # 矩阵的步长（stride），用于在内存中移动
    # stride_am 表示在 A 矩阵中，从一行移动到下一行需要跳过多少个元素
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # 分块大小，这是性能调优的关键
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BIAS: tl.constexpr,
):
    """
    计算 C = A @ B.T
    A: [M, K]
    B: [N, K] -> B.T: [K, N]
    C: [M, N]
    bias: [N]
    """
    # 1. 计算当前程序实例（Program Instance）负责的块
    # -----------------------------------------------------------
    # 每个 Triton 内核启动时会有一组并行的程序实例
    pid = tl.program_id(axis=0) # 获取当前程序实例的 ID
    
    # 计算总共有多少个程序实例组
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # 将一维的 pid 映射到二维的块分组上
    # GROUP_SIZE_M 是一个调优参数，用于控制块在内存中的局部性
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 计算当前块要加载的内存偏移量
    # ----------------------------------------------------------
    # `tl.arange` 创建一个常量范围，用于表示块内的偏移
    # `[None, :]` 和 `[:, None]` 用于广播，形成二维坐标
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化 A 和 B 的指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 对于 B，因为是 B.T，所以 K 维度是内循环
    # 我们加载 B 的 [N, K] 块
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # 3. 初始化累加器
    # -----------------------------------------------------------
    # 累加器 accumulator 用于存储 C 块的结果，初始化为 0
    # 使用 .to(tl.float32) 是为了更高的精度
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. 主循环：沿 K 维度进行计算和累加
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # -- 加载数据块 --
        # 从 A 加载一个 [BLOCK_SIZE_M, BLOCK_SIZE_K] 的块
        # mask 用于处理边界情况（当 M 或 K 不是块大小的整数倍时）
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # 从 B 加载一个 [BLOCK_SIZE_N, BLOCK_SIZE_K] 的块
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # -- 计算和累加 --
        # 这是 NT-GEMM 的核心：A @ B.T
        # a 的形状是 [M_block, K_block]
        # b 的形状是 [N_block, K_block]，所以 b.T 的形状是 [K_block, N_block]
        accumulator += tl.dot(a, tl.trans(b))
        
        # -- 更新指针，移动到 K 维度的下一个块 --
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. 写回最终结果
    # -----------------------------------------------------------
    # 将累加器的类型转换回 C 的数据类型
    c = accumulator.to(c_ptr.dtype.element_ty)
    
    # 计算 C 块的指针
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # 加载 bias 块
    if BIAS:
        # bias [BN,], c[BM, BN]
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        c += bias[None, :]
    
    # 创建写回的掩码，处理 M 和 N 的边界情况
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@register_triton_op
def linear(a, b, bias=None):
    """
    Python 包装函数，用于启动 Triton 内核
    C = A @ B.T
    A: [M, K]
    B: [N, K]
    """
    # 检查输入张量
    assert a.shape[1] == b.shape[1], "K dimension must match"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, K = a.shape
    N, _ = b.shape
    
    # 分配输出张量 C
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 定义网格（Grid），即启动多少个程序实例
    # grid 是一个元组，这里我们只使用一维的 grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # 启动内核
    gemm_nt_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # 这些是超参数，需要根据 GPU 架构和矩阵形状进行调优
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        # num_warps 和 num_stages 也是重要的调优参数
        num_warps=4,
        num_stages=2,
        BIAS = bias is not None,
    )
    return c

