// SCAO CUDA Kernels
// =================
// Fused low-rank preconditioner application and matrix root utilities.
//
// Compile via setup.py in this directory.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel: fused low-rank preconditioned matmul
//   Computes  U * diag(s) * (U^T * G)  for left preconditioning
//   Dimensions: U (m,k), s (k,), G (m,n) → out (m,n)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void low_rank_precond_mm_kernel(
    const scalar_t* __restrict__ U,       // m x k
    const scalar_t* __restrict__ s,       // k
    const scalar_t* __restrict__ G,       // m x n
    scalar_t* __restrict__ out,           // m x n
    int m, int n, int k
) {
    // Thread: (row in out, col in out)
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 0..m
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // 0..n
    if (row >= m || col >= n) return;

    // out[row, col] = sum_j  U[row,j] * s[j] * (sum_i U[i,j] * G[i,col])
    scalar_t val = 0;
    for (int j = 0; j < k; ++j) {
        scalar_t proj = 0;
        for (int i = 0; i < m; ++i) {
            proj += U[i * k + j] * G[i * n + col];
        }
        val += U[row * k + j] * s[j] * proj;
    }
    out[row * n + col] = val;
}

torch::Tensor low_rank_precond_mm_cuda(
    torch::Tensor U,
    torch::Tensor s,
    torch::Tensor G,
    bool left
) {
    TORCH_CHECK(U.is_cuda(), "U must be on CUDA");
    TORCH_CHECK(G.is_cuda(), "G must be on CUDA");
    int m = G.size(0), n = G.size(1), kk = U.size(1);
    auto out = torch::zeros_like(G);

    const dim3 block(16, 16);
    const dim3 grid((m + 15) / 16, (n + 15) / 16);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(G.scalar_type(), "low_rank_precond_mm", [&] {
        low_rank_precond_mm_kernel<scalar_t><<<grid, block>>>(
            U.data_ptr<scalar_t>(),
            s.data_ptr<scalar_t>(),
            G.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            m, n, kk
        );
    });

    return out;
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("low_rank_precond_mm", &low_rank_precond_mm_cuda,
          "Fused low-rank preconditioned matmul (CUDA)");
}
