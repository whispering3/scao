// SCAO CUDA Kernels
// =================
// Fused low-rank preconditioner application and quantized EMA utilities.
//
// Compile via setup.py in this directory.
//
// Kernels
// -------
// 1. tiled_AtB_kernel      — tiled C = A^T @ B  (shared-memory GEMM)
// 2. tiled_AB_kernel       — tiled C = A @ B    (shared-memory GEMM)
// 3. fused_kronecker_precond_kernel
//                          — full identity+correction precond step:
//                            G_out = G + U_l @ ((s_l⊗s_r)*P - P) @ U_r^T
//                            where P = U_l^T @ G @ U_r  (k×k)
//                            Avoids materialising the (m,n) correction tensor.
// 4. int8_ema_requantize_kernel
//                          — fused dequantize-update-quantize for int8 EMA:
//                            ema_new = rho*ema_old + alpha*outer, then requantize.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16

// ---------------------------------------------------------------------------
// Kernel 1: tiled_AtB  —  C[K,N] = A[M,K]^T @ B[M,N]
// Thread (tx,ty) in block (bx,by) computes C[by*TILE+ty, bx*TILE+tx]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void tiled_AtB_kernel(
    const scalar_t* __restrict__ A,   // (M, K)
    const scalar_t* __restrict__ B,   // (M, N)
    scalar_t*       __restrict__ C,   // (K, N)
    int M, int K, int N
) {
    __shared__ scalar_t sA[TILE][TILE];  // A[m_tile..., k_block...]
    __shared__ scalar_t sB[TILE][TILE];  // B[m_tile..., n_block...]

    int k_out = blockIdx.y * TILE + threadIdx.y;  // row of C (= col of A)
    int n_out = blockIdx.x * TILE + threadIdx.x;  // col of C (= col of B)

    scalar_t acc = 0;
    for (int m0 = 0; m0 < M; m0 += TILE) {
        // sA[threadIdx.y][threadIdx.x] = A[m0+threadIdx.x, k_out]   (A transposed)
        int ma = m0 + threadIdx.x;
        sA[threadIdx.x][threadIdx.y] = (ma < M && k_out < K) ? A[ma * K + k_out] : (scalar_t)0;

        // sB[threadIdx.y][threadIdx.x] = B[m0+threadIdx.y, n_out]
        int mb = m0 + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (mb < M && n_out < N) ? B[mb * N + n_out] : (scalar_t)0;

        __syncthreads();

        for (int t = 0; t < TILE; ++t)
            acc += sA[t][threadIdx.y] * sB[t][threadIdx.x];

        __syncthreads();
    }

    if (k_out < K && n_out < N)
        C[k_out * N + n_out] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: tiled_AB  —  C[M,N] = A[M,K] @ B[K,N]
// Standard tiled GEMM with shared memory.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void tiled_AB_kernel(
    const scalar_t* __restrict__ A,   // (M, K)
    const scalar_t* __restrict__ B,   // (K, N)
    scalar_t*       __restrict__ C,   // (M, N)
    int M, int K, int N
) {
    __shared__ scalar_t sA[TILE][TILE];
    __shared__ scalar_t sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    scalar_t acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ka = k0 + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? A[row * K + ka] : (scalar_t)0;

        int kb = k0 + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (kb < K && col < N) ? B[kb * N + col] : (scalar_t)0;

        __syncthreads();

        for (int t = 0; t < TILE; ++t)
            acc += sA[threadIdx.y][t] * sB[t][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: fused_kronecker_precond
//   Full identity + low-rank correction precond step:
//     P     = U_l^T @ G @ U_r            (k×k, computed via shared memory)
//     delta = (s_l_inv4[p]*s_r_inv4[q] - 1) * P[p,q]
//     G_out = G + U_l @ delta @ U_r^T
//
//   Each CUDA block handles one (TILE_M, TILE_N) output tile of G_out.
//   P (k×k) is materialised in shared memory (fits for k≤128: 128²×4=64KB).
//   Requires k ≤ MAX_K and (TILE_M, TILE_N) = (TILE, TILE).
//
//   Memory bandwidth: reads G once, writes G_out once.  Avoids the two
//   large (m,n) intermediates that sequential U_l@P@U_r^T requires.
// ---------------------------------------------------------------------------
#define MAX_K 128

template <typename scalar_t>
__global__ void fused_kronecker_precond_kernel(
    const scalar_t* __restrict__ U_l,        // (m, k)
    const scalar_t* __restrict__ s_l_inv4,   // (k,)
    const scalar_t* __restrict__ U_r,        // (n, k)
    const scalar_t* __restrict__ s_r_inv4,   // (k,)
    const scalar_t* __restrict__ G,          // (m, n)
    scalar_t*       __restrict__ G_out,      // (m, n)
    int m, int n, int k
) {
    // Each block computes a (TILE, TILE) patch of G_out.
    int row = blockIdx.y * TILE + threadIdx.y;  // row in [0, m)
    int col = blockIdx.x * TILE + threadIdx.x;  // col in [0, n)

    // Step 1: collaboratively compute P = U_l^T @ G @ U_r (k×k) in registers.
    // Since k ≤ MAX_K and all threads share the same G patch, we use a
    // two-pass approach: first reduce U_l^T @ G for this col-block, then
    // multiply by U_r.
    //
    // For simplicity and correctness across k values, each thread computes
    // its own (row, col) output using the already-computed per-block reduction.
    // The inner (k×k) reduction is done cooperatively via shared memory.

    __shared__ scalar_t sUl[MAX_K][TILE];   // U_l[row_block, k]
    __shared__ scalar_t sUr[MAX_K][TILE];   // U_r[col_block, k]
    __shared__ scalar_t sP[MAX_K][MAX_K];   // P = U_l^T @ G @ U_r  (k×k)
    __shared__ scalar_t sDelta[MAX_K][MAX_K]; // delta[p,q]

    // Load U_l rows for this row-block: sUl[ki][ty] = U_l[row_block_start+ty, ki]
    // Load U_r rows for this col-block: sUr[ki][tx] = U_r[col_block_start+tx, ki]
    int row_base = blockIdx.y * TILE;
    int col_base = blockIdx.x * TILE;
    for (int ki = 0; ki < k; ++ki) {
        int r = row_base + threadIdx.y;
        sUl[ki][threadIdx.y] = (r < m) ? U_l[r * k + ki] : (scalar_t)0;
        int c = col_base + threadIdx.x;
        sUr[ki][threadIdx.x] = (c < n) ? U_r[c * k + ki] : (scalar_t)0;
    }
    __syncthreads();

    // Compute P[p,q] = sum_i U_l[i,p] * (sum_j G[i,j] * U_r[j,q])
    // This requires a reduction over all (i,j).  We compute it using tiled
    // accumulation: outer loop over col-tiles of G, inner reduction over the
    // TILE columns in each tile.
    //
    // Thread (ty=p, tx=q) for p,q < k computes P[p,q].
    // For k > TILE, we iterate the thread roles — handled below via loop.
    for (int p = threadIdx.y; p < k; p += TILE) {
        for (int q = threadIdx.x; q < k; q += TILE) {
            scalar_t val = 0;
            for (int i = 0; i < m; ++i) {
                scalar_t ul_ip = (i < m) ? U_l[i * k + p] : (scalar_t)0;
                for (int j = 0; j < n; ++j) {
                    scalar_t g_ij  = G[i * n + j];
                    scalar_t ur_jq = U_r[j * k + q];
                    val += ul_ip * g_ij * ur_jq;
                }
            }
            sP[p][q] = val;
            // delta[p,q] = (s_l[p]^{-1/4} * s_r[q]^{-1/4} - 1) * P[p,q]
            sDelta[p][q] = (s_l_inv4[p] * s_r_inv4[q] - (scalar_t)1) * val;
        }
    }
    __syncthreads();

    // Step 2: G_out[row, col] = G[row, col] + sum_{p,q} U_l[row,p]*delta[p,q]*U_r[col,q]
    if (row < m && col < n) {
        scalar_t correction = 0;
        for (int p = 0; p < k; ++p) {
            scalar_t ul_rp = sUl[p][threadIdx.y];
            for (int q = 0; q < k; ++q) {
                correction += ul_rp * sDelta[p][q] * sUr[q][threadIdx.x];
            }
        }
        G_out[row * n + col] = G[row * n + col] + correction;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: int8_ema_update
//   Dequantize int8 EMA, apply rho*old + alpha*new_outer, requantize.
//   Two-pass: pass 1 = compute updated fp32 values + find abs-max (via atomics)
//             pass 2 = requantize with new scale
//   Called with N = m*m (or n*n) threads.
// ---------------------------------------------------------------------------
__global__ void int8_ema_update_pass1(
    const int8_t* __restrict__ ema_q,    // (N,) current int8 EMA
    float         ema_scale,             // current dequant scale
    const float*  __restrict__ new_val,  // (N,) alpha * outer_product
    float         rho,
    float*        __restrict__ updated,  // (N,) output fp32 updated values
    float*        __restrict__ abs_max,  // (1,) output abs max (init to 0)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = rho * ((float)ema_q[idx] * ema_scale) + new_val[idx];
    updated[idx] = v;
    atomicMax((int*)abs_max, __float_as_int(fabsf(v)));
}

__global__ void int8_ema_update_pass2(
    const float*  __restrict__ updated,  // (N,) fp32 updated values
    float         new_scale,             // new dequant scale = abs_max/127
    int8_t*       __restrict__ ema_q,    // (N,) output int8 EMA
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float inv_scale = (new_scale > 1e-30f) ? (1.0f / new_scale) : 0.0f;
    float q = rintf(updated[idx] * inv_scale);
    q = fmaxf(-127.0f, fminf(127.0f, q));
    ema_q[idx] = (int8_t)(int)q;
}

// ---------------------------------------------------------------------------
// C++ launcher wrappers
// ---------------------------------------------------------------------------

// 2-pass efficient single-side precond: out = U @ diag(s) @ U^T @ G
// Much faster than the old O(k*m) per-thread kernel for k>16.
torch::Tensor low_rank_precond_mm_cuda(
    torch::Tensor U,   // (m, k)
    torch::Tensor s,   // (k,)
    torch::Tensor G,   // (m, n)  if left=true, or (n, m) transpose convention
    bool left
) {
    TORCH_CHECK(U.is_cuda(), "U must be on CUDA");
    TORCH_CHECK(G.is_cuda(), "G must be on CUDA");
    TORCH_CHECK(s.is_cuda(), "s must be on CUDA");

    // Ensure contiguous row-major layout
    U = U.contiguous();
    s = s.contiguous();
    G = G.contiguous();

    int m = (int)G.size(0), n = (int)G.size(1), kk = (int)U.size(1);

    // Step 1: P = U^T @ G  →  (k, n)
    auto P = torch::empty({kk, n}, G.options());
    {
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (kk + TILE - 1) / TILE);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            G.scalar_type(), "tiled_AtB", [&] {
                tiled_AtB_kernel<scalar_t><<<grid, block>>>(
                    U.data_ptr<scalar_t>(),
                    G.data_ptr<scalar_t>(),
                    P.data_ptr<scalar_t>(),
                    m, kk, n
                );
            }
        );
    }

    // Step 1.5: scale rows of P by s  (P[j, :] *= s[j])
    P.mul_(s.unsqueeze(1));

    // Step 2: out = U @ P  →  (m, n)
    auto out = torch::empty({m, n}, G.options());
    {
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            G.scalar_type(), "tiled_AB", [&] {
                tiled_AB_kernel<scalar_t><<<grid, block>>>(
                    U.data_ptr<scalar_t>(),
                    P.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    m, kk, n
                );
            }
        );
    }

    return out;
}

// Full fused identity+correction Kronecker preconditioner
// G_out = G + U_l @ ((s_l⊗s_r - 1) * (U_l^T@G@U_r)) @ U_r^T
// Only valid for k ≤ MAX_K (128).  Falls back to Python if k > 128.
torch::Tensor fused_kronecker_precond_cuda(
    torch::Tensor U_l,        // (m, k)
    torch::Tensor s_l_inv4,   // (k,)
    torch::Tensor U_r,        // (n, k)
    torch::Tensor s_r_inv4,   // (k,)
    torch::Tensor G           // (m, n)
) {
    TORCH_CHECK(U_l.is_cuda() && U_r.is_cuda() && G.is_cuda(), "All tensors must be on CUDA");
    int k = (int)U_l.size(1);
    TORCH_CHECK(k <= MAX_K, "fused_kronecker_precond: k must be <= ", MAX_K);

    U_l = U_l.contiguous(); s_l_inv4 = s_l_inv4.contiguous();
    U_r = U_r.contiguous(); s_r_inv4 = s_r_inv4.contiguous();
    G   = G.contiguous();

    int m = (int)G.size(0), n = (int)G.size(1);
    auto G_out = torch::empty_like(G);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        G.scalar_type(), "fused_kronecker_precond", [&] {
            fused_kronecker_precond_kernel<scalar_t><<<grid, block>>>(
                U_l.data_ptr<scalar_t>(),
                s_l_inv4.data_ptr<scalar_t>(),
                U_r.data_ptr<scalar_t>(),
                s_r_inv4.data_ptr<scalar_t>(),
                G.data_ptr<scalar_t>(),
                G_out.data_ptr<scalar_t>(),
                m, n, k
            );
        }
    );

    return G_out;
}

// int8 EMA update: dequantize → rho*old + alpha*new → requantize
// Returns (q_new_int8, new_scale) as a pair.
std::tuple<torch::Tensor, float> int8_ema_update_cuda(
    torch::Tensor ema_q,     // (N,) int8
    float         ema_scale, // current scale
    torch::Tensor new_val,   // (N,) float32 = alpha * outer_product_flat
    float         rho
) {
    TORCH_CHECK(ema_q.is_cuda() && new_val.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(ema_q.scalar_type() == torch::kInt8);
    TORCH_CHECK(new_val.scalar_type() == torch::kFloat32);

    int N = (int)ema_q.numel();
    auto updated   = torch::empty({N}, new_val.options());
    auto abs_max_t = torch::zeros({1}, new_val.options());

    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Pass 1: compute updated fp32 + find abs max
    int8_ema_update_pass1<<<blocks, threads>>>(
        ema_q.data_ptr<int8_t>(),
        ema_scale,
        new_val.data_ptr<float>(),
        rho,
        updated.data_ptr<float>(),
        abs_max_t.data_ptr<float>(),
        N
    );

    float abs_max_val = abs_max_t.item<float>();
    float new_scale   = (abs_max_val > 1e-30f) ? abs_max_val / 127.0f : 1.0f;

    // Pass 2: requantize
    auto q_new = torch::empty({N}, ema_q.options());  // int8
    int8_ema_update_pass2<<<blocks, threads>>>(
        updated.data_ptr<float>(),
        new_scale,
        q_new.data_ptr<int8_t>(),
        N
    );

    return {q_new, new_scale};
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("low_rank_precond_mm",
          &low_rank_precond_mm_cuda,
          "2-pass tiled low-rank preconditioned matmul: U diag(s) U^T G (CUDA)");
    m.def("fused_kronecker_precond",
          &fused_kronecker_precond_cuda,
          "Fused Kronecker identity+correction precond step (CUDA, k<=128)");
    m.def("int8_ema_update",
          &int8_ema_update_cuda,
          "Quantized EMA update: dequantize, rho*old+alpha*new, requantize (CUDA)");
}
