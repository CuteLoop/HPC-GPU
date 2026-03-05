Below is a **super-brief README** you can drop into your CUDA BLAS mini-project. It defines the core Level 1/2/3 routines (axpy, dot, nrm2, scal, copy, gemv, trsv, gemm, trsm) and states the intended CUDA-facing API + a tiny optimization checklist.

---

## CUDA-BLAS (mini) — README

A small CUDA implementation of **BLAS-like** kernels, organized by BLAS “levels”:

* **Level 1 (O(n))**: vector–vector ops (strided)
* **Level 2 (O(n²))**: matrix–vector ops
* **Level 3 (O(n³))**: matrix–matrix ops (main optimization target: GEMM)

### Conventions

* All routines support **strides**:

  * vectors: `incx`, `incy`
  * matrices: **column-major** layout by default (BLAS-style), with leading dimensions `lda`, `ldb`, `ldc`
* Scalar types: start with `float` (extend later to `double` / complex).
* CUDA launch: `<<<grid, block, shared_bytes, stream>>>` and optionally accept `cudaStream_t stream`.

---

## API (Function Definitions)

### Level 1 — Vector ops

```c
// y <- alpha*x + y
void cublas1_saxpy(int n, float alpha,
                   const float* x, int incx,
                   float* y, int incy);

// returns sum_i x_i * y_i
float cublas1_sdot(int n,
                   const float* x, int incx,
                   const float* y, int incy);

// returns sqrt(sum_i x_i^2)
float cublas1_snrm2(int n, const float* x, int incx);

// x <- alpha * x
void cublas1_sscal(int n, float alpha,
                   float* x, int incx);

// y <- x
void cublas1_scopy(int n,
                   const float* x, int incx,
                   float* y, int incy);
```

### Level 2 — Matrix–vector ops

```c
// y <- alpha*A*x + beta*y
// A is m-by-n (column-major), lda >= m
void cublas2_sgemv(char trans, int m, int n,
                   float alpha,
                   const float* A, int lda,
                   const float* x, int incx,
                   float beta,
                   float* y, int incy);

// Solve T*x = y for x, where T is triangular.
// Overwrites x (or y) depending on your chosen convention.
// T is n-by-n (column-major), lda >= n
void cublas2_strsv(char uplo, char trans, char diag,
                   int n,
                   const float* T, int lda,
                   float* x, int incx);
```

### Level 3 — Matrix–matrix ops

```c
// C <- alpha*op(A)*op(B) + beta*C
// A is m-by-k, B is k-by-n, C is m-by-n (column-major)
void cublas3_sgemm(char transA, char transB,
                   int m, int n, int k,
                   float alpha,
                   const float* A, int lda,
                   const float* B, int ldb,
                   float beta,
                   float* C, int ldc);

// B <- alpha * inv(T) * B   (left-side triangular solve)
// T is n-by-n triangular; B is n-by-m
void cublas3_strsm(char side, char uplo, char trans, char diag,
                   int n, int m,
                   float alpha,
                   const float* T, int lda,
                   float* B, int ldb);
```

---

## Optimization Targets (very short)

### General CUDA knobs

* Prefer **coalesced memory** for contiguous access (special-case `incx=1`, `incy=1`).
* Use **grid-stride loops** for Level 1 kernels.
* Use **warp-level primitives** (`__shfl_down_sync`) for reductions (dot/nrm2).

### GEMV (Level 2)

* One warp (or half-warp) per output element `y_i` often works well.
* Cache `x` via **read-only cache** (or stage a tile into shared memory if reuse is high).

### GEMM (Level 3) — main focus

* **Tiling + shared memory** for A/B tiles, accumulate in registers.
* Tune `(BM, BN, BK)` block sizes; avoid shared bank conflicts.
* Special-case `beta == 0` and `beta == 1`.
* Consider a “packed” path (copy tiles to contiguous buffers) only if striding/lda causes poor locality.

---

## Scope Notes

* This project is for learning/porting: correctness first, then performance.
* Reference behavior is BLAS-like, but the exact interface may evolve (e.g., row-major wrappers, batched GEMM).

---

If you want, paste your repo structure (files + naming) and I’ll tailor this README to your exact layout (e.g., `src/level1/`, `kernels/`, `tests/`, and how you want `trans/uplo/diag` encoded).
