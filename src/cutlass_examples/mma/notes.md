# GEMM Tutorial
https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md

## 1. Define CTA tiler
```C++
  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)
```

## 2. Tiling global tensors
```C++
  // Get the appropriate blocks for this threadblock
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

## 3. Partitioning
```C++
  // Define thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{},Int<8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{},Int<8>{}));   // (n,k) -> thr_idx
```
Here, 32x8 threads will be used to partition a 128x8 tile of gmem and smem data into a 4x1 subtensor for each thread. 
Each thread needs to handle a 4x1 subtensor.
```C++
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);    // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);    // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);    // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);    // (THR_N,THR_K)
```
The naming convention `tAsA` is pretty typical across CuTe and CUTLASS. This is read as "Partitioning pattern `tA` applied to tensor `sA`". In the next section, we'll see a different partitioner applied to `sA` to produce `tCsA`. By applying the same partitioning pattern, `tA`, to tensors `sA` and `gA`, we preserve the logical consistency of those tensors (checked by the assertions above) where logical elements between the two tensors correspond despite any differences in their data layouts. When used in `cute::copy`, for example, this naming convention let's us lexically verify that the two tensors are using the same partitioning pattern.

With the data partitioned across the threads, every thread can now participate in the copy by writing
```C++
copy(tAgA(_,_,0), tAsA);
```
because every thread owns a different subtensor of the tile that will be copied.

The kernel now has tiles of shared memory copied in from global memory. We now want to create an efficient way to compute and accumulate the matrix product on that tile of shared memory.
```
  // Define thread layouts (static)
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx; m-major
```
This is a m-major 16x16 layout of threads which will be used to partition a 128x128 tile of C-data, resulting in each thread computing its own 8x8 subtensor of `gC`. <br/><br/>


# wgmma_sm90.cu

## TMA operations
