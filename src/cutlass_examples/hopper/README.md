# NOTEs
`wgmma_tma_sm90.cu` is copied from [cutlass repo](https://github.com/NVIDIA/cutlass/tree/8cd5bef43a2b0d3f9846b026c271593c6e4a8e8a/examples/cute/tutorial/hopper)<br/>
Git commit: https://github.com/NVIDIA/cutlass/tree/8cd5bef43a2b0d3f9846b026c271593c6e4a8e8a/examples/cute/tutorial/hopper

# About wgmma_tma_sm90.cu
## Majorness
Matrix A[M, K] with **M-major** layout means M-dimension is contiguous, which means Matrix A has column-major layout.<br/>
Leading dimension ldA = M.<br/>

Matrix B[N, K] with **N-major** layout means N-dimension is contiguous, which means Matrix B has column-major layout.<br/>
Leading dimension ldB = N.<br/>

## Update data types of matrix A, matrix B, matrix C
```C++
// Matrix A: BF16
// Matrix B: BF16
// Matrix C: F32

// Define the MMA
TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

## Kernel function template gemm_device
```C++
void gemm_device(ProblemShape, CtaTiler, const TA *, TmaA, const TB *, TmaB, TC *, CStride, TiledMma, Alpha, Beta) [with 
  ProblemShape = cute::tuple<int, int, int>;
  CtaTiler = cute::tuple<cute::C<128>, cute::C<128>, cute::C<64>>;
  TA = cutlass::bfloat16_t;
  SmemLayoutA = cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>, 
    cute::smem_ptr_flag_bits<16>, 
    cute::Layout<
      cute::tuple<
        cute::tuple<cute::C<64>, cute::C<2>>, 
        cute::tuple<cute::C<8>, cute::C<8>>, 
        cute::tuple<cute::C<1>, cute::C<1>>
      >, 
      cute::tuple<
        cute::tuple<cute::C<1>, cute::C<512>>,      
        cute::tuple<cute::C<64>, cute::C<1024>>, 
        cute::tuple<cute::C<0>, cute::C<0>>
      >
    >
  >;
  TmaA = cute::Copy_Atom<
    cute::Copy_Traits<
      cute::SM90_TMA_LOAD, 
      cute::C<8192>, 
      cute::AuxTmaParams<
        cute::tuple<
          cute::ScaledBasis<cute::C<1>, 0>, 
          cute::ScaledBasis<cute::C<1>, 1>
        >, 
        const cute::Layout<
          cute::tuple<cute::C<64>, cute::C<8>>, 
          cute::tuple<
            cute::ScaledBasis<cute::C<1>, 0>, 
            cute::ScaledBasis<cute::C<1>, 1>
          >
        > &, 
        const cute::Swizzle<3, 4, 3> &
      >
    >, 
    cutlass::bfloat16_t
  >;
  TB = cutlass::bfloat16_t;
  SmemLayoutB = cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>, 
    cute::smem_ptr_flag_bits<16>, 
    cute::Layout<
      cute::tuple<
        cute::tuple<cute::C<64>, cute::C<2>>, 
        cute::tuple<cute::C<8>, cute::C<8>>, 
        cute::tuple<cute::C<1>, cute::C<1>>
      >, 
      cute::tuple<
        cute::tuple<cute::C<1>, cute::C<512>>, 
        cute::tuple<cute::C<64>, cute::C<1024>>, 
        cute::tuple<cute::C<0>, cute::C<0>>
      >
    >
  >;
  TmaB = cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<8192>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 1>>, const cute::Layout<cute::tuple<cute::C<64>, cute::C<8>>, cute::tuple<cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 1>>> &, const cute::Swizzle<3, 4, 3> &>>, cutlass::bfloat16_t>;
  TC = float;
  CStride = cute::tuple<cute::C<1>, int>;
  TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>>, cute::Layout<cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, cute::tuple<cute::C<0>, cute::C<0>, cute::C<0>>>, cute::tuple<cute::Underscore, cute::Underscore, cute::Underscore>>;
  Alpha = float;
  Beta = float
]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:208: constexpr cute::GmmaDescriptor cute::SM90::GMMA::make_gmma_desc(const cute::Tensor<TEngine, TLayout> &) [with cute::SM90::GMMA::Major MajorMode = cute::SM90::GMMA::Major::MN; TEngine = cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>>; TLayout = cute::Layout<cute::tuple<cute::C<64>, cute::tuple<cute::C<8>, cute::C<2>>>, cute::tuple<cute::C<1>, cute::tuple<cute::C<64>, cute::C<1024>>>>]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:208: constexpr cute::GmmaDescriptor cute::SM90::GMMA::make_gmma_desc(const cute::Tensor<TEngine, TLayout> &) [with cute::SM90::GMMA::Major MajorMode = cute::SM90::GMMA::Major::MN; TEngine = cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>>; TLayout = cute::Layout<cute::tuple<cute::C<64>, cute::tuple<cute::C<8>, cute::C<2>>>, cute::tuple<cute::C<1>, cute::tuple<cute::C<64>, cute::C<1024>>>>]

mma: [with T = cute::TiledMMA<cute::MMA_Atom<cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>>, cute::Layout<cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, cute::tuple<cute::C<0>, cute::C<0>, cute::C<0>>>, cute::tuple<cute::Underscore, cute::Underscore, cute::Underscore>>

tCrA: [with T = cute::Tensor<cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, cute::Layout<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>>>

tCrB: [with T = cute::Tensor<cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>, cute::Layout<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>>>

tCrC: [with T = cute::Tensor<cute::ArrayEngine<float, 128UL>, cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>, cute::C<2>, cute::C<2>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::C<32>, cute::C<64>>>>

../../../cutlass/include/cute/algorithm/gemm.hpp:428: Dispatch [5]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with 
MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; 
TD = cute::ArrayEngine<float, 128UL>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>, cute::C<2>, cute::C<2>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::C<32>, cute::C<64>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::tuple<cute::C<0>, cute::C<64>, cute::C<256>>>; 
TC = cute::ArrayEngine<float, 128UL>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>, cute::C<2>, cute::C<2>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::C<32>, cute::C<64>>>; 
void *<anonymous> = (void *)nullptr
K = 4

../../../cutlass/include/cute/algorithm/gemm.hpp:296: Dispatch [4]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with 
MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; 
TD = cute::ArrayEngine<float, 128UL>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>, cute::C<2>, cute::C<2>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::C<32>, cute::C<64>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>, cute::C<2>>, cute::tuple<cute::C<0>, cute::C<64>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>, cute::C<2>>, cute::tuple<cute::C<0>, cute::C<64>>>; 
TC = cute::ArrayEngine<float, 128UL>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>, cute::C<2>, cute::C<2>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>, cute::C<32>, cute::C<64>>>; 
void *<anonymous> = (void *)nullptr]
M = 2
N = 2

../../../cutlass/include/cute/algorithm/gemm.hpp:197: Dispatch [1]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with 
MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; 
TD = cute::ViewEngine<float *>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TC = cute::ViewEngine<const float *>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
void *<anonymous> = (void *)nullptr]

../../../cutlass/include/cute/atom/mma_atom.hpp:105: constexpr void cute::MMA_Atom<cute::MMA_Traits<MMAOperation, Args...>>::call(cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) const [with 
TD = cute::ViewEngine<float *>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
MMAOperation = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:432: constexpr void cute::SM90::GMMA::mma_unpack(const cute::MMA_Traits<MMA_Op, MMA_Args...> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with 
MMA_Op = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; 
TD = cute::ViewEngine<float *>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TC = cute::ViewEngine<const float *>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>]

../../../cutlass/include/cute/algorithm/gemm.hpp:197: Dispatch [1]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with 
MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; 
TD = cute::ViewEngine<float *>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TC = cute::ViewEngine<const float *>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
void *<anonymous> = (void *)nullptr]

../../../cutlass/include/cute/atom/mma_atom.hpp:105: constexpr void cute::MMA_Atom<cute::MMA_Traits<MMAOperation, Args...>>::call(cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) const [with 
TD = cute::ViewEngine<float *>; 
DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; 
BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; 
TC = cute::ViewEngine<const float *>; 
CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; 
MMAOperation = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:432: constexpr void cute::SM90::GMMA::mma_unpack(const cute::MMA_Traits<MMA_Op, MMA_Args...> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with MMA_Op = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>]

../../../cutlass/include/cute/algorithm/gemm.hpp:197: Dispatch [1]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; void *<anonymous> = (void *)nullptr]

../../../cutlass/include/cute/atom/mma_atom.hpp:105: constexpr void cute::MMA_Atom<cute::MMA_Traits<MMAOperation, Args...>>::call(cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) const [with TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; MMAOperation = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:432: constexpr void cute::SM90::GMMA::mma_unpack(const cute::MMA_Traits<MMA_Op, MMA_Args...> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with MMA_Op = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>]

../../../cutlass/include/cute/algorithm/gemm.hpp:197: Dispatch [1]: void cute::gemm(const cute::MMA_Atom<MMA> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with MMA = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; void *<anonymous> = (void *)nullptr]

../../../cutlass/include/cute/atom/mma_atom.hpp:105: constexpr void cute::MMA_Atom<cute::MMA_Traits<MMAOperation, Args...>>::call(cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) const [with TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; MMAOperation = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>]

../../../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:432: constexpr void cute::SM90::GMMA::mma_unpack(const cute::MMA_Traits<MMA_Op, MMA_Args...> &, cute::Tensor<TD, DLayout> &, const cute::Tensor<TA, ALayout> &, const cute::Tensor<TB, BLayout> &, const cute::Tensor<TC, CLayout> &) [with MMA_Op = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::Major::MN, cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One>; TD = cute::ViewEngine<float *>; DLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>; TA = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; ALayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TB = cute::ViewEngine<cute::SM90::GMMA::DescriptorIterator>; BLayout = cute::Layout<cute::tuple<cute::C<1>>, cute::tuple<cute::C<0>>>; TC = cute::ViewEngine<const float *>; CLayout = cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2>, cute::C<8>>>, cute::tuple<cute::tuple<cute::C<1>, cute::C<2>, cute::C<4>>>>]
```

## TMA
### Creating CUtensorMap object
Call stack:
```C++
cute::detail::make_tma_copy_desc<
  cutlass::bfloat16_t, 
  cute::ViewEngine<cutlass::bfloat16_t const*>, 
  cute::Layout<
    cute::tuple<int, int>, 
    cute::tuple<cute::C<1>, int> 
  >, 
  cute::tuple<cute::C<64>, cute::C<8> >, 
  cute::tuple<
    cute::ScaledBasis<cute::C<1>, 0>, 
    cute::ScaledBasis<cute::C<1>, 1> 
  >, 
  3, 
  4, 
  3
> (/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:936)

cute::detail::make_tma_copy_atom<cutlass::bfloat16_t, cute::SM90_TMA_LOAD, cute::ViewEngine<cutlass::bfloat16_t const*>, cute::Layout<cute::tuple<int, int>, cute::tuple<cute::C<1>, int> >, cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::C<0u>, cute::Layout<cute::tuple<cute::tuple<cute::C<64>, cute::C<2> >, cute::tuple<cute::C<8>, cute::C<8> > >, cute::tuple<cute::tuple<cute::C<1>, cute::C<512> >, cute::tuple<cute::C<64>, cute::C<1024> > > > >, cute::tuple<cute::C<128>, cute::C<64> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 1> > > (/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1145)

cute::make_tma_atom<void, cute::SM90_TMA_LOAD, cute::ViewEngine<cutlass::bfloat16_t const*>, cute::Layout<cute::tuple<int, int>, cute::tuple<cute::C<1>, int> >, cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::C<0u>, cute::Layout<cute::tuple<cute::tuple<cute::C<64>, cute::C<2> >, cute::tuple<cute::C<8>, cute::C<8> > >, cute::tuple<cute::tuple<cute::C<1>, cute::C<512> >, cute::tuple<cute::C<64>, cute::C<1024> > > > >, cute::tuple<cute::C<128>, cute::C<64> >, cute::C<1> > (/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1378)

gemm_nt<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, float> (/data05/home/son.nguyen/workspace/learncuda/src/cutlass_examples/hopper/wgmma_tma_sm90.cu:335)

gemm<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, float> (/data05/home/son.nguyen/workspace/learncuda/src/cutlass_examples/hopper/wgmma_tma_sm90.cu:471)

main (/data05/home/son.nguyen/workspace/learncuda/src/cutlass_examples/hopper/wgmma_tma_sm90.cu:608)
```

Calls `cuTensorMapEncodeTiled`:
```C++
// 
// Alias of CUtensorMap
// 
// /data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/arch/copy_sm90_desc.hpp
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif

// 
// Arguments
// 
// /data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,                    // TmaDescriptor tma_desc{};
        tma_format,                   // CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        tma_dim,                      // 2
        gmem_address,                 // 0x155327e00000
        gmem_prob_shape.data(),       // [128, 64, 1, 1, 1]
        gmem_prob_stride.data() + 1,  // [256, 0, 0, 0] gmem_prob_stride[0] implicitly 1
        smem_box_shape.data(),        // [64, 8, 1, 1, 1]
        smem_box_stride.data(),       // [ 1, 1, 1, 1, 1]
        tma_interleave,               // CU_TENSOR_MAP_INTERLEAVE_NONE
        smem_swizzle,                 // CU_TENSOR_MAP_SWIZZLE_128B
        tma_l2Promotion,              // CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        tma_oobFill);                 // CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
```

### TMA Load
```C++
// 
// tma_transaction_bytes
// 
// The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
constexpr int tx_bytes_a = sizeof(make_tensor_like(tensor<0>(tAsA)));
constexpr int tx_bytes_b = sizeof(make_tensor_like(tensor<0>(tBsB)));
PRINT_INT_VALUE("tx_bytes_a", tx_bytes_a); // 128 * 64 * 2 bytes
PRINT_INT_VALUE("tx_bytes_b", tx_bytes_b); // 128 * 64 * 2 bytes
constexpr int tma_transaction_bytes = tx_bytes_a + tx_bytes_b;


//
// cute::copy
//
auto tmp_tma_a = tma_a.with(producer_mbar[pipe]);
PRINT_EXPR_TYPE("tma_a.with(producer_mbar[pipe])", tmp_tma_a);
PRINT_EXPR_TYPE("tAgA(_,k_tile)", tAgA(_,k_tile));
PRINT_EXPR_TYPE("tAsA(_,pipe)", tAsA(_,pipe));
cute::copy(tmp_tma_a, tAgA(_,k_tile), tAsA(_,pipe));

tma_a.with(producer_mbar[pipe]): T = cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD_OP, cute::C<8192>>, cutlass::bfloat16_t>

tAgA: T = cute::Tensor<
  cute::ViewEngine<cute::ArithmeticTupleIterator<cute::ArithmeticTuple<unsigned int, cute::C<0>>>>, 
  cute::Layout<
    cute::tuple<
      cute::tuple<
        cute::tuple<cute::C<64>, cute::C<8>>, 
        cute::tuple<cute::C<2>, cute::C<8>>
      >, 
      int
    >, 
    cute::tuple<
      cute::tuple<
        cute::tuple<
          cute::ScaledBasis<cute::C<1>, 0>, 
          cute::ScaledBasis<cute::C<1>, 1>
        >, 
        cute::tuple<
          cute::ScaledBasis<cute::C<64>, 0>, 
          cute::ScaledBasis<cute::C<8>, 1>
        >
      >, 
      cute::ScaledBasis<cute::C<64>, 1>
    >
  >
>

tAsA: T = cute::Tensor<
  cute::ViewEngine<cute::swizzle_ptr<cute::Swizzle<3, 4, 3>, cute::smem_ptr<cutlass::bfloat16_t *>>>, 
  cute::Layout<
    cute::tuple<
      cute::tuple<cute::C<512>, cute::C<16>>, 
      cute::tuple<cute::C<1>, cute::C<1>>
    >, 
    cute::tuple<
      cute::tuple<cute::C<1>, cute::C<512>>, 
      cute::tuple<cute::C<0>, cute::C<0>>
    >
  >
>

// cute::copy is defined here
// /data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/algorithm/copy.hpp
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
     Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  // First call to cute::copy will be dispatched to 
  copy_atom.call(src, dst);

  // Omit for brevity
}


// copy_atom T = cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD_OP, cute::C<8192>>, cutlass::bfloat16_t>
// copy_atom.call(src, dst) is defined here
// /data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_atom.hpp
template <class SEngine, class SLayout,
          class DEngine, class DLayout>
CUTE_HOST_DEVICE
void
call(Tensor<SEngine,SLayout> const& src,
      Tensor<DEngine,DLayout>      & dst) const
{
  // First call goes to 
  return copy(*this, tensor<0>(src), tensor<0>(dst));
  
  // Next calls go to
  return copy_unpack(static_cast<Traits const&>(*this), src, dst);
}

// 
// copy_unpack is a friend function defined here
// 
// /data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp
// Utility for unpacking TMA_LOAD arguments into a CopyOp
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack
{
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    // Omit for brevity
  }
}
```

## Matrix Descriptors
### union GmmaDescriptor
/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/arch/mma_sm90_desc.hpp
```C++
union GmmaDescriptor
{
  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  CUTE_HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
```

### make_gmma_desc<Major::MN>
/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp
```C++
/**
* ///////////////////////////////
* // make_gmma_desc<Major::MN> //
* ///////////////////////////////
* Each GmmaDescriptor Major-MN describes a canonical layout of the form
*
* LayoutType::INTERLEAVE   : Swizzle<0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
* LayoutType::B32          : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
* LayoutType::B64          : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
* LayoutType::B128         : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
*
* where
*   T  : sizeof(uint128_t) / sizeof(value_type)
*   m  : integer in [1,16] corresponding to GMMA shape
*   k  : integer in [1,32] corresponding to GMMA shape
*   SBO: stride byte offset
*   LBO: leading byte offset
*
* See GMMA::Layout_MN_XXX_Atom<value_type> for building canonical GmmaDescriptor Major-MN layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_MN_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_gmma_desc<Major::MN> for appropriate value_type.
*
* //////////////////////////////
* // make_gmma_desc<Major::K> //
* //////////////////////////////
* Each GmmaDescriptor Major-K describes a canonical layout of the form
*
* LayoutType::INTERLEAVE : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
* LayoutType::B32        : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
* LayoutType::B64        : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
* LayoutType::B128       : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
*
* See GMMA::Layout_K_XXX_Atom<value_type> for building canonical GmmaDescriptor Major-K layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_K_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_gmma_desc<Major::K> for appropriate value_type.
*/
template <Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
GmmaDescriptor
make_gmma_desc(Tensor<TEngine,TLayout> const& tensor)
{
}
```

### DescriptorIterator
/data05/home/son.nguyen/workspace/learncuda/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp
```C++
struct DescriptorIterator
{
  using reference    = GmmaDescriptor;
  using element_type = GmmaDescriptor;
  using value_type   = GmmaDescriptor;

  GmmaDescriptor desc_;

  // Dereference returns the GmmaDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new GmmaDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    // Use 32bit calculation rather than 64 bit calculation as we only update the part of desc
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return { ret };
  }
};
```