#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"

#define tdn(n) td##n;
#define tdidx(n) tdn(n)
#define td tdidx(__COUNTER__)

using namespace cute;
using TA = cute::half_t;

template <typename T>
class TD;

namespace cute {
  namespace SM90::GMMA {
    using Layout_MN_SW128_Atom_Bits_Sony = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

    template <class Type>
    using Layout_MN_SW128_Atom_Sony = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits_Sony{}));
  }
}

int main(int argc, char** argv) {
  auto bM = cute::Int<128>{};
  auto bN = cute::Int<128>{};
  auto bK = cute::Int< 64>{};
  auto bP = cute::Int<  3>{};  // Pipeline

  constexpr int N = sizeof_bits<TA>::value;
  printf("N: %d\n", N); // N bits

  auto sony_layout = GMMA::Layout_MN_SW128_Atom_Bits_Sony{};
  // TD<decltype(sony_layout)> td;
  /*
  cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>, 
    cute::smem_ptr_flag_bits<1>, 
    cute::Layout<
      cute::tuple<
        cute::C<1024>, 
        cute::C<8> 
      >, 
      cute::tuple<
        cute::C<1>, 
        cute::C<1024> 
      > 
    > 
  >
  */

  printf("sony_layout: ");
  cute::print(sony_layout);
  printf("\n");

  auto sony_layout_a = sony_layout.layout_a();
  // TD<decltype(sony_layout_a)> td;
  // cute::Swizzle<3, 4, 3>

  auto sony_layout_b = sony_layout.layout_b();
  // TD<decltype(sony_layout_b)> td;
  /*
  cute::Layout<
    cute::tuple<
      cute::C<1024>, 
      cute::C<8> 
    >, 
    cute::tuple<
      cute::C<1>,
      cute::C<1024> 
    > 
  >
  */

  auto upcasted_layout_b = upcast<N>(sony_layout_b); // = upcast<N>(sony_layout_b.shape(), sony_layout_b.stride())
  //=================================================================================================
  // What does `upcast<N>(sony_layout_b.shape(), sony_layout_b.stride())` do?
  //=================================================================================================
  /*
  cutlass/include/cute/layout.hpp
  For stride-1 mode, divide size by N. Divide all other strides by N.
  */

  // TD<decltype(upcasted_layout_b)> td;
  /*
  cute::Layout<
    cute::tuple<
      cute::C<64>, 
      cute::C<8> 
    >, 
    cute::tuple<
      cute::C<1>, 
      cute::C<64> 
    > 
  >
  */

  auto upcasted_layout = upcast<N>(sony_layout);
  
  //=================================================================================================
  // What does `upcast<16>(sony_layout)` do?
  //=================================================================================================
  
  // cutlass/include/cute/pointer_flagged.hpp
  /*
  template <int N, class SwizzleFn, int B, class Layout>
  CUTE_HOST_DEVICE constexpr
  auto
  upcast(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
  {
    return composition(layout.layout_a(), smem_ptr_flag_bits<B*N>{}, upcast<N>(layout.layout_b()));
  }
  */

  // cutlass/include/cute/layout_composed.hpp
  /*
  template <class LayoutA,
            class Offset,
            class LayoutB>
  CUTE_HOST_DEVICE constexpr
  auto
  composition(LayoutA const& layoutA,
              Offset  const& offset,
              LayoutB const& layoutB)
  {
    return ComposedLayout<LayoutA, Offset, LayoutB>{layoutA, offset, layoutB};
  }
  */

  // TD<decltype(upcasted_layout)> td;
  /*
  cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>, 
    cute::smem_ptr_flag_bits<16>, 
    cute::Layout<
      cute::tuple<
        cute::C<64>, 
        cute::C<8> 
      >, 
      cute::tuple<
        cute::C<1>, 
        cute::C<64> 
      > 
    > 
  >
  */

  auto layout = decltype(upcasted_layout){};
  auto trg_shape = make_shape(bM,bK,bP);

  auto sA = tile_to_shape(layout, trg_shape);
  // TD<decltype(sA)> td;
  /*
  cute::ComposedLayout<
    cute::Swizzle<3, 4, 3>, 
    cute::smem_ptr_flag_bits<16>,
    cute::Layout<
      cute::tuple<
        cute::tuple<
          cute::C<64>, 
          cute::C<2> 
        >, 
        cute::tuple<
          cute::C<8>, 
          cute::C<8> 
        >, 
        cute::C<3> 
      >, 
      cute::tuple<
        cute::tuple<
          cute::C<1>, 
          cute::C<512>
        >, 
        cute::tuple<
          cute::C<64>, 
          cute::C<1024> 
        >, 
        cute::C<8192> 
      > 
    > 
  >
  */

/*
template <class A, class O, class B,
          class Shape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_shape(ComposedLayout<A,O,B> const& layout,
              Shape                 const& trg_shape,
              ModeOrder             const& ord_shape = {})
{
  return composition(layout.layout_a(), layout.offset(), tile_to_shape(layout.layout_b(), trg_shape, ord_shape));
}
*/

  auto layout_b = layout.layout_b();
  // TD<decltype(layout_b)> td;
  /*
  cute::Layout<
    cute::tuple<
      cute::C<64>, 
      cute::C<8> 
    >, 
    cute::tuple<
      cute::C<1>, 
      cute::C<64> 
    > 
  > 
  */

  auto tiled_layout_b = tile_to_shape(layout_b, trg_shape);
  // TD<decltype(tiled_layout_b)> td;
  /*
  cute::Layout<
    cute::tuple<
      cute::tuple<
        cute::C<64>, 
        cute::C<2> 
      >, 
      cute::tuple<
        cute::C<8>, 
        cute::C<8> 
      >, 
      cute::C<3> 
    >, 
    cute::tuple<
      cute::tuple<
        cute::C<1>, 
        cute::C<512> 
      >, 
      cute::tuple<
        cute::C<64>, 
        cute::C<1024> 
      >, 
      cute::C<8192> 
    > 
  >
  */
}
